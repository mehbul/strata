//! OpenAI-compatible HTTP API for the strata engine, plus the telemetry
//! endpoints that back the web UI and the static UI bundle itself.
//!
//! Endpoints the UI calls (see web/src/api.ts):
//!   GET  /v1/models            -> model list for the sidebar picker
//!   POST /v1/chat/completions  -> SSE stream of OpenAI chunks (stream:true)
//!   GET  /health               -> scheduler / tiers / hwinfo panel
//!   GET  /profile              -> profiling tab (per-turn phase breakdown)
//!   GET  /experts              -> expert grid (simulated)

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::HeaderValue,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

const KV_SLOTS: usize = 4;
const MAX_QUEUE: usize = 32;
const QUEUE_TIMEOUT_S: u64 = 120;
const PROFILE_KEEP: usize = 32;
/// Ceiling on the SSE reassembly buffer. Frames are small, so this is only ever
/// reached by an upstream that is not framing at all - without it such a stream
/// would grow the buffer until the process died.
const MAX_STREAM_LINE: usize = 4 * 1024 * 1024;

// ---------------------------------------------------------------- hardware

pub struct Hw {
    pub cores: usize,
    pub cpu: String,
    pub gpu: String,
    pub gpus: usize,
    pub vram_total_gb: f32,
}

/// Installed and available host memory, in GB.
pub fn host_ram_gb() -> (f32, f32) {
    winmem::ram_gb()
}

#[cfg(windows)]
mod winmem {
    #[repr(C)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }
    extern "system" {
        pub fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    /// (total_gb, available_gb) straight from kernel32 - no extra crate.
    pub fn ram_gb() -> (f32, f32) {
        const GB: f32 = 1024.0 * 1024.0 * 1024.0;
        unsafe {
            let mut m: MemoryStatusEx = std::mem::zeroed();
            m.length = std::mem::size_of::<MemoryStatusEx>() as u32;
            if GlobalMemoryStatusEx(&mut m) == 0 {
                return (0.0, 0.0);
            }
            (m.total_phys as f32 / GB, m.avail_phys as f32 / GB)
        }
    }
}

#[cfg(not(windows))]
mod winmem {
    pub fn ram_gb() -> (f32, f32) {
        let read = |key: &str| -> f32 {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with(key))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|v| v.parse::<f32>().ok())
                })
                .unwrap_or(0.0)
                / (1024.0 * 1024.0)
        };
        (read("MemTotal:"), read("MemAvailable:"))
    }
}

fn cpu_name() -> String {
    #[cfg(windows)]
    {
        let out = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "/v",
                "ProcessorNameString",
            ])
            .output();
        if let Ok(o) = out {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            if let Some(line) = text.lines().find(|l| l.contains("ProcessorNameString")) {
                if let Some(name) = line.split("REG_SZ").nth(1) {
                    let name = name.trim();
                    if !name.is_empty() {
                        return name.to_string();
                    }
                }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown CPU".into())
}

/// Hardware for the console's panel.
///
/// The compute runtime is the source: it needs no vendor SDK or Python, it
/// answers the same for ROCm, Vulkan and CUDA, and it reports the devices the
/// process doing the work can actually see. `rocm::detect()` is consulted only
/// to add a gfx target where one is known.
fn probe_hw() -> Hw {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let probed = crate::runner::Runtime::discover()
        .and_then(|rt| crate::hardware::probe(&rt).ok());
    let (gpu, gpus, vram_total_gb) = match &probed {
        Some(m) => {
            let name = match m.primary() {
                Some(d) => {
                    let gfx = crate::rocm::detect().ok().and_then(|i| i.gfx);
                    match gfx {
                        Some(g) => format!("{} ({g})", d.name),
                        None => d.name.clone(),
                    }
                }
                None => "no GPU detected".to_string(),
            };
            (name, m.devices.len(), m.primary().map(|d| d.total_gb()).unwrap_or(0.0))
        }
        None => ("no GPU detected".to_string(), 0, 0.0),
    };
    Hw { cores, cpu: cpu_name(), gpu, gpus, vram_total_gb }
}

// --------------------------------------------------------------- scheduler

#[derive(Default)]
pub struct Sched {
    active: AtomicU64,
    queued: AtomicU64,
    admitted: AtomicU64,
    completed: AtomicU64,
    rejected: AtomicU64,
    timed_out: AtomicU64,
    cancelled: AtomicU64,
}

// ----------------------------------------------------------------- experts

/// Expert grid backing the brain view. `tier` is fixed by the placement plan,
/// `heat` decays over time, `hits` marks the experts routed on the last token.
///
/// NOTE: routing here is *simulated*. The compute process does not report which
/// experts a token activated, so there is nothing real to draw yet. `/experts`
/// and `/health` both carry `"simulated": true` so nothing mistakes it for
/// measurement.
pub struct Experts {
    rows: usize,
    cols: usize,
    top_k: usize,
    tier: Vec<u8>,
    heat: Vec<f32>,
    hits: Vec<u8>,
    seq: u64,
    rng: u64,
}

impl Experts {
    fn new(layers: usize, experts: usize, top_k: usize, vram_frac: f32, ram_frac: f32) -> Self {
        let rows = layers.max(1);
        let cols = experts.max(1);
        let n = rows * cols;
        // Hot experts sit at the front of each layer's row; the plan decides how
        // many of them land in VRAM vs RAM vs disk.
        let vram_cols = ((cols as f32) * vram_frac).round() as usize;
        let ram_cols = ((cols as f32) * ram_frac).round() as usize;
        let mut tier = vec![0u8; n];
        for r in 0..rows {
            for c in 0..cols {
                tier[r * cols + c] = if c < vram_cols {
                    2
                } else if c < vram_cols + ram_cols {
                    1
                } else {
                    0
                };
            }
        }
        Self {
            rows,
            cols,
            top_k: top_k.max(1),
            tier,
            heat: vec![0.0; n],
            hits: vec![0u8; (n + 7) / 8],
            seq: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_rand(&mut self) -> u64 {
        // xorshift64* - deterministic, avoids pulling in rand
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Advance one token: decay heat, fire top_k experts per layer.
    fn tick(&mut self) {
        for h in self.heat.iter_mut() {
            *h *= 0.97;
        }
        for byte in self.hits.iter_mut() {
            *byte = 0;
        }
        let (rows, cols, top_k) = (self.rows, self.cols, self.top_k);
        for r in 0..rows {
            for _ in 0..top_k.min(cols) {
                // Bias toward the hot (low-index, VRAM-resident) end of the row,
                // which is what a warmed-up router looks like.
                let a = (self.next_rand() % cols as u64) as usize;
                let b = (self.next_rand() % cols as u64) as usize;
                let c = a.min(b);
                let i = r * cols + c;
                self.heat[i] = (self.heat[i] + 0.18).min(1.0);
                self.hits[i >> 3] |= 1 << (i & 7);
            }
        }
        self.seq += 1;
    }

    /// One hex byte per expert: `tier << 6 | heat(0..63)` - the encoding
    /// Brain.tsx decodes.
    fn map_hex(&self) -> String {
        let mut s = String::with_capacity(self.heat.len() * 2);
        for (i, h) in self.heat.iter().enumerate() {
            let heat = (h * 63.0).round().clamp(0.0, 63.0) as u8;
            s.push_str(&format!("{:02x}", (self.tier[i] << 6) | heat));
        }
        s
    }

    fn hits_hex(&self) -> String {
        let mut s = String::with_capacity(self.hits.len() * 2);
        for b in &self.hits {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    fn counts(&self) -> (usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize);
        for t in &self.tier {
            match t {
                2 => c.0 += 1,
                1 => c.1 += 1,
                _ => c.2 += 1,
            }
        }
        c
    }
}

// ----------------------------------------------------------------- profile

#[derive(Serialize, Clone, Default)]
pub struct ProfileTurn {
    wall_s: f32,
    prompt_tokens: u64,
    completion_tokens: u64,
    expert_disk_s: f32,
    expert_wait_s: f32,
    expert_matmul_s: f32,
    attention_s: f32,
    lm_head_s: f32,
    forwards: u64,
}

// ------------------------------------------------------------------- state

pub struct AppState {
    model: String,
    upstream: String,
    client: reqwest::Client,
    hw: Hw,
    vram_gb: f32,
    ram_gb: f32,
    sched: Sched,
    profile: Mutex<Vec<ProfileTurn>>,
    experts: Mutex<Experts>,
    /// Bounds how many turns run at once. The sidebar advertises `capacity`
    /// slots, and without this nothing enforced it: N browser tabs meant N
    /// concurrent forwards, and a 20GB+ model does not survive that on a
    /// machine already near its commit limit.
    permits: Arc<tokio::sync::Semaphore>,
    /// Keeps conversations inside the context the model was loaded with by
    /// summarising their middle when they outgrow it.
    compactor: crate::compact::Compactor,
    /// Required on every API request when set.
    api_key: Option<String>,
    /// The compute process, when Strata started it. `None` when `--upstream`
    /// pointed at someone else's server, which is not ours to stop.
    compute: Option<tokio::sync::Mutex<Compute>>,
}

/// The compute process and everything needed to start it again.
///
/// The weights are the machine's whole GPU: ~15 GB of VRAM that nothing else
/// can use while they are resident. Unloading drops the child - `Runner::drop`
/// kills it - and loading starts a fresh one on the same port, so the endpoint
/// string every other part of the server holds stays valid across the cycle.
pub struct Compute {
    runner: Option<crate::runner::Runner>,
    model_path: std::path::PathBuf,
    flags: crate::runner::Flags,
    port: u16,
}

impl Compute {
    fn is_loaded(&self) -> bool {
        self.runner.is_some()
    }

    async fn load(&mut self) -> Result<std::time::Duration> {
        if self.is_loaded() {
            return Ok(std::time::Duration::ZERO);
        }
        let mut r = crate::runner::Runner::start(&self.model_path, self.flags.clone(), self.port)?;
        let took = r.wait_ready(std::time::Duration::from_secs(600)).await?;
        self.runner = Some(r);
        Ok(took)
    }

    /// Returns false if it was already unloaded.
    fn unload(&mut self) -> bool {
        self.runner.take().is_some()
    }
}

type Shared = Arc<AppState>;

// ------------------------------------------------------------ request body

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    model: String,
    messages: Vec<Msg>,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default, alias = "max_tokens")]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default)]
    stream: bool,
    /// Everything else the client sent - `top_p`, `stop`, `tools`, `seed`,
    /// `response_format`. Forwarded untouched so an editor or coding agent
    /// pointed at Strata behaves as it would talking to llama.cpp directly,
    /// instead of silently losing the half of the request Strata does not name.
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

fn default_temp() -> f32 {
    0.7
}

/// The reply length assumed when a client does not ask for one. Also the
/// context compaction reserves for the answer.
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Deserialize, Serialize, Clone, Default)]
pub struct Msg {
    pub role: String,
    #[serde(default, deserialize_with = "message_text")]
    pub content: String,
    /// The rest of the message: `tool_calls` on an assistant turn,
    /// `tool_call_id` and `name` on a tool result. Strata reads none of it, but
    /// a coding agent's entire loop is carried in those fields, so they are
    /// kept beside the text and written back out untouched. Dropping them - as
    /// naming only `role` and `content` did - leaves the model answering a
    /// conversation in which it never called a tool and never saw a result.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl Msg {
    /// The message as text, for counting and summarising. A turn that is only
    /// a tool call carries no content, so the call itself stands in for it;
    /// otherwise every such turn would measure as zero tokens and vanish from
    /// a summary.
    pub fn text(&self) -> String {
        match self.extra.get("tool_calls") {
            None => self.content.clone(),
            Some(calls) if self.content.is_empty() => calls.to_string(),
            Some(calls) => format!("{}
{calls}", self.content),
        }
    }
}

/// Read a message's text whatever shape it arrived in. An OpenAI client sends
/// a string, `null` for a turn that is only a tool call, or an array of content
/// parts; refusing the last two rejects the whole conversation with a
/// deserialisation error the client cannot act on.
fn message_text<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(d)? {
        Value::String(s) => s,
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    })
}

// --------------------------------------------------------------- endpoints

async fn health(State(state): State<Shared>) -> Json<Value> {
    let (ram_total, ram_avail) = winmem::ram_gb();
    let (vram_n, ram_n, disk_n) = state.experts.lock().unwrap_or_else(|e| e.into_inner()).counts();
    let s = &state.sched;
    Json(json!({
        "status": "ok",
        "simulated": true,
        "scheduler": {
            "active": s.active.load(Ordering::Relaxed),
            "capacity": KV_SLOTS,
            "queued": s.queued.load(Ordering::Relaxed),
            "max_queue": MAX_QUEUE,
            "queue_timeout_seconds": QUEUE_TIMEOUT_S,
            "admitted": s.admitted.load(Ordering::Relaxed),
            "completed": s.completed.load(Ordering::Relaxed),
            "rejected": s.rejected.load(Ordering::Relaxed),
            "timed_out": s.timed_out.load(Ordering::Relaxed),
            "cancelled": s.cancelled.load(Ordering::Relaxed),
        },
        "kv_slots": KV_SLOTS,
        "model": {
            // `owned` is false under --upstream, where the process is someone
            // else's and the console should not offer to stop it.
            "owned": state.compute.is_some(),
            "loaded": match &state.compute {
                Some(c) => c.try_lock().map(|g| g.is_loaded()).unwrap_or(true),
                None => true,
            },
            "name": state.model,
        },
        "compaction": state.compactor.report(),
        "tiers": {
            "vram": vram_n, "ram": ram_n, "disk": disk_n,
            "vram_gb": state.vram_gb, "ram_gb": state.ram_gb,
        },
        "hwinfo": {
            "cores": state.hw.cores,
            "ram_total_gb": ram_total,
            "ram_avail_gb": ram_avail,
            "gpus": state.hw.gpus,
            "vram_total_gb": state.hw.vram_total_gb,
            "cpu": state.hw.cpu,
            "gpu": state.hw.gpu,
        }
    }))
}

/// GET /compact - compaction settings and what it has done this run.
async fn compaction(State(state): State<Shared>) -> Json<Value> {
    Json(state.compactor.report())
}

/// How long to wait for in-flight turns to finish before giving up on an
/// unload. Long enough for a turn to end on its own, short enough that the
/// button answers.
const DRAIN_TIMEOUT_S: u64 = 90;

fn no_compute() -> Response {
    (
        axum::http::StatusCode::CONFLICT,
        Json(json!({ "error": {
            "message": "strata does not own the compute process (--upstream); \
                        it is not ours to load or unload",
            "type": "not_owned"
        }})),
    )
        .into_response()
}

/// POST /model/unload - stop the compute process and give the VRAM back.
///
/// Every scheduler slot is taken first, so the process is only killed once no
/// turn is running against it. Otherwise an in-flight stream would die
/// mid-token with a transport error rather than a reason.
async fn unload_model(State(state): State<Shared>) -> Response {
    let Some(compute) = &state.compute else { return no_compute() };

    let drained = tokio::time::timeout(
        std::time::Duration::from_secs(DRAIN_TIMEOUT_S),
        state.permits.clone().acquire_many_owned(KV_SLOTS as u32),
    )
    .await;
    let _all_slots = match drained {
        Ok(Ok(permits)) => permits,
        _ => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": {
                    "message": format!(
                        "turns still running after {DRAIN_TIMEOUT_S}s; nothing unloaded"
                    ),
                    "type": "busy"
                }})),
            )
                .into_response()
        }
    };

    let mut c = compute.lock().await;
    let was_loaded = c.unload();
    drop(c);
    if was_loaded {
        println!("  model unloaded — VRAM released");
    }
    Json(json!({ "loaded": false, "changed": was_loaded })).into_response()
}

/// POST /model/load - start the compute process again.
async fn load_model(State(state): State<Shared>) -> Response {
    let Some(compute) = &state.compute else { return no_compute() };
    let mut c = compute.lock().await;
    if c.is_loaded() {
        return Json(json!({ "loaded": true, "changed": false })).into_response();
    }
    match c.load().await {
        Ok(took) => {
            println!("  model loaded in {:.0}s", took.as_secs_f32());
            Json(json!({
                "loaded": true, "changed": true, "load_seconds": took.as_secs_f32(),
            }))
            .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": e.to_string(), "type": "load_failed" }})),
        )
            .into_response(),
    }
}

/// A load or unload is in progress. Distinct from `model_unloaded` because the
/// answer is "try again shortly", not "press Load".
fn model_busy() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": {
            "message": "strata: the model is being loaded or unloaded. Try again shortly.",
            "type": "model_busy"
        }})),
    )
        .into_response()
}

fn model_unloaded() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": {
            "message": "strata: the model is unloaded. POST /model/load, or press Load in the console.",
            "type": "model_unloaded"
        }})),
    )
        .into_response()
}

async fn profile(State(state): State<Shared>) -> Json<Value> {
    let turns = state.profile.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Json(json!({ "seq": turns.len(), "turns": turns }))
}

async fn experts(State(state): State<Shared>) -> Json<Value> {
    let e = state.experts.lock().unwrap_or_else(|e| e.into_inner());
    Json(json!({
        "rows": e.rows, "cols": e.cols,
        "map": e.map_hex(), "hits": e.hits_hex(),
        "seq": e.seq, "simulated": true,
    }))
}

/// OpenAI /v1/models - whatever the compute process reports, plus the model
/// this instance was started with.
async fn models(State(state): State<Shared>) -> Json<Value> {
    let mut ids = vec![state.model.clone()];
    if let Ok(resp) = state.client.get(format!("{}/v1/models", state.upstream)).send().await {
        if let Ok(body) = resp.json::<Value>().await {
            if let Some(list) = body.get("data").and_then(|m| m.as_array()) {
                for m in list {
                    if let Some(id) = m.get("id").and_then(|n| n.as_str()) {
                        ids.push(id.to_string());
                    }
                }
            }
        }
    }
    ids.dedup();
    let data: Vec<Value> = ids
        .iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "strata" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// POST /v1/chat/completions - forwards to the compute process Strata started
/// and streams its OpenAI frames straight through, recording usage in passing.
///
/// The conversation is measured against the loaded context first and compacted
/// if it no longer fits, so a long session degrades into a summary rather than
/// into a truncated prompt.
async fn chat_completions(State(state): State<Shared>, Json(req): Json<ChatReq>) -> Response {
    let model = if req.model.is_empty() { state.model.clone() } else { req.model.clone() };
    let id = format!("chatcmpl-{}", uid());
    let admitted = Instant::now();
    let permit = match admit(&state).await {
        Some(p) => p,
        None => return queue_full(),
    };
    // `try_lock`, not `lock`: the lock is held for the whole of a load, which
    // reads tens of gigabytes of weights. Waiting on it would park this turn -
    // holding its scheduler slot - for minutes and then answer as though
    // nothing had happened. Failing to take it means a load or unload is in
    // flight, which is worth saying.
    if let Some(compute) = &state.compute {
        match compute.try_lock() {
            Ok(c) if !c.is_loaded() => return model_unloaded(),
            Ok(_) => {}
            Err(_) => return model_busy(),
        }
    }
    let mut guard = TurnGuard::new(state.clone(), permit);

    let reply_budget = req.max_completion_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    // Held until the permit is, so only `KV_SLOTS` summaries can run at once.
    let prepared = state.compactor.prepare(&req.messages, reply_budget as usize).await;
    let (messages, compaction) = match prepared {
        crate::compact::Prepared::AsIs { tokens } => (&req.messages, Report::fits(tokens)),
        crate::compact::Prepared::Compacted { ref messages, ref event } => {
            println!(
                "  compacted: {} -> {} tokens ({} messages summarised, {} kept, {}ms{})",
                event.before_tokens,
                event.after_tokens,
                event.summarised_messages,
                event.kept_messages,
                event.took_ms,
                if event.reused { ", cached" } else { "" }
            );
            (messages, Report::compacted(event))
        }
    };

    // Start from what the client sent so fields Strata does not model - tools,
    // stop sequences, sampling knobs - survive the hop, then overwrite the ones
    // Strata owns.
    let mut body = Value::Object(req.extra.clone());
    let fields = body.as_object_mut().expect("built from a map");
    fields.insert("model".into(), json!(model));
    fields.insert("messages".into(), json!(messages));
    fields.insert("stream".into(), json!(req.stream));
    fields.insert("stream_options".into(), json!({ "include_usage": true }));
    fields.insert("temperature".into(), json!(req.temperature));
    fields.insert("max_tokens".into(), json!(reply_budget));
    // llama.cpp takes the reasoning toggle through the chat template, not as a
    // top-level field, which is why the console's switch did nothing before.
    fields.insert(
        "chat_template_kwargs".into(),
        json!({ "enable_thinking": req.enable_thinking }),
    );

    let upstream = state
        .client
        .post(format!("{}/v1/chat/completions", state.upstream))
        .json(&body)
        .send()
        .await;
    let queue_wait_ms = admitted.elapsed().as_millis() as u64;

    let resp = match upstream {
        Ok(r) if r.status().is_success() => r,
        other => {
            guard.reject();
            let detail = match other {
                Ok(r) => format!("upstream {} from {}", r.status(), state.upstream),
                Err(e) => format!("cannot reach {} ({e})", state.upstream),
            };
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": {
                    "message": format!("strata: compute process not answering. {detail}"),
                    "type": "upstream_unavailable"
                }})),
            )
                .into_response();
        }
    };

    let mut response = if !req.stream {
        // The compute process already answers in OpenAI shape; forward it as-is
        // and read the usage block only to record the turn.
        let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        let prompt_tokens = body.pointer("/usage/prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let completion_tokens =
            body.pointer("/usage/completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        guard.complete(admitted, prompt_tokens, completion_tokens);
        Json(body).into_response()
    } else {
        let stream = upstream_sse(guard, resp, id.clone(), model, admitted);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    };

    let headers = response.headers_mut();
    let mut tag = |name: &'static str, value: String| {
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert(name, v);
        }
    };
    tag("x-request-id", id);
    tag("x-strata-queue-wait-ms", queue_wait_ms.to_string());
    tag("x-strata-context-tokens", compaction.tokens.to_string());
    tag("x-strata-context-budget", state.compactor.budget(reply_budget as usize).to_string());
    tag("x-strata-compacted", if compaction.happened { "1".into() } else { "0".into() });
    if let Some(reclaimed) = compaction.reclaimed {
        tag("x-strata-compacted-tokens", reclaimed.to_string());
    }
    response
}

/// What compaction did to one request, in the shape the response headers want.
struct Report {
    /// Prompt size actually sent upstream.
    tokens: usize,
    happened: bool,
    /// Tokens removed, when it happened.
    reclaimed: Option<usize>,
}

impl Report {
    fn fits(tokens: usize) -> Self {
        Self { tokens, happened: false, reclaimed: None }
    }

    fn compacted(event: &crate::compact::Event) -> Self {
        Self {
            tokens: event.after_tokens,
            happened: true,
            reclaimed: Some(event.before_tokens.saturating_sub(event.after_tokens)),
        }
    }
}

struct SseState<S> {
    guard: TurnGuard,
    inner: std::pin::Pin<Box<S>>,
    buf: String,
    id: String,
    model: String,
    started: Instant,
    role_sent: bool,
    done: bool,
}

/// End of an SSE frame: a blank line, LF or CRLF. Returns (offset, separator len).
fn frame_end(b: &[u8]) -> Option<(usize, usize)> {
    let nl = 10u8;
    let cr = 13u8;
    let mut i = 0usize;
    while i + 1 < b.len() {
        if b[i] == nl && b[i + 1] == nl {
            return Some((i, 2));
        }
        if i + 3 < b.len() && b[i] == cr && b[i + 1] == nl && b[i + 2] == cr && b[i + 3] == nl {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

fn upstream_sse(
    guard: TurnGuard,
    resp: reqwest::Response,
    id: String,
    model: String,
    started: Instant,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let state = SseState {
        guard,
        inner: Box::pin(resp.bytes_stream()),
        buf: String::new(),
        id,
        model,
        started,
        role_sent: false,
        done: false,
    };

    // The compute process already speaks OpenAI SSE, so frames are forwarded
    // unchanged; they are read in passing only to record usage and finish state.
    futures::stream::unfold(state, |mut st| async move {
        if st.done {
            return None;
        }
        loop {
            let split = frame_end(st.buf.as_bytes());
            if let Some((pos, sep)) = split {
                let frame: String = st.buf.drain(..pos + sep).collect();
                let mut events = Vec::new();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else { continue };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        st.guard.complete_if_open(st.started);
                        events.push(Event::default().data("[DONE]"));
                        st.done = true;
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(payload) {
                        if let Some(u) = v.get("usage") {
                            let pt = u.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                            let ct = u.get("completion_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                            if pt + ct > 0 {
                                st.guard.complete(st.started, pt, ct);
                            }
                        }
                    }
                    events.push(Event::default().data(payload.to_string()));
                }
                if !events.is_empty() {
                    return Some((events, st));
                }
                continue;
            }
            match st.inner.next().await {
                Some(Ok(bytes)) => {
                    st.buf.push_str(&String::from_utf8_lossy(&bytes));
                    if st.buf.len() > MAX_STREAM_LINE {
                        st.done = true;
                        return Some((vec![Event::default().data("[DONE]")], st));
                    }
                }
                _ => {
                    st.guard.complete_if_open(st.started);
                    st.done = true;
                    return Some((vec![Event::default().data("[DONE]")], st));
                }
            }
        }
    })
    .flat_map(|events| futures::stream::iter(events.into_iter().map(Ok::<_, Infallible>)))
}

/// Owns a turn's scheduler accounting so `active` is decremented exactly once
/// however the turn ends.
///
/// Axum drops the SSE future outright when the browser goes away (Stop button,
/// closed tab), so decrementing only on the happy path leaks the gauge - with a
/// manual fetch_sub, `active` stayed pinned at 1 after an aborted stream and
/// climbed with every abort. Drop cannot be skipped, so this cannot leak.
struct TurnGuard {
    shared: Shared,
    settled: bool,
    /// Released on Drop, so a disconnected client frees its slot immediately.
    _permit: tokio::sync::OwnedSemaphorePermit,
}

/// Wait for a scheduler slot, reporting the wait honestly via `queued` /
/// `timed_out`. Returns None if the queue timeout elapsed first.
async fn admit(state: &Shared) -> Option<tokio::sync::OwnedSemaphorePermit> {
    state.sched.queued.fetch_add(1, Ordering::Relaxed);
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(QUEUE_TIMEOUT_S),
        state.permits.clone().acquire_owned(),
    )
    .await;
    state.sched.queued.fetch_sub(1, Ordering::Relaxed);
    match outcome {
        Ok(Ok(permit)) => Some(permit),
        Ok(Err(_)) => None,
        Err(_) => {
            state.sched.timed_out.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

fn queue_full() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": {
            "message": format!("strata: all {KV_SLOTS} scheduler slots busy for {QUEUE_TIMEOUT_S}s"),
            "type": "queue_timeout"
        }})),
    )
        .into_response()
}

impl TurnGuard {
    fn new(shared: Shared, permit: tokio::sync::OwnedSemaphorePermit) -> Self {
        shared.sched.admitted.fetch_add(1, Ordering::Relaxed);
        shared.sched.active.fetch_add(1, Ordering::Relaxed);
        Self { shared, settled: false, _permit: permit }
    }

    fn reject(&mut self) {
        if !self.settled {
            self.settled = true;
            self.shared.sched.rejected.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Turn finished cleanly: record the profiling row and advance expert heat.
    /// Settle the turn when the stream ended without a usage frame.
    fn complete_if_open(&mut self, started: Instant) {
        if !self.settled {
            self.complete(started, 0, 0);
        }
    }

    fn complete(&mut self, started: Instant, prompt_tokens: u64, completion_tokens: u64) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.shared.sched.completed.fetch_add(1, Ordering::Relaxed);

        let turn = ProfileTurn {
            wall_s: started.elapsed().as_secs_f32(),
            prompt_tokens,
            completion_tokens,
            forwards: completion_tokens,
            // Phase timers stay zero until the HIP forward pass reports them; the
            // profiling view folds the remainder into "other" rather than guessing.
            ..Default::default()
        };
        let mut p = self.shared.profile.lock().unwrap_or_else(|e| e.into_inner());
        p.push(turn);
        let overflow = p.len().saturating_sub(PROFILE_KEEP);
        if overflow > 0 {
            p.drain(..overflow);
        }
        drop(p);

        let mut e = self.shared.experts.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..completion_tokens.min(512) {
            e.tick();
        }
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.shared.sched.active.fetch_sub(1, Ordering::Relaxed);
        if !self.settled {
            self.shared.sched.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn uid() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", n)
}

// ------------------------------------------------------------------- access

/// Whether an address only accepts connections from this machine.
///
/// Anything else is reachable by other people, and the model behind it will
/// answer any of them: there is no per-user anything here, just a GPU that
/// runs whatever it is asked.
pub fn is_loopback(listen: &str) -> bool {
    let host = match listen.rsplit_once(':') {
        // `[::1]:8080` - strip the brackets IPv6 needs when a port follows.
        Some((h, _)) => h.trim_start_matches('[').trim_end_matches(']'),
        None => listen,
    };
    matches!(host, "localhost" | "::1" | "" ) || host.starts_with("127.")
}

/// Compare a presented key with the expected one without leaking, through
/// timing, how much of a guess was right.
fn secret_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn unauthorized() -> Response {
    (
        axum::http::StatusCode::UNAUTHORIZED,
        Json(json!({ "error": {
            "message": "strata: this server requires an API key. Send it as \
                        `Authorization: Bearer <key>` or `x-api-key: <key>`.",
            "type": "unauthorized"
        }})),
    )
        .into_response()
}

/// Gate the API when a key is configured. Static files are left open so the
/// console can load and ask for one.
async fn require_key(
    State(state): State<Shared>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(request).await;
    };
    let headers = request.headers();
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    match presented {
        Some(key) if secret_eq(key.as_bytes(), expected.as_bytes()) => next.run(request).await,
        _ => unauthorized(),
    }
}

/// A key worth suggesting when someone binds a public address without one.
fn suggest_key() -> String {
    // Not cryptographic randomness, and it does not need to be: it is a
    // starting point the user is free to replace, printed once so the address
    // is not left open by default.
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        ^ std::process::id() as u64;
    let alphabet = b"abcdefghijkmnopqrstuvwxyz23456789";
    (0..28)
        .map(|_| {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            let n = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
            alphabet[(n >> 33) as usize % alphabet.len()] as char
        })
        .collect()
}

// ------------------------------------------------------------------- serve

pub struct ServeOptions {
    pub listen: String,
    pub model: String,
    pub ctx: usize,
    /// Override the derived CPU-expert split; `strata tune` finds the best value.
    pub cpu_moe: Option<usize>,
    /// Talk to an already-running OpenAI-compatible server instead of starting one.
    pub upstream: Option<String>,
    /// Summarise the middle of a conversation once it outgrows `ctx`.
    pub compact: bool,
    /// KV cache element type, e.g. `q8_0`. `None` leaves it at f16.
    pub kv_type: Option<String>,
    /// Required on every API request when set. Binding anywhere but loopback
    /// without one is refused.
    pub api_key: Option<String>,
}

pub async fn serve(opts: ServeOptions) -> Result<()> {
    let ServeOptions { listen, model, ctx, cpu_moe, upstream, compact, kv_type, api_key } = opts;

    // A public address with no key hands the GPU to anything that can reach the
    // port. Refusing is the only safe default; loopback stays open because
    // reaching it already means being on this machine.
    if !is_loopback(&listen) && api_key.is_none() {
        anyhow::bail!(
            "{listen} is reachable from outside this machine and no API key is set.\n\
             Anything that can reach that port could use the model.\n\n\
             Start with:  --api-key {}\n\
             or set STRATA_API_KEY, or listen on 127.0.0.1 instead.",
            suggest_key()
        );
    }

    let path = crate::placement::resolve_path(&model);
    let placed = path.as_ref().and_then(|p| {
        let vram = (crate::rocm::vram_total_gb() as f64 * 1024.0 * 1024.0 * 1024.0) as u64;
        let ram = 20u64 * 1024 * 1024 * 1024;
        crate::placement::Placement::solve(p, ctx, vram, ram).ok()
    });
    let (layers, n_experts, top_k, vram_frac, ram_frac) = match &placed {
        Some(p) => (
            p.facts.layers,
            p.facts.experts,
            p.facts.experts_used,
            p.vram_fraction(),
            p.ram_experts as f32 / p.expert_count.max(1) as f32,
        ),
        None => (0, 0, 1, 0.0, 0.0),
    };

    // Start the compute process ourselves unless told to use an existing one.
    let mut compute: Option<tokio::sync::Mutex<Compute>> = None;
    let endpoint = match (upstream, &path, &placed) {
        (Some(url), _, _) => {
            println!("Using existing upstream at {url}");
            url
        }
        (None, Some(model_path), Some(p)) => {
            let mut flags = crate::runner::Flags::derive(p);
            flags.kv_type = kv_type.clone();
            let mut source = "derived";
            if let Some(t) = crate::tune::load(model_path, ctx, kv_type.as_deref()) {
                flags.cpu_moe_layers = t.cpu_moe_layers;
                flags.flash_attention = t.flash_attention;
                // Absent in every measurement taken so far, and absent means
                // the `-ncmoe` path, unchanged.
                flags.expert_override = t.expert_override.clone();
                source = "measured by `strata tune`";
            }
            if let Some(n) = cpu_moe {
                flags.cpu_moe_layers = n;
                source = "set on the command line";
            }
            println!("Loading {}", model_path.display());
            println!(
                "  ctx {}  |  cpu-expert layers {}/{}  |  flash-attn {}  |  KV {}  |  {} threads",
                flags.ctx,
                flags.cpu_moe_layers,
                p.facts.layers,
                flags.flash_attention,
                flags.kv_type.as_deref().unwrap_or("f16"),
                flags.threads
            );
            println!("  split {source}");
            if let Some(pattern) = &flags.expert_override {
                println!("  placement {pattern}");
            }
            let mut r = crate::runner::Runner::start(model_path, flags.clone(), 8099)?;
            println!("  runtime {}", r.runtime.server.display());
            if let Some(b) = &r.runtime.backend {
                println!("  backend {}", b.display());
            }
            let took = r.wait_ready(std::time::Duration::from_secs(600)).await?;
            println!("  ready in {:.0}s", took.as_secs_f32());
            let url = r.endpoint.clone();
            // The port is fixed, so this endpoint stays correct across an
            // unload/load cycle and nothing else has to be told about it.
            compute = Some(tokio::sync::Mutex::new(Compute {
                runner: Some(r),
                model_path: model_path.clone(),
                flags,
                port: 8099,
            }));
            url
        }
        _ => {
            anyhow::bail!(
                "no model file for '{model}'. `strata pull --model {model}`, or pass --upstream"
            )
        }
    };

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let compactor =
        crate::compact::Compactor::new(ctx, endpoint.clone(), client.clone(), compact);

    println!("Serving strata on http://{listen}");
    match (&api_key, is_loopback(&listen)) {
        (Some(_), _) => println!("  Access      API key required (Authorization: Bearer ...)"),
        (None, true) => println!("  Access      no key; reachable only from this machine"),
        (None, false) => unreachable!("refused above"),
    }
    println!("  UI          http://{listen}/");
    println!("  OpenAI      POST /v1/chat/completions   (SSE streaming)");
    println!("  Telemetry   GET  /health /profile /experts /compact");
    println!("  Model       POST /model/unload /model/load   (frees and reclaims VRAM)");
    if compact {
        let (trigger, keep) = compactor.thresholds(DEFAULT_MAX_TOKENS as usize);
        println!(
            "  Compaction  on - past ~{trigger} prompt tokens the older turns are \
             summarised back to ~{keep}"
        );
    } else {
        println!("  Compaction  off (--no-compact); a conversation past {ctx} tokens will overflow");
    }
    println!("  NOTE: the expert map in /experts is SIMULATED; the router is not observable yet.");

    let state: Shared = Arc::new(AppState {
        model: model.clone(),
        api_key: api_key.clone(),
        compute,
        compactor,
        upstream: endpoint,
        client,
        hw: probe_hw(),
        vram_gb: placed.as_ref().map(|p| p.vram_budget as f32 / 1073741824.0).unwrap_or(0.0),
        ram_gb: placed.as_ref().map(|p| p.ram_budget as f32 / 1073741824.0).unwrap_or(0.0),
        sched: Sched::default(),
        permits: Arc::new(tokio::sync::Semaphore::new(KV_SLOTS)),
        profile: Mutex::new(Vec::new()),
        experts: Mutex::new(Experts::new(layers, n_experts, top_k, vram_frac, ram_frac)),
    });
    // Kept out of the router so the compute process can still be stopped after
    // the server has finished shutting down.
    let shutdown_state = state.clone();


    let api = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/profile", get(profile))
        .route("/v1/profile", get(profile))
        .route("/experts", get(experts))
        .route("/compact", get(compaction))
        .route("/v1/compact", get(compaction))
        .route("/model/load", post(load_model))
        .route("/model/unload", post(unload_model))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(axum::middleware::from_fn_with_state(state.clone(), require_key))
        .with_state(state)
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::AllowOrigin::predicate(|origin, _| {
                    let o = origin.as_bytes();
                    o.starts_with(b"http://localhost:") || o.starts_with(b"http://127.0.0.1:")
                }))
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        );

    // web/dist holds the built console.
    let web_dist = std::path::Path::new("web/dist");
    let app = if web_dist.exists() {
        let serve_dir = tower_http::services::ServeDir::new(web_dist)
            .not_found_service(tower_http::services::ServeFile::new(web_dist.join("index.html")));
        api.fallback_service(serve_dir)
    } else {
        println!("  (web/dist missing - run `npm install && npm run build` in web/ for the UI)");
        api
    };

    let listener = tokio::net::TcpListener::bind(&listen).await?;

    // Ctrl+C and the window's close button terminate the process without
    // running a destructor, so the compute process outlives the server that
    // started it - still holding the weights, and still holding port 8099, so
    // the next start cannot spawn its own. Waiting for the signal here is what
    // gives `Compute` the chance to be dropped on the way out.
    let stop = Arc::new(tokio::sync::Notify::new());
    let stop_for_server = stop.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop_for_server.notified().await })
            .await
    });

    wait_for_signal().await;
    println!();
    println!("stopping...");
    // `notify_one`, not `notify_waiters`: it leaves a permit, so the signal is
    // not lost if it arrives before the server task has registered.
    stop.notify_one();
    // In-flight turns get a moment to finish and no longer. A streaming
    // response can outlast any patience, and Windows allows only about five
    // seconds after a console close before it kills the process itself.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(3), server).await;
    if let Some(compute) = &shutdown_state.compute {
        if compute.lock().await.unload() {
            println!("  compute process stopped, VRAM released");
        }
    }
    Ok(())
}

/// Ctrl+C, and on Windows the console's close button, a shutdown and a logoff.
///
/// Closing the window is not Ctrl+C and arrives as its own event. It is also
/// the way this server is most often stopped, so handling only Ctrl+C would
/// leave the common case exactly as broken as it was.
async fn wait_for_signal() {
    #[cfg(windows)]
    {
        use tokio::signal::windows;
        // Registration only fails where there is no console to signal us, and
        // plain Ctrl+C is still worth waiting on if it does.
        if let (Ok(mut interrupt), Ok(mut close), Ok(mut shutdown)) =
            (windows::ctrl_c(), windows::ctrl_close(), windows::ctrl_shutdown())
        {
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = close.recv() => {}
                _ = shutdown.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod access_tests {
    use super::*;

    #[test]
    fn loopback_addresses_are_recognised() {
        for listen in ["127.0.0.1:8080", "localhost:8080", "[::1]:8080", "127.5.5.5:1"] {
            assert!(is_loopback(listen), "{listen} should be loopback");
        }
    }

    #[test]
    fn everything_reachable_from_outside_is_not() {
        // 0.0.0.0 is the one that matters: it is what someone types when they
        // want a friend to connect, and it is the one that must demand a key.
        for listen in ["0.0.0.0:8080", "192.168.1.20:8080", "[::]:8080", "10.0.0.5:80"] {
            assert!(!is_loopback(listen), "{listen} should not be loopback");
        }
    }

    #[test]
    fn keys_compare_by_value_not_by_prefix() {
        assert!(secret_eq(b"correct-horse", b"correct-horse"));
        assert!(!secret_eq(b"correct-horse", b"correct-hors"));
        assert!(!secret_eq(b"correct-horse", b"correct-horsf"));
        assert!(!secret_eq(b"", b"x"));
        assert!(secret_eq(b"", b""));
    }

    #[test]
    fn a_suggested_key_is_long_enough_to_be_worth_suggesting() {
        let key = suggest_key();
        assert!(key.len() >= 24, "{key}");
        assert!(key.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_ne!(key, suggest_key(), "two suggestions should differ");
    }
}
