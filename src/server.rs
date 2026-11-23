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
    routing::{delete, get, head, post},
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
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let probed = crate::runner::Runtime::discover().and_then(|rt| crate::hardware::probe(&rt).ok());
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
            (
                name,
                m.devices.len(),
                m.primary().map(|d| d.total_gb()).unwrap_or(0.0),
            )
        }
        None => ("no GPU detected".to_string(), 0, 0.0),
    };
    Hw {
        cores,
        cpu: cpu_name(),
        gpu,
        gpus,
        vram_total_gb,
    }
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

/// Process-wide counters surfaced at GET /metrics (Prometheus text format).
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub requests_in_flight: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    pub rejected_total: AtomicU64,
    pub timed_out_total: AtomicU64,
    pub cancelled_total: AtomicU64,
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

/// The real placement solve, kept so `/health` can report where the weights
/// actually went instead of inferring it from the simulated expert grid.
#[derive(Clone, Copy)]
pub struct PlacementReport {
    pub layers: usize,
    pub attention_layers: usize,
    pub expert_count: usize,
    pub vram_experts: usize,
    pub ram_experts: usize,
    pub disk_experts: usize,
    pub dense_gb: f32,
    pub kv_gb: f32,
    pub total_gb: f32,
}

pub struct AppState {
    model: std::sync::Mutex<String>,
    /// `None` under `--upstream`, where Strata did not size the model itself.
    placement: Option<PlacementReport>,
    models_root: std::path::PathBuf,
    upstream: String,
    client: reqwest::Client,
    hw: Hw,
    vram_gb: f32,
    ram_gb: f32,
    sched: Sched,
    metrics: Metrics,
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
    /// Bumped whenever a request refreshes the keep-alive timer. A scheduled
    /// unload carries the value it was created with and does nothing if a
    /// later request has superseded it.
    keep_alive_gen: AtomicU64,
    /// When the current keep-alive timer will unload the model, in epoch
    /// seconds. Reported as `expires_at` by `/api/ps`.
    keep_alive_deadline: Mutex<Option<u64>>,
    /// Additional resident models, each with its own compute process and port.
    /// The primary model above stays on `upstream`; this pool is only populated
    /// when `OLLAMA_MAX_LOADED_MODELS` allows more than one resident.
    residents: tokio::sync::Mutex<Vec<Resident>>,
    max_resident: usize,
    default_ctx: usize,
    default_kv_type: Option<String>,
    default_backend: crate::backend::Backend,
    max_queue: usize,
    num_parallel: usize,
}

/// A model loaded alongside the primary one. Each resident owns a compute
/// process on its own port so requests can be routed without swapping weights.
struct Resident {
    model: String,
    compute: Compute,
    upstream: String,
    last_used: Instant,
    keep_alive_secs: Option<u64>,
    expires_at: Option<u64>,
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
    backend: crate::backend::Backend,
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
        let mut r = crate::runner::Runner::start(
            &self.model_path,
            self.flags.clone(),
            self.port,
            self.backend,
        )?;
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
    pub content: Value,
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
        let base = match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        match self.extra.get("tool_calls") {
            None => base,
            Some(calls) if base.is_empty() => calls.to_string(),
            Some(calls) => format!(
                "{}
{calls}",
                base
            ),
        }
    }
}

/// Read a message's text whatever shape it arrived in. An OpenAI client sends
/// a string, `null` for a turn that is only a tool call, or an array of content
/// parts; refusing the last two rejects the whole conversation with a
/// deserialisation error the client cannot act on.
fn message_text<'de, D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
    let v = Value::deserialize(d)?;
    Ok(match v {
        Value::Null => Value::Null,
        Value::String(s) => Value::String(s),
        Value::Array(parts) => Value::Array(parts),
        other => other,
    })
}

// --------------------------------------------------------------- endpoints

async fn health(State(state): State<Shared>) -> Json<Value> {
    let (ram_total, ram_avail) = winmem::ram_gb();
    let s = &state.sched;
    // Reported from the placement solve, not from the expert grid. The grid is
    // simulated and pads a dense model to one column per layer, so counting its
    // cells says 33 experts live on disk when the model has no experts at all.
    let tiers = match &state.placement {
        Some(p) => json!({
            "vram": p.vram_experts,
            "ram": p.ram_experts,
            "disk": p.disk_experts,
            "vram_gb": state.vram_gb,
            "ram_gb": state.ram_gb,
            "dense_gb": p.dense_gb,
            "kv_gb": p.kv_gb,
            "total_gb": p.total_gb,
            "expert_count": p.expert_count,
            "layers": p.layers,
            "attention_layers": p.attention_layers,
            "source": "placement",
        }),
        None => json!({
            "vram": 0, "ram": 0, "disk": 0,
            "vram_gb": state.vram_gb, "ram_gb": state.ram_gb,
            "dense_gb": 0.0, "kv_gb": 0.0, "total_gb": 0.0,
            "expert_count": 0, "layers": 0, "attention_layers": 0,
            "source": "unknown",
        }),
    };
    Json(json!({
        "status": "ok",
        "simulated": true,
        "scheduler": {
            "active": s.active.load(Ordering::Relaxed),
            "capacity": KV_SLOTS,
            "queued": s.queued.load(Ordering::Relaxed),
            "max_queue": state.max_queue,
            "num_parallel": state.num_parallel,
            "queue_timeout_seconds": QUEUE_TIMEOUT_S,
            "admitted": s.admitted.load(Ordering::Relaxed),
            "completed": s.completed.load(Ordering::Relaxed),
            "rejected": s.rejected.load(Ordering::Relaxed),
            "timed_out": s.timed_out.load(Ordering::Relaxed),
            "cancelled": s.cancelled.load(Ordering::Relaxed),
        },
        // The compute process is started with `-np 1`: one KV slot, because
        // every slot splits one shared cache and a turn can then land on a
        // slot that does not hold the prefix. `scheduler.capacity` is the
        // separate, larger bound on concurrent turns.
        "kv_slots": 1,
        "resident_capacity": state.max_resident,
        "keep_alive_expires_at": state
            .keep_alive_deadline
            .lock()
            .unwrap()
            .map(crate::ollama::iso_utc),
        "additional_residents": state.residents.try_lock().map(|r| r.len()).unwrap_or(0),
        "model": {
            // `owned` is false under --upstream, where the process is someone
            // else's and the console should not offer to stop it.
            "owned": state.compute.is_some(),
            "loaded": match &state.compute {
                Some(c) => c.try_lock().map(|g| g.is_loaded()).unwrap_or(true),
                None => true,
            },
            "name": state.model.lock().unwrap().clone(),
        },
        "compaction": state.compactor.report(),
        "tiers": tiers,
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

/// Drain in-flight turns, then stop the compute process.
///
/// Shared by `/model/unload` and the Ollama keep-alive timer: both must wait
/// for the scheduler, or an in-flight stream dies mid-token with a transport
/// error instead of a reason.
async fn drain_and_unload(state: &Shared) -> std::result::Result<bool, Response> {
    let Some(compute) = &state.compute else {
        return Err(no_compute());
    };
    let drained = tokio::time::timeout(
        std::time::Duration::from_secs(DRAIN_TIMEOUT_S),
        state
            .permits
            .clone()
            .acquire_many_owned(state.num_parallel as u32),
    )
    .await;
    let Ok(Ok(_all_slots)) = drained else {
        return Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": {
                "message": format!("turns still running after {DRAIN_TIMEOUT_S}s; nothing unloaded"),
                "type": "busy"
            }})),
        )
            .into_response());
    };
    let mut c = compute.lock().await;
    let was_loaded = c.unload();
    drop(c);
    let extras = state.residents.lock().await.len();
    state.residents.lock().await.clear();
    if was_loaded || extras > 0 {
        println!("  model unloaded — VRAM released");
    }
    Ok(was_loaded || extras > 0)
}

/// POST /model/unload - stop the compute process and give the VRAM back.
async fn unload_model(State(state): State<Shared>) -> Response {
    match drain_and_unload(&state).await {
        Ok(changed) => Json(json!({ "loaded": false, "changed": changed })).into_response(),
        Err(e) => e,
    }
}

/// Token ids for Ollama's deprecated-but-still-used `context` field. Computed
/// with Strata's own BPE over the prompt the compute process will actually see.
fn prompt_context(state: &Shared, model: &str, req: &Value) -> Vec<u32> {
    let path = crate::registry::resolve(&state.models_root, model).or_else(|| {
        // Under --upstream the client may name a model the local registry does
        // not know; tokenise with the resident model's vocabulary instead of
        // silently returning no context.
        let current = state.model.lock().unwrap().clone();
        crate::registry::resolve(&state.models_root, &current)
    });
    let Some(path) = path else { return Vec::new() };
    let Ok(tok) = crate::tokenizer::Tokenizer::from_gguf(&path.to_string_lossy()) else {
        return Vec::new();
    };
    let mut text = String::new();
    if let Some(system) = req.get("system").and_then(Value::as_str) {
        text.push_str(system);
        text.push('\n');
    }
    if let Some(prompt) = req.get("prompt").and_then(Value::as_str) {
        text.push_str(prompt);
    }
    if let Some(messages) = req.get("messages").and_then(Value::as_array) {
        for m in messages {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            let content = m
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            text.push_str(role);
            text.push_str(": ");
            text.push_str(content);
            text.push('\n');
        }
    }
    tok.encode(&text).unwrap_or_default()
}

/// Remote endpoints registered through `POST /api/connect`, in Ollama's shape.
fn connections_path() -> std::path::PathBuf {
    crate::registry::default_root().join("connections.json")
}

fn load_connections() -> serde_json::Map<String, Value> {
    std::fs::read_to_string(connections_path())
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn connection_key_for_url(url: &str) -> Option<String> {
    load_connections()
        .values()
        .find(|v| v.get("url").and_then(Value::as_str) == Some(url))
        .and_then(|v| v.get("key"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn connection_upstream(model: &str) -> Option<String> {
    load_connections()
        .get(model)
        .and_then(|v| v.get("url"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// GET/POST/DELETE /api/keys - local API keys, for private networks.
async fn api_keys_list() -> Json<Value> {
    Json(json!({ "keys": crate::auth::load_keys() }))
}

async fn api_keys_add(Json(req): Json<Value>) -> Response {
    let key = req.get("key").and_then(Value::as_str).unwrap_or("");
    if key.len() < 8 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "key must be at least 8 characters" })),
        )
            .into_response();
    }
    let mut keys = crate::auth::load_keys();
    if !keys.iter().any(|k| k == key) {
        keys.push(key.to_string());
    }
    match crate::auth::save_keys(&keys) {
        Ok(()) => Json(json!({ "status": "success" })).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn api_keys_remove(Json(req): Json<Value>) -> Json<Value> {
    let key = req.get("key").and_then(Value::as_str).unwrap_or("");
    let mut keys = crate::auth::load_keys();
    keys.retain(|k| k != key);
    if let Err(e) = crate::auth::save_keys(&keys) {
        return Json(json!({ "error": e.to_string() }));
    }
    Json(json!({ "status": "success" }))
}

/// POST /api/connect - register a remote OpenAI-compatible endpoint as a model.
async fn api_connect(Json(req): Json<Value>) -> Response {
    let model = req.get("model").and_then(Value::as_str).unwrap_or("");
    let url = req.get("url").and_then(Value::as_str).unwrap_or("");
    if model.is_empty() || !url.starts_with("http://") && !url.starts_with("https://") {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model and an http(s) url are required" })),
        )
            .into_response();
    }
    let mut all = load_connections();
    let mut entry = json!({ "url": url.trim_end_matches('/') });
    if let Some(key) = req.get("key").and_then(Value::as_str) {
        entry["key"] = json!(key);
    }
    all.insert(model.to_string(), entry);
    let encoded = serde_json::to_string_pretty(&all).unwrap_or_default();
    match std::fs::write(connections_path(), encoded) {
        Ok(()) => Json(json!({ "status": "success" })).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// DELETE /api/connect - forget a remote endpoint.
async fn api_disconnect(Json(req): Json<Value>) -> Response {
    let model = req.get("model").and_then(Value::as_str).unwrap_or("");
    let mut all = load_connections();
    if all.remove(model).is_none() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("no connection for {model}") })),
        )
            .into_response();
    }
    let encoded = serde_json::to_string_pretty(&all).unwrap_or_default();
    match std::fs::write(connections_path(), encoded) {
        Ok(()) => Json(json!({ "status": "success" })).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

fn compaction_value(state: &Shared) -> Value {
    let s = &state.compactor.stats;
    json!({
        "enabled": state.compactor.stats.last_budget.load(Ordering::Relaxed) > 0,
        "summarised": s.summarised.load(Ordering::Relaxed),
        "reused": s.reused.load(Ordering::Relaxed),
        "reclaimed": s.reclaimed.load(Ordering::Relaxed),
    })
}

fn ndjson_response(lines: Vec<String>) -> Response {
    let mut out = String::new();
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        out,
    )
        .into_response()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `None` means the request said nothing; `Some(None)` is Ollama's `-1`
/// ("keep forever"); `Some(Some(n))` is n seconds.
fn request_keep_alive(req: &Value) -> Option<Option<u64>> {
    if req.get("keep_alive").is_some() {
        return req.get("keep_alive").and_then(keep_alive_seconds);
    }
    std::env::var("OLLAMA_KEEP_ALIVE")
        .ok()
        .and_then(|s| keep_alive_seconds(&Value::String(s)))
}

/// Parse Ollama's `keep_alive`: seconds, a duration string, or -1 for forever.
/// Returns `None` for "leave the current timer alone".
pub fn keep_alive_seconds(v: &Value) -> Option<Option<u64>> {
    let raw = match v {
        Value::Number(n) => n.as_i64()?,
        Value::String(s) => {
            let s = s.trim();
            if s == "-1" {
                return Some(None);
            }
            let (digits, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
            let n: i64 = digits.parse().ok()?;
            match unit {
                "s" | "" => n,
                "m" => n * 60,
                "h" => n * 3600,
                "d" => n * 86400,
                _ => return None,
            }
        }
        _ => return None,
    };
    if raw < 0 {
        return Some(None);
    }
    Some(Some(raw as u64))
}

/// Refresh the unload timer after an Ollama request. A `keep_alive` of 0 is the
/// documented "unload now" idiom; -1 cancels unloading entirely.
async fn apply_keep_alive(state: &Shared, req: &Value, model: &str) {
    let seconds = request_keep_alive(req).or(Some(Some(300)));
    if let Some(Some(secs)) = seconds {
        // A resident can expire on its own without touching the primary. The
        // deadline check makes a later request's refresh win over this timer.
        let resident_model = model.to_string();
        let resident_state = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
            let mut residents = resident_state.residents.lock().await;
            if let Some(i) = residents.iter().position(|r| r.model == resident_model) {
                let deadline = residents[i].expires_at;
                if deadline.is_some_and(|d| d <= now_secs()) {
                    let gone = residents.remove(i);
                    println!("  {} unloaded after keep_alive", gone.model);
                }
            }
        });
    }
    let Some(seconds) = seconds else { return };
    let gen = state.keep_alive_gen.fetch_add(1, Ordering::SeqCst) + 1;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    match seconds {
        Some(0) => {
            *state.keep_alive_deadline.lock().unwrap() = None;
            drain_and_unload(state).await.ok();
        }
        Some(secs) => {
            *state.keep_alive_deadline.lock().unwrap() = Some(now + secs);
            let state = state.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                if state.keep_alive_gen.load(Ordering::SeqCst) == gen {
                    drain_and_unload(&state).await.ok();
                }
            });
        }
        None => {
            // -1: stay resident until the console unloads it.
            *state.keep_alive_deadline.lock().unwrap() = None;
        }
    }
}

/// POST /model/load - start the compute process again.
async fn load_model(State(state): State<Shared>) -> Response {
    let Some(compute) = &state.compute else {
        return no_compute();
    };
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
    let turns = state
        .profile
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
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

/// GET /metrics - Prometheus text format.
async fn metrics(State(state): State<Shared>) -> String {
    let m = &state.metrics;
    let mut out = String::new();
    out.push_str("# HELP strata_requests_total Total chat completion requests.\n");
    out.push_str("# TYPE strata_requests_total counter\n");
    out.push_str(&format!(
        "strata_requests_total {}\n",
        m.requests_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_requests_in_flight Requests currently in flight.\n");
    out.push_str("# TYPE strata_requests_in_flight gauge\n");
    out.push_str(&format!(
        "strata_requests_in_flight {}\n",
        m.requests_in_flight.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_prompt_tokens_total Total prompt tokens processed.\n");
    out.push_str("# TYPE strata_prompt_tokens_total counter\n");
    out.push_str(&format!(
        "strata_prompt_tokens_total {}\n",
        m.prompt_tokens_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_completion_tokens_total Total completion tokens generated.\n");
    out.push_str("# TYPE strata_completion_tokens_total counter\n");
    out.push_str(&format!(
        "strata_completion_tokens_total {}\n",
        m.completion_tokens_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_rejected_total Rejected requests (upstream unavailable).\n");
    out.push_str("# TYPE strata_rejected_total counter\n");
    out.push_str(&format!(
        "strata_rejected_total {}\n",
        m.rejected_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_timed_out_total Requests that timed out in queue.\n");
    out.push_str("# TYPE strata_timed_out_total counter\n");
    out.push_str(&format!(
        "strata_timed_out_total {}\n",
        m.timed_out_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_cancelled_total Cancelled requests.\n");
    out.push_str("# TYPE strata_cancelled_total counter\n");
    out.push_str(&format!(
        "strata_cancelled_total {}\n",
        m.cancelled_total.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_sched_active Active scheduler slots.\n");
    out.push_str("# TYPE strata_sched_active gauge\n");
    out.push_str(&format!(
        "strata_sched_active {}\n",
        state.sched.active.load(Ordering::Relaxed)
    ));
    out.push_str("# HELP strata_resident_models Models currently loaded.\n");
    out.push_str("# TYPE strata_resident_models gauge\n");
    let residents = state.residents.try_lock().map(|r| r.len()).unwrap_or(0);
    out.push_str(&format!("strata_resident_models {}\n", residents));
    out.push_str("# HELP strata_resident_capacity Maximum resident models.\n");
    out.push_str("# TYPE strata_resident_capacity gauge\n");
    out.push_str(&format!("strata_resident_capacity {}\n", state.max_resident));
    out.push_str("# HELP strata_sched_completed Completed turns.\n");
    out.push_str("# TYPE strata_sched_completed counter\n");
    out.push_str(&format!(
        "strata_sched_completed {}\n",
        state.sched.completed.load(Ordering::Relaxed)
    ));
    out
}

/// OpenAI /v1/models - whatever the compute process reports, plus the model
/// this instance was started with.
async fn models(State(state): State<Shared>) -> Json<Value> {
    // Every installed model, not just the resident one. This endpoint used to
    // report only `state.model` plus whatever the compute process listed, so a
    // machine holding several models advertised one and the console's picker
    // had nothing to pick.
    let mut ids: Vec<String> = crate::registry::scan(&state.models_root)
        .into_iter()
        .map(|e| e.id)
        .collect();

    // The resident model goes first, and is present even when its file is not
    // in the registry root (an explicit --model path, for instance).
    let current = state.model.lock().unwrap().clone();
    if !current.is_empty() {
        ids.retain(|id| id != &current);
        ids.insert(0, current);
    }

    // Anything the compute process itself reports, deduplicated against the
    // registry rather than appended blindly.
    if let Ok(resp) = state
        .client
        .get(format!("{}/v1/models", state.upstream))
        .send()
        .await
    {
        if let Ok(body) = resp.json::<Value>().await {
            if let Some(list) = body.get("data").and_then(|m| m.as_array()) {
                for m in list {
                    if let Some(id) = m.get("id").and_then(|n| n.as_str()) {
                        if !ids.iter().any(|existing| existing == id) {
                            ids.push(id.to_string());
                        }
                    }
                }
            }
        }
    }

    let data: Vec<Value> = ids
        .iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "strata" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// GET /api/models - installed models (PolyStrata contract).
async fn api_models(State(state): State<Shared>) -> Json<Value> {
    let entries = crate::registry::scan(&state.models_root);
    let current = state.model.lock().unwrap().clone();
    let loaded = state
        .compute
        .as_ref()
        .map(|c| {
            // try_lock to avoid blocking; treat locked as loaded-ish
            c.try_lock().map(|g| g.is_loaded()).unwrap_or(true)
        })
        .unwrap_or(false);
    let data: Vec<Value> = entries
        .iter()
        .map(|e| {
            json!({
                "id": e.id,
                "display_name": e.name,
                "arch": e.arch,
                "path": e.path,
                "experts": e.experts,
                "experts_used": e.experts_used,
                "context": e.context,
                "file_bytes": e.file_bytes,
                "is_moe": e.is_moe,
                "current": e.id == current,
                "loaded": if e.id == current { loaded } else { false },
            })
        })
        .collect();
    Json(json!({ "models": data, "current": current }))
}

/// GET /api/models/current
async fn api_models_current(State(state): State<Shared>) -> Json<Value> {
    let loaded = state
        .compute
        .as_ref()
        .map(|c| c.try_lock().map(|g| g.is_loaded()).unwrap_or(true))
        .unwrap_or(false);
    Json(json!({ "model": state.model.lock().unwrap().clone(), "loaded": loaded }))
}

#[derive(Deserialize)]
struct LoadModelReq {
    model: String,
}

/// POST /api/models/load { "model": "..." } - switch the active model.
async fn api_models_load(State(state): State<Shared>, Json(req): Json<LoadModelReq>) -> Response {
    let new_path = match crate::registry::resolve(&state.models_root, &req.model) {
        Some(p) => p,
        None => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(json!({"error": format!("model not found: {}", req.model)})),
            )
                .into_response()
        }
    };

    // With room for more than one resident, a second model is loaded beside
    // the primary one instead of replacing it. Requests name their model and
    // are routed to the right process by `ensure_upstream`.
    if state.max_resident > 1 {
        let current = state.model.lock().unwrap().clone();
        if req.model != current && state.compute.is_some() {
            match load_resident(&state, &req.model, request_keep_alive(&json!({}))).await {
                Ok(u) => {
                    println!("  {} resident at {u}", req.model);
                    return Json(json!({
                        "ok": true, "model": req.model, "resident": true, "upstream": u,
                    }))
                    .into_response();
                }
                Err(e) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": format!("failed to load {}: {e}", req.model) })),
                    )
                        .into_response()
                }
            }
        }
    }

    // Wait for in-flight to drain (best effort, up to 30s).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while state
        .sched
        .active
        .load(std::sync::atomic::Ordering::Relaxed)
        > 0
        && std::time::Instant::now() < deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let Some(compute) = &state.compute else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "no compute process (upstream mode)"})),
        )
            .into_response();
    };

    let mut c = compute.lock().await;

    // Remember what was resident, so a failed switch can put it back. A caller
    // asking to change models did not ask to give the GPU back, and leaving
    // nothing loaded turns a bad model name into a dead server.
    let previous_path = c.model_path.clone();
    let previous_model = state.model.lock().unwrap().clone();

    // Unload current.
    c.unload();
    // Swap model path.
    c.model_path = new_path.clone();
    // Reload.
    match c.load().await {
        Ok(_) => {
            *state.model.lock().unwrap() = req.model.clone();
            Json(json!({"ok": true, "model": req.model, "path": new_path})).into_response()
        }
        Err(e) => {
            // Roll back to what was loaded before this request.
            c.model_path = previous_path.clone();
            let (status, body) = match c.load().await {
                Ok(_) => {
                    *state.model.lock().unwrap() = previous_model.clone();
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        json!({
                            "error": format!("failed to load {}: {e}", req.model),
                            "rolled_back_to": previous_model,
                            "path": previous_path.to_string_lossy(),
                        }),
                    )
                }
                Err(rollback_err) => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    json!({
                        "error": format!("failed to load {}: {e}", req.model),
                        "rollback_failed": format!("{rollback_err}"),
                        "note": "no model is loaded",
                    }),
                ),
            };
            (status, Json(body)).into_response()
        }
    }
}

/// POST /api/models/unload
async fn api_models_unload(State(state): State<Shared>) -> Json<Value> {
    if let Some(compute) = &state.compute {
        if let Ok(mut c) = compute.try_lock() {
            c.unload();
        }
    }
    let cleared = state.residents.lock().await.len();
    state.residents.lock().await.clear();
    Json(json!({ "ok": true, "residents_unloaded": cleared }))
}

/// POST /v1/chat/completions - forwards to the compute process Strata started
/// and streams its OpenAI frames straight through, recording usage in passing.
///
/// The conversation is measured against the loaded context first and compacted
/// if it no longer fits, so a long session degrades into a summary rather than
/// into a truncated prompt.
async fn chat_completions(State(state): State<Shared>, Json(req): Json<ChatReq>) -> Response {
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .requests_in_flight
        .fetch_add(1, Ordering::Relaxed);
    let model = if req.model.is_empty() {
        state.model.lock().unwrap().clone()
    } else {
        req.model.clone()
    };
    let keep = request_keep_alive(&Value::Object(req.extra.clone()));
    apply_keep_alive(&state, &Value::Object(req.extra.clone()), &model).await;
    let upstream = match ensure_upstream(&state, &model, keep).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let is_extra = upstream != state.upstream;
    let id = format!("chatcmpl-{}", uid());
    let admitted = Instant::now();
    let permit = match admit(&state).await {
        Some(p) => p,
        None => {
            state
                .metrics
                .requests_in_flight
                .fetch_sub(1, Ordering::Relaxed);
            state
                .metrics
                .timed_out_total
                .fetch_add(1, Ordering::Relaxed);
            return queue_full();
        }
    };
    // `try_lock`, not `lock`: the lock is held for the whole of a load, which
    // reads tens of gigabytes of weights. Waiting on it would park this turn -
    // holding its scheduler slot - for minutes and then answer as though
    // nothing had happened. Failing to take it means a load or unload is in
    // flight, which is worth saying.
    if !is_extra {
        if let Some(compute) = &state.compute {
            match compute.try_lock() {
                Ok(c) if !c.is_loaded() => return model_unloaded(),
                Ok(_) => {}
                Err(_) => return model_busy(),
            }
        }
    }
    let mut guard = TurnGuard::new(state.clone(), permit);

    let reply_budget = req.max_completion_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    // Held until the permit is, so only `KV_SLOTS` summaries can run at once.
    let prepared = state
        .compactor
        .prepare(&req.messages, reply_budget as usize)
        .await;
    let (messages, compaction) = match prepared {
        crate::compact::Prepared::AsIs { tokens } => (&req.messages, Report::fits(tokens)),
        crate::compact::Prepared::Compacted {
            ref messages,
            ref event,
        } => {
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

    // Modelfile overrides are applied at the API boundary: llama.cpp has no
    // `--system-prompt-file` in the bundled runtime, so a derived model's
    // SYSTEM and PARAMETER lines have to travel with the request.
    let defaults = crate::registry::overrides(&state.models_root, &model);
    let mut turns: Vec<Value> = serde_json::from_value(json!(messages)).unwrap_or_default();
    if let Some(system) = defaults.get("system").and_then(Value::as_str) {
        if !turns
            .iter()
            .any(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        {
            turns.insert(0, json!({ "role": "system", "content": system }));
        }
    }
    if let Some(params) = defaults.get("parameters").and_then(Value::as_object) {
        for (key, value) in params {
            if req.extra.contains_key(key) {
                continue; // the caller's explicit setting wins
            }
            let parsed = value
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .map(Value::from);
            let mapped = match key.as_str() {
                "num_predict" => Some("max_tokens"),
                other => Some(other),
            };
            if let (Some(target), Some(v)) = (mapped, parsed.or_else(|| Some(value.clone()))) {
                fields.insert(target.into(), v);
            }
        }
    }
    fields.insert("messages".into(), json!(turns));
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
    // Re-apply the two fields Strata owns above, so a Modelfile's PARAMETER
    // temperature / num_predict is not lost to the defaults.
    if let Some(params) = defaults.get("parameters").and_then(Value::as_object) {
        for (key, target) in [("temperature", "temperature"), ("num_predict", "max_tokens")] {
            if req.extra.contains_key(key) {
                continue;
            }
            if let Some(v) = params.get(key) {
                let parsed = v
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(Value::from)
                    .unwrap_or_else(|| v.clone());
                fields.insert(target.into(), parsed);
            }
        }
    }

    let mut request = state
        .client
        .post(format!("{upstream}/v1/chat/completions"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(
            std::env::var("STRATA_UPSTREAM_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(900),
        ));
    if let Some(key) = connection_key_for_url(&upstream) {
        request = request.bearer_auth(key);
    }
    let upstream_result = request.send().await;
    let queue_wait_ms = admitted.elapsed().as_millis() as u64;

    let resp = match upstream_result {
        Ok(r) if r.status().is_success() => r,
        other => {
            guard.reject();
            let detail = match other {
                Ok(r) => format!("upstream {} from {upstream}", r.status()),
                Err(e) => format!("cannot reach {upstream} ({e})"),
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

    let mut invalid_structured = false;
    let mut response = if !req.stream {
        // The compute process already answers in OpenAI shape; forward it as-is
        // and read the usage block only to record the turn.
        let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        if let Some(format) = req.extra.get("response_format") {
            let text = body
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !crate::dialects::valid_structured(text, format) {
                eprintln!("strata: model did not honour the requested response format");
                invalid_structured = true;
            }
        }
        let prompt_tokens = body
            .pointer("/usage/prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let completion_tokens = body
            .pointer("/usage/completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        guard.complete(admitted, prompt_tokens, completion_tokens);
        Json(body).into_response()
    } else {
        let stream = upstream_sse(guard, resp, id.clone(), model, admitted);
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    };

    let headers = response.headers_mut();
    let mut tag = |name: &'static str, value: String| {
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert(name, v);
        }
    };
    if invalid_structured {
        tag("x-strata-invalid-structured", "1".into());
    }
    tag("x-request-id", id);
    tag("x-strata-queue-wait-ms", queue_wait_ms.to_string());
    tag("x-strata-context-tokens", compaction.tokens.to_string());
    tag(
        "x-strata-context-budget",
        state.compactor.budget(reply_budget as usize).to_string(),
    );
    tag(
        "x-strata-compacted",
        if compaction.happened {
            "1".into()
        } else {
            "0".into()
        },
    );
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
        Self {
            tokens,
            happened: false,
            reclaimed: None,
        }
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
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
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
                            let ct = u
                                .get("completion_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0);
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
    if state.sched.queued.load(Ordering::Relaxed) >= state.max_queue as u64 {
        return None;
    }
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
        Self {
            shared,
            settled: false,
            _permit: permit,
        }
    }

    fn reject(&mut self) {
        if !self.settled {
            self.settled = true;
            self.shared.sched.rejected.fetch_add(1, Ordering::Relaxed);
            self.shared
                .metrics
                .rejected_total
                .fetch_add(1, Ordering::Relaxed);
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

        self.shared
            .metrics
            .prompt_tokens_total
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.shared
            .metrics
            .completion_tokens_total
            .fetch_add(completion_tokens, Ordering::Relaxed);

        let turn = ProfileTurn {
            wall_s: started.elapsed().as_secs_f32(),
            prompt_tokens,
            completion_tokens,
            forwards: completion_tokens,
            // Phase timers stay zero until the HIP forward pass reports them; the
            // profiling view folds the remainder into "other" rather than guessing.
            ..Default::default()
        };
        let mut p = self
            .shared
            .profile
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        p.push(turn);
        let overflow = p.len().saturating_sub(PROFILE_KEEP);
        if overflow > 0 {
            p.drain(..overflow);
        }
        drop(p);

        let mut e = self
            .shared
            .experts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for _ in 0..completion_tokens.min(512) {
            e.tick();
        }
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.shared.sched.active.fetch_sub(1, Ordering::Relaxed);
        self.shared
            .metrics
            .requests_in_flight
            .fetch_sub(1, Ordering::Relaxed);
        if !self.settled {
            self.shared.sched.cancelled.fetch_add(1, Ordering::Relaxed);
            self.shared
                .metrics
                .cancelled_total
                .fetch_add(1, Ordering::Relaxed);
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
    matches!(host, "localhost" | "::1" | "") || host.starts_with("127.")
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
        Some(key)
            if secret_eq(key.as_bytes(), expected.as_bytes()) || crate::auth::valid_key(key) =>
        {
            next.run(request).await
        }
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
    /// Speculative decoding draft model path.
    pub draft: Option<String>,
    /// Max draft tokens per step.
    pub spec: Option<usize>,
    /// Multimodal projector path (optional).
    pub mmproj: Option<String>,
    pub devices: Option<String>,
    pub tensor_split: Option<String>,
    pub backend: Option<String>,
    pub embeddings: bool,
}

pub async fn serve(opts: ServeOptions) -> Result<()> {
    let ServeOptions {
        listen,
        model,
        ctx,
        cpu_moe,
        upstream,
        compact,
        kv_type,
        api_key,
        draft,
        spec,
        mmproj,
        devices,
        tensor_split,
        backend,
        embeddings,
    } = opts;

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
        let vram = vram_budget_bytes();
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

    // Backend selection: auto by architecture, or explicit override.
    let arch = placed
        .as_ref()
        .map(|p| p.facts.arch.clone())
        .unwrap_or_default();
    let mut selected_backend = match backend.as_deref() {
        Some("niko") => crate::backend::Backend::StrataNiko,
        Some("poly") => crate::backend::Backend::StrataPoly,
        Some("llamacpp") | Some("llama") => crate::backend::Backend::LlamaCpp,
        _ => crate::backend::select(&arch),
    };
    if selected_backend != crate::backend::Backend::LlamaCpp
        && crate::runner::Runtime::discover_for(selected_backend).is_none()
    {
        println!(
            "  backend {} not found in runtime/{}; falling back to llama.cpp",
            selected_backend.as_str(),
            selected_backend.as_str()
        );
        selected_backend = crate::backend::Backend::LlamaCpp;
    }
    println!("  backend  {}", selected_backend.as_str());

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
            flags.draft_model = draft.as_ref().map(std::path::PathBuf::from);
            flags.spec = spec;
            flags.mmproj = mmproj.as_ref().map(std::path::PathBuf::from);
            flags.devices = devices.clone();
            flags.tensor_split = tensor_split.clone();
            let defaults = crate::registry::overrides(&crate::registry::default_root(), &model);
            flags.chat_template = defaults
                .get("template")
                .and_then(Value::as_str)
                .and_then(usable_jinja_template);
            flags.lora_adapter = defaults
                .get("adapter")
                .and_then(Value::as_str)
                .map(std::path::PathBuf::from);
            flags.embeddings = embeddings;
            flags.batch = std::env::var("OLLAMA_BATCH").ok().and_then(|s| s.parse().ok());
            flags.ubatch = std::env::var("OLLAMA_UBATCH").ok().and_then(|s| s.parse().ok());
            // Auto-detect mmproj if not given.
            if flags.mmproj.is_none() {
                if let Some(p) = &placed {
                    let _ = p;
                }
                let model_dir = std::path::Path::new(&model)
                    .parent()
                    .unwrap_or(std::path::Path::new("."));
                // model is a name, not path; search rocm/models/<model>
                let dir = std::path::PathBuf::from("rocm/models").join(&model);
                if let Ok(rd) = std::fs::read_dir(&dir) {
                    for e in rd.flatten() {
                        let n = e.file_name();
                        let ns = n.to_string_lossy();
                        if ns.contains("mmproj") && ns.ends_with(".gguf") {
                            flags.mmproj = Some(e.path());
                            break;
                        }
                    }
                }
                let _ = model_dir;
            }
            let mut source = "derived";
            if let Some(t) = crate::tune::load(model_path, ctx, kv_type.as_deref()) {
                flags.cpu_moe_layers = t.cpu_moe_layers;
                flags.flash_attention = t.flash_attention;
                // Absent in every measurement taken so far, and absent means
                // the `-ncmoe` path, unchanged.
                flags.expert_override = t.expert_override.clone();
                if let Some(n) = t.gpu_layers {
                    flags.gpu_layers = n;
                }
                source = "measured by `strata tune`";
            }
            if let Some(n) = cpu_moe {
                if p.expert_count == 0 {
                    // Dense models have no experts: the equivalent knob is how
                    // many whole layers stay on the GPU.
                    flags.gpu_layers = p.facts.layers.saturating_sub(n);
                    flags.cpu_moe_layers = 0;
                } else {
                    flags.cpu_moe_layers = n;
                }
                source = "set on the command line";
            }
            // Ollama-compatible environment overrides, applied last so an
            // explicit CLI flag still wins.
            if let Some(n) = std::env::var("OLLAMA_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|n| *n > 0)
            {
                flags.threads = n;
            }
            if let Some(v) = std::env::var("OLLAMA_FLASH_ATTENTION").ok() {
                flags.flash_attention = matches!(v.as_str(), "1" | "true" | "TRUE" | "on");
            }
            if flags.devices.is_none()
                && std::env::var("OLLAMA_SCHED_SPREAD")
                    .ok()
                    .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"))
            {
                flags.devices = Some(detected_device_ids().join(","));
            }
            println!("Loading {}", model_path.display());
            println!(
                "  ctx {}  |  gpu layers {}/{}  |  cpu-expert layers {}  |  flash-attn {}  |  KV {}  |  {} threads",
                flags.ctx,
                flags.gpu_layers,
                p.facts.layers,
                flags.cpu_moe_layers,
                flags.flash_attention,
                flags.kv_type.as_deref().unwrap_or("f16"),
                flags.threads
            );
            println!("  split {source}");
            if let Some(pattern) = &flags.expert_override {
                println!("  placement {pattern}");
            }
            let mut r =
                crate::runner::Runner::start(model_path, flags.clone(), 8099, selected_backend)?;
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
                backend: selected_backend,
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
    let compactor = crate::compact::Compactor::new(ctx, endpoint.clone(), client.clone(), compact)
        .with_tokenizer(path.as_deref());

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
        println!(
            "  Compaction  off (--no-compact); a conversation past {ctx} tokens will overflow"
        );
    }
    println!("  NOTE: the expert map in /experts is SIMULATED; the router is not observable yet.");

    let num_parallel = std::env::var("OLLAMA_NUM_PARALLEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n > 0)
        .unwrap_or(KV_SLOTS);
    let state: Shared = Arc::new(AppState {
        model: std::sync::Mutex::new(model.clone()),
        models_root: crate::registry::default_root(),
        api_key: api_key.clone(),
        compute,
        keep_alive_gen: AtomicU64::new(0),
        keep_alive_deadline: Mutex::new(None),
        residents: tokio::sync::Mutex::new(Vec::new()),
        max_resident: default_resident_capacity(crate::rocm::vram_total_gb()),
        default_ctx: ctx,
        default_kv_type: kv_type.clone(),
        default_backend: selected_backend,
        max_queue: std::env::var("OLLAMA_MAX_QUEUE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MAX_QUEUE),
        num_parallel: std::env::var("OLLAMA_NUM_PARALLEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n: &usize| *n > 0)
            .unwrap_or(KV_SLOTS),
        compactor,
        upstream: endpoint,
        client,
        placement: placed.as_ref().map(|p| PlacementReport {
            layers: p.facts.layers,
            attention_layers: p.attention_layers,
            expert_count: p.expert_count,
            vram_experts: p.vram_experts,
            ram_experts: p.ram_experts,
            disk_experts: p.disk_experts,
            dense_gb: p.dense_bytes as f32 / 1073741824.0,
            kv_gb: p.kv_bytes as f32 / 1073741824.0,
            total_gb: p.total_weight_bytes as f32 / 1073741824.0,
        }),
        hw: probe_hw(),
        vram_gb: placed
            .as_ref()
            .map(|p| p.vram_budget as f32 / 1073741824.0)
            .unwrap_or(0.0),
        ram_gb: placed
            .as_ref()
            .map(|p| p.ram_budget as f32 / 1073741824.0)
            .unwrap_or(0.0),
        sched: Sched::default(),
        metrics: Metrics::default(),
        permits: Arc::new(tokio::sync::Semaphore::new(num_parallel)),
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
        .route("/metrics", get(metrics))
        .route("/v1/metrics", get(metrics))
        .route("/compact", get(compaction))
        .route("/v1/compact", get(compaction))
        .route("/model/load", post(load_model))
        .route("/model/unload", post(unload_model))
        .route("/api/models", get(api_models))
        .route("/api/models/current", get(api_models_current))
        .route("/api/models/load", post(api_models_load))
        .route("/api/models/unload", post(api_models_unload))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/responses", post(openai_responses))
        // Harness compatibility. Claude Code speaks the Anthropic Messages API
        // and Ollama's native clients speak /api/*; neither falls back when the
        // endpoint answers in the wrong shape, so both dialects are served here
        // rather than left to the caller to proxy.
        .route("/v1/messages", post(anthropic_messages))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/tags", get(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/copy", post(ollama_copy))
        .route("/api/delete", delete(ollama_delete))
        .route("/api/create", post(ollama_create))
        .route("/api/pull", post(ollama_pull))
        .route("/api/push", post(ollama_push))
        .route(
            "/api/blobs/:digest",
            head(blob_head).post(blob_post),
        )
        .route("/api/embed", post(ollama_embed))
        .route("/api/embeddings", post(ollama_embeddings))
        .route("/api/connect", post(api_connect).delete(api_disconnect))
        .route("/api/keys", get(api_keys_list).post(api_keys_add).delete(api_keys_remove))
        .route("/api/version", get(ollama_version))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_key,
        ))
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
        let serve_dir = tower_http::services::ServeDir::new(web_dist).not_found_service(
            tower_http::services::ServeFile::new(web_dist.join("index.html")),
        );
        api.fallback_service(serve_dir)
    } else {
        println!("  (web/dist missing - run `npm install && npm run build` in web/ for the UI)");
        api
    };

    let tls_cert = std::env::var("STRATA_TLS_CERT").ok();
    let tls_key = std::env::var("STRATA_TLS_KEY").ok();
    if tls_cert.is_some() != tls_key.is_some() {
        anyhow::bail!("set both STRATA_TLS_CERT and STRATA_TLS_KEY, or neither");
    }
    if let (Some(cert_path), Some(key_path)) = (tls_cert, tls_key) {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
            .await
            .map_err(|e| anyhow::anyhow!("TLS config: {e}"))?;
        let addr: std::net::SocketAddr = listen.parse()?;
        println!("  TLS        https://{listen}");
        let handle = axum_server::Handle::new();
        let shutdown_handle = handle.clone();
        tokio::spawn(async move {
            wait_for_signal().await;
            shutdown_handle.graceful_shutdown(Some(std::time::Duration::from_secs(3)));
        });
        axum_server::bind_rustls(addr, config)
            .handle(handle)
            .serve(app.into_make_service())
            .await?;
        if let Some(compute) = &shutdown_state.compute {
            if let Ok(mut c) = compute.try_lock() {
                c.unload();
            }
        }
        return Ok(());
    }

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
        if let (Ok(mut interrupt), Ok(mut close), Ok(mut shutdown)) = (
            windows::ctrl_c(),
            windows::ctrl_close(),
            windows::ctrl_shutdown(),
        ) {
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
mod http_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> Shared {
        let client = reqwest::Client::new();
        Arc::new(AppState {
            model: std::sync::Mutex::new("test-model".into()),
            placement: None,
            models_root: std::path::PathBuf::from("rocm/models"),
            upstream: "http://127.0.0.1:1".into(),
            client: client.clone(),
            hw: Hw { cores: 1, cpu: "test".into(), gpu: "test".into(), gpus: 0, vram_total_gb: 0.0 },
            vram_gb: 0.0,
            ram_gb: 0.0,
            sched: Sched::default(),
            metrics: Metrics::default(),
            profile: Mutex::new(Vec::new()),
            experts: Mutex::new(Experts::new(1, 1, 1, 0.0, 0.0)),
            permits: Arc::new(tokio::sync::Semaphore::new(KV_SLOTS)),
            compactor: crate::compact::Compactor::new(8192, "http://127.0.0.1:1".into(), client, false),
            api_key: None,
            compute: None,
            keep_alive_gen: AtomicU64::new(0),
            keep_alive_deadline: Mutex::new(None),
            residents: tokio::sync::Mutex::new(Vec::new()),
            max_resident: 1,
            default_ctx: 8192,
            default_kv_type: None,
            default_backend: crate::backend::Backend::LlamaCpp,
            max_queue: MAX_QUEUE,
            num_parallel: KV_SLOTS,
        })
    }

    async fn fetch(app: Router, path: &str) -> (axum::http::StatusCode, Value) {
        let resp = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
        (status, body)
    }

    #[tokio::test]
    async fn version_endpoint_returns_ollama_shape() {
        let app = Router::new()
            .route("/api/version", get(ollama_version))
            .with_state(test_state());
        let (status, body) = fetch(app, "/api/version").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(!body["version"].as_str().unwrap_or("").is_empty());
    }

    #[tokio::test]
    async fn ps_endpoint_reports_an_empty_list_when_nothing_is_loaded() {
        let app = Router::new()
            .route("/api/ps", get(ollama_ps))
            .with_state(test_state());
        let (status, body) = fetch(app, "/api/ps").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body["models"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn tags_metrics_and_compact_routes_answer() {
        let state = test_state();
        let app = Router::new()
            .route("/api/tags", get(ollama_tags))
            .route("/metrics", get(metrics))
            .route("/compact", get(compaction))
            .with_state(state);
        let (status, body) = fetch(app.clone(), "/api/tags").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body["models"].is_array());
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let (status, body) = fetch(app, "/compact").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body.get("enabled").is_some());
    }

    #[tokio::test]
    async fn more_routes_answer() {
        let state = test_state();
        let app = Router::new()
            .route("/api/version", get(ollama_version))
            .route("/api/tags", get(ollama_tags))
            .route("/api/ps", get(ollama_ps))
            .route("/api/models", get(api_models))
            .route("/api/models/current", get(api_models_current))
            .route("/v1/models", get(models))
            .route("/metrics", get(metrics))
            .route("/profile", get(profile))
            .route("/experts", get(experts))
            .route("/compact", get(compaction))
            .with_state(state);
        for path in [
            "/api/version", "/api/tags", "/api/ps", "/api/models",
            "/api/models/current", "/v1/models", "/metrics", "/profile",
            "/experts", "/compact",
        ] {
            let resp = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(resp.status(), axum::http::StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn health_endpoint_admits_it_is_simulated() {
        let app = Router::new()
            .route("/health", get(health))
            .with_state(test_state());
        let (status, body) = fetch(app, "/health").await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["simulated"], true);
        assert_eq!(body["status"], "ok");
    }
}

#[cfg(test)]
mod access_tests {
    use super::*;

    #[test]
    fn loopback_addresses_are_recognised() {
        for listen in [
            "127.0.0.1:8080",
            "localhost:8080",
            "[::1]:8080",
            "127.5.5.5:1",
        ] {
            assert!(is_loopback(listen), "{listen} should be loopback");
        }
    }

    #[test]
    fn everything_reachable_from_outside_is_not() {
        // 0.0.0.0 is the one that matters: it is what someone types when they
        // want a friend to connect, and it is the one that must demand a key.
        for listen in [
            "0.0.0.0:8080",
            "192.168.1.20:8080",
            "[::]:8080",
            "10.0.0.5:80",
        ] {
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

    #[test]
    fn keep_alive_accepts_every_shape_ollama_sends() {
        assert_eq!(keep_alive_seconds(&json!(0)), Some(Some(0)));
        assert_eq!(keep_alive_seconds(&json!(30)), Some(Some(30)));
        assert_eq!(keep_alive_seconds(&json!("5m")), Some(Some(300)));
        assert_eq!(keep_alive_seconds(&json!("2h")), Some(Some(7200)));
        assert_eq!(keep_alive_seconds(&json!("10s")), Some(Some(10)));
        assert_eq!(keep_alive_seconds(&json!(-1)), Some(None));
        assert_eq!(keep_alive_seconds(&json!("-1")), Some(None));
        assert_eq!(keep_alive_seconds(&json!("nonsense")), None);
    }

    #[test]
    fn ollama_go_templates_are_not_sent_to_the_jinja_engine() {
        assert!(usable_jinja_template("{{ .Prompt }}").is_none());
        assert!(usable_jinja_template("{{- $last := eq 1 1 }}").is_none());
        assert!(usable_jinja_template("{% for message in messages %}{{ message.content }}{% endfor %}").is_some());
        assert!(usable_jinja_template("").is_none());
    }

    #[test]
    fn keep_alive_precedence_is_request_then_environment() {
        std::env::remove_var("OLLAMA_KEEP_ALIVE");
        assert_eq!(request_keep_alive(&json!({})), None);
        assert_eq!(request_keep_alive(&json!({ "keep_alive": "2m" })), Some(Some(120)));
        assert_eq!(request_keep_alive(&json!({ "keep_alive": -1 })), Some(None));
        std::env::set_var("OLLAMA_KEEP_ALIVE", "10m");
        assert_eq!(request_keep_alive(&json!({})), Some(Some(600)));
        assert_eq!(request_keep_alive(&json!({ "keep_alive": "1m" })), Some(Some(60)));
        std::env::remove_var("OLLAMA_KEEP_ALIVE");
    }

    #[test]
    fn resident_capacity_scales_with_vram() {
        std::env::remove_var("OLLAMA_MAX_LOADED_MODELS");
        assert_eq!(default_resident_capacity(8.0), 1);
        assert_eq!(default_resident_capacity(16.0), 1);
        assert_eq!(default_resident_capacity(32.0), 2);
        assert_eq!(default_resident_capacity(64.0), 4);
        std::env::set_var("OLLAMA_MAX_LOADED_MODELS", "3");
        assert_eq!(default_resident_capacity(4.0), 3);
        std::env::remove_var("OLLAMA_MAX_LOADED_MODELS");
    }

    #[test]
    fn embed_inputs_read_both_ollama_field_names() {
        assert_eq!(
            embed_inputs(&json!({ "input": "hello" }), false),
            vec!["hello"]
        );
        assert_eq!(
            embed_inputs(&json!({ "prompt": "hello" }), true),
            vec!["hello"]
        );
        let mixed = json!({ "input": ["a", { "text": "b" }, 7] });
        assert_eq!(embed_inputs(&mixed, false), vec![json!("a"), json!({ "text": "b" })]);
    }
}

// ------------------------------------------------- harness compatibility
//
// Strata's own API is OpenAI-shaped, which covers most harnesses but not all.
// Ollama's native clients and Claude Code's Anthropic client each speak their
// own dialect, and neither falls back when the endpoint answers in the wrong
// shape. These handlers translate on the way in and out so one compute process
// serves all three, which is the whole point of being a drop-in local engine.

/// POST /v1/messages - the Anthropic Messages API, for Claude Code and friends.
async fn anthropic_messages(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let upstream = match ensure_upstream(&state, &model, request_keep_alive(&req)).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let body = match crate::anthropic::request_to_openai(&req) {
        Ok(b) => b,
        Err(e) => return anthropic_error(400, &format!("invalid request: {e}")),
    };

    let mut request = state
        .client
        .post(format!("{upstream}/v1/chat/completions"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(
            std::env::var("STRATA_UPSTREAM_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(900),
        ));
    if let Some(key) = connection_key_for_url(&upstream) {
        request = request.bearer_auth(key);
    }
    let cancel = tokio_util::sync::CancellationToken::new();
    let _cancel_guard = cancel.clone().drop_guard();
    let resp = match tokio::select! {
        resp = request.send() => resp,
        _ = cancel.cancelled() => {
            return (
                axum::http::StatusCode::from_u16(499).unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
                Json(json!({ "error": "client closed the request" })),
            ).into_response()
        }
    } {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => return anthropic_error(502, &format!("compute process returned {}", r.status())),
        Err(e) => return anthropic_error(502, &format!("cannot reach compute process ({e})")),
    };

    if !stream {
        let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        return Json(crate::anthropic::response_to_anthropic(&body, &model)).into_response();
    }

    // Streaming: OpenAI deltas in, Anthropic SSE events out. The framing differs
    // enough that this cannot be a pass-through - Anthropic brackets every block
    // with start/stop events and reports the stop reason at the end.
    let mut translator = crate::anthropic::StreamTranslator::new(&model);
    let mut inner = resp.bytes_stream();
    let stream = async_stream::stream! {
        let mut buf = String::new();
        loop {
            let Some(chunk) = inner.next().await else { break };
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(payload) {
                        for (name, data) in translator.push(&v) {
                            yield Ok::<Event, Infallible>(
                                Event::default().event(name).data(data.to_string()),
                            );
                        }
                    }
                }
            }
        }
        for (name, data) in translator.finish() {
            yield Ok(Event::default().event(name).data(data.to_string()));
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn anthropic_error(status: u16, message: &str) -> Response {
    let code = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (
        code,
        Json(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "message": message }
        })),
    )
        .into_response()
}

/// POST /api/chat - Ollama's native chat API.
async fn ollama_chat(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    ollama_forward(state, req, true).await
}

/// POST /api/generate - Ollama's native completion API.
async fn ollama_generate(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    ollama_forward(state, req, false).await
}

/// Raw prompts and suffix-based infill go through llama.cpp's native
/// `/completion` and `/infill` endpoints, which bypass the chat template and
/// support true prefix/suffix completion.
async fn raw_completion_forward(
    state: Shared,
    req: &Value,
    upstream: &str,
    model: &str,
    chat: bool,
) -> Response {
    let model = model.to_string();
    let prompt = req.get("prompt").and_then(Value::as_str).unwrap_or("");
    let suffix = req.get("suffix").and_then(Value::as_str).unwrap_or("");
    let raw = req.get("raw").and_then(Value::as_bool).unwrap_or(false);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let endpoint = if suffix.is_empty() || raw { "completion" } else { "infill" };
    let mut body = if endpoint == "completion" {
        json!({ "prompt": prompt })
    } else {
        json!({ "input_prefix": prompt, "input_suffix": suffix })
    };
    if let Some(opts) = req.get("options") {
        for (from, to) in [
            ("temperature", "temperature"),
            ("top_k", "top_k"),
            ("top_p", "top_p"),
            ("seed", "seed"),
            ("repeat_penalty", "repeat_penalty"),
            ("num_predict", "n_predict"),
        ] {
            if let Some(v) = opts.get(from) {
                body[to] = v.clone();
            }
        }
        if let Some(stop) = opts.get("stop") {
            body["stop"] = stop.clone();
        }
    }
    body["stream"] = json!(stream);

    let started = Instant::now();
    let resp = match state
        .client
        .post(format!("{upstream}/{endpoint}"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("runtime returned {}", r.status()) })),
            )
                .into_response()
        }
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    if !stream {
        let value = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        let text = value
            .get("content")
            .or_else(|| value.get("response"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut out = crate::ollama::complete(
            &model,
            chat,
            &json!({ "role": "assistant", "content": text }),
            "stop",
            &json!({}),
        );
        add_durations(&mut out, started.elapsed(), None);
        return Json(out).into_response();
    }

    let mut inner = resp.bytes_stream();
    let stream_model = model.clone();
    let stream = async_stream::stream! {
        let model = &stream_model;
        let mut buf = String::new();
        let mut first: Option<std::time::Duration> = None;
        while let Some(chunk) = inner.next().await {
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else { continue };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" { continue }
                    let Ok(v) = serde_json::from_str::<Value>(payload) else { continue };
                    let text = v.get("content")
                        .or_else(|| v.get("response"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if text.is_empty() { continue }
                    if first.is_none() { first = Some(started.elapsed()); }
                    let delta = if chat {
                        crate::ollama::chat_chunk(&json!({ "content": text }), model, false)
                    } else {
                        crate::ollama::generate_chunk(&json!({ "content": text }), model, false)
                    };
                    yield Ok::<_, Infallible>(http_body::Frame::data(bytes::Bytes::from(format!("{delta}\n"))));
                }
            }
        }
        let mut last = crate::ollama::final_chunk(model, chat, "stop", &json!({}));
        add_durations(&mut last, started.elapsed(), first);
        yield Ok(http_body::Frame::data(bytes::Bytes::from(format!("{last}\n"))));
    };
    let body = axum::body::Body::new(http_body_util::StreamBody::new(stream));
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        body,
    )
        .into_response()
}

/// Shared path for both Ollama endpoints: translate, forward, translate back.
async fn ollama_forward(state: Shared, mut req: Value, chat: bool) -> Response {
    if req.get("keep_alive").is_none() {
        if let Ok(keep) = std::env::var("OLLAMA_KEEP_ALIVE") {
            if !keep.trim().is_empty() {
                if let Some(obj) = req.as_object_mut() {
                    obj.insert("keep_alive".into(), json!(keep));
                }
            }
        }
    }
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let keep = request_keep_alive(&req).or(Some(Some(300))); // Ollama's default
    let empty_turn = if chat {
        req.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|m| m.is_empty())
    } else {
        req.get("prompt").and_then(Value::as_str).unwrap_or("").is_empty()
    };
    if empty_turn {
        let unload = matches!(keep, Some(Some(0)));
        if unload {
            drain_and_unload(&state).await.ok();
        } else {
            ensure_upstream(&state, &model, keep).await.ok();
        }
        let reason = if unload { "unload" } else { "load" };
        let mut out = crate::ollama::complete(
            &model,
            chat,
            &json!({ "role": "assistant", "content": "" }),
            reason,
            &json!({}),
        );
        out["done_reason"] = json!(reason);
        return Json(out).into_response();
    }
    apply_keep_alive(&state, &req, &model).await;
    let upstream = match ensure_upstream(&state, &model, request_keep_alive(&req)).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    if !chat
        && (req.get("raw").and_then(Value::as_bool).unwrap_or(false)
            || req.get("suffix").and_then(Value::as_str).is_some())
    {
        return raw_completion_forward(state, &req, &upstream, &model, chat).await;
    }
    apply_keep_alive(&state, &req, &model).await;
    let context = prompt_context(&state, &model, &req);
    let started = Instant::now();

    let body = match if chat {
        crate::ollama::chat_to_openai(&req)
    } else {
        crate::ollama::generate_to_openai(&req)
    } {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid request: {e}") })),
            )
                .into_response()
        }
    };

    let mut request = state
        .client
        .post(format!("{upstream}/v1/chat/completions"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(
            std::env::var("STRATA_UPSTREAM_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(900),
        ));
    if let Some(key) = connection_key_for_url(&upstream) {
        request = request.bearer_auth(key);
    }
    let cancel = tokio_util::sync::CancellationToken::new();
    let _cancel_guard = cancel.clone().drop_guard();
    let resp = match tokio::select! {
        resp = request.send() => resp,
        _ = cancel.cancelled() => {
            return (
                axum::http::StatusCode::from_u16(499).unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
                Json(json!({ "error": "client closed the request" })),
            ).into_response()
        }
    } {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("compute process returned {}", r.status()) })),
            )
                .into_response()
        }
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("cannot reach compute process ({e})") })),
            )
                .into_response()
        }
    };

    if !stream {
        let mut body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        // Retry once when an explicit response format was ignored. The hint is
        // appended as a user turn so it survives every chat template.
        if let Some(format) = req.get("format").or_else(|| req.get("response_format")) {
            let text = body
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !crate::dialects::valid_structured(text, format) {
                eprintln!("strata: response format not honoured; retrying once");
                // Rebuild from the translated request instead of the reply.
                let mut translated = crate::ollama::chat_to_openai(&req)
                    .or_else(|_| crate::ollama::generate_to_openai(&req))
                    .unwrap_or_else(|_| json!({ "messages": [] }));
                if let Some(msgs) = translated.get_mut("messages").and_then(Value::as_array_mut) {
                    msgs.push(json!({
                        "role": "user",
                        "content": "Your previous reply was not valid for the requested format. Reply again with only valid output."
                    }));
                }
                translated["stream"] = json!(false);
                if let Ok(retry_resp) = state
                    .client
                    .post(format!("{upstream}/v1/chat/completions"))
                    .json(&translated)
                    .send()
                    .await
                {
                    if retry_resp.status().is_success() {
                        body = retry_resp.json::<Value>().await.unwrap_or(body);
                    }
                }
            }
        }
        let choice = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());
        let message = choice
            .and_then(|c| c.get("message"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let finish = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .unwrap_or("stop");
        let usage = body.get("usage").cloned().unwrap_or_else(|| json!({}));
        let mut out = crate::ollama::complete(&model, chat, &message, finish, &usage);
        if !context.is_empty() {
            out.as_object_mut().unwrap().insert("context".into(), json!(context));
        }
        out.as_object_mut().unwrap().insert("compaction".into(), compaction_value(&state));
        add_durations(&mut out, started.elapsed(), None);
        return Json(out).into_response();
    }

    // Ollama streams newline-delimited JSON, not SSE. Each line is one object.
    // Build a plain NDJSON body so clients never see `data:` frames.
    let mut inner = resp.bytes_stream();
    let stream = async_stream::stream! {
        let mut buf = String::new();
        let mut usage = json!({});
        let mut finish = "stop".to_string();
        let mut first: Option<std::time::Duration> = None;
        loop {
            let Some(chunk) = inner.next().await else { break };
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<Value>(payload) else {
                        continue;
                    };
                    if let Some(u) = v.get("usage") {
                        if u.get("completion_tokens").is_some() {
                            usage = u.clone();
                        }
                    }
                    let Some(choice) = v
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                    else {
                        continue;
                    };
                    if let Some(r) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        finish = r.to_string();
                    }
                    if let Some(delta) = choice.get("delta") {
                        let has_text = delta.get("content").and_then(Value::as_str)
                            .or_else(|| delta.get("reasoning_content").and_then(Value::as_str))
                            .is_some_and(|s| !s.is_empty());
                        if has_text && first.is_none() {
                            first = Some(started.elapsed());
                        }
                        let out = if chat {
                            crate::ollama::chat_chunk(delta, &model, false)
                        } else {
                            crate::ollama::generate_chunk(delta, &model, false)
                        };
                        yield Ok::<_, Infallible>(
                            http_body::Frame::data(bytes::Bytes::from(format!("{out}\n"))),
                        );
                    }
                }
            }
        }
        let mut last = crate::ollama::final_chunk(&model, chat, &finish, &usage);
        if !context.is_empty() {
            last.as_object_mut().unwrap().insert("context".into(), json!(context));
        }
        last.as_object_mut().unwrap().insert("compaction".into(), compaction_value(&state));
        add_durations(&mut last, started.elapsed(), first);
        yield Ok(http_body::Frame::data(bytes::Bytes::from(format!("{last}\n"))));
    };
    let body = axum::body::Body::new(http_body_util::StreamBody::new(stream));
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        body,
    )
        .into_response()
}

/// Send an already-translated chat request to the compute process.
///
/// Every dialect adapter ends up here, so a dead compute process produces one
/// honest gateway error rather than a different shape per endpoint.
async fn upstream_chat(
    state: &Shared,
    upstream: &str,
    body: Value,
) -> std::result::Result<reqwest::Response, Response> {
    let mut last_error = String::new();
    for attempt in 0..3 {
        let mut request = state
            .client
            .post(format!("{upstream}/v1/chat/completions"))
            .json(&body);
        if let Some(key) = connection_key_for_url(&upstream) {
            request = request.bearer_auth(key);
        }
        match request.send().await {
            Ok(r) if r.status().is_success() => return Ok(r),
            Ok(r) => last_error = format!("compute process returned {}", r.status()),
            Err(e) => last_error = format!("cannot reach compute process ({e})"),
        }
        if attempt < 2 {
            tokio::time::sleep(std::time::Duration::from_millis(200 << attempt)).await;
        }
    }
    Err((
        axum::http::StatusCode::BAD_GATEWAY,
        Json(json!({ "error": last_error })),
    )
        .into_response())
}

/// How many models may stay resident at once. An explicit
/// `OLLAMA_MAX_LOADED_MODELS` wins; otherwise VRAM decides, conservatively,
/// because two 20 GB experts do not fit on a 16 GB card however many slots the
/// scheduler would like to offer.
fn default_resident_capacity(vram_gb: f32) -> usize {
    if let Some(n) = std::env::var("OLLAMA_MAX_LOADED_MODELS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n > 0)
    {
        return n;
    }
    if vram_gb >= 48.0 {
        4
    } else if vram_gb >= 24.0 {
        2
    } else {
        1
    }
}

/// Only pass a template to llama.cpp when it is Jinja. Ollama manifests carry
/// Go-template syntax (`{{ .Prompt }}`, `{{- $last := ... }}`), which the
/// runtime's lexer rejects outright. Those templates are kept in strata.json
/// for `show`, but the model's own embedded chat template is used instead.
fn usable_jinja_template(text: &str) -> Option<String> {
    let go_markers = ["{{.", "{{ .", "{{- $", "{{ $", "range ", ".Prompt", ".System", ".Response"];
    if go_markers.iter().any(|m| text.contains(m)) {
        return None;
    }
    let text = text.trim();
    if text.is_empty() { None } else { Some(text.to_string()) }
}

fn detected_device_ids() -> Vec<String> {
    let Some(runtime) = crate::runner::Runtime::discover() else {
        return Vec::new();
    };
    let Ok(out) = std::process::Command::new(&runtime.server)
        .arg("--list-devices")
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.contains("ROCm") || l.contains("CUDA") || l.contains("Vulkan"))
        .filter_map(|l| l.split(':').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn vram_budget_bytes() -> u64 {
    let total = (crate::rocm::vram_total_gb() as f64 * 1024.0 * 1024.0 * 1024.0) as u64;
    let overhead = std::env::var("OLLAMA_GPU_OVERHEAD")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_mul(1024 * 1024);
    total.saturating_sub(overhead)
}

fn placement_budgets() -> (u64, u64) {
    (vram_budget_bytes(), 20u64 * 1024 * 1024 * 1024)
}

fn free_port() -> u16 {
    (1..32)
        .filter_map(|_| {
            let l = std::net::TcpListener::bind("127.0.0.1:0").ok()?;
            l.local_addr().ok().map(|a| a.port())
        })
        .find(|p| *p > 0)
        .unwrap_or(18_000)
}

/// Load a second (or third…) model beside the primary one and return its
/// upstream URL. The least-recently-used resident is unloaded first.
async fn load_resident(
    state: &Shared,
    model: &str,
    keep_alive: Option<Option<u64>>,
) -> Result<String> {
    let path = crate::registry::resolve(&state.models_root, model)
        .ok_or_else(|| anyhow::anyhow!("model '{model}' not found"))?;
    let (vram, ram) = placement_budgets();
    let placement = crate::placement::Placement::solve(&path, state.default_ctx, vram, ram)?;
    let mut flags = crate::runner::Flags::derive(&placement);
    flags.kv_type = state.default_kv_type.clone();
    let overrides = crate::registry::overrides(&state.models_root, model);
    flags.chat_template = overrides
        .get("template")
        .and_then(Value::as_str)
        .map(str::to_string);
    let port = free_port();
    let mut compute = Compute {
        runner: None,
        model_path: path,
        flags,
        backend: state.default_backend,
        port,
    };
    compute.load().await?;
    let upstream = format!("http://127.0.0.1:{port}");

    let mut residents = state.residents.lock().await;
    if residents.len() >= state.max_resident {
        let victim = residents
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| r.last_used)
            .map(|(i, _)| i);
        if let Some(i) = victim {
            let gone = residents.remove(i);
            println!("  evicted {} from the resident pool", gone.model);
        }
    }
    let secs = keep_alive.and_then(|v| v);
    let expires_at = secs.map(|s| now_secs() + s);
    residents.push(Resident {
        model: model.to_string(),
        compute,
        upstream: upstream.clone(),
        last_used: Instant::now(),
        keep_alive_secs: secs,
        expires_at,
    });
    Ok(upstream)
}

/// Where a request for `model` should go: an additional resident if one holds
/// it, otherwise the primary compute process / external upstream.
async fn ensure_upstream(
    state: &Shared,
    model: &str,
    keep_alive: Option<Option<u64>>,
) -> Result<String> {
    if let Some(url) = connection_upstream(model) {
        return Ok(url);
    }
    {
        let mut residents = state.residents.lock().await;
        if let Some(r) = residents.iter_mut().find(|r| r.model == model) {
            r.last_used = Instant::now();
            let secs = match keep_alive {
                Some(v) => v,
                None => r.keep_alive_secs,
            };
            r.keep_alive_secs = secs;
            r.expires_at = secs.map(|s| now_secs() + s);
            return Ok(r.upstream.clone());
        }
    }
    let current = state.model.lock().unwrap().clone();
    if model.is_empty()
        || model == current
        || state.compute.is_none()
        || state.max_resident <= 1
    {
        return Ok(state.upstream.clone());
    }
    load_resident(state, model, keep_alive).await
}

/// Ollama reports every duration in nanoseconds. Strata measures wall time and
/// time-to-first-token at the API boundary; there is no separate load phase to
/// report once the compute process is already resident.
fn add_durations(out: &mut Value, total: std::time::Duration, first: Option<std::time::Duration>) {
    let Some(obj) = out.as_object_mut() else { return };
    let total_ns = total.as_nanos() as u64;
    obj.insert("total_duration".into(), json!(total_ns));
    obj.insert("load_duration".into(), json!(0));
    obj.insert("prompt_eval_duration".into(), json!(first.map(|d| d.as_nanos() as u64).unwrap_or(0)));
    obj.insert(
        "eval_duration".into(),
        json!(total_ns.saturating_sub(first.map(|d| d.as_nanos() as u64).unwrap_or(0))),
    );
}

fn dialect_id(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}_{nanos:x}")
}

/// POST /v1/completions - OpenAI's legacy text-completion dialect.
async fn openai_completions(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let upstream = match ensure_upstream(&state, &model, request_keep_alive(&req)).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = dialect_id("cmpl");
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = match crate::dialects::completions_to_chat(&req) {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let resp = match upstream_chat(&state, &upstream, body).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if !stream {
        let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        return Json(crate::dialects::chat_to_completions(
            &body, &model, &id, created,
        ))
        .into_response();
    }

    let mut inner = resp.bytes_stream();
    let stream = async_stream::stream! {
        let mut buf = String::new();
        loop {
            let Some(chunk) = inner.next().await else { break };
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else { continue };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" { continue }
                    let Ok(v) = serde_json::from_str::<Value>(payload) else { continue };
                    let delta = v.get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| {
                            d.get("content")
                                .or_else(|| d.get("reasoning_content"))
                                .or_else(|| d.get("reasoning"))
                        })
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let finish = v.get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !delta.is_empty() || !finish.is_empty() {
                        let out = json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": delta,
                                "logprobs": null,
                                "finish_reason": if finish.is_empty() { Value::Null } else { Value::String(finish.into()) },
                            }],
                        });
                        yield Ok::<_, Infallible>(Event::default().data(out.to_string()));
                    }
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// POST /v1/responses - the OpenAI Responses API Ollama 0.40 also speaks.
async fn openai_responses(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let upstream = match ensure_upstream(&state, &model, request_keep_alive(&req)).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = dialect_id("resp");
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = match crate::dialects::responses_to_chat(&req) {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let resp = match upstream_chat(&state, &upstream, body).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if !stream {
        let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
        return Json(crate::dialects::chat_to_response(
            &body, &model, &id, created,
        ))
        .into_response();
    }

    let mut inner = resp.bytes_stream();
    let stream = async_stream::stream! {
        let mut buf = String::new();
        let mut text = String::new();
        let mut usage = json!({});
        let start = crate::dialects::chat_to_response(&json!({}), &model, &id, created);
        yield Ok::<_, Infallible>(crate::dialects::response_event(
            "response.created",
            &json!({ "type": "response.created", "response": start }),
        ));
        // The item exists from the moment the response is created, even if
        // the model emits no visible text (a tool call or a pure refusal).
        yield Ok(crate::dialects::response_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message", "id": format!("msg_{id}"),
                    "status": "in_progress", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "" }],
                },
            }),
        ));
        loop {
            let Some(chunk) = inner.next().await else { break };
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else { continue };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" { continue }
                    let Ok(v) = serde_json::from_str::<Value>(payload) else { continue };
                    if let Some(u) = v.get("usage") {
                        if u.get("completion_tokens").is_some() { usage = u.clone(); }
                    }
                    let delta = v.get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| {
                            d.get("content")
                                .or_else(|| d.get("reasoning_content"))
                                .or_else(|| d.get("reasoning"))
                        })
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if delta.is_empty() { continue }
                    text.push_str(delta);
                    let event = json!({
                        "type": "response.output_text.delta",
                        "item_id": format!("msg_{id}"),
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    });
                    yield Ok(crate::dialects::response_event(
                        "response.output_text.delta",
                        &event,
                    ));
                }
            }
        }
        yield Ok(crate::dialects::response_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "item_id": format!("msg_{id}"),
                "output_index": 0,
                "content_index": 0,
                "text": text.clone(),
            }),
        ));
        yield Ok(crate::dialects::response_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": format!("msg_{id}"),
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text.clone() }],
                },
            }),
        ));
        let done = crate::dialects::chat_to_response(
            &json!({
                "choices": [{ "message": { "content": text } }],
                "usage": usage,
            }),
            &model,
            &id,
            created,
        );
        yield Ok(crate::dialects::response_event(
            "response.completed",
            &json!({ "type": "response.completed", "response": done }),
        ));
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Extract embedding inputs from either Ollama's `input` or the legacy
/// `prompt` field. A single string stays a single string; a list may mix plain
/// text with objects carrying `text`.
fn embed_inputs(req: &Value, legacy: bool) -> Vec<Value> {
    let raw = req.get(if legacy { "prompt" } else { "input" });
    let mut out = Vec::new();
    match raw {
        Some(Value::String(s)) => out.push(json!(s)),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(s) => out.push(json!(s)),
                    Value::Object(o) => {
                        let mut entry = serde_json::Map::new();
                        if let Some(s) = o.get("text").and_then(Value::as_str) {
                            entry.insert("text".into(), json!(s));
                        }
                        for key in ["image", "audio"] {
                            if let Some(data) = o.get(key).and_then(Value::as_str) {
                                entry.insert(key.into(), json!(data));
                            }
                        }
                        if !entry.is_empty() {
                            out.push(Value::Object(entry));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    out
}

/// POST /api/embed and the legacy /api/embeddings.
///
/// llama.cpp exposes embeddings at `/v1/embeddings` in recent builds and at
/// `/embedding` in older ones. Try the OpenAI shape first and fall back on a
/// 404, so one Strata binary works with either runtime.
async fn embed_forward(state: Shared, req: Value, legacy: bool) -> Response {
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let inputs = embed_inputs(&req, legacy);
    if inputs.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "input (or prompt) must contain some text" })),
        )
            .into_response();
    }
    let upstream = match ensure_upstream(&state, &model, request_keep_alive(&req)).await {
        Ok(u) => u,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    apply_keep_alive(&state, &req, &model).await;
    let mut body = json!({ "model": model, "input": inputs });
    if let Some(d) = req.get("dimensions").and_then(Value::as_u64) {
        body["dimensions"] = json!(d);
    }
    if let Some(t) = req.get("truncate").and_then(Value::as_bool) {
        body["truncate"] = json!(t);
    }
    let mut resp = match state
        .client
        .post(format!("{upstream}/v1/embeddings"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("cannot reach compute process ({e})") })),
            )
                .into_response()
        }
    };
    if resp.status() == axum::http::StatusCode::NOT_FOUND {
        let content = if legacy {
            json!(inputs[0])
        } else {
            json!(inputs)
        };
        let old = json!({ "content": content });
        match state
            .client
            .post(format!("{upstream}/embedding"))
            .json(&old)
            .send()
            .await
        {
            Ok(r) => resp = r,
            Err(e) => {
                return (
                    axum::http::StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("cannot reach compute process ({e})") })),
                )
                    .into_response()
            }
        }
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return (
            status,
            Json(json!({ "error": format!("compute process returned {status}: {text}") })),
        )
            .into_response();
    }
    let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
    let vectors: Vec<Value> = if let Some(data) = body.get("data").and_then(Value::as_array) {
        data.iter().filter_map(|d| d.get("embedding").cloned()).collect()
    } else if let Some(v) = body.get("embeddings").and_then(Value::as_array) {
        v.clone()
    } else if let Some(v) = body.get("embedding") {
        vec![v.clone()]
    } else {
        Vec::new()
    };
    if legacy {
        Json(json!({ "embedding": vectors.first().cloned().unwrap_or(json!([])) })).into_response()
    } else {
        Json(json!({ "model": model, "embeddings": vectors })).into_response()
    }
}

async fn ollama_embed(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    embed_forward(state, req, false).await
}

async fn ollama_embeddings(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    embed_forward(state, req, true).await
}

/// POST /api/copy - give an installed model a second name.
async fn ollama_copy(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let source = req.get("source").and_then(Value::as_str).unwrap_or("");
    let destination = req
        .get("destination")
        .and_then(Value::as_str)
        .unwrap_or("");
    match crate::registry::copy(&state.models_root, source, destination) {
        Ok(()) => Json(json!({ "status": "success" })).into_response(),
        Err(e) => (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// DELETE /api/delete - remove a model from this machine.
async fn ollama_delete(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let name = req.get("model").and_then(Value::as_str).unwrap_or("");
    if state.model.lock().unwrap().as_str() == name {
        return (
            axum::http::StatusCode::CONFLICT,
            Json(json!({ "error": "unload the model before deleting it" })),
        )
            .into_response();
    }
    match crate::registry::remove(&state.models_root, name) {
        Ok(()) => Json(json!({ "status": "success" })).into_response(),
        Err(e) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// POST /api/create - derive a named model from one already installed.
async fn ollama_create(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let name = req.get("model").and_then(Value::as_str).unwrap_or("");
    let from = req.get("from").and_then(Value::as_str).unwrap_or("");
    let files = req
        .get("files")
        .and_then(Value::as_object)
        .cloned();
    if files.is_none() && from.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "create needs either `from` or `files`" })),
        )
            .into_response();
    }
    let mut overrides = serde_json::Map::new();
    for key in ["system", "template", "renderer", "parser", "license", "parameters", "messages"] {
        if let Some(v) = req.get(key) {
            overrides.insert(key.into(), v.clone());
        }
    }
    let result = match files {
        Some(f) => crate::registry::create_from_blobs(
            &state.models_root,
            name,
            &f,
            &Value::Object(overrides),
        ),
        None => crate::registry::create_from(
            &state.models_root,
            name,
            from,
            &Value::Object(overrides),
        ),
    };
    match result {
        Ok(()) => ndjson_response(vec![
            json!({ "status": "reading model metadata" }).to_string(),
            json!({ "status": "creating model" }).to_string(),
            json!({ "status": "success" }).to_string(),
        ]),
        Err(e) => ndjson_response(vec![json!({ "error": e.to_string() }).to_string()]),
    }
}

/// POST /api/pull - the CLI's pull path, with Ollama's streaming status shape.
async fn ollama_pull(Json(req): Json<Value>) -> Response {
    let model = req.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    if model.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model is required" })),
        )
            .into_response();
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let progress_tx = tx.clone();
    let hub = crate::hub::Hub::with_progress(move |line| {
        let _ = progress_tx.send(line);
    });
    let pull_model = model.clone();
    tokio::spawn(async move {
        match hub.pull(&pull_model).await {
            Ok(_) => {
                let _ = tx.send("success".into());
            }
            Err(e) => {
                let _ = tx.send(format!("error: {e}"));
            }
        }
    });
    let stream = async_stream::stream! {
        yield Ok::<_, Infallible>(
            http_body::Frame::data(bytes::Bytes::from_static(b"{\"status\":\"pulling manifest\"}\n"))
        );
        while let Some(line) = rx.recv().await {
            let value = if line == "success" {
                json!({ "status": "success" })
            } else if let Some(error) = line.strip_prefix("error: ") {
                json!({ "error": error })
            } else {
                json!({ "status": line })
            };
            yield Ok(http_body::Frame::data(bytes::Bytes::from(format!("{value}\n"))));
        }
    };
    let body = axum::body::Body::new(http_body_util::StreamBody::new(stream));
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        body,
    )
        .into_response()
}

/// POST /api/push - push a model to the registry configured on this machine.
async fn ollama_push(Json(req): Json<Value>) -> Response {
    let name = req.get("model").and_then(Value::as_str).unwrap_or("");
    match crate::push::push(name, None, None, false).await {
        Ok(()) => ndjson_response(vec![
            json!({ "status": "retrieving manifest" }).to_string(),
            json!({ "status": "pushing manifest" }).to_string(),
            json!({ "status": "success" }).to_string(),
        ]),
        Err(e) => ndjson_response(vec![json!({ "error": e.to_string() }).to_string()]),
    }
}

/// HEAD /api/blobs/:digest - does this blob exist already?
async fn blob_head(
    State(state): State<Shared>,
    axum::extract::Path(digest): axum::extract::Path<String>,
) -> Response {
    let path = match crate::blob::blob_path(&state.models_root, &digest) {
        Ok(p) => p,
        Err(_) => return axum::http::StatusCode::BAD_REQUEST.into_response(),
    };
    if path.exists() {
        axum::http::StatusCode::OK.into_response()
    } else {
        axum::http::StatusCode::NOT_FOUND.into_response()
    }
}

/// POST /api/blobs/:digest - upload a blob, verifying its digest before it is
/// given a content-addressed name.
async fn blob_post(
    State(state): State<Shared>,
    axum::extract::Path(digest): axum::extract::Path<String>,
    body: axum::body::Body,
) -> Response {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tokio::io::AsyncWriteExt;

    let hex = match crate::blob::parse_digest(&digest) {
        Ok(h) => h,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let dir = crate::blob::blob_dir(&state.models_root);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response();
    }
    let tmp = dir.join(format!("sha256-{hex}.partial"));
    let final_path = dir.join(format!("sha256-{hex}"));
    if final_path.exists() {
        return axum::http::StatusCode::CREATED.into_response();
    }
    let mut body: Body = body;
    let result: anyhow::Result<()> = async {
        let mut file = tokio::fs::File::create(&tmp).await?;
        while let Some(frame) = body.frame().await {
            let frame = frame?;
            let bytes = frame
                .into_data()
                .map_err(|_| anyhow::anyhow!("trailers are not supported in a blob upload"))?;
            file.write_all(&bytes).await?;
        }
        file.flush().await?;
        drop(file);
        let hash_path = tmp.clone();
        let got = tokio::task::spawn_blocking(move || {
            crate::blob::sha256_hex(std::fs::File::open(&hash_path)?)
        })
        .await??;
        anyhow::ensure!(got == hex, "digest mismatch: got {got}, expected {hex}");
        std::fs::rename(&tmp, &final_path)?;
        Ok(())
    }
    .await;
    match result {
        Ok(()) => axum::http::StatusCode::CREATED.into_response(),
        Err(e) => {
            std::fs::remove_file(crate::blob::blob_dir(&state.models_root).join(format!(
                "sha256-{hex}.partial"
            )))
            .ok();
            (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// GET /api/tags - the model list Ollama clients expect.
async fn ollama_tags(State(state): State<Shared>) -> Json<Value> {
    let entries = crate::registry::scan(&state.models_root);
    let models: Vec<Value> = entries
        .iter()
        .map(crate::ollama::tag_entry)
        .collect();
    Json(json!({ "models": models }))
}

/// GET /api/version - some clients probe this before anything else.
async fn ollama_version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// GET /api/ps - what is loaded right now.
async fn ollama_ps(State(state): State<Shared>) -> Json<Value> {
    let current = state.model.lock().unwrap().clone();
    let entries = crate::registry::scan(&state.models_root);
    let expires = state
        .keep_alive_deadline
        .lock()
        .unwrap()
        .map(crate::ollama::iso_utc);
    let mut models: Vec<Value> = Vec::new();

    let loaded = state
        .compute
        .as_ref()
        .map(|c| c.try_lock().map(|g| g.is_loaded()).unwrap_or(true))
        .unwrap_or(false);
    if loaded {
        if let Some(entry) = entries.iter().find(|e| e.id == current) {
            models.push(json!({
                "name": current,
                "model": current,
                "size": entry.file_bytes,
                "digest": entry.digest,
                "details": crate::ollama::details(entry),
                "vram": (state.vram_gb * 1_000_000_000.0) as u64,
                "expires_at": expires,
            }));
        } else {
            models.push(json!({
                "name": current,
                "model": current,
                "size": 0,
                "digest": "strata-external",
                "details": { "format": "gguf", "family": "unknown" },
                "vram": (state.vram_gb * 1_000_000_000.0) as u64,
                "expires_at": expires,
            }));
        }
    }

    for r in state.residents.lock().await.iter() {
        let entry = entries.iter().find(|e| e.id == r.model);
        models.push(json!({
            "name": r.model,
            "model": r.model,
            "size": entry.map(|e| e.file_bytes).unwrap_or(0),
            "digest": entry.map(|e| e.digest.clone()).unwrap_or_else(|| "strata-external".into()),
            "details": entry
                .map(crate::ollama::details)
                .unwrap_or_else(|| json!({ "format": "gguf", "family": "unknown" })),
            "expires_at": r.expires_at,
        }));
    }
    Json(json!({ "models": models }))
}

/// POST /api/show - what a model file declares about itself.
async fn ollama_show(State(state): State<Shared>, Json(req): Json<Value>) -> Response {
    let name = req.get("model").and_then(|v| v.as_str()).unwrap_or("");
    let entries = crate::registry::scan(&state.models_root);
    let Some(e) = entries.iter().find(|x| x.id == name) else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model '{name}' not found") })),
        )
            .into_response();
    };
    let verbose = req
        .get("verbose")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Json(crate::ollama::show_body_verbose(e, verbose)).into_response()
}
