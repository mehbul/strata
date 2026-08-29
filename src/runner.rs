//! Owns the compute process.
//!
//! Strata loads the model itself rather than talking to a separate daemon: it
//! locates a llama.cpp build, derives placement flags from `placement.rs`, and
//! runs the server as a child process it starts, health-checks and kills.

use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::placement::Placement;

/// Lines of the compute process's log kept for a post-mortem. Enough to hold
/// the backend banner and the failure that follows it.
const LOG_KEEP: usize = 60;

/// A llama.cpp installation: the server binary plus the GPU backend to preload.
#[derive(Debug, Clone)]
pub struct Runtime {
    pub server: PathBuf,
    pub lib_dir: PathBuf,
    pub backend: Option<PathBuf>,
}

impl Runtime {
    /// Look for a usable llama.cpp. `STRATA_LLAMA_DIR` wins; otherwise fall
    /// back to the copy Ollama ships, which is already built for this GPU.
    pub fn discover() -> Option<Self> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(dir) = std::env::var("STRATA_LLAMA_DIR") {
            roots.push(PathBuf::from(dir));
        }
        roots.push(PathBuf::from("runtime"));
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            roots.push(PathBuf::from(local).join("Programs/Ollama/lib/ollama"));
        }
        for root in roots {
            let server = root.join(if cfg!(windows) { "llama-server.exe" } else { "llama-server" });
            if server.is_file() {
                return Some(Self { backend: pick_backend(&root), server, lib_dir: root });
            }
        }
        None
    }
}

/// Prefer a ROCm backend, then Vulkan. CUDA is skipped: this targets AMD.
///
/// Official llama.cpp releases put the backend beside the binary; Ollama's
/// layout puts it in a per-GPU subdirectory. Both are handled.
fn pick_backend(root: &Path) -> Option<PathBuf> {
    let lib = if cfg!(windows) { "ggml-hip.dll" } else { "libggml-hip.so" };
    if root.join(lib).is_file() {
        return Some(root.join(lib));
    }
    let vk_lib = if cfg!(windows) { "ggml-vulkan.dll" } else { "libggml-vulkan.so" };
    if root.join(vk_lib).is_file() {
        return Some(root.join(vk_lib));
    }
    let mut best: Option<PathBuf> = None;
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with("rocm") && dir.join(lib).is_file() {
            best = Some(dir.join(lib));
            break;
        }
        if name.starts_with("vulkan") && best.is_none() {
            let vk = dir.join(if cfg!(windows) { "ggml-vulkan.dll" } else { "libggml-vulkan.so" });
            if vk.is_file() {
                best = Some(vk);
            }
        }
    }
    best
}

/// The flags Strata chooses for a model on this machine.
#[derive(Debug, Clone)]
pub struct Flags {
    pub ctx: usize,
    /// Layers whose expert weights stay on the CPU. Counter-intuitively this is
    /// not zero: leaving some experts in host memory frees VRAM for the layers
    /// that remain, and measures faster than forcing everything onto the GPU.
    pub cpu_moe_layers: usize,
    pub flash_attention: bool,
    pub threads: usize,
    /// A non-prefix expert placement, rendered as an `-ot` pattern.
    ///
    /// `None` for every plan the runtime can already express as `-ncmoe`,
    /// which is the common case and the one every existing measurement was
    /// taken with. Only a plan with a gap in it reaches this field.
    pub expert_override: Option<String>,
    /// Element type for the KV cache, e.g. `q8_0`. `None` leaves it at f16.
    ///
    /// The cache is allocated for the whole context at load time, so at a large
    /// `--ctx` it is VRAM the experts cannot have. Halving it buys expert
    /// layers back. Requires flash attention for the V side.
    pub kv_type: Option<String>,
}

impl Flags {
    /// Derive a starting point from measured placement. This is an estimate —
    /// `strata tune` measures the real optimum, which is usually higher.
    pub fn derive(p: &Placement) -> Self {
        let layers = p.facts.layers.max(1);
        let per_layer_experts = p.expert_bytes / layers as u64;
        // Leave room for compute buffers and the graph, not just the weights.
        let reserve = p.vram_budget / 5;
        let free_for_experts = p
            .vram_budget
            .saturating_sub(p.dense_bytes + p.kv_bytes + reserve);
        let gpu_expert_layers = if per_layer_experts > 0 {
            (free_for_experts / per_layer_experts) as usize
        } else {
            layers
        };
        Self {
            ctx: p.ctx,
            cpu_moe_layers: layers.saturating_sub(gpu_expert_layers.min(layers)),
            flash_attention: true,
            threads: std::thread::available_parallelism().map(|n| n.get() / 2).unwrap_or(6).max(1),
            expert_override: None,
            kv_type: None,
        }
    }

    fn to_args(&self, model: &Path, port: u16) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "-m".into(),
            model.to_string_lossy().into_owned(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "-c".into(),
            self.ctx.to_string(),
            "-ngl".into(),
            "99".into(),
            "-t".into(),
            self.threads.to_string(),
            "--no-webui".into(),
            // One slot, not the four this build picks on its own.
            //
            // With `-np auto` the KV cache is a single unified pool of `-c`
            // cells shared by every slot, so four slots do not buy four
            // conversations - they share one conversation's worth of cells and
            // a turn can land on a slot that does not hold the prefix. The
            // prompt cache then tries to restore the saved state into a pool
            // the live conversation already occupies, fails to find the cells,
            // and reprocesses the whole prompt: "failed to find N available
            // cells in kv cache". One slot is also what the context budget here
            // already assumes, and it costs no memory - the pool is sized by
            // `-c` either way.
            "-np".into(),
            "1".into(),
            // Render the model's own chat template. Without it llama.cpp
            // refuses any request carrying `tools` - "tools param requires
            // --jinja flag" - so a coding agent cannot talk to Strata at all,
            // and the console's thinking switch, which travels in
            // `chat_template_kwargs`, has nothing to reach.
            "--jinja".into(),
        ];
        // A pattern replaces the count rather than adding to it; passing both
        // would place the same tensors twice by two different rules.
        match &self.expert_override {
            Some(pattern) => {
                args.push("-ot".into());
                args.push(pattern.clone());
            }
            None if self.cpu_moe_layers > 0 => {
                args.push("-ncmoe".into());
                args.push(self.cpu_moe_layers.to_string());
            }
            None => {}
        }
        if self.flash_attention {
            args.push("-fa".into());
            args.push("on".into());
        }
        if let Some(t) = &self.kv_type {
            args.push("-ctk".into());
            args.push(t.clone());
            args.push("-ctv".into());
            args.push(t.clone());
        }
        args
    }
}

/// The loader search path a compute process needs.
///
/// The backend libraries live beside the server binary and in a per-GPU
/// subdirectory; both must be on the path or the GPU backend silently falls
/// back to CPU. A stock llama.cpp ROCm build links rocBLAS/hipBLAS but does not
/// ship them either, so the SDK directories go in front. Without one of these
/// the HIP backend fails to load and the GPU vanishes with no error in the log
/// - which is also why `--list-devices` needs the same environment.
pub fn loader_path(runtime: &Runtime) -> Option<std::ffi::OsString> {
    let mut search = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .unwrap_or_default();
    if let Some(backend) = runtime.backend.as_ref().and_then(|b| b.parent()) {
        search.insert(0, backend.to_path_buf());
    }
    search.insert(0, runtime.lib_dir.clone());
    for dir in crate::rocm::sdk_search_paths().into_iter().rev() {
        search.insert(0, dir);
    }
    std::env::join_paths(search).ok()
}

/// A running compute process, killed when this value is dropped.
pub struct Runner {
    child: Child,
    pub endpoint: String,
    pub flags: Flags,
    pub runtime: Runtime,
    /// The tail of what the compute process wrote to stderr. Kept because
    /// llama.cpp reports a missing rocBLAS, a bad `-ncmoe` or an out-of-memory
    /// load there and then simply never answers `/health` - so without this a
    /// failed start is indistinguishable from a slow one.
    log: Arc<Mutex<VecDeque<String>>>,
}

impl Runner {
    pub fn start(model: &Path, flags: Flags, port: u16) -> Result<Self> {
        let runtime = Runtime::discover().context(
            "no llama.cpp runtime found. Set STRATA_LLAMA_DIR to a directory holding llama-server",
        )?;
        if !model.is_file() {
            bail!("model file not found: {}", model.display());
        }

        let path = loader_path(&runtime).context("building loader search path")?;

        let mut cmd = Command::new(&runtime.server);
        cmd.args(flags.to_args(model, port))
            .env("PATH", path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(backend) = &runtime.backend {
            cmd.env("GGML_BACKEND_PATH", backend);
        }
        let mut child =
            cmd.spawn().with_context(|| format!("starting {}", runtime.server.display()))?;

        // stderr must be drained or the pipe fills and the child blocks on its
        // own logging, so a reader thread owns it for the life of the process.
        let log = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_KEEP)));
        if let Some(stderr) = child.stderr.take() {
            let sink = log.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if is_interesting(&line) {
                        println!("  llama: {}", line.trim_end());
                    }
                    let mut buf = sink.lock().unwrap_or_else(|e| e.into_inner());
                    if buf.len() == LOG_KEEP {
                        buf.pop_front();
                    }
                    buf.push_back(line);
                }
            });
        }

        Ok(Self { child, endpoint: format!("http://127.0.0.1:{port}"), flags, runtime, log })
    }

    /// The tail of the compute process's log, newest last.
    pub fn log_tail(&self, lines: usize) -> String {
        let buf = self.log.lock().unwrap_or_else(|e| e.into_inner());
        buf.iter()
            .skip(buf.len().saturating_sub(lines))
            .map(|l| format!("    {}", l.trim_end()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// PID of the compute process, for per-process resource queries.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Block until the model is loaded and answering, or give up.
    ///
    /// Reports elapsed time while waiting, because loading 20 GB of weights
    /// takes long enough that a silent prompt looks like a hang, and stops
    /// early if the process exits rather than waiting out the whole timeout on
    /// something already dead.
    pub async fn wait_ready(&mut self, timeout: Duration) -> Result<Duration> {
        let started = Instant::now();
        let client = reqwest::Client::new();
        let url = format!("{}/health", self.endpoint);
        while started.elapsed() < timeout {
            if let Ok(r) = client.get(&url).timeout(Duration::from_secs(3)).send().await {
                if r.status().is_success() {
                    print!("\r                                     \r");
                    let _ = std::io::stdout().flush();
                    return Ok(started.elapsed());
                }
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                bail!(
                    "the compute process exited during load ({status}). Its last output:\n{}",
                    self.log_tail(25)
                );
            }
            print!("\r  loading… {:>3.0}s", started.elapsed().as_secs_f32());
            let _ = std::io::stdout().flush();
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        bail!(
            "model did not finish loading within {timeout:?}. The compute process is still \
             running; its last output:\n{}",
            self.log_tail(25)
        )
    }
}

/// Whether a line from the compute process is worth putting in front of the
/// user. Loading a model produces hundreds of lines of tensor bookkeeping;
/// only the device summary and anything that went wrong belong on the console.
fn is_interesting(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    ["error", "failed", "cannot", "unable to", "out of memory", "not supported", "warning: failed"]
        .iter()
        .any(|k| l.contains(k))
        || l.starts_with("load_tensors: offloading")
        || l.starts_with("ggml_cuda_init")
}

impl Drop for Runner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Flags {
        Flags {
            ctx: 65536,
            cpu_moe_layers: 15,
            flash_attention: true,
            threads: 6,
            expert_override: None,
            kv_type: None,
        }
    }

    /// The exact command line the tuned configuration on the development
    /// machine was measured with. Every later feature is additive, and this is
    /// what proves it: if adding one changes these arguments, the measurement
    /// no longer describes what runs.
    #[test]
    fn the_tuned_configuration_produces_the_arguments_it_was_measured_with() {
        let args = base().to_args(Path::new("model.gguf"), 8099);
        assert_eq!(
            args,
            vec![
                "-m", "model.gguf",
                "--host", "127.0.0.1",
                "--port", "8099",
                "-c", "65536",
                "-ngl", "99",
                "-t", "6",
                "--no-webui",
                "-np", "1",
                "--jinja",
                "-ncmoe", "15",
                "-fa", "on",
            ]
        );
    }

    #[test]
    fn a_pattern_replaces_the_count_rather_than_joining_it() {
        let mut flags = base();
        flags.expert_override = Some(r"blk\.(0|1|5)\.ffn_.*_exps\.=CPU".into());
        let args = flags.to_args(Path::new("model.gguf"), 8099);
        assert!(args.contains(&"-ot".to_string()));
        assert!(!args.contains(&"-ncmoe".to_string()));
    }

    #[test]
    fn a_kv_type_adds_both_halves_of_the_cache() {
        let mut flags = base();
        flags.kv_type = Some("q8_0".into());
        let args = flags.to_args(Path::new("model.gguf"), 8099);
        let at = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
        assert_eq!(at("-ctk").as_deref(), Some("q8_0"));
        assert_eq!(at("-ctv").as_deref(), Some("q8_0"));
    }
}
