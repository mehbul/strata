//! Strata — a local inference engine for mixture-of-experts models on ROCm.
//!
//! The command surface deliberately mirrors Ollama's: `pull`, `create`, `show`,
//! `run`, `stop`, `ps`, `cp`, `rm`, `push`, `signin`, `signout` and `launch`
//! all mean what an Ollama user expects. Strata's engine-specific commands
//! (`setup`, `plan`, `tune`, `bench`, `inspect`, `mcp`) sit alongside them.

mod anthropic;
mod auth;
mod backend;
mod bench;
mod blob;
mod compact;
mod daemon;
mod dequant;
mod dialects;
mod doctor;
mod expert;
mod gguf;
mod hardware;
mod hub;
mod mcp;
mod modelfile;
mod ollama;
mod placement;
mod push;
mod registry;
mod rocm;
mod run;
mod runner;
mod server;
mod setup;
mod tools;
mod tune;
mod tokenizer;
mod updater;

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use serde_json::{json, Value};
use tracing_subscriber::EnvFilter;

fn config_value(key: &str) -> Option<String> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let text = std::fs::read_to_string(std::path::Path::new(&home).join(".strata/config.toml")).ok()?;
    text.lines()
        .find_map(|line| {
            let line = line.trim();
            if line.starts_with('#') { return None }
            let (k, v) = line.split_once('=')?;
            (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
        })
        .filter(|v| !v.is_empty())
}

fn default_host() -> String {
    config_value("host")
        .or_else(|| std::env::var("STRATA_HOST").ok())
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8080".into())
}

fn default_ctx() -> usize {
    config_value("context_length")
        .or_else(|| std::env::var("OLLAMA_CONTEXT_LENGTH").ok())
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n: &usize| *n > 0)
        .unwrap_or(8192)
}

fn env_kv_type() -> Option<String> {
    config_value("kv_cache_type")
        .or_else(|| std::env::var("OLLAMA_KV_CACHE_TYPE").ok())
        .filter(|s| !s.trim().is_empty())
}

fn env_api_key() -> Option<String> {
    config_value("api_key")
        .or_else(|| std::env::var("STRATA_API_KEY").ok())
        .or_else(|| std::env::var("OLLAMA_API_KEY").ok())
        .filter(|s| !s.trim().is_empty())
}

/// Resolve a positional argument or its long-flag alias.
fn either(positional: Option<String>, flag: Option<String>, what: &str) -> Result<String> {
    positional
        .or(flag)
        .filter(|s| !s.trim().is_empty())
        .with_context(|| format!("{what} is required (see `strata help`)"))
}

#[derive(Parser)]
#[command(
    name = "strata",
    version,
    about = "Strata - Ollama-compatible MoE inference engine (Rust, ROCm)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Print timing information for each response
    #[arg(long, global = true)]
    verbose: bool,
    /// Do not wrap long lines in interactive output
    #[arg(long, global = true)]
    nowordwrap: bool,
    /// Address of a running Strata server (also STRATA_HOST / OLLAMA_HOST)
    #[arg(long, global = true)]
    host: Option<String>,
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the server in the background
    Start {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
    },
    /// Stop the background server and its compute child
    Stop,
    /// Report PID, health and log location
    Status {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Detect this machine, pick a context that fits it, and measure the
    /// fastest expert split for it. Run this once after cloning.
    Setup {
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        allow_disk: bool,
    },
    /// Detected hardware and ROCm toolchain
    Info,
    /// Check runtime, GPU, models, console and server in one pass
    Doctor {
        #[arg(long)]
        host: Option<String>,
    },
    /// Models present on disk, read from their GGUF headers
    List {
        #[arg(long)]
        json: bool,
    },
    /// Installed models (registry view)
    Models {
        #[arg(long)]
        verify: bool,
    },
    /// Show information for a model
    Show {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(long = "model", value_name = "NAME")]
        model_flag: Option<String>,
        /// Print the generated Modelfile
        #[arg(long)]
        modelfile: bool,
        /// Print the chat template
        #[arg(long)]
        template: bool,
        /// Print the license
        #[arg(long)]
        license: bool,
        /// Print the parameters
        #[arg(long)]
        parameters: bool,
        /// Include the full tokenizer arrays
        #[arg(long)]
        full: bool,
    },
    /// Create a model from a Modelfile
    Create {
        #[arg(value_name = "NAME")]
        name: Option<String>,
        #[arg(short = 'f', long, value_name = "FILE")]
        file: Option<String>,
        /// Use an installed model as the base instead of a Modelfile
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        template: Option<String>,
    },
    /// Run a model: one-shot prompt, or an interactive chat shell
    Run {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(value_name = "PROMPT")]
        prompt: Vec<String>,
        /// Server address
        #[arg(long)]
        host: Option<String>,
        /// Response format: json, or a JSON schema
        #[arg(long)]
        format: Option<String>,
        /// How long to keep the model resident (0 unloads, -1 is forever)
        #[arg(long)]
        keepalive: Option<String>,
        /// Enable thinking
        #[arg(long)]
        think: Option<bool>,
        /// Image file to send with the prompt (vision models)
        #[arg(long)]
        image: Option<String>,
    },
    /// Stop a running model and free its memory
    Stop {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(long = "model", value_name = "NAME")]
        model_flag: Option<String>,
        #[arg(long)]
        host: Option<String>,
    },
    /// Pull a model from Ollama's library or a Hugging Face repository
    Pull {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(long = "model", value_name = "NAME")]
        model_flag: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Push a model to a registry (requires STRATA_REGISTRY and a token)
    Push {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(long)]
        registry: Option<String>,
        #[arg(long)]
        token: Option<String>,
        #[arg(long)]
        insecure: bool,
    },
    /// Sign in to a model registry
    Signin {
        #[arg(long)]
        token: Option<String>,
        #[arg(long)]
        registry: Option<String>,
    },
    /// Sign out of a model registry
    Signout,
    /// List running models
    Ps {
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Copy a model under a second name
    Cp {
        #[arg(value_name = "SOURCE")]
        source: Option<String>,
        #[arg(value_name = "DESTINATION")]
        destination: Option<String>,
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
    },
    /// Remove a model from this machine
    Rm {
        #[arg(value_name = "MODEL")]
        model: Option<String>,
        #[arg(long = "model", value_name = "NAME")]
        model_flag: Option<String>,
        #[arg(long)]
        purge: bool,
        /// Keep shared blobs (the default; explicit for scripts)
        #[arg(long)]
        keep_blobs: bool,
    },
    /// Export a model, its config and tags as a .tar.gz bundle
    Export {
        #[arg(long)]
        model: String,
        #[arg(long)]
        out: String,
    },
    /// Import a model bundle
    Import {
        #[arg(long)]
        file: String,
    },
    /// Quantize a GGUF with the bundled llama.cpp tool
    Quantize {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "Q4_K_M")]
        kind: String,
    },
    /// Split a GGUF into shards
    Split {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = 2)]
        parts: usize,
    },
    /// Merge GGUF shards back into one file
    Merge {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
    },
    /// Remove blobs no model references
    Gc {
        #[arg(long)]
        purge: bool,
    },
    /// Launch the web console or an integration
    Launch {
        #[arg(value_name = "APP")]
        app: Option<String>,
        #[arg(long)]
        host: Option<String>,
    },
    /// Ask a running server to load a different model
    Switch {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(value_name = "MODEL")]
        model_pos: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
    },
    /// Everything a GGUF file declares about itself
    Inspect {
        #[arg(long)]
        path: String,
        #[arg(long)]
        all_layers: bool,
    },
    /// Tokenize text with the GGUF's own vocabulary and merges
    Tokenize {
        #[arg(long)]
        path: String,
        #[arg(long)]
        text: String,
        #[arg(long)]
        ids: bool,
    },
    /// Decode a few weights from a Q4_K / Q6_K tensor with Strata's own decoder
    Dequant {
        #[arg(long)]
        path: String,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 8)]
        count: usize,
    },
    /// Where a model's weights would sit, measured from its GGUF
    Plan {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        #[arg(long)]
        vram: Option<f32>,
        #[arg(long, default_value_t = 20.0)]
        ram: f32,
    },
    /// Measure the fastest expert split for this machine
    Tune {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        #[arg(long)]
        kv_type: Option<String>,
        #[arg(long)]
        save: bool,
    },
    /// Load a model and serve the HTTP API and web console
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long)]
        port: Option<u16>,
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = default_ctx())]
        ctx: usize,
        #[arg(long)]
        cpu_moe: Option<usize>,
        #[arg(long)]
        upstream: Option<String>,
        #[arg(long)]
        no_compact: bool,
        #[arg(long)]
        kv_type: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        draft: Option<String>,
        #[arg(long)]
        spec: Option<usize>,
        #[arg(long)]
        mmproj: Option<String>,
        #[arg(long)]
        devices: Option<String>,
        #[arg(long)]
        tensor_split: Option<String>,
        #[arg(long, default_value = "auto")]
        backend: Option<String>,
        /// Enable embeddings in the compute process
        #[arg(long)]
        embeddings: bool,
    },
    /// Install or remove the logon-time background task
    Service {
        #[arg(long)]
        install: bool,
        #[arg(long)]
        uninstall: bool,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
    },
    /// Control the background server (start / stop / status)
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Print a shell completion script
    Completion {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// Read or write a value in ~/.strata/config.toml
    Config {
        #[arg(value_name = "KEY")]
        key: Option<String>,
        #[arg(value_name = "VALUE")]
        value: Option<String>,
    },
    /// Refresh the global `strata` command shim
    Install,
    /// Pull the latest source and rebuild (needs a clean working tree)
    Update {
        #[arg(long)]
        stash: bool,
    },
    /// Run as an MCP server over stdio
    Mcp,
    /// Benchmark a running instance
    Bench {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 2048)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 256)]
        output_tokens: usize,
        #[arg(long)]
        json: Option<String>,
        #[arg(long)]
        compare: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let debug = std::env::var("OLLAMA_DEBUG")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"));
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| if debug { EnvFilter::try_new("debug") } else { EnvFilter::try_new("info") })
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();
    let host = cli.host.unwrap_or_else(default_host);

    match cli.command {
        Commands::Setup { model, force, dry_run, allow_disk } => {
            setup::run(model, force, dry_run, allow_disk).await?
        }
        Commands::Info => rocm::print_info()?,
        Commands::Doctor { host: doctor_host } => {
            doctor::run(&doctor_host.unwrap_or(host)).await?
        }
        Commands::List { json } => {
            let entries = registry::scan(&registry::default_root());
            if json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
                return Ok(());
            }
            // Ollama's `list` is a short table; Strata's detailed placement view
            // stays available as `models` / `info`.
            println!("NAME\tID\tSIZE\tMODIFIED");
            for e in registry::scan(&registry::default_root()) {
                let id: String = e.digest.chars().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
                println!(
                    "{}\t{}\t{}\t{}",
                    e.id,
                    id,
                    e.file_bytes,
                    ollama::iso_utc(e.modified_secs)
                );
            }
        }
        Commands::Models { verify } => {
            let entries = registry::scan(&registry::default_root());
            if verify {
                for e in &entries {
                    use std::io::Read;
                    let mut f = std::fs::File::open(&e.path).ok();
                    let mut magic = [0u8; 4];
                    let ok = f
                        .as_mut()
                        .and_then(|f| f.read_exact(&mut magic).ok())
                        .is_some()
                        && &magic == b"GGUF";
                    println!("{}  {}", if ok { "OK  " } else { "BAD " }, e.id);
                }
                return Ok(());
            }
            for e in entries {
                println!(
                    "{}  {}  {} experts (top-{})  ctx {}  {:.1} GiB{}",
                    e.id, e.arch, e.experts, e.experts_used, e.context,
                    e.file_bytes as f64 / 1073741824.0,
                    if e.is_moe { "  [MoE]" } else { "" }
                );
            }
        }
        Commands::Show { model, model_flag, modelfile, template, license, parameters, full } => {
            let name = either(model, model_flag, "model")?;
            let root = registry::default_root();
            let entries = registry::scan(&root);
            let Some(e) = entries.iter().find(|e| e.id == name) else {
                bail!("model '{name}' not found (see `strata list`)");
            };
            let body = ollama::show_body_verbose(e, full);
            if modelfile {
                println!("{}", body["modelfile"].as_str().unwrap_or(""));
            } else if template {
                println!("{}", body["template"].as_str().unwrap_or(""));
            } else if license {
                // The GGUF declares its own licence; a derived model may add
                // one on top. Show both rather than only the override.
                let declared: Vec<&String> = e
                    .metadata_info
                    .iter()
                    .filter(|(k, _)| {
                        k == "general.license"
                            || k == "general.license.name"
                            || k == "general.license.link"
                    })
                    .map(|(_, v)| v)
                    .collect();
                let overrides = registry::overrides(&root, &name);
                if !declared.is_empty() {
                    for value in declared {
                        println!("{value}");
                    }
                }
                if !overrides["license"].is_null() {
                    println!("{}", serde_json::to_string_pretty(&overrides["license"]).unwrap_or_default());
                }
            } else if parameters {
                println!("{}", body["parameters"].as_str().unwrap_or(""));
            } else {
                println!("{}", serde_json::to_string_pretty(&body)?);
            }
        }
        Commands::Create { name, file, from, system, template } => {
            let name = name.context("model name is required: strata create <name> -f Modelfile")?;
            let parsed = if let Some(path) = file {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {path}"))?;
                modelfile::Modelfile::parse(&text)?
            } else if let Some(from) = from {
                let mut m = modelfile::Modelfile { from, ..Default::default() };
                if let Some(v) = system { m.system = Some(v); }
                if let Some(v) = template { m.template = Some(v); }
                m
            } else {
                bail!("create needs -f <Modelfile> or --from <model>");
            };
            registry::create_from_modelfile(&registry::default_root(), &name, &parsed)?;
            println!("created {name}");
        }
        Commands::Run { model, prompt, host: run_host, format, keepalive, think, image } => {
            let model = model.unwrap_or_default();
            let model = if model.is_empty() {
                registry::scan(&registry::default_root())
                    .first()
                    .map(|e| e.id.clone())
                    .context("no models installed; run `strata pull --model <name>` first")?
            } else { model };
            run::run(run::RunArgs {
                host: run_host.unwrap_or(host),
                model,
                prompt: if prompt.is_empty() { None } else { Some(prompt) },
                format,
                keepalive,
                image,
                think,
                verbose: cli.verbose,
            }).await?
        }
        Commands::Stop { model, model_flag, host: stop_host } => {
            let name = either(model, model_flag, "model")?;
            let client = reqwest::Client::new();
            let base = stop_host.unwrap_or(host);
            let current: Value = client
                .get(format!("{}/api/models/current", base.trim_end_matches('/')))
                .send().await?.json().await.unwrap_or(json!({}));
            let running = current.get("model").and_then(|v| v.as_str()).unwrap_or("");
            if running != name {
                println!("{name} is not running (resident: {running})");
                return Ok(());
            }
            let resp = client
                .post(format!("{}/api/models/unload", base.trim_end_matches('/')))
                .send().await?;
            if resp.status().is_success() {
                println!("stopped {name}");
            } else {
                bail!("could not stop {name}: {}", resp.text().await?);
            }
        }
        Commands::Pull { model, model_flag, dry_run } => {
            let name = either(model, model_flag, "model")?;
            if dry_run {
                let root = registry::default_root();
                let local = root.join(hub::Hub::dir_name(&name)).join("model.gguf");
                println!("would pull {name}");
                println!("  local destination: {}", local.display());
                println!("  already present: {}", local.exists());
                return Ok(());
            }
            let dir = hub::Hub::new().pull(&name).await?;
            if let Some(stored) = dir.file_name().and_then(|s| s.to_str()) {
                registry::tag(&registry::default_root(), &name, stored)?;
            }
        }
        Commands::Push { model, registry: reg, token, insecure } => {
            let name = model.context("model name is required: strata push <model>")?;
            push::push(&name, reg.as_deref(), token.as_deref(), insecure).await?
        }
        Commands::Signin { token, registry } => auth::signin(token.as_deref(), registry.as_deref())?,
        Commands::Signout => auth::signout()?,
        Commands::Ps { host: ps_host, json } => {
            let base = ps_host.unwrap_or(host);
            let client = reqwest::Client::new();
            let resp = client
                .get(format!("{}/api/ps", base.trim_end_matches('/')))
                .send().await
                .context("cannot reach Strata; is `strata serve` running?")?;
            let body: Value = resp.json().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&body)?);
            } else {
                println!("NAME\tID\tSIZE\tPROCESSOR\tCONTEXT\tUNTIL");
                for m in body.get("models").and_then(|v| v.as_array()).unwrap_or(&vec![]) {
                    let id = m
                        .get("digest")
                        .and_then(|v| v.as_str())
                        .and_then(|d| d.strip_prefix("sha256:"))
                        .or_else(|| m.get("digest").and_then(|v| v.as_str()))
                        .unwrap_or("");
                    let id: String = id.chars().take(12).collect();
                    let until = m
                        .get("expires_at")
                        .and_then(|v| v.as_str())
                        .unwrap_or("never");
                    println!("{}\t{}\t{}\tGPU\t{}\t{}",
                        m.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        id,
                        m.get("size").and_then(|v| v.as_u64()).unwrap_or(0),
                        m.get("details").and_then(|d| d.get("context_length"))
                            .and_then(|v| v.as_u64()).unwrap_or(0),
                        until);
                }
            }
        }
        Commands::Cp { source, destination, from, to } => {
            let source = either(source, from, "source")?;
            let destination = either(destination, to, "destination")?;
            registry::copy(&registry::default_root(), &source, &destination)?;
            println!("copied {source} -> {destination}");
        }
        Commands::Rm { model, model_flag, purge, keep_blobs } => {
            let name = either(model, model_flag, "model")?;
            registry::remove(&registry::default_root(), &name)?;
            println!("removed {name}");
            if purge && !keep_blobs {
                for path in blob::unreferenced(&registry::default_root()) {
                    std::fs::remove_file(&path)?;
                    println!("purged {}", path.display());
                }
            }
        }
        Commands::Export { model, out } => tools::export_bundle(&model, &out)?,
        Commands::Import { file } => tools::import_bundle(&file)?,
        Commands::Quantize { input, output, kind } => tools::quantize(&input, &output, &kind)?,
        Commands::Split { input, output, parts } => tools::split(&input, parts, &output)?,
        Commands::Merge { input, output } => tools::merge(&input, &output)?,
        Commands::Gc { purge } => {
            let stale = blob::unreferenced(&registry::default_root());
            if stale.is_empty() {
                println!("no unreferenced blobs");
            } else {
                for path in &stale {
                    println!("{}", path.display());
                    if purge {
                        std::fs::remove_file(path)?;
                    }
                }
                println!("{} blob(s){}", stale.len(), if purge { " purged" } else { " (use --purge to delete)" });
            }
        }
        Commands::Launch { app, host: launch_host } => {
            let url = format!("{}/", launch_host.unwrap_or(host).trim_end_matches('/'));
            match app.as_deref().unwrap_or("console") {
                app => {
                    let target = match app {
                        "console" | "web" | "ui" | "" => url.clone(),
                        "open-webui" | "webui" => "http://localhost:3000".into(),
                        "anythingllm" => "http://localhost:3001".into(),
                        "librechat" => "http://localhost:3080".into(),
                        other => bail!(
                            "unknown app '{other}'. Known: console, open-webui, anythingllm, librechat"
                        ),
                    };
                    println!("Opening {target} (the integration must already be installed)");
                    let result = if cfg!(windows) {
                        std::process::Command::new("cmd").args(["/C", "start", &target]).status()
                    } else {
                        std::process::Command::new("xdg-open").arg(&target).status()
                    };
                    if result.is_err() {
                        println!("Open this address in a browser: {target}");
                    }
                }
            }
        }
        Commands::Switch { url, model, model_pos, api_key } => {
            let name = either(model_pos, model, "model")?;
            let api_key = api_key.or_else(env_api_key);
            registry::switch(&url, &name, api_key.as_deref()).await?
        }
        Commands::Tokenize { path, text, ids } => {
            let tok = tokenizer::Tokenizer::from_gguf(&path)?;
            let encoded = tok.encode(&text)?;
            if ids {
                println!("{:?}", encoded);
            } else {
                println!("{} tokens", encoded.len());
                println!("{:?}", tok.decode(&encoded)?);
            }
            let _ = tok.vocab_size();
        }
        Commands::Inspect { path, all_layers } => {
            if all_layers { gguf::inspect_all(&path)? } else { gguf::inspect(&path)? }
        }
        Commands::Dequant { path, tensor, count } => {
            let values = dequant::preview(&path, &tensor, count)?;
            println!("{tensor}: {} values decoded with Strata's reference decoder", values.len());
            println!("{values:?}");
        }
        Commands::Plan { model, ctx, vram, ram } => {
            let vram = vram.unwrap_or_else(rocm::vram_total_gb);
            placement::report(&model, ctx, vram, ram)?
        }
        Commands::Tune { model, ctx, kv_type, save } => {
            tune::run(&model, ctx, kv_type, save).await?
        }
        Commands::Bench { url, model, prompt_tokens, output_tokens, json, compare } => {
            bench::run(bench::BenchArgs { url, model, prompt_tokens, output_tokens, json, compare }).await?
        }
        Commands::Service { install, uninstall, listen, model, ctx } => {
            if uninstall {
                daemon::service_uninstall()?;
            } else if install {
                let model = match model {
                    Some(m) => m,
                    None => registry::scan(&registry::default_root())
                        .first()
                        .map(|e| e.id.clone())
                        .context("no models installed")?,
                };
                daemon::service_install(&listen, &model, ctx)?;
            } else {
                println!("use --install or --uninstall");
            }
        }
        Commands::Daemon { action } => match action {
            DaemonAction::Start { listen, model, ctx } => {
                let model = match model {
                    Some(m) => m,
                    None => registry::scan(&registry::default_root())
                        .first()
                        .map(|e| e.id.clone())
                        .context("no models installed; run `strata pull` first")?,
                };
                daemon::start(&listen, &model, ctx).await?
            }
            DaemonAction::Stop => daemon::stop("127.0.0.1:8080")?,
            DaemonAction::Status { listen } => daemon::status(&listen).await?,
        },
        Commands::Completion { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
        }
        Commands::Config { key, value } => {
            let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
            let dir = std::path::Path::new(&home).join(".strata");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join("config.toml");
            match (key, value) {
                (Some(k), Some(v)) => {
                    let mut lines: Vec<String> = std::fs::read_to_string(&path)
                        .unwrap_or_default()
                        .lines()
                        .map(str::to_string)
                        .collect();
                    lines.retain(|l| !l.trim_start().starts_with(&format!("{k} =")));
                    lines.push(format!("{k} = \"{v}\""));
                    std::fs::write(&path, lines.join("\n") + "\n")?;
                    println!("set {k}");
                }
                (Some(k), None) => {
                    println!("{}", config_value(&k).unwrap_or_default());
                }
                _ => {
                    print!("{}", std::fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
        Commands::Install => {
            updater::write_shim()?;
            println!("Refreshed the global strata command shim.");
        }
        Commands::Update { stash } => updater::run(stash)?,
        Commands::Mcp => mcp::run().await?,
        Commands::Serve {
            listen, port, model, ctx, cpu_moe, upstream, no_compact, kv_type, api_key,
            draft, spec, mmproj, devices, tensor_split, backend, embeddings,
        } => {
            let listen = match port {
                Some(p) => format!(
                    "{}:{}",
                    listen.rsplit_once(':').map(|x| x.0).unwrap_or("127.0.0.1"),
                    p
                ),
                None => listen,
            };
            server::serve(server::ServeOptions {
                listen, model, ctx, cpu_moe, upstream,
                compact: !no_compact,
                kv_type: kv_type.or_else(env_kv_type),
                api_key: api_key.or_else(env_api_key),
                draft, spec, mmproj, devices, tensor_split, backend, embeddings,
            }).await?
        }
    }
    Ok(())
}
