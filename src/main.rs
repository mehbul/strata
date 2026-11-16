//! Strata — a local inference engine for mixture-of-experts models on ROCm.

mod anthropic;
mod backend;
mod bench;
mod compact;
mod expert;
mod gguf;
mod hardware;
mod hub;
mod mcp;
mod ollama;
mod placement;
mod registry;
mod rocm;
mod runner;
mod server;
mod setup;
mod tune;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "strata",
    version,
    about = "Strata - local MoE inference engine (ROCm)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect this machine, pick a context that fits it, and measure the
    /// fastest expert split for it. Run this once after cloning.
    Setup {
        /// Model to configure for (default: the only one present)
        #[arg(long)]
        model: Option<String>,
        /// Measure again even if a tuned result already exists
        #[arg(long)]
        force: bool,
        /// Report what would be chosen without measuring or writing anything
        #[arg(long)]
        dry_run: bool,
        /// Measure even when most of the model would stream from disk
        #[arg(long)]
        allow_disk: bool,
    },
    /// Detected hardware and ROCm toolchain
    Info,
    /// Models present on disk, read from their GGUF headers
    List,
    /// Installed models (registry view)
    Models,
    /// Ask a running server to load a different model
    Switch {
        /// Base URL of a running Strata instance
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Model name or path, as `strata models` lists it
        model: String,
        /// API key, when the server requires one. Also read from STRATA_API_KEY
        #[arg(long)]
        api_key: Option<String>,
    },
    /// Everything a GGUF file declares about itself
    Inspect {
        #[arg(long)]
        path: String,
    },
    /// Where a model's weights would sit, measured from its GGUF
    Plan {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        /// VRAM budget in GiB (default: detected)
        #[arg(long)]
        vram: Option<f32>,
        /// RAM budget in GiB available for weights
        #[arg(long, default_value_t = 20.0)]
        ram: f32,
    },
    /// Copy a model out of the Ollama blob store into rocm/models
    Pull {
        #[arg(long)]
        model: String,
    },
    /// Measure the fastest expert split for this machine
    Tune {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        /// KV cache element type (e.g. q8_0). Halves what a large --ctx
        /// reserves in VRAM, leaving more of it for experts
        #[arg(long)]
        kv_type: Option<String>,
        /// Write the result next to the model so `serve` uses it
        #[arg(long)]
        save: bool,
    },
    /// Load a model and serve the HTTP API and web console
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        /// Layers whose expert weights stay on the CPU (default: derived)
        #[arg(long)]
        cpu_moe: Option<usize>,
        /// Use an existing OpenAI-compatible server instead of starting one
        #[arg(long)]
        upstream: Option<String>,
        /// Let a conversation overflow the context instead of summarising its
        /// older turns to make room
        #[arg(long)]
        no_compact: bool,
        /// KV cache element type (e.g. q8_0). Halves what a large --ctx
        /// reserves in VRAM, leaving more of it for experts
        #[arg(long)]
        kv_type: Option<String>,
        /// Require this key on every API request. Mandatory when --listen is
        /// not a loopback address. Also read from STRATA_API_KEY
        #[arg(long)]
        api_key: Option<String>,
        /// Speculative decoding draft model (GGUF). Off by default; there is
        /// no automatic selection yet, so name the draft model explicitly.
        #[arg(long)]
        draft: Option<String>,
        /// Max draft tokens per speculation step.
        #[arg(long)]
        spec: Option<usize>,
        /// Multimodal projector (mmproj) GGUF for vision models.
        #[arg(long)]
        mmproj: Option<String>,
        /// Devices for multi-GPU (comma list).
        #[arg(long)]
        devices: Option<String>,
        /// Tensor split across devices.
        #[arg(long)]
        tensor_split: Option<String>,
        /// Compute backend: auto (default), llamacpp, niko, poly.
        #[arg(long, default_value = "auto")]
        backend: Option<String>,
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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match Cli::parse().command {
        Commands::Setup {
            model,
            force,
            dry_run,
            allow_disk,
        } => setup::run(model, force, dry_run, allow_disk).await?,
        Commands::Info => rocm::print_info()?,
        Commands::List => placement::list_local()?,
        Commands::Models => {
            let entries = registry::scan(&registry::default_root());
            for e in entries {
                println!(
                    "{}  {}  {} experts (top-{})  ctx {}  {:.1} GiB{}",
                    e.id,
                    e.arch,
                    e.experts,
                    e.experts_used,
                    e.context,
                    e.file_bytes as f64 / 1073741824.0,
                    if e.is_moe { "  [MoE]" } else { "" }
                );
            }
        }
        Commands::Switch {
            url,
            model,
            api_key,
        } => {
            let api_key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
            registry::switch(&url, &model, api_key.as_deref()).await?
        }
        Commands::Inspect { path } => gguf::inspect(&path)?,
        Commands::Plan {
            model,
            ctx,
            vram,
            ram,
        } => {
            let vram = vram.unwrap_or_else(rocm::vram_total_gb);
            placement::report(&model, ctx, vram, ram)?
        }
        Commands::Pull { model } => {
            hub::Hub::new().pull(&model).await?;
        }
        Commands::Tune {
            model,
            ctx,
            kv_type,
            save,
        } => tune::run(&model, ctx, kv_type, save).await?,
        Commands::Bench {
            url,
            model,
            prompt_tokens,
            output_tokens,
            json,
        } => {
            bench::run(bench::BenchArgs {
                url,
                model,
                prompt_tokens,
                output_tokens,
                json,
            })
            .await?
        }
        Commands::Mcp => mcp::run().await?,
        Commands::Serve {
            listen,
            model,
            ctx,
            cpu_moe,
            upstream,
            no_compact,
            kv_type,
            api_key,
            draft,
            spec,
            mmproj,
            devices,
            tensor_split,
            backend,
        } => {
            server::serve(server::ServeOptions {
                listen,
                model,
                ctx,
                cpu_moe,
                upstream,
                compact: !no_compact,
                kv_type,
                api_key: api_key
                    .or_else(|| std::env::var("STRATA_API_KEY").ok())
                    .filter(|k| !k.trim().is_empty()),
                draft,
                spec,
                mmproj,
                devices,
                tensor_split,
                backend,
            })
            .await?
        }
    }
    Ok(())
}
