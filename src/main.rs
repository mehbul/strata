//! Strata — a local inference engine for mixture-of-experts models on ROCm.

mod compact;
mod hardware;
mod expert;
mod gguf;
mod hub;
mod placement;
mod rocm;
mod runner;
mod setup;
mod server;
mod tune;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "strata", version, about = "Strata - local MoE inference engine (ROCm)")]
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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match Cli::parse().command {
        Commands::Setup { model, force, dry_run, allow_disk } => {
            setup::run(model, force, dry_run, allow_disk).await?
        }
        Commands::Info => rocm::print_info()?,
        Commands::List => placement::list_local()?,
        Commands::Inspect { path } => gguf::inspect(&path)?,
        Commands::Plan { model, ctx, vram, ram } => {
            let vram = vram.unwrap_or_else(rocm::vram_total_gb);
            placement::report(&model, ctx, vram, ram)?
        }
        Commands::Pull { model } => {
            hub::Hub::new().pull(&model).await?;
        }
        Commands::Tune { model, ctx, kv_type, save } => {
            tune::run(&model, ctx, kv_type, save).await?
        }
        Commands::Serve { listen, model, ctx, cpu_moe, upstream, no_compact, kv_type, api_key } => {
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
            })
            .await?
        }
    }
    Ok(())
}
