//! Placement computed from the model file, not from a hand-written table.
//!
//! Every number here is derived from the GGUF tensor index and metadata. Where
//! something cannot be measured it is left out rather than estimated.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::gguf::{Gguf, ModelFacts};

/// Where a weight can live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Vram,
    Ram,
    Disk,
}

#[derive(Debug, Clone)]
pub struct Placement {
    pub facts: ModelFacts,
    pub path: PathBuf,
    /// Summed from the tensor index — the real stored size.
    pub total_weight_bytes: u64,
    /// Tensors that hold routed experts (`*_exps.*`).
    pub expert_bytes: u64,
    /// Everything else: attention, norms, shared experts, embeddings.
    pub dense_bytes: u64,
    /// Number of routed experts across all layers.
    pub expert_count: usize,
    /// Bytes of KV cache per token, over layers that actually keep one.
    pub kv_bytes_per_token: u64,
    /// Layers that run full attention (the rest are recurrent, if hybrid).
    pub attention_layers: usize,
    pub ctx: usize,

    // budgets the plan was solved against
    pub vram_budget: u64,
    pub ram_budget: u64,

    // solution
    pub vram_experts: usize,
    pub ram_experts: usize,
    pub disk_experts: usize,
    pub kv_bytes: u64,
}

/// Find a model file from a name or a path.
pub fn resolve_path(model: &str) -> Option<PathBuf> {
    let direct = Path::new(model);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    let safe = model.replace([':', '/', '\\'], "_");
    for candidate in [
        PathBuf::from("rocm/models").join(&safe).join("model.gguf"),
        PathBuf::from("rocm/models").join(model).join("model.gguf"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rocm/models").join(&safe).join("model.gguf"),
    ] {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// How the routed experts divide across the three tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiers {
    pub vram_experts: usize,
    pub ram_experts: usize,
    /// Experts with nowhere to live. There is no disk tier in the runtime, so
    /// this is not a placement - it is the count that will be faulted out of
    /// the mmapped file on demand, and it is the number that decides whether a
    /// model is worth running at all.
    pub disk_experts: usize,
}

/// Fill VRAM first, then host RAM, then give up.
///
/// Greedy and deliberately so: the dense weights and the KV cache have already
/// been taken out of the VRAM budget by the caller, because those must be
/// resident for the model to run at all. Experts are what is left to place.
pub fn fill_tiers(
    expert_count: usize,
    per_expert: u64,
    vram_after_fixed: u64,
    ram_budget: u64,
) -> Tiers {
    if per_expert == 0 || expert_count == 0 {
        return Tiers { vram_experts: 0, ram_experts: 0, disk_experts: expert_count };
    }
    let vram_experts = ((vram_after_fixed / per_expert) as usize).min(expert_count);
    let ram_experts =
        ((ram_budget / per_expert) as usize).min(expert_count - vram_experts);
    Tiers {
        vram_experts,
        ram_experts,
        disk_experts: expert_count - vram_experts - ram_experts,
    }
}

impl Placement {
    /// Solve placement for a model file against a VRAM and RAM budget.
    pub fn solve(path: &Path, ctx: usize, vram_budget: u64, ram_budget: u64) -> Result<Self> {
        let gguf = Gguf::open(path).with_context(|| format!("reading {}", path.display()))?;
        let facts = ModelFacts::read(&gguf)?;

        // Routed experts are the tensors llama.cpp names `*_exps.*`; a shared
        // expert (`*_shexp.*`) is dense — it runs for every token.
        let mut expert_bytes = 0u64;
        let mut total = 0u64;
        for t in &gguf.tensors {
            let bytes = t.bytes().unwrap_or(0);
            total += bytes;
            if t.name.contains("_exps.") {
                expert_bytes += bytes;
            }
        }
        if total == 0 {
            bail!("could not size any tensor in {}", path.display());
        }
        let dense_bytes = total - expert_bytes;
        let expert_count = facts.layers * facts.experts;

        // Only full-attention layers hold a KV cache. On a hybrid model the
        // remaining layers carry a fixed recurrent state instead, which does
        // not grow with context.
        let attention_layers = match facts.full_attention_interval {
            Some(n) if n > 0 => facts.layers / n,
            _ => facts.layers,
        };
        let key_len = gguf.key_u64("attention.key_length").unwrap_or(0);
        let value_len = gguf.key_u64("attention.value_length").unwrap_or(0);
        // f16 cache: 2 bytes per element, K and V, per kv head, per layer.
        let kv_bytes_per_token =
            attention_layers as u64 * facts.kv_heads as u64 * (key_len + value_len) * 2;
        let kv_bytes = kv_bytes_per_token * ctx as u64;

        // Dense weights and the KV cache are what must be resident to run at
        // all; experts fill whatever VRAM is left, then RAM, then stay on disk.
        let per_expert = if expert_count > 0 { expert_bytes / expert_count as u64 } else { 0 };
        let vram_after_fixed = vram_budget.saturating_sub(dense_bytes + kv_bytes);
        let Tiers { vram_experts, ram_experts, disk_experts } =
            fill_tiers(expert_count, per_expert, vram_after_fixed, ram_budget);

        Ok(Self {
            facts,
            path: path.to_path_buf(),
            total_weight_bytes: total,
            expert_bytes,
            dense_bytes,
            expert_count,
            kv_bytes_per_token,
            attention_layers,
            ctx,
            vram_budget,
            ram_budget,
            vram_experts,
            ram_experts,
            disk_experts,
            kv_bytes,
        })
    }

    /// Fraction of routed experts that fit in VRAM — what a runtime would take
    /// as its offload ratio.
    pub fn vram_fraction(&self) -> f32 {
        if self.expert_count == 0 {
            return 0.0;
        }
        self.vram_experts as f32 / self.expert_count as f32
    }

    /// Layers to offload, for a runtime that offloads whole layers.
    pub fn gpu_layers(&self) -> usize {
        let resident = self.dense_bytes + self.kv_bytes;
        if resident >= self.vram_budget {
            return 0;
        }
        let per_layer = self.total_weight_bytes / self.facts.layers.max(1) as u64;
        if per_layer == 0 {
            return 0;
        }
        (((self.vram_budget - self.kv_bytes) / per_layer) as usize).min(self.facts.layers)
    }

    pub fn fits_without_disk(&self) -> bool {
        self.disk_experts == 0
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

pub fn report(model: &str, ctx: usize, vram_gb: f32, ram_gb: f32) -> Result<()> {
    // An error, reported as one. Printing the guidance and returning Ok meant
    // `strata plan` exited 0 having planned nothing, so a script or the
    // launcher could not tell the difference between that and success.
    let Some(path) = resolve_path(model) else {
        bail!(
            "no model file found for '{model}'.\n\
             Looked for rocm/models/<name>/model.gguf, or pass a path directly.\n\
             `strata pull --model {model}` copies it out of the Ollama blob store."
        )
    };
    let vram_budget = (vram_gb as f64 * GIB) as u64;
    let ram_budget = (ram_gb as f64 * GIB) as u64;
    let p = Placement::solve(&path, ctx, vram_budget, ram_budget)?;
    let f = &p.facts;

    println!("=== Placement: {} ===", f.name);
    println!("file            {}", p.path.display());
    println!("architecture    {}  ({} layers, {} vocab)", f.arch, f.layers, f.vocab);
    if f.is_moe() {
        println!(
            "experts         {} per layer, top-{} routed  ({} total)",
            f.experts, f.experts_used, p.expert_count
        );
    }
    if f.is_hybrid_ssm() {
        println!(
            "attention       {} of {} layers keep a KV cache (rest are recurrent)",
            p.attention_layers, f.layers
        );
    }
    println!();
    println!("--- measured from the tensor index ---");
    println!("total weights   {:>8.2} GiB", gib(p.total_weight_bytes));
    println!("  dense         {:>8.2} GiB   (attention, norms, shared expert, embeddings)", gib(p.dense_bytes));
    println!("  routed        {:>8.2} GiB   ({} experts, {:.1} MiB each)",
        gib(p.expert_bytes), p.expert_count,
        if p.expert_count > 0 { p.expert_bytes as f64 / p.expert_count as f64 / 1024.0 / 1024.0 } else { 0.0 });
    println!("KV @ {:<7}    {:>8.2} GiB   ({} bytes/token)", ctx, gib(p.kv_bytes), p.kv_bytes_per_token);
    println!();
    println!("--- solved against {:.1} GiB VRAM / {:.1} GiB RAM ---", vram_gb, ram_gb);
    println!("must be resident: dense {:.2} + KV {:.2} = {:.2} GiB",
        gib(p.dense_bytes), gib(p.kv_bytes), gib(p.dense_bytes + p.kv_bytes));
    if p.dense_bytes + p.kv_bytes > p.vram_budget {
        println!("  does not fit in VRAM alone — dense weights must spill to RAM");
    }
    println!("experts in VRAM {:>8}  ({:.0}%)", p.vram_experts, 100.0 * p.vram_fraction());
    println!("experts in RAM  {:>8}", p.ram_experts);
    println!("experts on disk {:>8}{}", p.disk_experts,
        if p.disk_experts > 0 { "   <- streaming required" } else { "" });
    println!();
    println!("suggested --n-gpu-layers {} of {}", p.gpu_layers(), f.layers);
    Ok(())
}

/// `strata list` — models actually present under rocm/models, read from their headers.
pub fn list_local() -> Result<()> {
    let root = Path::new("rocm/models");
    if !root.is_dir() {
        println!("No rocm/models directory yet.");
        println!("`strata pull --model <name>` copies a model out of the Ollama blob store.");
        return Ok(());
    }
    let mut found = 0;
    for entry in std::fs::read_dir(root)? {
        let dir = entry?.path();
        let file = dir.join("model.gguf");
        if !file.is_file() {
            continue;
        }
        found += 1;
        match Gguf::open(&file).and_then(|g| ModelFacts::read(&g).map(|f| (g, f))) {
            Ok((g, f)) => {
                println!(
                    "{:<22} {:<12} {:>3} layers  {:>5} experts x top-{:<2} {:>7.2} GiB  ctx {}",
                    dir.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                    f.arch,
                    f.layers,
                    f.experts,
                    f.experts_used,
                    gib(g.tensor_bytes()),
                    f.context,
                );
            }
            Err(e) => println!("{:<22} unreadable: {e}", dir.display()),
        }
    }
    if found == 0 {
        println!("rocm/models exists but holds no model.gguf files.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Ornith-1.5-35B on the development machine: 20.21 GiB total, 1.56 GiB of
    /// it dense, 10,496 routed experts, 1.25 GiB of cache at ctx 65536.
    #[test]
    fn the_model_we_run_needs_no_disk() {
        let expert_bytes = (20.21 - 1.56) * GIB as f64;
        let count = 41 * 256;
        let per_expert = (expert_bytes / count as f64) as u64;
        let vram_left = 16 * GIB - (1.56 * GIB as f64) as u64 - (1.25 * GIB as f64) as u64;

        let t = fill_tiers(count, per_expert, vram_left, 21 * GIB);
        assert_eq!(t.disk_experts, 0, "this configuration is measured and does not spill");
        assert!(t.vram_experts > count / 2, "most experts are on the card");
    }

    /// A 200B MoE at Q4_K on the same machine. The point of the test is that
    /// the answer is "most of it has nowhere to go", which is what `setup`
    /// refuses on.
    #[test]
    fn a_200b_model_spills_most_of_itself() {
        let total = 112.0 * GIB as f64;
        let dense = total * 0.08;
        let count = 96 * 256;
        let per_expert = ((total - dense) / count as f64) as u64;
        let vram_left = 16 * GIB - dense as u64 - (1.25 * GIB as f64) as u64;

        let t = fill_tiers(count, per_expert, vram_left, 21 * GIB);
        let spill = t.disk_experts as f32 / count as f32;
        assert!(spill > 0.5, "expected most of a 200B to spill, got {:.0}%", 100.0 * spill);
    }

    #[test]
    fn every_expert_is_accounted_for() {
        let count = 1000;
        for vram in [0u64, GIB, 8 * GIB, 64 * GIB] {
            for ram in [0u64, 4 * GIB, 512 * GIB] {
                let t = fill_tiers(count, 16 * 1024 * 1024, vram, ram);
                assert_eq!(t.vram_experts + t.ram_experts + t.disk_experts, count);
            }
        }
    }

    #[test]
    fn a_model_with_no_room_anywhere_is_entirely_spilled() {
        let t = fill_tiers(512, GIB, 0, 0);
        assert_eq!(t.disk_experts, 512);
    }

    #[test]
    fn a_dense_model_has_no_experts_to_place() {
        let t = fill_tiers(0, 0, 8 * GIB, 32 * GIB);
        assert_eq!(t, Tiers { vram_experts: 0, ram_experts: 0, disk_experts: 0 });
    }
}
