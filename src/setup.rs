//! One command that configures Strata for the machine it finds itself on.
//!
//! Everything Strata needs to run well was worked out by hand on one desktop:
//! which context the card can afford, and which expert split measures fastest
//! at that context. Neither answer transfers. A 16 GB discrete card and an 8 GB
//! laptop want different contexts, and the split that is fastest on one is not
//! the split that is fastest on the other - the whole premise of `tune` is that
//! this has to be measured rather than predicted.
//!
//! So `strata setup` does for a stranger's machine what was done by hand for
//! the first one: look at the hardware, pick a context that fits it, measure
//! the split, and write the result. It never overwrites a measurement that is
//! already there.

use anyhow::{bail, Result};
use std::path::Path;

use crate::hardware::{self, Machine};
use crate::placement::{resolve_path, Placement};
use crate::runner::Runtime;

/// Contexts worth considering, smallest first. Powers of two because the KV
/// cache is allocated per context and the differences below 2x do not pay for
/// the tuning run they would each need.
const LADDER: [usize; 8] = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

/// The KV cache may take this much of the card, and no more.
///
/// The cache is allocated in full at load time, so it is VRAM the experts
/// cannot have for the whole run - and expert residency is what decode speed
/// depends on. Measured on a 16 GB card: at 1.25 GiB of cache (7.8%) the model
/// decodes at 42.5 tok/s, and at 5.00 GiB (31%) it drops to 32.6 even after
/// re-tuning. Ten percent reproduces the context that was chosen by hand there,
/// which is the only real evidence this number has behind it.
const KV_SHARE_OF_VRAM: f32 = 0.10;

/// On a dense model the KV cache competes only with the compute buffers, not
/// with experts, so it may draw on this share of the VRAM left after the
/// weights. Half keeps a real margin: filling the card to the last byte leaves
/// nothing for the buffers llama.cpp allocates at load.
const DENSE_KV_SHARE_OF_FREE: f64 = 0.5;

/// Below this there is no point continuing: the weights that must be resident
/// to run at all would not fit beside even a minimal cache.
const MIN_CONTEXT: usize = 2048;

/// Past this share of experts landing outside VRAM and RAM, setup refuses.
///
/// There is no disk tier. `-ncmoe` and `-ot` place expert tensors in VRAM or
/// host memory and nowhere else, so "on disk" means the runtime mmaps the file
/// and the operating system faults pages in as they are touched. Decode routes
/// a different handful of experts per layer per token, which is a scattered
/// read with almost no locality - bound by random reads, not bandwidth. A model
/// mostly in that state technically runs, at a fraction of a token per second,
/// and measuring it would load the file once per configuration first.
const HOPELESS_DISK_SHARE: f32 = 0.50;

/// What `setup` decided, and why.
pub struct Choice {
    pub ctx: usize,
    pub kv_gb: f32,
    pub dense_gb: f32,
    /// Share of routed experts that would live outside VRAM and RAM.
    pub disk_share: f32,
    /// Weights this machine can hold without touching disk, at this context.
    pub ceiling_gb: f32,
    /// True when the weights fit in VRAM plus host RAM, so nothing streams from
    /// disk. False means it will run, but slowly enough to be worth saying so.
    pub fits_in_memory: bool,
    pub vram_gb: f32,
    pub ram_gb: f32,
}

/// Largest context whose KV cache stays within its share of the card.
///
/// Bigger is not better. A context the machine can technically allocate still
/// costs residency for the entire run, and a conversation that never reaches the
/// end of a smaller window has paid that for nothing - compaction covers the
/// overflow. So this picks the largest context that stays cheap, not the largest
/// that fits.
///
/// "Cheap" means different things depending on whether there are experts to
/// protect. On a mixture-of-experts model every byte of KV cache is VRAM an
/// expert cannot have, so the cache is held to a small share of the card. On a
/// dense model there are no experts to starve, and the only thing the cache can
/// crowd out is the compute buffers - so the budget is drawn from what is left
/// after the weights, not from a flat slice of the whole card. Applying the MoE
/// rule to a dense model is what made a 5.7 GB model load at 32K when it had
/// room for far more.
pub fn choose_context(
    facts_ctx: usize,
    kv_bytes_per_token: u64,
    vram_gb: f32,
    dense_bytes: u64,
    expert_count: usize,
) -> usize {
    let ceiling = facts_ctx.max(MIN_CONTEXT);
    let vram_bytes = vram_gb as f64 * 1024.0 * 1024.0 * 1024.0;

    let budget_bytes = if expert_count == 0 {
        // Dense: the weights are already resident, so the cache is competing
        // only with the compute buffers. Take at most half of what is left,
        // which leaves room for those buffers and keeps some headroom on the
        // card instead of filling it to the last byte.
        let free = (vram_bytes - dense_bytes as f64).max(0.0);
        (free * DENSE_KV_SHARE_OF_FREE) as u64
    } else {
        // MoE: experts need the rest of the card, so the cache keeps to a small
        // fixed share regardless of how much is free.
        (vram_bytes * KV_SHARE_OF_VRAM as f64) as u64
    };

    let affordable = if kv_bytes_per_token > 0 {
        (budget_bytes / kv_bytes_per_token) as usize
    } else {
        ceiling
    };
    LADDER
        .iter()
        .rev()
        .copied()
        .find(|&c| c <= affordable && c <= ceiling)
        .unwrap_or(MIN_CONTEXT)
}

/// Work out what this machine should run, without changing anything.
pub fn plan_for(model_path: &Path, machine: &Machine) -> Result<Choice> {
    let Some(device) = machine.primary() else {
        bail!("no GPU to plan against")
    };
    let vram_gb = device.total_gb();
    let ram_gb = machine.ram_budget_gb();

    // Solved once at a nominal context only to read the model's own byte
    // counts; the context that matters is chosen from them below.
    let probe = Placement::solve(
        model_path,
        MIN_CONTEXT,
        (vram_gb as f64 * 1073741824.0) as u64,
        (ram_gb as f64 * 1073741824.0) as u64,
    )?;
    let ctx = choose_context(
        probe.facts.context,
        probe.kv_bytes_per_token,
        vram_gb,
        probe.dense_bytes,
        probe.expert_count,
    );

    let solved = Placement::solve(
        model_path,
        ctx,
        (vram_gb as f64 * 1073741824.0) as u64,
        (ram_gb as f64 * 1073741824.0) as u64,
    )?;
    let gib = |b: u64| b as f32 / 1073741824.0;

    let disk_share = solved.disk_experts as f32 / solved.expert_count.max(1) as f32;
    // What fits with nothing streaming: both memories, less the cache and a
    // little for compute buffers and the graph.
    let ceiling_gb = (vram_gb + ram_gb - gib(solved.kv_bytes) - 1.0).max(0.0);

    if solved.dense_bytes + solved.kv_bytes > (vram_gb * 1073741824.0) as u64 {
        bail!(
            "this model does not fit on a {vram_gb:.1} GB card: {:.1} GB of weights must be \
             resident before any expert, plus {:.1} GB of KV cache at the smallest useful \
             context. A smaller or more heavily quantised model is the answer, not a \
             different setting.",
            gib(solved.dense_bytes),
            gib(solved.kv_bytes)
        );
    }

    Ok(Choice {
        ctx,
        kv_gb: gib(solved.kv_bytes),
        dense_gb: gib(solved.dense_bytes),
        disk_share,
        ceiling_gb,
        fits_in_memory: solved.fits_without_disk(),
        vram_gb,
        ram_gb,
    })
}

/// Detect, choose, measure, save.
pub async fn run(
    model: Option<String>,
    force: bool,
    dry_run: bool,
    allow_disk: bool,
) -> Result<()> {
    println!("Strata setup\n");

    // 1. The compute runtime, which is also how the hardware is inspected.
    let Some(runtime) = Runtime::discover() else {
        bail!(
            "no llama.cpp runtime found.\n\n\
             Strata does not ship one. Download a build for your GPU from\n  \
             https://github.com/ggml-org/llama.cpp/releases\n\
             and unzip it into `runtime/` beside this binary, or point\n\
             STRATA_LLAMA_DIR at wherever you put it."
        )
    };
    println!("  runtime   {}", runtime.server.display());

    // 2. What the runtime can actually see. No vendor SDK, no Python.
    let machine = hardware::probe(&runtime)?;
    for d in &machine.devices {
        let tag = if machine.primary().map(|p| p.id == d.id).unwrap_or(false) {
            " <-- planning against this one"
        } else if d.looks_integrated() {
            "  (integrated; its memory is the host's)"
        } else {
            ""
        };
        println!("  {}  {} — {:.1} GB{tag}", d.id, d.name, d.total_gb());
    }
    println!(
        "  host      {:.0} GB RAM, {:.0} GB free, {} threads",
        machine.ram_total_gb, machine.ram_avail_gb, machine.cores
    );

    // 3. A model to plan for.
    let model = match model {
        Some(m) => m,
        None => match sole_local_model()? {
            Some(m) => m,
            None => bail!(
                "no model found in rocm/models.\n\n\
                 Put a .gguf at rocm/models/<name>/model.gguf, or fetch one with\n  \
                 strata pull --model <name>\n\
                 then run setup again."
            ),
        },
    };
    let Some(path) = resolve_path(&model) else {
        bail!("no model file for '{model}'")
    };

    let choice = plan_for(&path, &machine)?;
    println!("\n  model     {}", path.display());
    println!(
        "  context   {} — {:.2} GB of KV cache, {:.0}% of the card",
        choice.ctx,
        choice.kv_gb,
        100.0 * choice.kv_gb / choice.vram_gb
    );
    println!(
        "  resident  {:.2} GB of dense weights before any expert",
        choice.dense_gb
    );
    println!(
        "  ceiling   {:.0} GB of weights fit without touching disk ({:.0} GB VRAM + {:.0} GB free RAM)",
        choice.ceiling_gb, choice.vram_gb, choice.ram_gb
    );

    if choice.disk_share > 0.0 {
        println!(
            "  spill     {:.0}% of experts fall outside both memories",
            100.0 * choice.disk_share
        );
    }
    if choice.disk_share >= HOPELESS_DISK_SHARE && !allow_disk {
        bail!(
            "{:.0}% of this model's experts would live outside VRAM and RAM.\n\n\
             There is no disk tier: expert tensors go to VRAM or host memory and nowhere \
             else, so the rest is the operating system faulting pages out of the mmapped \
             file. Decode touches a different handful of experts per layer per token, which \
             is a scattered read with almost no locality, and the result is a fraction of a \
             token per second. Measuring it would load the whole file once per configuration \
             to arrive at that answer.\n\n\
             This machine holds about {:.0} GB of weights without spilling. Use a smaller \
             model or a smaller quantisation, or pass --allow-disk to measure it anyway.",
            100.0 * choice.disk_share,
            choice.ceiling_gb
        );
    }
    if choice.disk_share > 0.0 {
        println!("  NOTE      spilled experts fault in from the mmapped file. It will be slow.");
    }

    if dry_run {
        println!("\nNothing measured or written (--dry-run).");
        println!("Run without --dry-run to measure the expert/layer split for this machine.");
        return Ok(());
    }

    // 4. Measure, unless a measurement is already there. Someone else's tuned
    //    file is the one thing setup must never quietly replace.
    let tuned = crate::tune::tuned_path(&path, choice.ctx, None);
    if tuned.exists() && !force {
        println!("\n{} already exists — keeping it.", tuned.display());
        println!("Pass --force to measure again.");
    } else {
        println!("\nMeasuring the expert/layer split for this machine. This loads the model once per");
        println!("configuration and takes a while; it is done once.\n");
        crate::tune::run(&model, choice.ctx, None, true).await?;
    }

    println!("\nReady:");
    println!("  strata serve --model {model} --ctx {}", choice.ctx);
    println!("  or .\\strata.ps1");
    Ok(())
}

/// The only model on disk, when there is exactly one.
fn sole_local_model() -> Result<Option<String>> {
    let dir = Path::new("rocm/models");
    if !dir.is_dir() {
        return Ok(None);
    }
    let mut found: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir)?.flatten() {
        if entry.path().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                if resolve_path(name).is_some() {
                    found.push(name.to_string());
                }
            }
        }
    }
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.remove(0))),
        _ => bail!(
            "several models present; choose one with --model: {}",
            found.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ornith-1.5-35B: hybrid attention, so only some layers keep a cache.
    const ORNITH_KV: u64 = 20480;
    /// Ornith has routed experts, so it takes the MoE path.
    const ORNITH_EXPERTS: usize = 256 * 41;
    /// 5.69 GiB of dense weights, which is the model this was reported against.
    const HERETIC_DENSE: u64 = 6_115_694_112;

    #[test]
    fn reproduces_the_context_chosen_by_hand_on_a_16gb_card() {
        // 65536 was arrived at by measurement on the development machine; a
        // rule that disagreed with it would be the wrong rule.
        assert_eq!(
            choose_context(262144, ORNITH_KV, 16.0, 3_000_000_000, ORNITH_EXPERTS),
            65536
        );
    }

    #[test]
    fn smaller_cards_get_smaller_contexts() {
        for (vram, want) in [(8.0, 32768), (4.0, 16384), (2.0, 8192)] {
            assert_eq!(
                choose_context(262144, ORNITH_KV, vram, 1_000_000_000, ORNITH_EXPERTS),
                want,
                "{vram} GB"
            );
        }
    }

    #[test]
    fn a_dense_model_with_a_fat_cache_gets_a_short_context() {
        // A non-hybrid 70B at 640 KB/token over 80 layers. The weights are 2 GB,
        // leaving 14 GB, half of which is 7 GB - and 8192 tokens of this cache
        // is 4.88 GB, so 8192 is the largest rung that fits. The point of the
        // test is that a fat cache still yields a short window, not that the
        // dense path is generous with every model.
        let dense_kv = 640_000u64;
        assert_eq!(
            choose_context(131072, dense_kv, 16.0, 2_000_000_000, 0),
            8192
        );
    }

    /// The reported bug: a 5.7 GB dense model on a 16 GB card was loaded at 32K
    /// because the MoE rule capped the cache at 10% of the whole card. With no
    /// experts to protect, the cache should draw on what is left after the
    /// weights instead.
    #[test]
    fn a_dense_model_gets_more_context_than_the_moe_rule_would_allow() {
        let dense_kv = 32768u64; // 32 KB per token, measured from this model
        let got = choose_context(262144, dense_kv, 16.0, HERETIC_DENSE, 0);
        let moe_rule = choose_context(262144, dense_kv, 16.0, HERETIC_DENSE, 1);
        assert!(
            got > moe_rule,
            "dense {got} should beat the MoE rule's {moe_rule}"
        );
        // 5.69 GB of weights leaves ~10.3 GB; half of that is ~5.15 GB, which
        // buys 131072 tokens of cache. It must not pick the 32K the old rule did.
        assert!(got >= 131072, "expected at least 131072, got {got}");
    }

    #[test]
    fn a_dense_model_still_respects_the_card_it_is_on() {
        let dense_kv = 32768u64;
        let small = choose_context(262144, dense_kv, 8.0, HERETIC_DENSE, 0);
        let big = choose_context(262144, dense_kv, 16.0, HERETIC_DENSE, 0);
        assert!(small < big, "an 8 GB card must not match a 16 GB one");
    }

    #[test]
    fn a_dense_model_never_exceeds_its_trained_context() {
        assert_eq!(choose_context(8192, 32768, 48.0, HERETIC_DENSE, 0), 8192);
    }

    #[test]
    fn never_exceeds_what_the_model_was_trained_for() {
        assert_eq!(
            choose_context(8192, ORNITH_KV, 48.0, 1_000_000_000, ORNITH_EXPERTS),
            8192
        );
        assert_eq!(
            choose_context(4096, ORNITH_KV, 96.0, 1_000_000_000, ORNITH_EXPERTS),
            4096
        );
    }

    #[test]
    fn always_lands_on_the_ladder() {
        for vram in [1.0f32, 3.5, 7.0, 11.0, 24.0, 80.0] {
            let c = choose_context(262144, ORNITH_KV, vram, 1_000_000_000, ORNITH_EXPERTS);
            assert!(LADDER.contains(&c), "{vram} GB gave {c}");
            let d = choose_context(262144, 32768, vram, HERETIC_DENSE, 0);
            assert!(LADDER.contains(&d), "dense {vram} GB gave {d}");
        }
    }
}
