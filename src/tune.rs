//! Measures the best expert split for a model on this machine.
//!
//! `Flags::derive` estimates from byte counts, but the estimate is consistently
//! low: the fastest configuration leaves more experts on the CPU than a pure
//! capacity calculation suggests, because freeing VRAM lets the remaining
//! layers run without contention. So measure instead of predict, and write the
//! answer next to the model for `serve` to pick up.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::placement::{resolve_path, Placement};
use crate::runner::{Flags, Runner};

const PROMPT: &str = "Write a Rust function that parses a GGUF header and returns the tensor count. Code only.";
const TOKENS: u32 = 160;
const PORT: u16 = 8098;

/// The turn `tune` optimises for: read this much unseen code, then write this
/// much answer. Sized for long-context coding work rather than for chat.
const PREFILL_TOKENS: u32 = 8192;
const DECODE_TOKENS: u32 = 512;

/// Prose used to build a prompt of a known size. Content is irrelevant; only
/// its length matters, and it must be the same across configurations for the
/// measurements to be comparable.
const FILLER: &str = "The record layout is fixed at compile time and validated \
on load; readers detect a lap by observing a discontinuity in sequence numbers \
rather than by taking a lock, which keeps the hot path free of contention. ";

/// What `tune` measured, stored beside the model as `tuned.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tuned {
    pub cpu_moe_layers: usize,
    pub ctx: usize,
    pub flash_attention: bool,
    pub threads: usize,
    /// KV cache element type this was measured with; absent means f16.
    #[serde(default)]
    pub kv_type: Option<String>,
    /// A non-prefix expert placement, as an `-ot` pattern. Absent for every
    /// plan `-ncmoe` can express, which is all of them today - the field
    /// exists so a measured placement can outlive the process that found it.
    #[serde(default)]
    pub expert_override: Option<String>,
    /// Decode rate, tokens per second.
    pub tok_s: f32,
    /// Prompt-ingest rate, tokens per second. Absent in files written before
    /// tuning measured reading as well as writing.
    #[serde(default)]
    pub prefill_tok_s: f32,
    /// Wall time for the turn this was ranked on.
    #[serde(default)]
    pub turn_s: f32,
    pub vram_gb: f32,
    pub measured_utc: String,
}

/// One file per context, because the best split is context-specific: the KV
/// cache reserved at load time is VRAM the experts cannot have, so a split
/// measured at 8192 is wrong at 262144. Keeping them apart means switching
/// context back and forth does not throw a measurement away.
pub fn tuned_path(model_file: &Path, ctx: usize, kv_type: Option<&str>) -> PathBuf {
    match kv_type {
        Some(t) => model_file.with_file_name(format!("tuned-{ctx}-{t}.json")),
        None => model_file.with_file_name(format!("tuned-{ctx}.json")),
    }
}

/// Load a previous measurement for this context.
pub fn load(model_file: &Path, ctx: usize, kv_type: Option<&str>) -> Option<Tuned> {
    let read = |p: PathBuf| -> Option<Tuned> {
        serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
    };
    read(tuned_path(model_file, ctx, kv_type))
        .filter(|t: &Tuned| t.ctx == ctx)
        // Measurements written before tuning was per-context live in a single
        // `tuned.json`; honour it when it happens to be for this context.
        .or_else(|| {
            read(model_file.with_file_name("tuned.json")).filter(|t: &Tuned| t.ctx == ctx)
        })
}

/// What one configuration achieves, split into the two rates that behave
/// differently. Prefill is compute-bound and prefers experts on the GPU;
/// decode is memory-bound and tolerates them in host RAM. A single "tok/s"
/// number hides that they disagree.
#[derive(Debug, Clone, Copy)]
pub struct Perf {
    /// Tokens per second ingesting a prompt the process has not seen.
    pub prefill_tok_s: f32,
    /// Tokens per second generating, with the prompt already cached.
    pub decode_tok_s: f32,
}

impl Perf {
    /// Wall time for one realistic turn: read `PREFILL_TOKENS` of unseen code,
    /// then write `DECODE_TOKENS` of answer.
    ///
    /// This is what `tune` minimises. Ranking by decode speed alone optimises
    /// the smaller half of a long-context turn: at 8k of new context and 512
    /// tokens out, reading is most of the wall clock, and the split that
    /// generates fastest is not the split that reads fastest.
    pub fn turn_seconds(&self) -> f32 {
        PREFILL_TOKENS as f32 / self.prefill_tok_s.max(0.01)
            + DECODE_TOKENS as f32 / self.decode_tok_s.max(0.01)
    }
}

/// A prompt of roughly `tokens` tokens.
fn sized_prompt(tokens: u32) -> String {
    let mut s = String::with_capacity(tokens as usize * 4);
    // ~3.4 characters per token for English prose, close enough to land in the
    // right range; the measurement uses the tokenizer's own count regardless.
    while s.len() < (tokens as f32 * 3.4) as usize {
        s.push_str(FILLER);
    }
    s
}

async fn measure(endpoint: &str) -> Result<Perf> {
    let client = reqwest::Client::new();

    // Prefill: a prompt this process has never seen, answered in a single
    // token, so elapsed time is almost entirely the cost of reading it.
    let started = Instant::now();
    let r = client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{
                "role": "user",
                "content": format!("{}\n\nReply with one word.", sized_prompt(PREFILL_TOKENS)),
            }],
            "max_tokens": 1,
            "stream": false,
            "temperature": 0.0,
        }))
        .timeout(Duration::from_secs(900))
        .send()
        .await?;
    let v: serde_json::Value = r.json().await?;
    let read = v.pointer("/usage/prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
    let prefill_secs = started.elapsed().as_secs_f32();
    let prefill_tok_s =
        if read > 0 && prefill_secs > 0.0 { read as f32 / prefill_secs } else { 0.0 };

    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": PROMPT }],
        "max_tokens": TOKENS,
        "stream": false,
        "temperature": 0.0,
    });
    let mut best = 0.0f32;
    // First pass warms caches; the second is the number worth keeping.
    for _ in 0..2 {
        let started = Instant::now();
        let r = client
            .post(format!("{endpoint}/v1/chat/completions"))
            .json(&body)
            .timeout(Duration::from_secs(600))
            .send()
            .await?;
        let v: serde_json::Value = r.json().await?;
        let produced = v
            .pointer("/usage/completion_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let secs = started.elapsed().as_secs_f32();
        if produced > 0 && secs > 0.0 {
            best = best.max(produced as f32 / secs);
        }
    }
    Ok(Perf { prefill_tok_s, decode_tok_s: best })
}

fn vram_of(pid: u32) -> f32 {
    // Windows exposes per-process GPU memory through performance counters; a
    // fresh HIP context cannot see another process's allocations, so this is
    // the only reliable source.
    #[cfg(windows)]
    {
        let script = format!(
            "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | \
             Where-Object {{ $_.InstanceName -like 'pid_{pid}_*' }} | \
             Measure-Object CookedValue -Sum | ForEach-Object {{ $_.Sum }}"
        );
        if let Ok(out) = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()
        {
            if let Ok(text) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = text.trim().parse::<f64>() {
                    return (bytes / 1024.0 / 1024.0 / 1024.0) as f32;
                }
            }
        }
    }
    let _ = pid;
    0.0
}

/// Sweep the CPU-expert split and keep the fastest.
pub async fn run(model: &str, ctx: usize, kv_type: Option<String>, keep: bool) -> Result<()> {
    let Some(path) = resolve_path(model) else {
        bail!("no model file for '{model}' — try `strata pull --model {model}`");
    };
    let vram = (crate::rocm::vram_total_gb() as f64 * 1024.0 * 1024.0 * 1024.0) as u64;
    let ram = 20u64 * 1024 * 1024 * 1024;
    let placed = Placement::solve(&path, ctx, vram, ram)?;
    let layers = placed.facts.layers;
    let mut derived = Flags::derive(&placed);
    derived.kv_type = kv_type.clone();

    // Around half the layers is where this class of model tends to peak, so
    // bracket that rather than sweeping uniformly from zero.
    let mut candidates: Vec<usize> = [0.0, 0.3, 0.4, 0.5, 0.6, 0.75]
        .iter()
        .map(|f| ((layers as f32) * f).round() as usize)
        .collect();
    candidates.push(derived.cpu_moe_layers);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|&c| c < layers);

    println!(
        "Tuning {} at ctx {} (KV cache {})",
        placed.facts.name,
        ctx,
        kv_type.as_deref().unwrap_or("f16")
    );
    println!("{} layers, {} experts each; derived estimate is {}", layers, placed.facts.experts, derived.cpu_moe_layers);
    println!(
        "Optimising for a {PREFILL_TOKENS}-token read plus a {DECODE_TOKENS}-token answer.\n"
    );
    println!(
        "{:>10}  {:>12}  {:>11}  {:>9}  {:>9}",
        "cpu-moe", "prefill tok/s", "decode tok/s", "turn s", "VRAM GB"
    );
    println!("{:->60}", "");

    let mut best: Option<(Tuned, Perf)> = None;
    let mut done: Vec<usize> = Vec::new();

    // Coarse sweep, then a second pass either side of the winner. The surface
    // is not smooth - at ctx 262144, 12 layers measured 32.6 tok/s while 16
    // measured 27.4 - so the coarse grid can straddle the peak without landing
    // on it.
    let mut queue = candidates.clone();
    let mut refined = false;
    while let Some(cpu_moe) = queue.pop() {
        if done.contains(&cpu_moe) || cpu_moe >= layers {
            if queue.is_empty() && !refined {
                refined = true;
                if let Some((b, _)) = &best {
                    queue = neighbours(b.cpu_moe_layers, layers, &done);
                    if !queue.is_empty() {
                        println!("{:->60}", "");
                    }
                }
            }
            continue;
        }
        done.push(cpu_moe);

        let mut flags = derived.clone();
        flags.cpu_moe_layers = cpu_moe;
        flags.ctx = ctx;

        let outcome = async {
            let mut runner = match Runner::start(&path, flags.clone(), PORT) {
                Ok(r) => r,
                Err(e) => {
                    println!("{cpu_moe:>10}  start failed: {e}");
                    return None;
                }
            };
            if runner.wait_ready(Duration::from_secs(600)).await.is_err() {
                println!("{cpu_moe:>10}  {:>12}", "no load");
                return None;
            }
            let vram_gb = vram_of(runner.pid());
            let perf = measure(&runner.endpoint).await.ok()?;
            Some((perf, vram_gb))
        }
        .await;

        if let Some((perf, vram_gb)) = outcome {
            if perf.decode_tok_s > 0.0 && perf.prefill_tok_s > 0.0 {
                let turn = perf.turn_seconds();
                let improved = best.as_ref().map_or(true, |(_, p)| turn < p.turn_seconds());
                println!(
                    "{cpu_moe:>10}  {:>12.0}  {:>11.1}  {turn:>9.1}  {vram_gb:>9.2}{}",
                    perf.prefill_tok_s,
                    perf.decode_tok_s,
                    if improved { " <-- best" } else { "" }
                );
                if improved {
                    best = Some((
                        Tuned {
                            cpu_moe_layers: cpu_moe,
                            ctx,
                            flash_attention: flags.flash_attention,
                            threads: flags.threads,
                            kv_type: kv_type.clone(),
                            expert_override: flags.expert_override.clone(),
                            tok_s: perf.decode_tok_s,
                            prefill_tok_s: perf.prefill_tok_s,
                            turn_s: turn,
                            vram_gb,
                            measured_utc: now_utc(),
                        },
                        perf,
                    ));
                }
            } else {
                println!("{cpu_moe:>10}  {:>12}", "failed");
            }
        }

        tokio::time::sleep(Duration::from_secs(3)).await;

        if queue.is_empty() && !refined {
            refined = true;
            if let Some((b, _)) = &best {
                queue = neighbours(b.cpu_moe_layers, layers, &done);
                if !queue.is_empty() {
                    println!("{:->60}", "");
                }
            }
        }
    }

    let Some((best, perf)) = best else { bail!("no configuration produced tokens") };
    println!(
        "\nBest: {} CPU-expert layers — {:.0} tok/s reading, {:.1} tok/s writing, \
         {:.1}s per turn ({:.2} GB VRAM, KV {})",
        best.cpu_moe_layers,
        perf.prefill_tok_s,
        perf.decode_tok_s,
        perf.turn_seconds(),
        best.vram_gb,
        kv_type.as_deref().unwrap_or("f16")
    );
    if keep {
        let out = tuned_path(&path, ctx, kv_type.as_deref());
        std::fs::write(&out, serde_json::to_string_pretty(&best)?)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("Saved to {} — `strata serve` will use it.", out.display());
    } else {
        println!("Not saved (pass --save to keep it).");
    }
    Ok(())
}

/// Splits either side of `centre` that have not been measured yet, nearest
/// first, for the refinement pass.
fn neighbours(centre: usize, layers: usize, done: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for delta in [1isize, 2] {
        for candidate in [centre as isize - delta, centre as isize + delta] {
            if candidate >= 0 && (candidate as usize) < layers && !done.contains(&(candidate as usize))
            {
                out.push(candidate as usize);
            }
        }
    }
    // `run` pops from the back, so reverse to try the nearest splits first.
    out.reverse();
    out
}

fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}
