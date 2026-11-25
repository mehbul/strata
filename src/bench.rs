//! `strata bench` - measure a running instance the way PolyStrata does.
//!
//! Sends one greedy chat completion with a prompt of roughly `--prompt-tokens`
//! tokens and `--output-tokens` of greedy output, then reports TTFT, prefill
//! throughput and decode throughput. Run it twice: the first run pays for the
//! file cache warming up.

use anyhow::Result;
use clap::Args;
use futures::StreamExt;
use std::time::Instant;

#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// Base URL of a running Strata instance
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub url: String,
    /// Model name to send (defaults to whatever the server reports first)
    #[arg(long)]
    pub model: Option<String>,
    /// Prompt size in tokens, measured with the model's own tokenizer
    #[arg(long, default_value_t = 2048)]
    pub prompt_tokens: usize,
    /// GGUF whose tokenizer should size the prompt (auto-detected from the model)
    #[arg(long)]
    pub tokenizer: Option<String>,
    /// Tokens to generate
    #[arg(long, default_value_t = 256)]
    pub output_tokens: usize,
    /// Write results as JSON to this path
    #[arg(long)]
    pub json: Option<String>,
    /// Compare the new run with a previous JSON result
    #[arg(long)]
    pub compare: Option<String>,
}

/// Last-resort sizing when no GGUF tokenizer is available.
fn estimate_prompt(tokens: usize) -> String {
    const BASE: &str = "The quick brown fox jumps over the lazy dog. ";
    let target = tokens.saturating_mul(4);
    let mut prompt = String::with_capacity(target);
    while prompt.len() < target {
        prompt.push_str(BASE);
    }
    prompt.truncate(target);
    prompt
}

pub async fn run(args: BenchArgs) -> Result<()> {
    let client = reqwest::Client::new();

    // Resolve model name if not given.
    let model = match args.model.clone() {
        Some(m) => m,
        None => {
            let resp: serde_json::Value = client
                .get(format!("{}/v1/models", args.url))
                .send()
                .await?
                .json()
                .await?;
            resp.get("data")
                .and_then(|d| d.as_array())
                .and_then(|a| a.first())
                .and_then(|m| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("strata")
                .to_string()
        }
    };

    // Size the prompt with the model's tokenizer rather than by characters.
    // A character estimate lands anywhere near the requested size on a large
    // vocabulary or a non-English prompt, which makes a benchmark misleading.
    let (prompt, sized) = match args.tokenizer.clone().or_else(|| {
        crate::placement::resolve_path(&model).map(|p| p.to_string_lossy().into_owned())
    }) {
        Some(path) => match crate::tokenizer::Tokenizer::from_gguf(&path) {
            Ok(tok) => {
                let base = "The quick brown fox jumps over the lazy dog. ";
                let mut text = String::with_capacity(args.prompt_tokens * 8);
                while text.len() < args.prompt_tokens * 8 {
                    text.push_str(base);
                }
                match tok.encode(&text) {
                    Ok(ids) => {
                        let n = ids.len().min(args.prompt_tokens);
                        match tok.decode(&ids[..n]) {
                            Ok(decoded) => (decoded, Some(n)),
                            Err(e) => {
                                eprintln!(
                                    "tokenizer decode failed ({e}); using character estimate"
                                );
                                (estimate_prompt(args.prompt_tokens), None)
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("tokenizer encode failed ({e}); using character estimate");
                        (estimate_prompt(args.prompt_tokens), None)
                    }
                }
            }
            Err(e) => {
                eprintln!("cannot open tokenizer {path} ({e}); using character estimate");
                (estimate_prompt(args.prompt_tokens), None)
            }
        },
        None => (estimate_prompt(args.prompt_tokens), None),
    };

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "temperature": 0.0,
        "max_tokens": args.output_tokens,
        "stream_options": {"include_usage": true},
    });

    println!("strata bench");
    println!("  url           {}", args.url);
    println!("  model         {}", model);
    match sized {
        Some(n) => println!(
            "  prompt_tokens {} (model tokenizer, chars {})",
            n,
            prompt.len()
        ),
        None => println!(
            "  prompt_tokens ~{} (character estimate, chars {})",
            args.prompt_tokens,
            prompt.len()
        ),
    }
    println!("  output_tokens {}", args.output_tokens);
    println!();

    let started = Instant::now();
    let resp = client
        .post(format!("{}/v1/chat/completions", args.url))
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("server returned {}", resp.status());
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut ttft_ms: Option<u128> = None;
    let mut prompt_tokens: u64 = 0;
    let mut completion_tokens: u64 = 0;
    let mut first_token_seen = false;

    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                eprintln!("stream error: {e}");
                break;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));
        // Process complete SSE frames.
        while let Some(pos) = buf.find("\n\n") {
            let frame: String = buf.drain(..=pos + 1).collect();
            for line in frame.lines() {
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim_start();
                if payload == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                    if let Some(u) = v.get("usage") {
                        if let Some(pt) = u.get("prompt_tokens").and_then(|x| x.as_u64()) {
                            prompt_tokens = pt;
                        }
                        if let Some(ct) = u.get("completion_tokens").and_then(|x| x.as_u64()) {
                            completion_tokens = ct;
                        }
                    }
                    if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                        if !choices.is_empty() {
                            let has_content = choices[0]
                                .get("delta")
                                .and_then(|d| d.get("content"))
                                .and_then(|c| c.as_str())
                                .map(|s| !s.is_empty())
                                .unwrap_or(false);
                            if has_content && !first_token_seen {
                                first_token_seen = true;
                                ttft_ms = Some(started.elapsed().as_millis());
                            }
                        }
                    }
                }
            }
        }
    }

    let total_ms = started.elapsed().as_millis();
    let ttft = ttft_ms.unwrap_or(total_ms);
    let prefill_ms = ttft;
    let decode_ms = total_ms.saturating_sub(ttft);

    let prefill_tps = if prefill_ms > 0 {
        prompt_tokens as f64 * 1000.0 / prefill_ms as f64
    } else {
        0.0
    };
    let decode_tps = if decode_ms > 0 {
        completion_tokens as f64 * 1000.0 / decode_ms as f64
    } else {
        0.0
    };

    println!("Results:");
    println!(
        "  prompt_tokens      {} (requested {})",
        prompt_tokens, args.prompt_tokens
    );
    println!("  completion_tokens {}", completion_tokens);
    println!("  TTFT              {} ms", ttft);
    println!("  prefill           {:.1} tok/s", prefill_tps);
    println!("  decode            {:.1} tok/s", decode_tps);
    println!("  total             {} ms", total_ms);

    if let Some(path) = &args.json {
        let json_out = serde_json::json!({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_ms": total_ms,
        });
        std::fs::write(path, serde_json::to_string_pretty(&json_out)?)?;
        println!("  wrote {}", path);
    }

    if let Some(path) = &args.compare {
        let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let get = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        println!("Comparison with {path}:");
        for key in ["ttft_ms", "prefill_tps", "decode_tps", "total_ms"] {
            let before = get(&old, key);
            let now = match key {
                "ttft_ms" => ttft as f64,
                "prefill_tps" => prefill_tps,
                "decode_tps" => decode_tps,
                _ => total_ms as f64,
            };
            let delta = if before > 0.0 {
                (now - before) / before * 100.0
            } else {
                0.0
            };
            println!("  {:<14} {:.1} -> {:.1} ({:+.1}%)", key, before, now, delta);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_fallback_hits_the_requested_size() {
        for tokens in [0, 1, 100, 2048] {
            assert_eq!(estimate_prompt(tokens).len(), tokens.saturating_mul(4));
        }
    }
}
