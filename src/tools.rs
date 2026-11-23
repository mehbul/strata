//! Thin wrappers over the llama.cpp utilities already in `runtime/`.
//!
//! Strata does not reimplement quantisation or sharding; the runtime ships
//! `llama-quantize` and `llama-gguf-split`, so the CLI just invokes them with
//! the same environment (loader path) the server uses.

use anyhow::{bail, Context, Result};
use std::process::Command;

fn binary(name: &str) -> Result<std::path::PathBuf> {
    let path = std::path::Path::new("runtime").join(name);
    if !path.exists() {
        bail!("{} is missing from runtime/", name);
    }
    Ok(path)
}

pub fn quantize(input: &str, output: &str, kind: &str) -> Result<()> {
    let exe = binary("llama-quantize.exe")?;
    let status = Command::new(exe)
        .args(["--allow-requantize", input, output, kind])
        .status()
        .with_context(|| "running llama-quantize")?;
    if !status.success() {
        bail!("llama-quantize failed");
    }
    println!("wrote {output}");
    Ok(())
}

pub fn split(input: &str, parts: usize, output: &str) -> Result<()> {
    let exe = binary("llama-gguf-split.exe")?;
    let status = Command::new(exe)
        .args(["--split", input, output, &parts.to_string()])
        .status()
        .with_context(|| "running llama-gguf-split")?;
    if !status.success() {
        bail!("llama-gguf-split failed");
    }
    println!("split {input} into {parts} parts at {output}");
    Ok(())
}

pub fn merge(input: &str, output: &str) -> Result<()> {
    let exe = binary("llama-gguf-split.exe")?;
    let status = Command::new(exe)
        .args(["--merge", input, output])
        .status()
        .with_context(|| "running llama-gguf-split --merge")?;
    if !status.success() {
        bail!("llama-gguf-split --merge failed");
    }
    println!("wrote {output}");
    Ok(())
}
