//! Minimal safetensors → GGUF conversion for generic transformer checkpoints.
//!
//! This is deliberately a straight copy ofSupported F32/F16 tensors into an
//! F32/F16 GGUF with the metadata llama.cpp needs to identify the architecture.
//! It does not quantise or rename vendor-specific tensor names; use
//! `strata quantize` afterwards for a smaller file.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_string_array(out: &mut Vec<u8>, items: &[String]) {
    out.extend_from_slice(&9u32.to_le_bytes()); // array
    out.extend_from_slice(&8u32.to_le_bytes()); // string element
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        put_string(out, item);
    }
}

fn put_u32_array(out: &mut Vec<u8>, items: &[u32]) {
    out.extend_from_slice(&9u32.to_le_bytes()); // array
    out.extend_from_slice(&4u32.to_le_bytes()); // u32 element
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        out.extend_from_slice(&item.to_le_bytes());
    }
}

/// Read Hugging Face tokenizer.json into GGUF's tokenizer arrays.
fn tokenizer_arrays(dir: &std::path::Path) -> Option<(Vec<String>, Vec<String>, Vec<u32>)> {
    let text = std::fs::read_to_string(dir.join("tokenizer.json")).ok()?;
    let tokenizer: Value = serde_json::from_str(&text).ok()?;
    let model = tokenizer.get("model")?;
    let vocab = model.get("vocab")?.as_object()?;
    let mut tokens = vec![String::new(); vocab.len()];
    for (token, id) in vocab {
        if let Some(i) = id.as_u64() {
            if (i as usize) < tokens.len() {
                tokens[i as usize] = token.clone();
            }
        }
    }
    let merges: Vec<String> = match model.get("merges")? {
        Value::Array(items) => items
            .iter()
            .filter_map(|m| match m {
                Value::String(s) => Some(s.clone()),
                Value::Array(pair) if pair.len() == 2 => Some(format!(
                    "{} {}",
                    pair[0].as_str().unwrap_or(""),
                    pair[1].as_str().unwrap_or("")
                )),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let types = vec![1u32; tokens.len()];
    Some((tokens, merges, types))
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::String(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_string(out, s);
        }
        Value::Number(n) if n.is_u64() => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&n.as_u64().unwrap().to_le_bytes());
        }
        Value::Number(n) if n.is_i64() => {
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&n.as_i64().unwrap().to_le_bytes());
        }
        Value::Number(n) if n.is_f64() => {
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&n.as_f64().unwrap().to_le_bytes());
        }
        Value::Bool(b) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(*b as u8);
        }
        _ => {}
    }
}

pub fn convert(dir: &str, output: &str) -> Result<()> {
    let dir = std::path::Path::new(dir);
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).with_context(|| "reading config.json")?,
    )?;
    let arch = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .unwrap_or("LlamaForCausalLM")
        .strip_suffix("ForCausalLM")
        .unwrap_or("llama")
        .to_ascii_lowercase();
    let layers = config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let embed = config
        .get("hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ctx = config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(4096);
    let vocab = config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let heads = config
        .get("num_attention_heads")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let kv_heads = config
        .get("num_key_value_heads")
        .and_then(Value::as_u64)
        .unwrap_or(heads);
    let rope_base = config
        .get("rope_theta")
        .and_then(Value::as_f64)
        .unwrap_or(10000.0);

    let files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    if files.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }

    struct Tensor {
        name: String,
        dims: Vec<u64>,
        dtype: u32,
        bytes: Vec<u8>,
    }
    let mut tensors = Vec::new();
    for file in &files {
        let bytes = std::fs::read(file)?;
        if bytes.len() < 8 {
            bail!("short safetensors file {}", file.display())
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
        let base = 8 + header_len;
        for (name, meta) in header.as_object().unwrap_or(&serde_json::Map::new()) {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match meta.get("dtype").and_then(Value::as_str) {
                Some("F32") => 0u32,
                Some("F16") => 1u32,
                Some("BF16") => 30u32,
                Some(other) => bail!("unsupported safetensors dtype {other} for {name}"),
                None => bail!("tensor {name} has no dtype"),
            };
            let shape: Vec<u64> = meta
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            let offsets: Vec<usize> = meta
                .get("data_offsets")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_u64)
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default();
            if offsets.len() != 2 {
                bail!("tensor {name} has no data_offsets")
            }
            let data = bytes[base + offsets[0]..base + offsets[1]].to_vec();
            let mut dims = shape;
            dims.reverse(); // safetensors row-major → GGUF dimension order
            tensors.push(Tensor {
                name: name.clone(),
                dims,
                dtype,
                bytes: data,
            });
        }
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let tokenizer = tokenizer_arrays(dir);
    let kv_count = 11 + if tokenizer.is_some() { 5 } else { 0 };
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kv_count as u64).to_le_bytes());
    let metadata: Vec<(String, Value)> = vec![
        ("general.architecture".into(), json!(arch)),
        (
            "general.name".into(),
            json!(&config["model_type"].as_str().unwrap_or("converted")),
        ),
        ("general.file_type".into(), json!(0u64)),
        (format!("{arch}.block_count"), json!(layers)),
        (format!("{arch}.context_length"), json!(ctx)),
        (format!("{arch}.embedding_length"), json!(embed)),
        (format!("{arch}.vocab_size"), json!(vocab)),
        (format!("{arch}.attention.head_count"), json!(heads)),
        (format!("{arch}.attention.head_count_kv"), json!(kv_heads)),
        (format!("{arch}.rope.freq_base"), json!(rope_base)),
        ("tokenizer.ggml.model".into(), json!("gpt2")),
    ];
    for (k, v) in &metadata {
        put_string(&mut out, k);
        put_value(&mut out, v);
    }
    if let Some((tokens, merges, types)) = tokenizer {
        let bos = config
            .get("bos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(1) as u32;
        let eos = config
            .get("eos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(2) as u32;
        for (key, value) in [
            ("tokenizer.ggml.tokens".to_string(), 0u8),
            ("tokenizer.ggml.merges".to_string(), 1),
            ("tokenizer.ggml.token_type".to_string(), 2),
            ("tokenizer.ggml.bos_token_id".to_string(), 3),
            ("tokenizer.ggml.eos_token_id".to_string(), 4),
        ] {
            put_string(&mut out, &key);
            match value {
                0 => put_string_array(&mut out, &tokens),
                1 => put_string_array(&mut out, &merges),
                2 => put_u32_array(&mut out, &types),
                3 => put_value(&mut out, &json!(bos)),
                _ => put_value(&mut out, &json!(eos)),
            }
        }
    }
    let mut offset = 0u64;
    for t in &tensors {
        put_string(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.dtype.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += t.bytes.len() as u64;
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for t in &tensors {
        out.extend_from_slice(&t.bytes);
    }
    std::fs::write(output, out)?;
    println!("wrote {output} ({} tensors)", tensors.len());
    Ok(())
}
