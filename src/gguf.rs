//! GGUF v2/v3 reader — header, metadata and tensor index.
//!
//! Reads the real model rather than trusting the hand-written table in
//! `registry.rs`. Everything here is parsed from the file on disk; nothing is
//! assumed about the architecture.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";

/// Quantisation of a stored tensor. Values are GGML's own type ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::F32 => "F32".into(),
            Self::F16 => "F16".into(),
            Self::Q4_0 => "Q4_0".into(),
            Self::Q4_1 => "Q4_1".into(),
            Self::Q5_0 => "Q5_0".into(),
            Self::Q5_1 => "Q5_1".into(),
            Self::Q8_0 => "Q8_0".into(),
            Self::Q8_1 => "Q8_1".into(),
            Self::Q2K => "Q2_K".into(),
            Self::Q3K => "Q3_K".into(),
            Self::Q4K => "Q4_K".into(),
            Self::Q5K => "Q5_K".into(),
            Self::Q6K => "Q6_K".into(),
            Self::Q8K => "Q8_K".into(),
            Self::BF16 => "BF16".into(),
            Self::Other(id) => format!("type{id}"),
        }
    }

    /// (elements per block, bytes per block). k-quants use 256-element blocks.
    pub fn block(&self) -> Option<(usize, usize)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Other(_) => return None,
        })
    }
}

/// A metadata value. GGUF's own type tags, kept lossless.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => *v as u64,
            Self::I8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::I16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::I32(v) => *v as u64,
            Self::U64(v) => *v,
            Self::I64(v) => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Self::F32(v) => *v as f64,
            Self::F64(v) => *v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn len(&self) -> Option<usize> {
        match self {
            Self::Array(v) => Some(v.len()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub kind: GgmlType,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }
    /// Stored size in bytes, or None for a type we do not know the layout of.
    pub fn bytes(&self) -> Option<u64> {
        let (per_block, block_bytes) = self.kind.block()?;
        let n = self.elements();
        Some((n / per_block as u64) * block_bytes as u64)
    }
}

pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute offset where tensor data begins.
    pub data_offset: u64,
    pub file_size: u64,
}

struct Cursor<R: Read + Seek> {
    inner: R,
    pos: u64,
}

impl<R: Read + Seek> Cursor<R> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        self.pos += 1;
        Ok(b[0])
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.inner.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 64 * 1024 * 1024 {
            bail!("implausible string length {n} in GGUF metadata");
        }
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        self.pos += n as u64;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn value(&mut self, tag: u32) -> Result<Value> {
        Ok(match tag {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(u16::from_le_bytes(self.take::<2>()?)),
            3 => Value::I16(i16::from_le_bytes(self.take::<2>()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.take::<4>()?)),
            6 => Value::F32(f32::from_le_bytes(self.take::<4>()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()?;
                let mut items = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n {
                    items.push(self.value(elem)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.take::<8>()?)),
            12 => Value::F64(f64::from_le_bytes(self.take::<8>()?)),
            other => bail!("unknown GGUF value tag {other}"),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_size = file.metadata()?.len();
        let mut cur = Cursor { inner: BufReader::with_capacity(1 << 20, file), pos: 0 };

        let magic = cur.take::<4>()?;
        if &magic != MAGIC {
            bail!("not a GGUF file: magic was {magic:?}");
        }
        let version = cur.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let tensor_count = cur.u64()?;
        let kv_count = cur.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cur.string()?;
            let tag = cur.u32()?;
            let value = cur.value(tag)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            if n_dims > 4 {
                bail!("tensor {name} claims {n_dims} dimensions");
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.u64()?);
            }
            let kind = GgmlType::from_id(cur.u32()?);
            let offset = cur.u64()?;
            tensors.push(TensorInfo { name, dims, kind, offset });
        }

        // Tensor data begins at the next alignment boundary after the index.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let data_offset = cur.pos.div_ceil(alignment) * alignment;
        cur.inner.seek(SeekFrom::Start(data_offset))?;

        Ok(Self { version, metadata, tensors, data_offset, file_size })
    }

    pub fn arch(&self) -> &str {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    }

    pub fn name(&self) -> &str {
        self.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("unnamed")
    }

    /// Architecture-scoped metadata lookup, e.g. `key("block_count")`.
    pub fn key(&self, suffix: &str) -> Option<&Value> {
        self.metadata.get(&format!("{}.{}", self.arch(), suffix))
    }

    pub fn key_u64(&self, suffix: &str) -> Option<u64> {
        self.key(suffix).and_then(|v| v.as_u64())
    }

    pub fn key_f64(&self, suffix: &str) -> Option<f64> {
        self.key(suffix).and_then(|v| v.as_f64())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total stored bytes of all tensors whose layout we understand.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(|t| t.bytes()).sum()
    }

    /// How many bytes sit in each quantisation, largest first.
    pub fn by_type(&self) -> Vec<(String, usize, u64)> {
        let mut acc: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let e = acc.entry(t.kind.name()).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.bytes().unwrap_or(0);
        }
        let mut out: Vec<_> = acc.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
        out.sort_by_key(|(_, _, b)| std::cmp::Reverse(*b));
        out
    }

    /// The per-layer prefix set, e.g. everything under `blk.0.`.
    pub fn layer_tensors(&self, layer: usize) -> Vec<&TensorInfo> {
        let prefix = format!("blk.{layer}.");
        self.tensors.iter().filter(|t| t.name.starts_with(&prefix)).collect()
    }
}

/// What the file says about the model, as opposed to what a registry guesses.
#[derive(Debug, Clone)]
pub struct ModelFacts {
    pub name: String,
    pub arch: String,
    pub layers: usize,
    pub embedding: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ff: usize,
    pub context: usize,
    pub vocab: usize,
    pub rope_base: f64,
    /// Present only on hybrid state-space architectures.
    pub ssm_state: Option<usize>,
    pub ssm_inner: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub file_bytes: u64,
    pub tensor_count: usize,
}

impl ModelFacts {
    pub fn read(gguf: &Gguf) -> Result<Self> {
        let vocab = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.len())
            .ok_or_else(|| anyhow!("no tokenizer.ggml.tokens array in file"))?;
        Ok(Self {
            name: gguf.name().to_string(),
            arch: gguf.arch().to_string(),
            layers: gguf.key_u64("block_count").unwrap_or(0) as usize,
            embedding: gguf.key_u64("embedding_length").unwrap_or(0) as usize,
            heads: gguf.key_u64("attention.head_count").unwrap_or(0) as usize,
            kv_heads: gguf.key_u64("attention.head_count_kv").unwrap_or(0) as usize,
            experts: gguf.key_u64("expert_count").unwrap_or(0) as usize,
            experts_used: gguf.key_u64("expert_used_count").unwrap_or(0) as usize,
            expert_ff: gguf.key_u64("expert_feed_forward_length").unwrap_or(0) as usize,
            context: gguf.key_u64("context_length").unwrap_or(0) as usize,
            vocab,
            rope_base: gguf.key_f64("rope.freq_base").unwrap_or(0.0),
            ssm_state: gguf.key_u64("ssm.state_size").map(|v| v as usize),
            ssm_inner: gguf.key_u64("ssm.inner_size").map(|v| v as usize),
            full_attention_interval: gguf.key_u64("full_attention_interval").map(|v| v as usize),
            file_bytes: gguf.file_size,
            tensor_count: gguf.tensors.len(),
        })
    }

    pub fn is_moe(&self) -> bool {
        self.experts > 1
    }

    pub fn is_hybrid_ssm(&self) -> bool {
        self.ssm_state.is_some()
    }
}

/// `strata inspect` — read a GGUF and report what is actually in it.
pub fn inspect(path: &str) -> Result<()> {
    let gguf = Gguf::open(path)?;
    let facts = ModelFacts::read(&gguf)?;
    let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;

    println!("=== {} ===", facts.name);
    println!("file          : {} ({:.2} GiB)", path, gb(facts.file_bytes));
    println!("gguf version  : {}", gguf.version);
    println!("architecture  : {}", facts.arch);
    println!("tensors       : {}  (data starts at byte {})", facts.tensor_count, gguf.data_offset);
    println!();
    println!("layers        : {}", facts.layers);
    println!("embedding     : {}", facts.embedding);
    println!("attn heads    : {} query / {} kv", facts.heads, facts.kv_heads);
    if facts.is_moe() {
        println!(
            "experts       : {} per layer, top-{} routed, ff {}",
            facts.experts, facts.experts_used, facts.expert_ff
        );
        println!("expert total  : {}", facts.layers * facts.experts);
    }
    if facts.is_hybrid_ssm() {
        println!(
            "state space   : state {}, inner {}, full-attention every {} layers",
            facts.ssm_state.unwrap_or(0),
            facts.ssm_inner.unwrap_or(0),
            facts.full_attention_interval.unwrap_or(0),
        );
    }
    println!("context       : {}", facts.context);
    println!("vocab         : {}", facts.vocab);
    println!("rope base     : {}", facts.rope_base);

    println!("\n--- stored weights by quantisation ---");
    for (kind, count, bytes) in gguf.by_type() {
        println!("  {kind:<6} {count:>5} tensors  {:>8.2} GiB", gb(bytes));
    }
    println!("  {:<6} {:>5} tensors  {:>8.2} GiB total", "", facts.tensor_count, gb(gguf.tensor_bytes()));

    println!("\n--- layer 0 ---");
    for t in gguf.layer_tensors(0) {
        let dims = t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" x ");
        println!("  {:<34} {:<20} {}", t.name, dims, t.kind.name());
    }
    Ok(())
}
