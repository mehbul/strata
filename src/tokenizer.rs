//! A byte-level BPE over the GGUF's own vocabulary.
//!
//! llama.cpp carries a full tokenizer; this one is deliberately small. It
//! reads `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` and the BOS/EOS flags
//! from the file itself, applies GPT-2's byte-to-unicode map and merge ranks,
//! and is checked against `llama-tokenize.exe` on the development machine.
//! It is not a replacement for the compute process's tokenizer.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

pub struct Tokenizer {
    tokens: Vec<String>,
    lookup: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_map: Vec<char>,
    bos: u32,
    eos: u32,
    add_bos: bool,
    add_eos: bool,
}

/// GPT-2's reversible byte → unicode mapping, so every input byte has a
/// printable character that can appear in the model's vocabulary.
fn bytes_to_unicode() -> Vec<char> {
    let printable: Vec<u32> = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .map(u32::from)
        .collect();
    let mut bs = printable.clone();
    let mut cs = printable.clone();
    let mut n = 0usize;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.insert(n, b);
            cs.insert(n, 256 + n as u32);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (byte, code) in bs.iter().zip(cs.iter()) {
        out[*byte as usize] = char::from_u32(*code).unwrap_or('\u{fffd}');
    }
    out
}

fn gguf_strings(value: Option<&crate::gguf::Value>) -> Vec<String> {
    match value {
        Some(crate::gguf::Value::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                crate::gguf::Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

impl Tokenizer {
    pub fn from_gguf(path: &str) -> Result<Self> {
        let gguf = crate::gguf::Gguf::open(path)?;
        let tokens = gguf_strings(gguf.metadata.get("tokenizer.ggml.tokens"));
        if tokens.is_empty() {
            bail!("no tokenizer.ggml.tokens in {path}");
        }
        let lookup = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let mut merges = HashMap::new();
        for (rank, line) in gguf_strings(gguf.metadata.get("tokenizer.ggml.merges"))
            .into_iter()
            .enumerate()
        {
            if let Some((a, b)) = line.split_once(' ') {
                merges.insert((a.to_string(), b.to_string()), rank);
            }
        }
        let number = |key: &str, default: u32| -> u32 {
            gguf.metadata
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default)
        };
        let flag = |key: &str, default: bool| -> bool {
            match gguf.metadata.get(key) {
                Some(crate::gguf::Value::Bool(b)) => *b,
                _ => default,
            }
        };
        Ok(Self {
            tokens,
            lookup,
            merges,
            byte_map: bytes_to_unicode(),
            bos: number("tokenizer.ggml.bos_token_id", 0),
            eos: number("tokenizer.ggml.eos_token_id", 0),
            add_bos: flag("tokenizer.ggml.add_bos_token", false),
            add_eos: flag("tokenizer.ggml.add_eos_token", false),
        })
    }

    /// Split a byte-mapped string the way GPT-2's pre-tokenizer does:
    /// contractions, then optional leading space + letters, digits, or other
    /// characters, then whitespace runs. Avoids a regex dependency.
    fn pre_split(&self, mapped: &str) -> Vec<String> {
        let chars: Vec<char> = mapped.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '\'' {
                let rest: String = chars[i..].iter().take(3).collect();
                let contraction = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
                    .iter()
                    .find(|p| rest.starts_with(**p))
                    .copied();
                if let Some(p) = contraction {
                    out.push(p.to_string());
                    i += p.chars().count();
                    continue;
                }
            }
            let start = i;
            let mut space = false;
            if c == 'Ġ' || c == ' ' {
                space = true;
                i += 1;
                if i >= chars.len() {
                    out.push(c.to_string());
                    break;
                }
            }
            let first = chars[i];
            let class = |ch: char| {
                if ch.is_alphabetic() {
                    1
                } else if ch.is_numeric() {
                    2
                } else if ch.is_whitespace() || ch == 'Ġ' {
                    4
                } else {
                    3
                }
            };
            let wanted = class(first);
            if wanted == 4 {
                // A whitespace run: keep all but the last space as one piece.
                while i < chars.len() && class(chars[i]) == 4 {
                    i += 1;
                }
            } else {
                while i < chars.len() && class(chars[i]) == wanted {
                    i += 1;
                }
            }
            let piece: String = if space {
                let mut s = chars[start].to_string();
                s.extend(&chars[start + 1..i]);
                s
            } else {
                chars[start..i].iter().collect()
            };
            if !piece.is_empty() {
                out.push(piece);
            }
        }
        out
    }

    fn bpe(&self, piece: &str) -> Vec<String> {
        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .merges
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mapped: String = text
            .as_bytes()
            .iter()
            .map(|b| self.byte_map[*b as usize])
            .collect();
        let mut ids = Vec::new();
        if self.add_bos {
            ids.push(self.bos);
        }
        for piece in self.pre_split(&mapped) {
            for symbol in self.bpe(&piece) {
                let id = *self
                    .lookup
                    .get(&symbol)
                    .with_context(|| format!("token {symbol:?} is not in the vocabulary"))?;
                ids.push(id);
            }
        }
        if self.add_eos {
            ids.push(self.eos);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut inverse = HashMap::new();
        for (byte, ch) in self.byte_map.iter().enumerate() {
            inverse.insert(*ch, byte as u8);
        }
        let mut out = Vec::new();
        for id in ids {
            let token = self
                .tokens
                .get(*id as usize)
                .ok_or_else(|| anyhow::anyhow!("token id {id} out of range"))?;
            for ch in token.chars() {
                if let Some(byte) = inverse.get(&ch) {
                    out.push(*byte);
                }
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_is_complete_and_reversible() {
        let map = bytes_to_unicode();
        assert_eq!(map.len(), 256);
        let distinct: std::collections::HashSet<char> = map.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
    }

    #[test]
    fn a_small_vocab_performs_bpe_merges_by_rank() {
        let mut t = Tokenizer {
            tokens: vec!["a".into(), "b".into(), "ab".into()],
            lookup: [("a".into(), 0), ("b".into(), 1), ("ab".into(), 2)]
                .into_iter()
                .collect(),
            merges: [(("a".into(), "b".into()), 0)].into_iter().collect(),
            byte_map: bytes_to_unicode(),
            bos: 0,
            eos: 0,
            add_bos: false,
            add_eos: false,
        };
        t.tokens = vec!["a".into(), "b".into(), "ab".into()];
        let ids = t.encode("ab").unwrap();
        assert_eq!(ids, vec![2]);
    }
}
