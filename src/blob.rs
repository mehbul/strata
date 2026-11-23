//! A small content-addressed blob store, in the shape Ollama's
//! `/api/blobs/:digest` expects.
//!
//! SHA-256 is implemented here rather than pulled in as a crate: the store
//! needs one hash function, and keeping it local avoids another dependency in
//! a binary whose job is to own a GPU, not to manage a dependency tree.

use anyhow::{bail, Result};
use std::io::Read;
use std::path::{Path, PathBuf};

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    let next = [
        state[0].wrapping_add(a),
        state[1].wrapping_add(b),
        state[2].wrapping_add(c),
        state[3].wrapping_add(d),
        state[4].wrapping_add(e),
        state[5].wrapping_add(f),
        state[6].wrapping_add(g),
        state[7].wrapping_add(h),
    ];
    *state = next;
}

/// SHA-256 of a stream, hex encoded. Reads in 64-byte blocks so a multi-gigabyte
/// blob never needs to be held in memory.
pub fn sha256_hex<R: Read>(mut reader: R) -> Result<String> {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut block = [0u8; 64];
    let mut used = 0usize;
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut block[used..])?;
        if n == 0 {
            break;
        }
        total = total.wrapping_add(n as u64);
        used += n;
        if used == 64 {
            compress(&mut state, &block);
            used = 0;
        }
    }
    let bit_len = total.wrapping_mul(8);
    block[used] = 0x80;
    used += 1;
    if used > 56 {
        for b in &mut block[used..] {
            *b = 0;
        }
        compress(&mut state, &block);
        used = 0;
    }
    for b in &mut block[used..56] {
        *b = 0;
    }
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress(&mut state, &block);
    let mut out = String::with_capacity(64);
    for word in state {
        out.push_str(&format!("{word:08x}"));
    }
    Ok(out)
}

/// Validate the `sha256:<64 hex>` form before it reaches the filesystem.
pub fn parse_digest(digest: &str) -> Result<String> {
    let hex = digest
        .strip_prefix("sha256:")
        .or_else(|| digest.strip_prefix("sha256-"))
        .unwrap_or(digest);
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("digest must be sha256:<64 hex digits>");
    }
    Ok(hex.to_ascii_lowercase())
}

/// Where uploaded blobs live. Kept under the models root so a user who moves
/// `rocm/` moves their whole model store with it.
pub fn blob_dir(models_root: &Path) -> PathBuf {
    models_root.join(".blobs")
}

pub fn blob_path(models_root: &Path, digest: &str) -> Result<PathBuf> {
    Ok(blob_dir(models_root).join(format!("sha256-{}", parse_digest(digest)?)))
}

/// Blobs with no matching file in the model store. A model's `model.gguf` is
/// normally a hard link to its blob, so comparing size plus the first four
/// bytes is enough to decide whether a blob is still referenced, without
/// hashing multi-gigabyte weights.
pub fn unreferenced(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    for dir in entries.flatten() {
        if !dir.path().is_dir() || dir.file_name() == ".blobs" { continue }
        let Ok(inner) = std::fs::read_dir(dir.path()) else { continue };
        for f in inner.flatten() {
            if !f.path().is_file() { continue }
            if let Ok(meta) = f.metadata() {
                let head = std::fs::File::open(f.path()).ok().and_then(|mut file| {
                    use std::io::Read;
                    let mut b = [0u8; 4];
                    file.read_exact(&mut b).ok().map(|_| b)
                });
                files.push((meta.len(), head));
            }
        }
    }
    let mut out = Vec::new();
    let Ok(blobs) = std::fs::read_dir(blob_dir(root)) else { return out };
    for blob in blobs.flatten() {
        let Ok(meta) = blob.metadata() else { continue };
        let head = std::fs::File::open(blob.path()).ok().and_then(|mut file| {
            use std::io::Read;
            let mut b = [0u8; 4];
            file.read_exact(&mut b).ok().map(|_| b)
        });
        if !files.iter().any(|(size, h)| *size == meta.len() && *h == head) {
            out.push(blob.path());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sha256_vectors() {
        assert_eq!(
            sha256_hex(&b"abc"[..]).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(&b""[..]).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(&b"hello world"[..]).unwrap(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn a_large_stream_crosses_block_boundaries() {
        let data = vec![7u8; 1000];
        let from_chunks = sha256_hex(&data[..]).unwrap();
        let mut acc = Vec::new();
        for chunk in data.chunks(37) {
            acc.extend_from_slice(chunk);
        }
        assert_eq!(from_chunks, sha256_hex(&acc[..]).unwrap());
    }

    #[test]
    fn digests_are_validated_before_use() {
        assert!(parse_digest("sha256:abc").is_err());
        assert!(parse_digest("../etc/passwd").is_err());
        assert_eq!(
            parse_digest(&format!("sha256:{}", "a".repeat(64))).unwrap(),
            "a".repeat(64)
        );
    }
}
