//! Reference dequantisation for the k-quants Strata actually ships with.
//!
//! The layouts and arithmetic here follow llama.cpp b10679
//! (`ggml-common.h`, `ggml-quants.c`) exactly. Nothing in the server calls this
//! yet — the matrix multiplies are still llama.cpp's job — but having the
//! decoder in the repository makes the placement and byte estimates
//! checkable, and gives the forward-pass work a verified starting point.

use anyhow::{bail, Result};
use std::io::{Read, Seek, SeekFrom};

pub const QK_K: usize = 256;
pub const Q4_K_BYTES: usize = 144;
pub const Q6_K_BYTES: usize = 210;

/// IEEE 754 half precision → f32, by bits. GGUF stores scales as raw u16.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        sign * (mant as f32) * (1.0 / 16_777_216.0)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (2.0f32).powi(exp as i32 - 15) * (1.0 + mant as f32 / 1024.0)
    }
}

fn u16le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// 6-bit scale/min pair for sub-block `j`, from a Q4_K or Q5_K scale array.
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One 144-byte Q4_K super-block → 256 weights.
pub fn dequant_q4_k(block: &[u8]) -> Result<Vec<f32>> {
    if block.len() != Q4_K_BYTES {
        bail!("Q4_K block must be {Q4_K_BYTES} bytes, got {}", block.len());
    }
    let d = f16_to_f32(u16le(&block[0..2]));
    let min = f16_to_f32(u16le(&block[2..4]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut out = Vec::with_capacity(QK_K);
    let mut q = 0usize;
    let mut is = 0usize;
    for _ in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let m1 = min * m1 as f32;
        let d2 = d * sc2 as f32;
        let m2 = min * m2 as f32;
        for l in 0..32 {
            out.push(d1 * (qs[q + l] & 0x0f) as f32 - m1);
        }
        for l in 0..32 {
            out.push(d2 * (qs[q + l] >> 4) as f32 - m2);
        }
        q += 32;
        is += 2;
    }
    Ok(out)
}

/// One 210-byte Q6_K super-block → 256 weights.
pub fn dequant_q6_k(block: &[u8]) -> Result<Vec<f32>> {
    if block.len() != Q6_K_BYTES {
        bail!("Q6_K block must be {Q6_K_BYTES} bytes, got {}", block.len());
    }
    let d = f16_to_f32(u16le(&block[208..210]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let mut out = vec![0f32; QK_K];
    for n in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8 - 32;
            out[n + l] = d * scales[is] as f32 * q1 as f32;
            out[n + l + 32] = d * scales[is + 2] as f32 * q2 as f32;
            out[n + l + 64] = d * scales[is + 4] as f32 * q3 as f32;
            out[n + l + 96] = d * scales[is + 6] as f32 * q4 as f32;
        }
    }
    Ok(out)
}

/// Decompress the first `count` weights of one tensor, reading only the bytes
/// needed for those blocks.
pub fn preview(path: &str, tensor: &str, count: usize) -> Result<Vec<f32>> {
    let gguf = crate::gguf::Gguf::open(path)?;
    let info = gguf
        .tensor(tensor)
        .ok_or_else(|| anyhow::anyhow!("no tensor named {tensor}"))?;
    type Decoder = fn(&[u8]) -> Result<Vec<f32>>;
    let (block_bytes, elements, decode): (usize, usize, Decoder) = match info.kind {
        crate::gguf::GgmlType::Q4K => (Q4_K_BYTES, QK_K, dequant_q4_k),
        crate::gguf::GgmlType::Q6K => (Q6_K_BYTES, QK_K, dequant_q6_k),
        _ => bail!(
            "only Q4_K and Q6_K are dequantised; tensor is {}",
            info.kind.name()
        ),
    };
    let blocks = count.div_ceil(elements);
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(gguf.data_offset + info.offset))?;
    let mut bytes = vec![0u8; blocks * block_bytes];
    file.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(blocks * elements);
    for chunk in bytes.chunks_exact(block_bytes) {
        out.extend(decode(chunk)?);
    }
    out.truncate(count);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_vectors() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x6400), 1024.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn q4_k_reference_block() {
        // d=1, dmin=0.5, scale[0]=1, min[0]=1, first nibble = 15.
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        block[4] = 1; // scale of sub-block 0
        block[8] = 1; // min of sub-block 0
        block[16] = 0x0f;
        let out = dequant_q4_k(&block).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 14.5); // 1*1*15 - 0.5*1
        assert_eq!(out[1], -0.5); // nibble 0
        assert_eq!(out[32], 0.0); // sub-block 1 has scale and min 0
        assert_eq!(out[255], 0.0);
    }

    #[test]
    fn q4_k_all_quants_are_exact_with_unit_scale() {
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        block[4] = 1; // sub-block 0 scale = 1
        block[5] = 1; // sub-block 1 scale = 1; both mins stay 0
        for i in 0..16 {
            block[16 + i] = (i as u8) | (((15 - i) as u8) << 4);
        }
        let out = dequant_q4_k(&block).unwrap();
        for i in 0..16 {
            assert_eq!(out[i], i as f32);
            assert_eq!(out[32 + i], (15 - i) as f32);
        }
    }

    #[test]
    fn q6_k_reference_block() {
        let mut block = vec![0u8; Q6_K_BYTES];
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        for s in &mut block[192..208] {
            *s = 1;
        }
        // q1 = 31 → raw 63; q2,q3,q4 = 0 → raw 32.
        block[0] = 0x0f;
        // Each qh byte holds four 2-bit high parts. 0xab = q1 high 3,
        // the others high 2 (raw 32, which dequantises to 0).
        block[128] = 0xab;
        for i in 1..32 {
            block[i] = 0x00;
            block[32 + i] = 0x00;
            block[128 + i] = 0xaa;
        }
        let out = dequant_q6_k(&block).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 31.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[32], 0.0);
        assert_eq!(out[64], 0.0);
        assert_eq!(out[96], 0.0);
    }

    #[test]
    fn malformed_blocks_are_rejected() {
        assert!(dequant_q4_k(&[0u8; 10]).is_err());
        assert!(dequant_q6_k(&[0u8; 10]).is_err());
    }
}
