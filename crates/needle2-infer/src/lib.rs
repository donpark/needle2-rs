//! Needle 2 inference primitives. The model implementation is deliberately
//! split from the `.cact` reader so each numerical operation can be parity-tested.

pub const D_MODEL: usize = 512;
pub const NUM_LAYERS: usize = 27;
pub const NUM_HEADS: usize = 8;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 64;
pub const VOCAB_SIZE: usize = 8192;
pub const ROPE_THETA: f32 = 100_000.0;

pub fn zc_rms_norm(x: &[f32], scale: &[f32], epsilon: f32, out: &mut [f32]) {
    assert_eq!(x.len(), scale.len());
    assert_eq!(x.len(), out.len());
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = (mean + epsilon).sqrt().recip();
    for ((dst, &value), &s) in out.iter_mut().zip(x).zip(scale) {
        *dst = (1.0 + s) * value * inv_rms;
    }
}

pub fn hadamard_in_place(values: &mut [f32]) {
    assert!(values.len().is_power_of_two());
    let mut width = 1;
    while width < values.len() {
        let step = width * 2;
        for base in (0..values.len()).step_by(step) {
            for i in 0..width {
                let a = values[base + i];
                let b = values[base + width + i];
                values[base + i] = a + b;
                values[base + width + i] = a - b;
            }
        }
        width = step;
    }
    let scale = (values.len() as f32).sqrt().recip();
    values.iter_mut().for_each(|v| *v *= scale);
}

pub fn rope_frequencies(head_dim: usize, seq_len: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0; seq_len * half];
    let mut sin = vec![0.0; seq_len * half];
    for position in 0..seq_len {
        for i in 0..half {
            let frequency = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = position as f32 * frequency;
            cos[position * half + i] = angle.cos();
            sin[position * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// Applies the split-half rotary layout used by Needle 2 to one head vector.
pub fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    assert_eq!(x.len(), cos.len() * 2);
    assert_eq!(x.len(), sin.len() * 2);
    let half = x.len() / 2;
    let original = x.to_vec();
    for i in 0..half {
        x[i] = original[i] * cos[i] - original[half + i] * sin[i];
        x[half + i] = original[half + i] * cos[i] + original[i] * sin[i];
    }
}

pub fn matvec(weights: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    for row in 0..rows {
        out[row] = weights[row * cols..(row + 1) * cols]
            .iter()
            .zip(x)
            .map(|(w, v)| w * v)
            .sum();
    }
}

pub fn hadamard_mlp(x: &[f32], d1: &[f32], d2: &[f32], d3: &[f32], out: &mut [f32]) {
    assert_eq!(d1.len(), d2.len());
    assert_eq!(d2.len(), d3.len());
    assert_eq!(out.len(), x.len());
    let width = d1.len();
    assert!(width.is_power_of_two() && width >= x.len());
    let mut z = vec![0.0; width];
    z[..x.len()].copy_from_slice(x);
    for i in 0..width {
        z[i] *= d1[i];
    }
    hadamard_in_place(&mut z);
    for i in 0..width {
        z[i] = silu(d2[i] * z[i]);
    }
    hadamard_in_place(&mut z);
    for (dst, (value, scale)) in out.iter_mut().zip(z.iter().zip(d3)) {
        *dst = value * scale;
    }
}

pub fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    causal: bool,
    out: &mut [f32],
) {
    assert_eq!(q.len(), q_heads * seq_len * head_dim);
    assert_eq!(k.len(), kv_heads * seq_len * head_dim);
    assert_eq!(v.len(), k.len());
    assert_eq!(out.len(), q.len());
    assert_eq!(q_heads % kv_heads, 0);
    let repeat = q_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut scores = vec![0.0; seq_len];
    for h in 0..q_heads {
        let kv_h = h / repeat;
        for t in 0..seq_len {
            let q_base = (h * seq_len + t) * head_dim;
            let end = if causal { t + 1 } else { seq_len };
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..end {
                let k_base = (kv_h * seq_len + s) * head_dim;
                let score = q[q_base..q_base + head_dim]
                    .iter()
                    .zip(&k[k_base..k_base + head_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores[s] = score;
                max_score = max_score.max(score);
            }
            let mut total = 0.0;
            for score in &mut scores[..end] {
                *score = (*score - max_score).exp();
                total += *score;
            }
            let out_base = q_base;
            for d in 0..head_dim {
                let mut value = 0.0;
                for s in 0..end {
                    value += scores[s] / total * v[(kv_h * seq_len + s) * head_dim + d];
                }
                out[out_base + d] = value;
            }
        }
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn engram_indices(tokens: &[u32], orders: &[usize], heads: usize, slots: u32) -> Vec<u32> {
    const SEED: u32 = 0x9E37_79B9;
    const PRIME: u32 = 0x0100_0193;
    let mut output = vec![0; tokens.len() * orders.len() * heads];
    for (position, _) in tokens.iter().enumerate() {
        for (order_index, &order) in orders.iter().enumerate() {
            for head in 0..heads {
                let seed = SEED.wrapping_mul((order_index * heads + head + 1) as u32);
                let mut acc = seed;
                for j in 0..order {
                    let token = position.checked_sub(j).map(|i| tokens[i]).unwrap_or(0);
                    acc = (acc ^ token).wrapping_mul(PRIME);
                }
                acc ^= acc >> 15;
                output[position * orders.len() * heads + order_index * heads + head] = acc % slots;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_orthonormal() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        hadamard_in_place(&mut x);
        hadamard_in_place(&mut x);
        for (got, want) in x.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let (cos, sin) = rope_frequencies(4, 1, ROPE_THETA);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, &cos, &sin);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rms_norm_matches_identity_scale() {
        let mut out = [0.0; 2];
        zc_rms_norm(&[3.0, 4.0], &[0.0, 0.0], 1e-6, &mut out);
        assert!((out[0] - 0.848528).abs() < 1e-5);
        assert!((out[1] - 1.131370).abs() < 1e-5);
    }

    #[test]
    fn engram_hash_matches_reference_fixture() {
        assert_eq!(
            engram_indices(&[1, 2, 3, 4], &[2, 3], 2, 8192),
            [
                3101, 3314, 234, 5746, 6228, 6217, 1096, 3580, 2635, 6396, 4366, 6630, 7601, 4923,
                1956, 4976
            ]
        );
    }

    #[test]
    fn matvec_is_row_major() {
        let mut out = [0.0; 2];
        matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0], &mut out);
        assert_eq!(out, [17.0, 39.0]);
    }

    #[test]
    fn hadamard_mlp_preserves_zero_when_output_gate_is_zero() {
        let mut out = [1.0; 2];
        hadamard_mlp(
            &[1.0, 2.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0; 4],
            &[0.0; 4],
            &mut out,
        );
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn causal_gqa_cannot_see_future_values() {
        let q = [1.0, 1.0, 1.0, 1.0];
        let k = [1.0, 1.0, 1.0, 1.0];
        let v = [2.0, 4.0, 8.0, 16.0];
        let mut out = [0.0; 4];
        gqa_attention(&q, &k, &v, 1, 1, 4, 1, true, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!(out[1] > 2.0 && out[1] < 4.0);
        assert!(out[2] > out[1] && out[2] < 8.0);
        assert!(out[3] > out[2] && out[3] < 16.0);
    }
}
