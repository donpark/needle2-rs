//! Needle 2 inference primitives. The model implementation is deliberately
//! split from the `.cact` reader so each numerical operation can be parity-tested.

pub const D_MODEL: usize = 512;
pub const NUM_LAYERS: usize = 27;
pub const NUM_HEADS: usize = 8;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 64;
pub const VOCAB_SIZE: usize = 8192;
pub const MHC_LANES: usize = 4;
pub const ENGRAM_SITES: usize = 2;
pub const ENGRAM_HEADS: usize = 2;
pub const ENGRAM_ORDERS: [usize; 2] = [2, 3];
pub const ENGRAM_SLOTS: usize = 8192;
pub const ENGRAM_SUB_DIM: usize = 128;
pub const ROPE_THETA: f32 = 100_000.0;

pub struct EngramWeights {
    pub tables: Vec<f32>,
    pub key_proj: Vec<f32>,
    pub value_proj: Vec<f32>,
    pub taps: Vec<f32>,
}

impl EngramWeights {
    pub fn from_cact(
        model: &needle2_format::CactModel<'_>,
        site: usize,
    ) -> Result<Self, needle2_format::CactError> {
        let base = LAYER_TENSOR_START + NUM_LAYERS * 14 + 9 + site * 4;
        Ok(Self {
            tables: model.tensor_f32(base)?,
            key_proj: model.tensor_f32(base + 1)?,
            value_proj: model.tensor_f32(base + 2)?,
            taps: model.tensor_f32(base + 3)?,
        })
    }
}

pub struct MhcWeights {
    pub a_pre: Vec<f32>,
    pub a_post: Vec<f32>,
    pub a_res: Vec<f32>,
    pub b_pre: Vec<f32>,
    pub b_post: Vec<f32>,
    pub b_res: Vec<f32>,
    pub phi_pre: Vec<f32>,
    pub phi_post: Vec<f32>,
    pub phi_res: Vec<f32>,
}

impl MhcWeights {
    pub fn from_cact(
        model: &needle2_format::CactModel<'_>,
    ) -> Result<Self, needle2_format::CactError> {
        let base = LAYER_TENSOR_START + NUM_LAYERS * 14;
        let get = |offset| model.tensor_f32(base + offset);
        Ok(Self {
            a_pre: get(0)?,
            a_post: get(1)?,
            a_res: get(2)?,
            b_pre: get(3)?,
            b_post: get(4)?,
            b_res: get(5)?,
            phi_pre: get(6)?,
            phi_post: get(7)?,
            phi_res: get(8)?,
        })
    }
}

pub struct LayerWeights {
    pub norm_in: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn_gate: f32,
    pub pre_hada: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub d3: Vec<f32>,
}

pub const LAYER_TENSOR_START: usize = 1;

impl LayerWeights {
    pub fn from_cact(
        model: &needle2_format::CactModel<'_>,
        layer: usize,
    ) -> Result<Self, needle2_format::CactError> {
        let base = 1 + layer * 14;
        let get = |offset| model.tensor_f32(base + offset);
        Ok(Self {
            norm_in: get(0)?,
            q_proj: get(1)?,
            k_proj: get(2)?,
            v_proj: get(3)?,
            q_norm: get(4)?,
            k_norm: get(5)?,
            gate_proj: get(6)?,
            out_proj: get(7)?,
            post_norm: get(8)?,
            attn_gate: get(9)?.first().copied().unwrap_or_default(),
            pre_hada: get(10)?,
            d1: get(11)?,
            d2: get(12)?,
            d3: get(13)?,
        })
    }
}

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

pub fn engram_forward(tokens: &[u32], weights: &EngramWeights, key: &mut [f32], value: &mut [f32]) {
    let seq_len = tokens.len();
    assert_eq!(key.len(), seq_len * D_MODEL);
    assert_eq!(value.len(), key.len());
    let indices = engram_indices(tokens, &ENGRAM_ORDERS, ENGRAM_HEADS, ENGRAM_SLOTS as u32);
    let e = vec![0.0; ENGRAM_ORDERS.len() * ENGRAM_HEADS * ENGRAM_SUB_DIM];
    let mut fetched = vec![0.0; e.len()];
    let mut raw_value = vec![0.0; seq_len * D_MODEL];
    for t in 0..seq_len {
        for table in 0..ENGRAM_ORDERS.len() * ENGRAM_HEADS {
            let order = ENGRAM_ORDERS[table / ENGRAM_HEADS];
            let slot = if t + 1 >= order {
                indices[t * ENGRAM_ORDERS.len() * ENGRAM_HEADS + table] as usize
            } else {
                0
            };
            let src = &weights.tables[(table * ENGRAM_SLOTS + slot) * ENGRAM_SUB_DIM
                ..(table * ENGRAM_SLOTS + slot + 1) * ENGRAM_SUB_DIM];
            fetched[table * ENGRAM_SUB_DIM..(table + 1) * ENGRAM_SUB_DIM].copy_from_slice(src);
            if t + 1 < order {
                fetched[table * ENGRAM_SUB_DIM..(table + 1) * ENGRAM_SUB_DIM].fill(0.0);
            }
        }
        matvec(
            &weights.key_proj,
            D_MODEL,
            e.len(),
            &fetched,
            &mut key[t * D_MODEL..(t + 1) * D_MODEL],
        );
        matvec(
            &weights.value_proj,
            D_MODEL,
            e.len(),
            &fetched,
            &mut raw_value[t * D_MODEL..(t + 1) * D_MODEL],
        );
        value[t * D_MODEL..(t + 1) * D_MODEL].fill(0.0);
        for tap in 0..4 {
            if t >= tap {
                for d in 0..D_MODEL {
                    value[t * D_MODEL + d] +=
                        weights.taps[tap * D_MODEL + d] * raw_value[(t - tap) * D_MODEL + d];
                }
            }
        }
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

fn rms_unit(x: &[f32], out: &mut [f32]) {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = (mean + 1e-6).sqrt().recip();
    for (dst, value) in out.iter_mut().zip(x) {
        *dst = value * scale;
    }
}

fn sinkhorn(logits: &[f32], lanes: usize, out: &mut [f32]) {
    out.copy_from_slice(logits);
    for _ in 0..20 {
        for row in out.chunks_exact_mut(lanes) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z = row.iter().map(|v| (*v - m).exp()).sum::<f32>().ln() + m;
            row.iter_mut().for_each(|v| *v -= z);
        }
        for col in 0..lanes {
            let m = (0..lanes)
                .map(|row| out[row * lanes + col])
                .fold(f32::NEG_INFINITY, f32::max);
            let z = (0..lanes)
                .map(|row| (out[row * lanes + col] - m).exp())
                .sum::<f32>()
                .ln()
                + m;
            for row in 0..lanes {
                out[row * lanes + col] -= z;
            }
        }
    }
    out.iter_mut().for_each(|v| *v = v.exp());
}

pub fn stack_mhc_forward(
    model: &needle2_format::CactModel<'_>,
    tokens: &[u32],
    x: &[f32],
    seq_len: usize,
    out: &mut [f32],
) -> Result<(), needle2_format::CactError> {
    assert_eq!(tokens.len(), seq_len);
    assert_eq!(x.len(), seq_len * D_MODEL);
    assert_eq!(out.len(), x.len());
    let mhc = MhcWeights::from_cact(model)?;
    let mut engram_keys = vec![vec![0.0; seq_len * D_MODEL]; ENGRAM_SITES];
    let mut engram_values = vec![vec![0.0; seq_len * D_MODEL]; ENGRAM_SITES];
    for site in 0..ENGRAM_SITES {
        let weights = EngramWeights::from_cact(model, site)?;
        engram_forward(
            &tokens,
            &weights,
            &mut engram_keys[site],
            &mut engram_values[site],
        );
    }
    let mut state = vec![0.0; seq_len * MHC_LANES * D_MODEL];
    for t in 0..seq_len {
        for lane in 0..MHC_LANES {
            state[(t * MHC_LANES + lane) * D_MODEL..(t * MHC_LANES + lane + 1) * D_MODEL]
                .copy_from_slice(&x[t * D_MODEL..(t + 1) * D_MODEL]);
        }
    }
    let mut flat_norm = vec![0.0; MHC_LANES * D_MODEL];
    let mut hpre = vec![0.0; MHC_LANES];
    let mut hpost = vec![0.0; MHC_LANES];
    let mut res_logits = vec![0.0; MHC_LANES * MHC_LANES];
    let mut res_mix = vec![0.0; res_logits.len()];
    for layer in 0..NUM_LAYERS {
        let weights = LayerWeights::from_cact(model, layer)?;
        let mut next = vec![0.0; state.len()];
        for t in 0..seq_len {
            let start = t * MHC_LANES * D_MODEL;
            let lanes = &state[start..start + MHC_LANES * D_MODEL];
            rms_unit(lanes, &mut flat_norm);
            for lane in 0..MHC_LANES {
                let row = &mhc.phi_pre[(layer * MHC_LANES + lane) * MHC_LANES * D_MODEL
                    ..(layer * MHC_LANES + lane + 1) * MHC_LANES * D_MODEL];
                let score = mhc.a_pre[layer]
                    * row.iter().zip(&flat_norm).map(|(a, b)| a * b).sum::<f32>()
                    + mhc.b_pre[layer * MHC_LANES + lane]
                    + if lane == layer % MHC_LANES { 4.0 } else { -4.0 };
                hpre[lane] = 1.0 / (1.0 + (-score).exp());
            }
            let mut selected = vec![0.0; D_MODEL];
            for lane in 0..MHC_LANES {
                for d in 0..D_MODEL {
                    selected[d] += hpre[lane] * lanes[lane * D_MODEL + d];
                }
            }
            if layer == 2 || layer == 15 {
                let site = if layer == 2 { 0 } else { 1 };
                let mut unit_x = vec![0.0; D_MODEL];
                let mut unit_k = vec![0.0; D_MODEL];
                rms_unit(&selected, &mut unit_x);
                rms_unit(
                    &engram_keys[site][t * D_MODEL..(t + 1) * D_MODEL],
                    &mut unit_k,
                );
                let alpha = 1.0
                    / (1.0
                        + (-unit_x.iter().zip(&unit_k).map(|(a, b)| a * b).sum::<f32>()
                            / (D_MODEL as f32).sqrt())
                        .exp());
                for d in 0..D_MODEL {
                    selected[d] += alpha * engram_values[site][t * D_MODEL + d];
                }
            }
            let mut block = vec![0.0; D_MODEL];
            block_forward(&selected, 1, &weights, &mut block);
            for d in 0..D_MODEL {
                block[d] -= selected[d];
            }
            for lane in 0..MHC_LANES {
                let row = &mhc.phi_post[(layer * MHC_LANES + lane) * MHC_LANES * D_MODEL
                    ..(layer * MHC_LANES + lane + 1) * MHC_LANES * D_MODEL];
                let score = mhc.a_post[layer]
                    * row.iter().zip(&flat_norm).map(|(a, b)| a * b).sum::<f32>()
                    + mhc.b_post[layer * MHC_LANES + lane]
                    + if lane == layer % MHC_LANES { 0.0 } else { -4.0 };
                hpost[lane] = 2.0 / (1.0 + (-score).exp());
            }
            let phi = &mhc.phi_res[(layer * MHC_LANES * MHC_LANES) * MHC_LANES * D_MODEL
                ..(layer * MHC_LANES * MHC_LANES + MHC_LANES * MHC_LANES) * MHC_LANES * D_MODEL];
            for r in 0..MHC_LANES * MHC_LANES {
                res_logits[r] = mhc.a_res[layer]
                    * phi[r * MHC_LANES * D_MODEL..(r + 1) * MHC_LANES * D_MODEL]
                        .iter()
                        .zip(&flat_norm)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                    + mhc.b_res[(layer * MHC_LANES * MHC_LANES) + r];
            }
            sinkhorn(&res_logits, MHC_LANES, &mut res_mix);
            for lane in 0..MHC_LANES {
                for d in 0..D_MODEL {
                    let mut value = 0.0;
                    for src in 0..MHC_LANES {
                        value += res_mix[lane * MHC_LANES + src] * lanes[src * D_MODEL + d];
                    }
                    next[start + lane * D_MODEL + d] = value + hpost[lane] * block[d];
                }
            }
        }
        state = next;
    }
    for t in 0..seq_len {
        let mut mean = vec![0.0; D_MODEL];
        for lane in 0..MHC_LANES {
            for d in 0..D_MODEL {
                mean[d] += state[(t * MHC_LANES + lane) * D_MODEL + d] / MHC_LANES as f32;
            }
        }
        out[t * D_MODEL..(t + 1) * D_MODEL].copy_from_slice(&mean);
    }
    Ok(())
}

pub fn render_tool_prompt(tools_json: &str, query: &str) -> String {
    format!("<|im_start|>user\\n<tools>{tools_json}</tools>\\n{query}<|im_end|>\\n<|im_start|>assistant\\n")
}

pub fn decode_tool_calls(text: &str) -> Result<Vec<serde_json::Value>, String> {
    let body = text
        .split_once("<tool_call>")
        .map(|(_, rest)| rest)
        .unwrap_or(text);
    let body = body
        .split_once("</tool_call>")
        .map(|(body, _)| body)
        .unwrap_or(body)
        .trim();
    let start = body
        .find(['{', '['])
        .ok_or_else(|| "missing JSON tool payload".to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&body[start..]).map_err(|error| error.to_string())?;
    match value {
        serde_json::Value::Array(calls) => Ok(calls),
        serde_json::Value::Object(_) => Ok(vec![value]),
        _ => Err("tool call payload must be an object or array".into()),
    }
}

pub fn generate_tool_calls(
    model: &needle2_format::CactModel<'_>,
    tools_json: &str,
    query: &str,
    max_new_tokens: usize,
) -> Result<Vec<serde_json::Value>, String> {
    let text =
        generate_greedy(model, tools_json, query, max_new_tokens).map_err(|e| e.to_string())?;
    decode_tool_calls(&text)
}

pub fn generate_greedy(
    model: &needle2_format::CactModel<'_>,
    tools_json: &str,
    query: &str,
    max_new_tokens: usize,
) -> Result<String, needle2_format::CactError> {
    let prompt = render_tool_prompt(tools_json, query);
    let mut tokens = vec![2u32];
    tokens.extend(model.tokenizer.encode(&prompt));
    let mut generated = Vec::new();
    let mut logits = vec![0.0; VOCAB_SIZE * tokens.len()];
    for _ in 0..max_new_tokens {
        logits.resize(tokens.len() * VOCAB_SIZE, 0.0);
        infer_logits(model, &tokens, &mut logits)?;
        let start = (tokens.len() - 1) * VOCAB_SIZE;
        let next = logits[start..start + VOCAB_SIZE]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        if next == 1 {
            break;
        }
        tokens.push(next);
        generated.push(next);
    }
    Ok(model.tokenizer.decode(&generated))
}

pub fn infer_logits(
    model: &needle2_format::CactModel<'_>,
    tokens: &[u32],
    out: &mut [f32],
) -> Result<(), needle2_format::CactError> {
    let embedding = model.tensor_f32(0)?;
    assert_eq!(embedding.len(), VOCAB_SIZE * D_MODEL);
    assert_eq!(out.len(), tokens.len() * VOCAB_SIZE);
    let mut input = vec![0.0; tokens.len() * D_MODEL];
    let scale = (D_MODEL as f32).sqrt();
    for (t, &token) in tokens.iter().enumerate() {
        let row = &embedding[token as usize * D_MODEL..(token as usize + 1) * D_MODEL];
        for d in 0..D_MODEL {
            input[t * D_MODEL + d] = row[d] * scale;
        }
    }
    let mut hidden = vec![0.0; input.len()];
    stack_mhc_forward(model, tokens, &input, tokens.len(), &mut hidden)?;
    for t in 0..tokens.len() {
        for vocab in 0..VOCAB_SIZE {
            out[t * VOCAB_SIZE + vocab] = hidden[t * D_MODEL..(t + 1) * D_MODEL]
                .iter()
                .zip(&embedding[vocab * D_MODEL..(vocab + 1) * D_MODEL])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    Ok(())
}

pub fn stack_forward(
    model: &needle2_format::CactModel<'_>,
    x: &[f32],
    seq_len: usize,
    out: &mut [f32],
) -> Result<(), needle2_format::CactError> {
    assert_eq!(x.len(), seq_len * D_MODEL);
    assert_eq!(out.len(), x.len());
    let mut state = x.to_vec();
    let mut next = vec![0.0; state.len()];
    for layer in 0..NUM_LAYERS {
        let weights = LayerWeights::from_cact(model, layer)?;
        block_forward(&state, seq_len, &weights, &mut next);
        std::mem::swap(&mut state, &mut next);
    }
    // Six mHC scalar/vector tensors plus three packed projection tensors,
    // followed by two engram sites (four tensors each).
    let final_norm = model.tensor_f32(LAYER_TENSOR_START + NUM_LAYERS * 14 + 9 + 8)?;
    for t in 0..seq_len {
        zc_rms_norm(
            &state[t * D_MODEL..(t + 1) * D_MODEL],
            &final_norm,
            1e-6,
            &mut out[t * D_MODEL..(t + 1) * D_MODEL],
        );
    }
    Ok(())
}

pub fn attention_block(x: &[f32], seq_len: usize, weights: &LayerWeights, out: &mut [f32]) {
    assert_eq!(x.len(), seq_len * D_MODEL);
    assert_eq!(out.len(), x.len());
    let mut normalized = vec![0.0; x.len()];
    for t in 0..seq_len {
        zc_rms_norm(
            &x[t * D_MODEL..(t + 1) * D_MODEL],
            &weights.norm_in,
            1e-6,
            &mut normalized[t * D_MODEL..(t + 1) * D_MODEL],
        );
    }
    let mut q = vec![0.0; seq_len * D_MODEL];
    let mut k = vec![0.0; seq_len * NUM_KV_HEADS * HEAD_DIM];
    let mut v = vec![0.0; k.len()];
    let mut row = vec![0.0; D_MODEL];
    for t in 0..seq_len {
        matvec(
            &weights.q_proj,
            D_MODEL,
            D_MODEL,
            &normalized[t * D_MODEL..(t + 1) * D_MODEL],
            &mut q[t * D_MODEL..(t + 1) * D_MODEL],
        );
        matvec(
            &weights.k_proj,
            NUM_KV_HEADS * HEAD_DIM,
            D_MODEL,
            &normalized[t * D_MODEL..(t + 1) * D_MODEL],
            &mut k[t * NUM_KV_HEADS * HEAD_DIM..(t + 1) * NUM_KV_HEADS * HEAD_DIM],
        );
        matvec(
            &weights.v_proj,
            NUM_KV_HEADS * HEAD_DIM,
            D_MODEL,
            &normalized[t * D_MODEL..(t + 1) * D_MODEL],
            &mut v[t * NUM_KV_HEADS * HEAD_DIM..(t + 1) * NUM_KV_HEADS * HEAD_DIM],
        );
    }
    for h in 0..NUM_HEADS {
        for t in 0..seq_len {
            let p = (t * NUM_HEADS + h) * HEAD_DIM;
            zc_rms_norm(
                &q[p..p + HEAD_DIM],
                &weights.q_norm,
                1e-6,
                &mut row[..HEAD_DIM],
            );
            q[p..p + HEAD_DIM].copy_from_slice(&row[..HEAD_DIM]);
        }
    }
    for h in 0..NUM_KV_HEADS {
        for t in 0..seq_len {
            let p = (t * NUM_KV_HEADS + h) * HEAD_DIM;
            zc_rms_norm(
                &k[p..p + HEAD_DIM],
                &weights.k_norm,
                1e-6,
                &mut row[..HEAD_DIM],
            );
            k[p..p + HEAD_DIM].copy_from_slice(&row[..HEAD_DIM]);
        }
    }
    let (cos, sin) = rope_frequencies(HEAD_DIM, seq_len, ROPE_THETA);
    for h in 0..NUM_HEADS {
        for t in 0..seq_len {
            apply_rope(
                &mut q[(t * NUM_HEADS + h) * HEAD_DIM..(t * NUM_HEADS + h + 1) * HEAD_DIM],
                &cos[t * HEAD_DIM / 2..(t + 1) * HEAD_DIM / 2],
                &sin[t * HEAD_DIM / 2..(t + 1) * HEAD_DIM / 2],
            );
        }
    }
    for h in 0..NUM_KV_HEADS {
        for t in 0..seq_len {
            apply_rope(
                &mut k[(t * NUM_KV_HEADS + h) * HEAD_DIM..(t * NUM_KV_HEADS + h + 1) * HEAD_DIM],
                &cos[t * HEAD_DIM / 2..(t + 1) * HEAD_DIM / 2],
                &sin[t * HEAD_DIM / 2..(t + 1) * HEAD_DIM / 2],
            );
        }
    }
    // Attention helpers use head-major layout; transpose the token-major projections.
    let mut qh = vec![0.0; q.len()];
    let mut kh = vec![0.0; k.len()];
    let mut vh = vec![0.0; v.len()];
    for t in 0..seq_len {
        for h in 0..NUM_HEADS {
            qh[(h * seq_len + t) * HEAD_DIM..(h * seq_len + t + 1) * HEAD_DIM].copy_from_slice(
                &q[(t * NUM_HEADS + h) * HEAD_DIM..(t * NUM_HEADS + h + 1) * HEAD_DIM],
            );
        }
        for h in 0..NUM_KV_HEADS {
            kh[(h * seq_len + t) * HEAD_DIM..(h * seq_len + t + 1) * HEAD_DIM].copy_from_slice(
                &k[(t * NUM_KV_HEADS + h) * HEAD_DIM..(t * NUM_KV_HEADS + h + 1) * HEAD_DIM],
            );
            vh[(h * seq_len + t) * HEAD_DIM..(h * seq_len + t + 1) * HEAD_DIM].copy_from_slice(
                &v[(t * NUM_KV_HEADS + h) * HEAD_DIM..(t * NUM_KV_HEADS + h + 1) * HEAD_DIM],
            );
        }
    }
    let mut attended = vec![0.0; q.len()];
    gqa_attention(
        &qh,
        &kh,
        &vh,
        NUM_HEADS,
        NUM_KV_HEADS,
        seq_len,
        HEAD_DIM,
        true,
        &mut attended,
    );
    for t in 0..seq_len {
        for h in 0..NUM_HEADS {
            row[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(
                &attended[(h * seq_len + t) * HEAD_DIM..(h * seq_len + t + 1) * HEAD_DIM],
            );
        }
        matvec(
            &weights.gate_proj,
            D_MODEL,
            D_MODEL,
            &normalized[t * D_MODEL..(t + 1) * D_MODEL],
            &mut q[t * D_MODEL..(t + 1) * D_MODEL],
        );
        for i in 0..D_MODEL {
            row[i] *= 1.0 / (1.0 + (-q[t * D_MODEL + i]).exp());
        }
        matvec(
            &weights.out_proj,
            D_MODEL,
            D_MODEL,
            &row,
            &mut out[t * D_MODEL..(t + 1) * D_MODEL],
        );
    }
}

pub fn block_forward(x: &[f32], seq_len: usize, weights: &LayerWeights, out: &mut [f32]) {
    assert_eq!(x.len(), seq_len * D_MODEL);
    assert_eq!(out.len(), x.len());
    let mut attention = vec![0.0; x.len()];
    attention_block(x, seq_len, weights, &mut attention);
    let mut state = vec![0.0; x.len()];
    for t in 0..seq_len {
        let range = t * D_MODEL..(t + 1) * D_MODEL;
        let mut post = vec![0.0; D_MODEL];
        zc_rms_norm(
            &attention[range.clone()],
            &weights.post_norm,
            1e-6,
            &mut post,
        );
        let gate = 1.0 / (1.0 + (-weights.attn_gate).exp());
        for i in 0..D_MODEL {
            state[range.start + i] = x[range.start + i] + gate * post[i];
        }
        let mut pre = vec![0.0; D_MODEL];
        zc_rms_norm(&state[range.clone()], &weights.pre_hada, 1e-6, &mut pre);
        let mut mlp = vec![0.0; D_MODEL];
        hadamard_mlp(&pre, &weights.d1, &weights.d2, &weights.d3, &mut mlp);
        for i in 0..D_MODEL {
            out[range.start + i] = state[range.start + i] + mlp[i];
        }
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
    fn decodes_single_and_multiple_tool_calls() {
        assert_eq!(
            decode_tool_calls("<tool_call>{\"name\":\"weather\"}</tool_call>")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            decode_tool_calls("reasoning {\"name\":\"weather\"}")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            decode_tool_calls("[{\"name\":\"a\"},{\"name\":\"b\"}]")
                .unwrap()
                .len(),
            2
        );
        assert!(decode_tool_calls("null").is_err());
    }

    #[test]
    fn renders_official_tool_prompt() {
        assert_eq!(
            render_tool_prompt("[{\"name\":\"weather\"}]", "hello"),
            "<|im_start|>user\\n<tools>[{\"name\":\"weather\"}]</tools>\\nhello<|im_end|>\\n<|im_start|>assistant\\n"
        );
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
