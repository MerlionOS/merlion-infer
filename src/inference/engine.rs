/// LLM inference engine implementing the Llama architecture.
/// Based on llama2.c by Andrej Karpathy, adapted for no_std Rust.
///
/// Optimized: pre-computed tensor indices, zero allocation in forward pass.

use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;
use crate::inference::kernels::scalar;

/// Model configuration parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
}

impl ModelConfig {
    pub fn from_gguf(model: &crate::inference::gguf::GgufModel) -> Result<Self, &'static str> {
        let get_u32 = |key: &str| -> Result<u32, &'static str> {
            model.get_metadata(key)
                .and_then(|v| v.as_u32())
                .ok_or("missing metadata key")
        };

        let arch = model.get_metadata("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        let prefix = alloc::format!("{}", arch);

        let dim = get_u32(&alloc::format!("{}.embedding_length", prefix))? as usize;
        let hidden_dim = get_u32(&alloc::format!("{}.feed_forward_length", prefix))? as usize;
        let n_layers = get_u32(&alloc::format!("{}.block_count", prefix))? as usize;
        let n_heads = get_u32(&alloc::format!("{}.attention.head_count", prefix))? as usize;
        let n_kv_heads = get_u32(&alloc::format!("{}.attention.head_count_kv", prefix))
            .unwrap_or(n_heads as u32) as usize;
        let max_seq_len = get_u32(&alloc::format!("{}.context_length", prefix))
            .unwrap_or(2048) as usize;

        let vocab_size = model.get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                crate::inference::gguf::GgufValue::Array(arr) => Some(arr.len()),
                _ => None,
            })
            .unwrap_or(32000);

        let head_dim = dim / n_heads;
        let kv_dim = head_dim * n_kv_heads;

        Ok(Self {
            dim, hidden_dim, n_layers, n_heads, n_kv_heads,
            vocab_size, max_seq_len, head_dim, kv_dim,
        })
    }
}

/// Run state: pre-allocated buffers for a single forward pass.
pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            x: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            q: vec![0.0; cfg.dim],
            k: vec![0.0; cfg.kv_dim],
            v: vec![0.0; cfg.kv_dim],
            att: vec![0.0; cfg.n_heads * cfg.max_seq_len],
            logits: vec![0.0; cfg.vocab_size],
            key_cache: vec![0.0; cfg.n_layers * cfg.max_seq_len * cfg.kv_dim],
            value_cache: vec![0.0; cfg.n_layers * cfg.max_seq_len * cfg.kv_dim],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.x.len() + self.xb.len() + self.xb2.len()
         + self.hb.len() + self.hb2.len()
         + self.q.len() + self.k.len() + self.v.len()
         + self.att.len() + self.logits.len()
         + self.key_cache.len() + self.value_cache.len()) * 4
    }
}

/// Pre-computed tensor index for a single layer (no allocations in forward pass).
struct LayerIndex {
    attn_norm: usize,  // index into tensor_map
    attn_q: usize,
    attn_k: usize,
    attn_v: usize,
    attn_output: usize,
    ffn_norm: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_down: usize,
}

/// Pre-computed tensor indices for the entire model.
struct TensorIndex {
    token_embd: usize,
    layers: Vec<LayerIndex>,
    output_norm: usize,
    output: usize,
}

pub struct ModelWeights {
    pub data: Vec<u8>,
    pub data_offset: usize,
    pub tensor_map: Vec<(String, usize, usize)>,
    pub is_quantized: bool,
}

impl ModelWeights {
    fn find_index(&self, name: &str) -> usize {
        for (i, (tname, _, _)) in self.tensor_map.iter().enumerate() {
            if tname == name { return i; }
        }
        usize::MAX // sentinel: tensor not found
    }

    fn get_bytes_by_index(&self, idx: usize) -> Option<&[u8]> {
        if idx == usize::MAX { return None; }
        let (_, offset, size) = &self.tensor_map[idx];
        let start = self.data_offset + *offset;
        let end = start + *size;
        if end <= self.data.len() { Some(&self.data[start..end]) } else { None }
    }

    fn get_f32_by_index(&self, idx: usize) -> Option<&[f32]> {
        let bytes = self.get_bytes_by_index(idx)?;
        let ptr = bytes.as_ptr() as *const f32;
        let len = bytes.len() / 4;
        Some(unsafe { core::slice::from_raw_parts(ptr, len) })
    }

    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let idx = self.find_index(name);
        self.get_bytes_by_index(idx)
    }

    pub fn get_tensor_f32(&self, name: &str) -> Option<&[f32]> {
        let idx = self.find_index(name);
        self.get_f32_by_index(idx)
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// The Llama inference engine.
pub struct LlamaEngine {
    pub config: ModelConfig,
    pub state: RunState,
    pub weights: ModelWeights,
    index: TensorIndex,
}

impl LlamaEngine {
    /// Create engine with pre-computed tensor indices.
    pub fn new(config: ModelConfig, state: RunState, weights: ModelWeights) -> Self {
        let mut layers = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            layers.push(LayerIndex {
                attn_norm: weights.find_index(&alloc::format!("blk.{}.attn_norm.weight", l)),
                attn_q: weights.find_index(&alloc::format!("blk.{}.attn_q.weight", l)),
                attn_k: weights.find_index(&alloc::format!("blk.{}.attn_k.weight", l)),
                attn_v: weights.find_index(&alloc::format!("blk.{}.attn_v.weight", l)),
                attn_output: weights.find_index(&alloc::format!("blk.{}.attn_output.weight", l)),
                ffn_norm: weights.find_index(&alloc::format!("blk.{}.ffn_norm.weight", l)),
                ffn_gate: weights.find_index(&alloc::format!("blk.{}.ffn_gate.weight", l)),
                ffn_up: weights.find_index(&alloc::format!("blk.{}.ffn_up.weight", l)),
                ffn_down: weights.find_index(&alloc::format!("blk.{}.ffn_down.weight", l)),
            });
        }

        let index = TensorIndex {
            token_embd: weights.find_index("token_embd.weight"),
            layers,
            output_norm: weights.find_index("output_norm.weight"),
            output: weights.find_index("output.weight"),
        };

        Self { config, state, weights, index }
    }

    /// Perform a single forward pass for token at position `pos`.
    /// Zero allocations — all tensor lookups use pre-computed indices.
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let dim = self.config.dim;
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let kv_dim = self.config.kv_dim;
        let n_heads = self.config.n_heads;
        let kv_mul = n_heads / self.config.n_kv_heads;

        // 1. Token embedding
        if let Some(embed) = self.weights.get_f32_by_index(self.index.token_embd) {
            let start = token as usize * dim;
            if start + dim <= embed.len() {
                self.state.x.copy_from_slice(&embed[start..start + dim]);
            }
        }

        // 2. Transformer layers
        for layer in 0..self.config.n_layers {
            let li = &self.index.layers[layer];

            // Attention rmsnorm
            if let Some(w) = self.weights.get_f32_by_index(li.attn_norm) {
                scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
            }

            // QKV projections
            if self.weights.is_quantized {
                if let Some(wq) = self.weights.get_bytes_by_index(li.attn_q) {
                    scalar::matmul_q4_0(&mut self.state.q, wq, &self.state.xb, dim, dim);
                }
                if let Some(wk) = self.weights.get_bytes_by_index(li.attn_k) {
                    scalar::matmul_q4_0(&mut self.state.k, wk, &self.state.xb, dim, kv_dim);
                }
                if let Some(wv) = self.weights.get_bytes_by_index(li.attn_v) {
                    scalar::matmul_q4_0(&mut self.state.v, wv, &self.state.xb, dim, kv_dim);
                }
            } else {
                if let Some(wq) = self.weights.get_f32_by_index(li.attn_q) {
                    scalar::matmul(&mut self.state.q, wq, &self.state.xb, dim, dim);
                }
                if let Some(wk) = self.weights.get_f32_by_index(li.attn_k) {
                    scalar::matmul(&mut self.state.k, wk, &self.state.xb, dim, kv_dim);
                }
                if let Some(wv) = self.weights.get_f32_by_index(li.attn_v) {
                    scalar::matmul(&mut self.state.v, wv, &self.state.xb, dim, kv_dim);
                }
            }

            // RoPE
            scalar::rope(&mut self.state.q, &mut self.state.k, pos, dim, head_dim, kv_dim);

            // Cache KV
            let kv_offset = layer * self.config.max_seq_len * kv_dim + pos * kv_dim;
            if kv_offset + kv_dim <= self.state.key_cache.len() {
                self.state.key_cache[kv_offset..kv_offset + kv_dim]
                    .copy_from_slice(&self.state.k[..kv_dim]);
                self.state.value_cache[kv_offset..kv_offset + kv_dim]
                    .copy_from_slice(&self.state.v[..kv_dim]);
            }

            // Multi-head attention
            self.state.xb.fill(0.0);
            for h in 0..n_heads {
                let kv_h = h / kv_mul;
                let q_off = h * head_dim;
                let att_off = h * self.config.max_seq_len;

                // Attention scores
                for t in 0..=pos {
                    let kc_off = layer * self.config.max_seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                    let mut score = 0.0f32;
                    let q_slice = &self.state.q[q_off..q_off + head_dim];
                    let k_slice = &self.state.key_cache[kc_off..kc_off + head_dim];
                    // Manual dot product — avoid iterator overhead
                    let mut i = 0;
                    while i + 3 < head_dim {
                        score += q_slice[i] * k_slice[i]
                               + q_slice[i+1] * k_slice[i+1]
                               + q_slice[i+2] * k_slice[i+2]
                               + q_slice[i+3] * k_slice[i+3];
                        i += 4;
                    }
                    while i < head_dim {
                        score += q_slice[i] * k_slice[i];
                        i += 1;
                    }
                    self.state.att[att_off + t] = score / libm::sqrtf(head_dim as f32);
                }

                // Softmax
                scalar::softmax(&mut self.state.att[att_off..att_off + pos + 1]);

                // Weighted sum of values
                let xb_off = h * head_dim;
                for t in 0..=pos {
                    let vc_off = layer * self.config.max_seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                    let w = self.state.att[att_off + t];
                    for d in 0..head_dim {
                        self.state.xb[xb_off + d] += w * self.state.value_cache[vc_off + d];
                    }
                }
            }

            // Output projection
            if self.weights.is_quantized {
                if let Some(wo) = self.weights.get_bytes_by_index(li.attn_output) {
                    scalar::matmul_q4_0(&mut self.state.xb2, wo, &self.state.xb, dim, dim);
                }
            } else {
                if let Some(wo) = self.weights.get_f32_by_index(li.attn_output) {
                    scalar::matmul(&mut self.state.xb2, wo, &self.state.xb, dim, dim);
                }
            }

            // Residual
            scalar::elementwise_add(&mut self.state.x, &self.state.xb2);

            // FFN rmsnorm
            if let Some(w) = self.weights.get_f32_by_index(li.ffn_norm) {
                scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
            }

            // FFN
            if self.weights.is_quantized {
                if let Some(wg) = self.weights.get_bytes_by_index(li.ffn_gate) {
                    scalar::matmul_q4_0(&mut self.state.hb, wg, &self.state.xb, dim, hidden_dim);
                }
                if let Some(wu) = self.weights.get_bytes_by_index(li.ffn_up) {
                    scalar::matmul_q4_0(&mut self.state.hb2, wu, &self.state.xb, dim, hidden_dim);
                }
            } else {
                if let Some(wg) = self.weights.get_f32_by_index(li.ffn_gate) {
                    scalar::matmul(&mut self.state.hb, wg, &self.state.xb, dim, hidden_dim);
                }
                if let Some(wu) = self.weights.get_f32_by_index(li.ffn_up) {
                    scalar::matmul(&mut self.state.hb2, wu, &self.state.xb, dim, hidden_dim);
                }
            }

            scalar::silu(&mut self.state.hb);
            scalar::elementwise_mul(&mut self.state.hb, &self.state.hb2);

            if self.weights.is_quantized {
                if let Some(wd) = self.weights.get_bytes_by_index(li.ffn_down) {
                    scalar::matmul_q4_0(&mut self.state.xb2, wd, &self.state.hb, hidden_dim, dim);
                }
            } else {
                if let Some(wd) = self.weights.get_f32_by_index(li.ffn_down) {
                    scalar::matmul(&mut self.state.xb2, wd, &self.state.hb, hidden_dim, dim);
                }
            }

            // Residual
            scalar::elementwise_add(&mut self.state.x, &self.state.xb2);
        }

        // Final rmsnorm
        if let Some(w) = self.weights.get_f32_by_index(self.index.output_norm) {
            scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
        } else {
            self.state.xb.copy_from_slice(&self.state.x);
        }

        // Classifier
        if self.weights.is_quantized {
            if let Some(wo) = self.weights.get_bytes_by_index(self.index.output) {
                scalar::matmul_q4_0(&mut self.state.logits, wo, &self.state.xb,
                    self.config.dim, self.config.vocab_size);
            }
        } else {
            if let Some(wo) = self.weights.get_f32_by_index(self.index.output) {
                scalar::matmul(&mut self.state.logits, wo, &self.state.xb,
                    self.config.dim, self.config.vocab_size);
            }
        }

        &self.state.logits
    }
}
