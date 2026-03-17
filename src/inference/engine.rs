/// LLM inference engine implementing the Llama architecture.
/// Based on llama2.c by Andrej Karpathy, adapted for no_std Rust.
///
/// Supports Llama/Llama2/Llama3/SmolLM models in Q4_0 and F32 formats.

use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;
use crate::inference::kernels::scalar;

/// Model configuration parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,         // transformer dimension (embedding size)
    pub hidden_dim: usize,  // FFN hidden dimension
    pub n_layers: usize,    // number of transformer layers
    pub n_heads: usize,     // number of attention heads
    pub n_kv_heads: usize,  // number of KV heads (for GQA)
    pub vocab_size: usize,  // vocabulary size
    pub max_seq_len: usize, // maximum sequence length
    pub head_dim: usize,    // dim / n_heads
    pub kv_dim: usize,      // head_dim * n_kv_heads
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

        // Vocab size from tokenizer metadata
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
    // Current activations
    pub x: Vec<f32>,       // (dim,) activation at current position
    pub xb: Vec<f32>,      // (dim,) after rmsnorm
    pub xb2: Vec<f32>,     // (dim,) second buffer
    pub hb: Vec<f32>,      // (hidden_dim,) FFN hidden
    pub hb2: Vec<f32>,     // (hidden_dim,) FFN second hidden
    pub q: Vec<f32>,       // (dim,) query
    pub k: Vec<f32>,       // (kv_dim,) key
    pub v: Vec<f32>,       // (kv_dim,) value
    pub att: Vec<f32>,     // (n_heads, max_seq_len) attention scores
    pub logits: Vec<f32>,  // (vocab_size,) output logits

    // KV cache
    pub key_cache: Vec<f32>,   // (n_layers, max_seq_len, kv_dim)
    pub value_cache: Vec<f32>, // (n_layers, max_seq_len, kv_dim)
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

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.x.len() + self.xb.len() + self.xb2.len()
         + self.hb.len() + self.hb2.len()
         + self.q.len() + self.k.len() + self.v.len()
         + self.att.len() + self.logits.len()
         + self.key_cache.len() + self.value_cache.len()) * 4
    }
}

/// Model weights — pointers into the loaded GGUF data.
/// Weights are NOT copied; they reference the memory-mapped model data.
pub struct ModelWeights {
    /// Raw model data (owned, loaded from disk)
    pub data: Vec<u8>,
    /// Data offset within the GGUF file where tensor data begins
    pub data_offset: usize,
    /// Tensor name → (offset_in_data, byte_size) mapping
    pub tensor_map: Vec<(String, usize, usize)>,
    /// Whether weights are quantized (Q4_0) or float32
    pub is_quantized: bool,
}

impl ModelWeights {
    /// Get raw bytes for a tensor by name.
    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        for (tname, offset, size) in &self.tensor_map {
            if tname == name {
                let start = self.data_offset + *offset;
                let end = start + *size;
                if end <= self.data.len() {
                    return Some(&self.data[start..end]);
                }
            }
        }
        None
    }

    /// Get tensor as f32 slice (only for non-quantized weights).
    pub fn get_tensor_f32(&self, name: &str) -> Option<&[f32]> {
        let bytes = self.get_tensor_bytes(name)?;
        let ptr = bytes.as_ptr() as *const f32;
        let len = bytes.len() / 4;
        Some(unsafe { core::slice::from_raw_parts(ptr, len) })
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
}

impl LlamaEngine {
    /// Perform a single forward pass for token at position `pos`.
    /// Returns logits over the vocabulary.
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let cfg = &self.config;
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let head_dim = cfg.head_dim;
        let kv_dim = cfg.kv_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let kv_mul = n_heads / n_kv_heads; // GQA multiplier

        // 1. Token embedding: copy row `token` from embedding table
        let embed_name = "token_embd.weight";
        if let Some(embed) = self.weights.get_tensor_f32(embed_name) {
            let start = token as usize * dim;
            if start + dim <= embed.len() {
                self.state.x.copy_from_slice(&embed[start..start + dim]);
            }
        }

        // 2. Transformer layers
        for layer in 0..cfg.n_layers {
            let l = layer; // layer index for weight name construction

            // Attention rmsnorm
            let attn_norm_name = alloc::format!("blk.{}.attn_norm.weight", l);
            if let Some(w) = self.weights.get_tensor_f32(&attn_norm_name) {
                scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
            }

            // QKV projections
            let wq_name = alloc::format!("blk.{}.attn_q.weight", l);
            let wk_name = alloc::format!("blk.{}.attn_k.weight", l);
            let wv_name = alloc::format!("blk.{}.attn_v.weight", l);

            if self.weights.is_quantized {
                if let Some(wq) = self.weights.get_tensor_bytes(&wq_name) {
                    scalar::matmul_q4_0(&mut self.state.q, wq, &self.state.xb, dim, dim);
                }
                if let Some(wk) = self.weights.get_tensor_bytes(&wk_name) {
                    scalar::matmul_q4_0(&mut self.state.k, wk, &self.state.xb, dim, kv_dim);
                }
                if let Some(wv) = self.weights.get_tensor_bytes(&wv_name) {
                    scalar::matmul_q4_0(&mut self.state.v, wv, &self.state.xb, dim, kv_dim);
                }
            } else {
                if let Some(wq) = self.weights.get_tensor_f32(&wq_name) {
                    scalar::matmul(&mut self.state.q, wq, &self.state.xb, dim, dim);
                }
                if let Some(wk) = self.weights.get_tensor_f32(&wk_name) {
                    scalar::matmul(&mut self.state.k, wk, &self.state.xb, dim, kv_dim);
                }
                if let Some(wv) = self.weights.get_tensor_f32(&wv_name) {
                    scalar::matmul(&mut self.state.v, wv, &self.state.xb, dim, kv_dim);
                }
            }

            // RoPE
            scalar::rope(&mut self.state.q, &mut self.state.k, pos, dim, head_dim, kv_dim);

            // Cache KV
            let kv_cache_offset = l * cfg.max_seq_len * kv_dim + pos * kv_dim;
            if kv_cache_offset + kv_dim <= self.state.key_cache.len() {
                self.state.key_cache[kv_cache_offset..kv_cache_offset + kv_dim]
                    .copy_from_slice(&self.state.k[..kv_dim]);
                self.state.value_cache[kv_cache_offset..kv_cache_offset + kv_dim]
                    .copy_from_slice(&self.state.v[..kv_dim]);
            }

            // Multi-head attention
            self.state.xb.fill(0.0);
            for h in 0..n_heads {
                let kv_h = h / kv_mul; // GQA: which KV head this Q head maps to

                let q_offset = h * head_dim;
                let q_head = &self.state.q[q_offset..q_offset + head_dim];

                // Compute attention scores for all positions up to `pos`
                let att_offset = h * cfg.max_seq_len;
                for t in 0..=pos {
                    let kc_offset = l * cfg.max_seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_head[d] * self.state.key_cache[kc_offset + d];
                    }
                    self.state.att[att_offset + t] = score / libm::sqrtf(head_dim as f32);
                }

                // Softmax over attention scores
                scalar::softmax(&mut self.state.att[att_offset..att_offset + pos + 1]);

                // Weighted sum of values
                let xb_offset = h * head_dim;
                for t in 0..=pos {
                    let vc_offset = l * cfg.max_seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                    let weight = self.state.att[att_offset + t];
                    for d in 0..head_dim {
                        self.state.xb[xb_offset + d] += weight * self.state.value_cache[vc_offset + d];
                    }
                }
            }

            // Output projection
            let wo_name = alloc::format!("blk.{}.attn_output.weight", l);
            if self.weights.is_quantized {
                if let Some(wo) = self.weights.get_tensor_bytes(&wo_name) {
                    scalar::matmul_q4_0(&mut self.state.xb2, wo, &self.state.xb, dim, dim);
                }
            } else {
                if let Some(wo) = self.weights.get_tensor_f32(&wo_name) {
                    scalar::matmul(&mut self.state.xb2, wo, &self.state.xb, dim, dim);
                }
            }

            // Residual connection
            scalar::elementwise_add(&mut self.state.x, &self.state.xb2);

            // FFN rmsnorm
            let ffn_norm_name = alloc::format!("blk.{}.ffn_norm.weight", l);
            if let Some(w) = self.weights.get_tensor_f32(&ffn_norm_name) {
                scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
            }

            // FFN: gate + up → SiLU → element_mul → down
            let wg_name = alloc::format!("blk.{}.ffn_gate.weight", l);
            let wu_name = alloc::format!("blk.{}.ffn_up.weight", l);
            let wd_name = alloc::format!("blk.{}.ffn_down.weight", l);

            if self.weights.is_quantized {
                if let Some(wg) = self.weights.get_tensor_bytes(&wg_name) {
                    scalar::matmul_q4_0(&mut self.state.hb, wg, &self.state.xb, dim, hidden_dim);
                }
                if let Some(wu) = self.weights.get_tensor_bytes(&wu_name) {
                    scalar::matmul_q4_0(&mut self.state.hb2, wu, &self.state.xb, dim, hidden_dim);
                }
            } else {
                if let Some(wg) = self.weights.get_tensor_f32(&wg_name) {
                    scalar::matmul(&mut self.state.hb, wg, &self.state.xb, dim, hidden_dim);
                }
                if let Some(wu) = self.weights.get_tensor_f32(&wu_name) {
                    scalar::matmul(&mut self.state.hb2, wu, &self.state.xb, dim, hidden_dim);
                }
            }

            scalar::silu(&mut self.state.hb);
            scalar::elementwise_mul(&mut self.state.hb, &self.state.hb2);

            if self.weights.is_quantized {
                if let Some(wd) = self.weights.get_tensor_bytes(&wd_name) {
                    scalar::matmul_q4_0(&mut self.state.xb2, wd, &self.state.hb, hidden_dim, dim);
                }
            } else {
                if let Some(wd) = self.weights.get_tensor_f32(&wd_name) {
                    scalar::matmul(&mut self.state.xb2, wd, &self.state.hb, hidden_dim, dim);
                }
            }

            // Residual
            scalar::elementwise_add(&mut self.state.x, &self.state.xb2);
        }

        // Final rmsnorm
        let final_norm_name = "output_norm.weight";
        if let Some(w) = self.weights.get_tensor_f32(final_norm_name) {
            scalar::rmsnorm(&mut self.state.xb, &self.state.x, w);
        } else {
            self.state.xb.copy_from_slice(&self.state.x);
        }

        // Classifier: project to vocab
        let output_name = "output.weight";
        if self.weights.is_quantized {
            if let Some(wo) = self.weights.get_tensor_bytes(output_name) {
                scalar::matmul_q4_0(&mut self.state.logits, wo, &self.state.xb, cfg.dim, cfg.vocab_size);
            }
        } else {
            if let Some(wo) = self.weights.get_tensor_f32(output_name) {
                scalar::matmul(&mut self.state.logits, wo, &self.state.xb, cfg.dim, cfg.vocab_size);
            }
        }

        &self.state.logits
    }
}
