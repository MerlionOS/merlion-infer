/// Built-in tiny test model for verifying the inference pipeline.
/// dim=32, hidden_dim=64, 1 layer, 2 heads, vocab=256 (byte-level).
/// Total ~107 KiB of F32 weights, allocated on heap.

use alloc::vec::Vec;
use alloc::string::String;
use crate::inference::engine::{ModelConfig, ModelWeights, RunState, LlamaEngine};

const DIM: usize = 32;
const HIDDEN_DIM: usize = 64;
const N_LAYERS: usize = 1;
const N_HEADS: usize = 2;
const KV_HEADS: usize = 2;
const VOCAB: usize = 256;
const MAX_SEQ: usize = 128;
const HEAD_DIM: usize = DIM / N_HEADS;
const KV_DIM: usize = HEAD_DIM * KV_HEADS;

/// Create a tiny test model with pseudo-random weights.
pub fn create_test_engine() -> LlamaEngine {
    let config = ModelConfig {
        dim: DIM,
        hidden_dim: HIDDEN_DIM,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
        n_kv_heads: KV_HEADS,
        vocab_size: VOCAB,
        max_seq_len: MAX_SEQ,
        head_dim: HEAD_DIM,
        kv_dim: KV_DIM,
    };

    // Build weight data: all tensors concatenated as f32
    let mut data: Vec<u8> = Vec::new();
    let mut tensor_map: Vec<(String, usize, usize)> = Vec::new();
    let mut rng: u32 = 42;

    // Helper: append a tensor of given size with small random values
    let append_tensor = |name: &str, n_floats: usize, data: &mut Vec<u8>,
                              map: &mut Vec<(String, usize, usize)>, rng: &mut u32| {
        let offset = data.len();
        let byte_size = n_floats * 4;
        for _ in 0..n_floats {
            // Simple xorshift for reproducible pseudo-random
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            // Scale to [-0.1, 0.1]
            let val = (*rng as f32 / u32::MAX as f32 - 0.5) * 0.2;
            data.extend_from_slice(&val.to_le_bytes());
        }
        map.push((String::from(name), offset, byte_size));
    };

    // Token embedding: (vocab, dim)
    append_tensor("token_embd.weight", VOCAB * DIM, &mut data, &mut tensor_map, &mut rng);

    // Per-layer weights
    for l in 0..N_LAYERS {
        let prefix = alloc::format!("blk.{}", l);
        append_tensor(&alloc::format!("{}.attn_norm.weight", prefix), DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.attn_q.weight", prefix), DIM * DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.attn_k.weight", prefix), DIM * KV_DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.attn_v.weight", prefix), DIM * KV_DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.attn_output.weight", prefix), DIM * DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.ffn_norm.weight", prefix), DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.ffn_gate.weight", prefix), DIM * HIDDEN_DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.ffn_up.weight", prefix), DIM * HIDDEN_DIM, &mut data, &mut tensor_map, &mut rng);
        append_tensor(&alloc::format!("{}.ffn_down.weight", prefix), HIDDEN_DIM * DIM, &mut data, &mut tensor_map, &mut rng);
    }

    // Final norm + output projection
    append_tensor("output_norm.weight", DIM, &mut data, &mut tensor_map, &mut rng);
    append_tensor("output.weight", DIM * VOCAB, &mut data, &mut tensor_map, &mut rng);

    let weights = ModelWeights {
        data,
        data_offset: 0, // tensors start at offset 0 in our data vec
        tensor_map,
        is_quantized: false,
    };

    let state = RunState::new(&config);

    crate::serial_println!("[test-model] Created: dim={} hidden={} layers={} heads={} vocab={}",
        DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, VOCAB);
    crate::serial_println!("[test-model] Weights: {} KiB, State: {} KiB",
        weights.memory_bytes() / 1024, state.memory_bytes() / 1024);

    LlamaEngine { config, state, weights }
}

/// Set up byte-level tokenizer (token i = byte i).
pub fn setup_byte_tokenizer() {
    use crate::inference::gguf::GgufValue;

    let mut tokens: Vec<GgufValue> = Vec::with_capacity(VOCAB);
    let mut scores: Vec<GgufValue> = Vec::with_capacity(VOCAB);

    for i in 0..VOCAB {
        // Each token is a single byte
        let s = if i >= 32 && i < 127 {
            alloc::format!("{}", i as u8 as char)
        } else {
            alloc::format!("<{:02x}>", i)
        };
        tokens.push(GgufValue::String(s));
        scores.push(GgufValue::Float32(0.0));
    }

    crate::inference::tokenizer::load_from_gguf(&tokens, &scores);
    crate::serial_println!("[test-model] Byte-level tokenizer: {} tokens", VOCAB);
}
