/// Scalar (no SIMD) math kernels for LLM inference.
/// Pure Rust fallback — works on any CPU.

use crate::inference::tensor::BlockQ4_0;
use libm::{expf, sqrtf};

/// RMSNorm: out[i] = (x[i] / rms) * weight[i]
/// where rms = sqrt(mean(x^2) + eps)
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    ss = 1.0 / sqrtf(ss / n as f32 + 1e-5);
    for i in 0..n {
        out[i] = x[i] * ss * weight[i];
    }
}

/// Softmax in-place over x[0..size].
pub fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 { return; }

    // Find max for numerical stability
    let mut max = x[0];
    for i in 1..n {
        if x[i] > max { max = x[i]; }
    }

    // exp and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    // normalize
    let inv_sum = 1.0 / sum;
    for i in 0..n {
        x[i] *= inv_sum;
    }
}

/// Matrix-vector multiply: out = W * x
/// W is (n_out, n_in) stored row-major as f32.
pub fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n_in: usize, n_out: usize) {
    for i in 0..n_out {
        let mut val = 0.0f32;
        let row = &w[i * n_in..(i + 1) * n_in];
        for j in 0..n_in {
            val += row[j] * x[j];
        }
        out[i] = val;
    }
}

/// Quantized matrix-vector multiply: out = W_q4 * x
/// W is stored as Q4_0 blocks, row-major.
/// Each row has n_in elements = n_in/32 blocks.
pub fn matmul_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize) {
    let blocks_per_row = n_in / BlockQ4_0::BLOCK_SIZE;
    let row_bytes = blocks_per_row * BlockQ4_0::BYTE_SIZE;

    for i in 0..n_out {
        let row_start = i * row_bytes;
        let mut val = 0.0f32;

        for b in 0..blocks_per_row {
            let block_offset = row_start + b * BlockQ4_0::BYTE_SIZE;
            let block = unsafe {
                &*(w_bytes.as_ptr().add(block_offset) as *const BlockQ4_0)
            };
            let mut deq = [0.0f32; 32];
            block.dequantize(&mut deq);

            let x_offset = b * BlockQ4_0::BLOCK_SIZE;
            for k in 0..BlockQ4_0::BLOCK_SIZE {
                val += deq[k] * x[x_offset + k];
            }
        }
        out[i] = val;
    }
}

/// RoPE (Rotary Positional Embedding) applied in-place.
/// Applies to pairs (q[2i], q[2i+1]) at position `pos`.
pub fn rope(q: &mut [f32], k: &mut [f32], pos: usize, dim: usize, head_dim: usize, kv_dim: usize) {
    let mut i = 0;
    while i < dim {
        let head_d = i % head_dim;
        let freq = 1.0 / libm::powf(10000.0, head_d as f32 / head_dim as f32);
        let val = pos as f32 * freq;
        let cos_val = libm::cosf(val);
        let sin_val = libm::sinf(val);

        // Rotate q
        if i + 1 < dim {
            let q0 = q[i];
            let q1 = q[i + 1];
            q[i] = q0 * cos_val - q1 * sin_val;
            q[i + 1] = q0 * sin_val + q1 * cos_val;
        }

        // Rotate k (only up to kv_dim)
        if i < kv_dim && i + 1 < kv_dim {
            let k0 = k[i];
            let k1 = k[i + 1];
            k[i] = k0 * cos_val - k1 * sin_val;
            k[i + 1] = k0 * sin_val + k1 * cos_val;
        }

        i += 2;
    }
}

/// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + expf(-*v));
    }
}

/// Element-wise multiply: a[i] *= b[i]
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Element-wise add: a[i] += b[i]
pub fn elementwise_add(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Argmax over a slice.
pub fn argmax(x: &[f32]) -> usize {
    let mut max_i = 0;
    let mut max_v = x[0];
    for i in 1..x.len() {
        if x[i] > max_v {
            max_v = x[i];
            max_i = i;
        }
    }
    max_i
}
