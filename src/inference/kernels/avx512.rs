/// AVX-512 optimized math kernels for LLM inference.
/// Uses 512-bit SIMD (16 floats per instruction).
/// Requires: Intel Skylake-X/Sapphire Rapids or AMD Zen 4+.
/// Enable with: RUSTFLAGS="--cfg avx512_available" cargo build

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::inference::tensor::BlockQ4_0;

/// RMSNorm with AVX-512: 2x throughput over AVX2.
#[target_feature(enable = "avx512f")]
pub unsafe fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = x.len();
    let mut ss = _mm512_setzero_ps();

    // Sum of squares
    let mut i = 0;
    while i + 16 <= n {
        let xv = _mm512_loadu_ps(x.as_ptr().add(i));
        ss = _mm512_fmadd_ps(xv, xv, ss);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(ss);
    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }

    let scale = 1.0 / libm::sqrtf(sum / n as f32 + 1e-5);
    let scale_v = _mm512_set1_ps(scale);

    // Normalize
    i = 0;
    while i + 16 <= n {
        let xv = _mm512_loadu_ps(x.as_ptr().add(i));
        let wv = _mm512_loadu_ps(weight.as_ptr().add(i));
        let result = _mm512_mul_ps(_mm512_mul_ps(xv, scale_v), wv);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 16;
    }
    while i < n {
        out[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

/// Matrix-vector multiply with AVX-512: 16 floats per cycle.
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n_in: usize, n_out: usize) {
    for i in 0..n_out {
        let row = &w[i * n_in..];
        let mut acc = _mm512_setzero_ps();

        let mut j = 0;
        while j + 16 <= n_in {
            let wv = _mm512_loadu_ps(row.as_ptr().add(j));
            let xv = _mm512_loadu_ps(x.as_ptr().add(j));
            acc = _mm512_fmadd_ps(wv, xv, acc);
            j += 16;
        }
        let mut val = _mm512_reduce_add_ps(acc);
        while j < n_in {
            val += row[j] * x[j];
            j += 1;
        }
        out[i] = val;
    }
}

/// Quantized matmul (Q4_0) with AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize) {
    let blocks_per_row = n_in / BlockQ4_0::BLOCK_SIZE;
    let row_bytes = blocks_per_row * BlockQ4_0::BYTE_SIZE;

    for i in 0..n_out {
        let row_start = i * row_bytes;
        let mut val = 0.0f32;

        for b in 0..blocks_per_row {
            let block_offset = row_start + b * BlockQ4_0::BYTE_SIZE;
            let block = &*(w_bytes.as_ptr().add(block_offset) as *const BlockQ4_0);
            let mut deq = [0.0f32; 32];
            block.dequantize(&mut deq);

            let x_offset = b * BlockQ4_0::BLOCK_SIZE;

            // Use AVX-512 for the 32-element dot product (2 x 16)
            let d0 = _mm512_loadu_ps(deq.as_ptr());
            let x0 = _mm512_loadu_ps(x.as_ptr().add(x_offset));
            let d1 = _mm512_loadu_ps(deq.as_ptr().add(16));
            let x1 = _mm512_loadu_ps(x.as_ptr().add(x_offset + 16));

            let p0 = _mm512_mul_ps(d0, x0);
            let p1 = _mm512_fmadd_ps(d1, x1, p0);
            val += _mm512_reduce_add_ps(p1);
        }
        out[i] = val;
    }
}

/// Softmax with AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 { return; }

    // Find max
    let mut max_v = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        max_v = _mm512_max_ps(max_v, v);
        i += 16;
    }
    let mut max = _mm512_reduce_max_ps(max_v);
    while i < n {
        if x[i] > max { max = x[i]; }
        i += 1;
    }

    // exp and sum (scalar — exp not in AVX-512)
    let mut sum = 0.0f32;
    for i in 0..n {
        x[i] = libm::expf(x[i] - max);
        sum += x[i];
    }

    // Normalize with AVX-512
    let inv_sum = _mm512_set1_ps(1.0 / sum);
    i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        _mm512_storeu_ps(x.as_mut_ptr().add(i), _mm512_mul_ps(v, inv_sum));
        i += 16;
    }
    let inv = 1.0 / sum;
    while i < n {
        x[i] *= inv;
        i += 1;
    }
}

/// SiLU with AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn silu(x: &mut [f32]) {
    // SiLU = x * sigmoid(x), no vectorized exp in AVX-512 base
    for v in x.iter_mut() {
        *v = *v / (1.0 + libm::expf(-*v));
    }
}

/// Element-wise multiply with AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(a.as_mut_ptr().add(i), _mm512_mul_ps(av, bv));
        i += 16;
    }
    while i < n {
        a[i] *= b[i];
        i += 1;
    }
}

/// Element-wise add with AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn elementwise_add(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(a.as_mut_ptr().add(i), _mm512_add_ps(av, bv));
        i += 16;
    }
    while i < n {
        a[i] += b[i];
        i += 1;
    }
}
