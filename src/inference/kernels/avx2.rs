/// AVX2-optimized math kernels for LLM inference.
/// Each function uses #[target_feature(enable = "avx2")] — never enable
/// AVX2 globally via rustflags (breaks early boot before SIMD init).
///
/// These process 8 f32 values per SIMD instruction (256-bit registers).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::inference::tensor::BlockQ4_0;

/// FMA emulation: a*b + c (avoids _mm256_fmadd_ps which requires FMA3 feature)
#[target_feature(enable = "avx2")]
unsafe fn avx2_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_add_ps(_mm256_mul_ps(a, b), c)
}

/// RMSNorm: out[i] = (x[i] / rms) * weight[i]
#[target_feature(enable = "avx2")]
pub unsafe fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = x.len();

    // Sum of squares using AVX2
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;
    let end8 = n & !7;
    while i < end8 {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        sum_vec = avx2_fmadd_ps(xv, xv, sum_vec);
        i += 8;
    }

    // Horizontal sum
    let mut ss = hsum_ps_avx(sum_vec);
    while i < n {
        ss += x[i] * x[i];
        i += 1;
    }

    // 1/sqrt(mean + eps)
    ss = 1.0 / libm::sqrtf(ss / n as f32 + 1e-5);
    let ss_vec = _mm256_set1_ps(ss);

    // Apply normalization: out = x * ss * weight
    i = 0;
    while i < end8 {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
        let scaled = _mm256_mul_ps(xv, ss_vec);
        let result = _mm256_mul_ps(scaled, wv);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < n {
        out[i] = x[i] * ss * weight[i];
        i += 1;
    }
}

/// Matrix-vector multiply: out = W * x (W is row-major, n_out x n_in)
#[target_feature(enable = "avx2")]
pub unsafe fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n_in: usize, n_out: usize) {
    for i in 0..n_out {
        let row = &w[i * n_in..];
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut j = 0;
        let end32 = n_in & !31;

        // Process 32 elements per iteration (4 x 8-wide SIMD)
        while j < end32 {
            let w0 = _mm256_loadu_ps(row.as_ptr().add(j));
            let x0 = _mm256_loadu_ps(x.as_ptr().add(j));
            acc0 = avx2_fmadd_ps(w0, x0, acc0);

            let w1 = _mm256_loadu_ps(row.as_ptr().add(j + 8));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(j + 8));
            acc1 = avx2_fmadd_ps(w1, x1, acc1);

            let w2 = _mm256_loadu_ps(row.as_ptr().add(j + 16));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(j + 16));
            acc2 = avx2_fmadd_ps(w2, x2, acc2);

            let w3 = _mm256_loadu_ps(row.as_ptr().add(j + 24));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(j + 24));
            acc3 = avx2_fmadd_ps(w3, x3, acc3);

            j += 32;
        }

        // Reduce 4 accumulators to 1
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // Process remaining 8-wide chunks
        while j + 8 <= n_in {
            let wv = _mm256_loadu_ps(row.as_ptr().add(j));
            let xv = _mm256_loadu_ps(x.as_ptr().add(j));
            acc0 = avx2_fmadd_ps(wv, xv, acc0);
            j += 8;
        }

        // Horizontal sum + scalar remainder
        let mut val = hsum_ps_avx(acc0);
        while j < n_in {
            val += row[j] * x[j];
            j += 1;
        }
        out[i] = val;
    }
}

/// Quantized matmul: out = W_q4 * x (W stored as Q4_0 blocks)
#[target_feature(enable = "avx2")]
pub unsafe fn matmul_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize) {
    let blocks_per_row = n_in / BlockQ4_0::BLOCK_SIZE;
    let row_bytes = blocks_per_row * BlockQ4_0::BYTE_SIZE;

    for i in 0..n_out {
        let row_start = i * row_bytes;
        let mut acc = _mm256_setzero_ps();
        let _val = 0.0f32;

        for b in 0..blocks_per_row {
            let block_offset = row_start + b * BlockQ4_0::BYTE_SIZE;
            let block = &*(w_bytes.as_ptr().add(block_offset) as *const BlockQ4_0);

            // Dequantize block to f32
            let mut deq = [0.0f32; 32];
            block.dequantize(&mut deq);

            let x_offset = b * BlockQ4_0::BLOCK_SIZE;

            // AVX2: process 8 elements at a time
            let d0 = _mm256_loadu_ps(deq.as_ptr());
            let x0 = _mm256_loadu_ps(x.as_ptr().add(x_offset));
            acc = avx2_fmadd_ps(d0, x0, acc);

            let d1 = _mm256_loadu_ps(deq.as_ptr().add(8));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 8));
            acc = avx2_fmadd_ps(d1, x1, acc);

            let d2 = _mm256_loadu_ps(deq.as_ptr().add(16));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 16));
            acc = avx2_fmadd_ps(d2, x2, acc);

            let d3 = _mm256_loadu_ps(deq.as_ptr().add(24));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 24));
            acc = avx2_fmadd_ps(d3, x3, acc);
        }

        out[i] = hsum_ps_avx(acc);
    }
}

/// Softmax in-place over x[0..n].
#[target_feature(enable = "avx2")]
pub unsafe fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 { return; }
    let end8 = n & !7;

    // Find max
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < end8 {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, xv);
        i += 8;
    }
    let mut max = hmax_ps_avx(max_vec);
    while i < n {
        if x[i] > max { max = x[i]; }
        i += 1;
    }

    // exp(x - max) and sum
    let _max_vec = _mm256_set1_ps(max);
    let mut sum = 0.0f32;
    // Scalar exp (no _mm256_exp_ps in core::arch)
    for i in 0..n {
        x[i] = libm::expf(x[i] - max);
        sum += x[i];
    }

    // Normalize
    let inv_sum = _mm256_set1_ps(1.0 / sum);
    i = 0;
    while i < end8 {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let result = _mm256_mul_ps(xv, inv_sum);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
        i += 8;
    }
    let inv = 1.0 / sum;
    while i < n {
        x[i] *= inv;
        i += 1;
    }
}

/// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
#[target_feature(enable = "avx2")]
pub unsafe fn silu(x: &mut [f32]) {
    // Scalar — exp isn't available as AVX intrinsic in core::arch
    for v in x.iter_mut() {
        *v = *v / (1.0 + libm::expf(-*v));
    }
}

/// Element-wise multiply: a[i] *= b[i]
#[target_feature(enable = "avx2")]
pub unsafe fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let end8 = n & !7;
    let mut i = 0;
    while i < end8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let result = _mm256_mul_ps(av, bv);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < n {
        a[i] *= b[i];
        i += 1;
    }
}

/// Element-wise add: a[i] += b[i]
#[target_feature(enable = "avx2")]
pub unsafe fn elementwise_add(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let end8 = n & !7;
    let mut i = 0;
    while i < end8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let result = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < n {
        a[i] += b[i];
        i += 1;
    }
}

// --- AVX2 helpers ---

/// Horizontal sum of 8 f32 lanes in __m256.
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
}

/// Horizontal max of 8 f32 lanes in __m256.
#[target_feature(enable = "avx2")]
unsafe fn hmax_ps_avx(v: __m256) -> f32 {
    // Extract to array and find max (avoids cross-compilation issues)
    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    let mut max = arr[0];
    for i in 1..8 {
        if arr[i] > max { max = arr[i]; }
    }
    max
}
