/// Runtime kernel dispatch: selects AVX2 or scalar based on CPUID.
/// Call init() once after SIMD detection, then use the dispatch functions.

use core::sync::atomic::{AtomicBool, Ordering};

static USE_AVX2: AtomicBool = AtomicBool::new(false);

/// Initialize dispatch based on detected CPU features.
pub fn init() {
    let cpu_avx2 = crate::arch::x86_64::simd::has_avx2();
    // Only use AVX2 if both CPU supports it AND the AVX2 module was compiled
    let use_avx2 = cpu_avx2 && cfg!(avx2_available);
    USE_AVX2.store(use_avx2, Ordering::SeqCst);
    if cpu_avx2 && !cfg!(avx2_available) {
        crate::serial_println!("[kernels] dispatch: scalar (CPU has AVX2 but kernel built without --cfg avx2_available)");
    } else {
        crate::serial_println!("[kernels] dispatch: {}", if use_avx2 { "AVX2" } else { "scalar" });
    }
}

pub fn backend_name() -> &'static str {
    if USE_AVX2.load(Ordering::Relaxed) { "AVX2" } else { "scalar" }
}

/// RMSNorm
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::rmsnorm(out, x, weight); }
        return;
    }
    super::scalar::rmsnorm(out, x, weight);
}

/// Matrix-vector multiply (f32)
pub fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n_in: usize, n_out: usize) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::matmul(out, w, x, n_in, n_out); }
        return;
    }
    super::scalar::matmul(out, w, x, n_in, n_out);
}

/// Quantized matrix-vector multiply (Q4_0)
pub fn matmul_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::matmul_q4_0(out, w_bytes, x, n_in, n_out); }
        return;
    }
    super::scalar::matmul_q4_0(out, w_bytes, x, n_in, n_out);
}

/// Softmax
pub fn softmax(x: &mut [f32]) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::softmax(x); }
        return;
    }
    super::scalar::softmax(x);
}

/// SiLU activation
pub fn silu(x: &mut [f32]) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::silu(x); }
        return;
    }
    super::scalar::silu(x);
}

/// Element-wise multiply
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::elementwise_mul(a, b); }
        return;
    }
    super::scalar::elementwise_mul(a, b);
}

/// Element-wise add
pub fn elementwise_add(a: &mut [f32], b: &[f32]) {
    #[cfg(avx2_available)]
    if USE_AVX2.load(Ordering::Relaxed) {
        unsafe { super::avx2::elementwise_add(a, b); }
        return;
    }
    super::scalar::elementwise_add(a, b);
}

/// RoPE (scalar only — trigonometric ops don't benefit from AVX2 alone)
pub fn rope(q: &mut [f32], k: &mut [f32], pos: usize, dim: usize, head_dim: usize, kv_dim: usize) {
    super::scalar::rope(q, k, pos, dim, head_dim, kv_dim);
}

/// Argmax (scalar only — no SIMD benefit for small vocab)
pub fn argmax(x: &[f32]) -> usize {
    super::scalar::argmax(x)
}
