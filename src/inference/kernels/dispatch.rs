/// Runtime kernel dispatch: selects AVX2 or scalar based on CPUID.
/// Call init() once after SIMD detection, then use the dispatch functions.

use core::sync::atomic::{AtomicU8, Ordering};

/// 0=scalar, 1=AVX2, 2=AVX-512
static BACKEND: AtomicU8 = AtomicU8::new(0);

/// Initialize dispatch based on detected CPU features.
pub fn init() {
    let cpu_avx2 = crate::arch::x86_64::simd::has_avx2();
    let cpu_avx512 = crate::arch::x86_64::simd::has_avx512();

    let backend = if cpu_avx512 && cfg!(avx512_available) {
        2
    } else if cpu_avx2 && cfg!(avx2_available) {
        1
    } else {
        0
    };
    BACKEND.store(backend, Ordering::SeqCst);

    if cpu_avx512 && !cfg!(avx512_available) {
        crate::serial_println!("[kernels] CPU has AVX-512 but kernel built without --cfg avx512_available");
    }
    if cpu_avx2 && !cfg!(avx2_available) && !cfg!(avx512_available) {
        crate::serial_println!("[kernels] CPU has AVX2 but kernel built without --cfg avx2_available");
    }
    crate::serial_println!("[kernels] dispatch: {}", backend_name());
}

fn use_avx512() -> bool { BACKEND.load(Ordering::Relaxed) == 2 }
fn use_avx2() -> bool { BACKEND.load(Ordering::Relaxed) >= 1 }

pub fn backend_name() -> &'static str {
    match BACKEND.load(Ordering::Relaxed) {
        2 => "AVX-512",
        1 => "AVX2",
        _ => "scalar",
    }
}

/// RMSNorm
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::rmsnorm(out, x, weight); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::rmsnorm(out, x, weight); } return; }
    super::scalar::rmsnorm(out, x, weight);
}

/// Matrix-vector multiply (f32)
pub fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n_in: usize, n_out: usize) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::matmul(out, w, x, n_in, n_out); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::matmul(out, w, x, n_in, n_out); } return; }
    super::scalar::matmul(out, w, x, n_in, n_out);
}

/// Quantized matrix-vector multiply (Q4_0)
pub fn matmul_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::matmul_q4_0(out, w_bytes, x, n_in, n_out); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::matmul_q4_0(out, w_bytes, x, n_in, n_out); } return; }
    super::scalar::matmul_q4_0(out, w_bytes, x, n_in, n_out);
}

/// Softmax
pub fn softmax(x: &mut [f32]) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::softmax(x); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::softmax(x); } return; }
    super::scalar::softmax(x);
}

/// SiLU activation
pub fn silu(x: &mut [f32]) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::silu(x); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::silu(x); } return; }
    super::scalar::silu(x);
}

/// Element-wise multiply
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::elementwise_mul(a, b); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::elementwise_mul(a, b); } return; }
    super::scalar::elementwise_mul(a, b);
}

/// Element-wise add
pub fn elementwise_add(a: &mut [f32], b: &[f32]) {
    #[cfg(avx512_available)]
    if use_avx512() { unsafe { super::avx512::elementwise_add(a, b); } return; }
    #[cfg(avx2_available)]
    if use_avx2() { unsafe { super::avx2::elementwise_add(a, b); } return; }
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
