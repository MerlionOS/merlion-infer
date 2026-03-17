pub mod scalar;
// AVX2 module uses x86_64 SIMD intrinsics. It compiles on x86_64 targets
// but causes LLVM errors when cross-compiling from aarch64 hosts.
// Enable with: RUSTFLAGS="--cfg avx2_available" cargo build
#[cfg(avx2_available)]
pub mod avx2;
pub mod dispatch;
