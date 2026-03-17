/// VRAM allocator stub.
/// Will manage GPU-local memory for model weights and KV cache.
/// Phase 6: implement buddy allocator over VRAM BAR2.

use core::sync::atomic::{AtomicU64, Ordering};

static VRAM_BASE: AtomicU64 = AtomicU64::new(0);
static VRAM_SIZE: AtomicU64 = AtomicU64::new(0);
static VRAM_NEXT: AtomicU64 = AtomicU64::new(0);

/// Initialize VRAM allocator with base address and size.
pub fn init(base: u64, size: u64) {
    VRAM_BASE.store(base, Ordering::SeqCst);
    VRAM_SIZE.store(size, Ordering::SeqCst);
    VRAM_NEXT.store(base, Ordering::SeqCst);
    crate::serial_println!("[vram] Initialized: {:#x}..{:#x} ({} MiB)",
        base, base + size, size / (1024 * 1024));
}

/// Allocate `size` bytes from VRAM (bump allocator, 256-byte aligned).
pub fn alloc(size: u64) -> Option<u64> {
    let aligned_size = (size + 255) & !255;
    let end = VRAM_BASE.load(Ordering::SeqCst) + VRAM_SIZE.load(Ordering::SeqCst);
    loop {
        let addr = VRAM_NEXT.load(Ordering::SeqCst);
        if addr + aligned_size > end { return None; }
        let next = addr + aligned_size;
        if VRAM_NEXT.compare_exchange(addr, next, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            return Some(addr);
        }
    }
}

pub fn used_bytes() -> u64 {
    VRAM_NEXT.load(Ordering::SeqCst) - VRAM_BASE.load(Ordering::SeqCst)
}

pub fn total_bytes() -> u64 {
    VRAM_SIZE.load(Ordering::SeqCst)
}
