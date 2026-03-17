/// SIMD state initialization and CPUID detection.
/// Enables SSE, AVX, and detects AVX2/AVX-512/AMX capabilities.

use core::sync::atomic::{AtomicBool, Ordering};

static HAS_AVX2: AtomicBool = AtomicBool::new(false);
static HAS_AVX512: AtomicBool = AtomicBool::new(false);
static HAS_AMX: AtomicBool = AtomicBool::new(false);

/// Initialize SIMD state: enable SSE, AVX via CR0/CR4/XCR0.
/// Must be called early in boot, before any SIMD code runs.
pub fn init() {
    unsafe {
        // CR0: clear EM (bit 2), set MP (bit 1)
        let mut cr0: u64;
        core::arch::asm!("mov {}, cr0", out(reg) cr0);
        cr0 &= !(1 << 2); // Clear EM
        cr0 |= 1 << 1;    // Set MP
        core::arch::asm!("mov cr0, {}", in(reg) cr0);

        // CR4: set OSFXSR (bit 9), OSXMMEXCPT (bit 10), OSXSAVE (bit 18)
        let mut cr4: u64;
        core::arch::asm!("mov {}, cr4", out(reg) cr4);
        cr4 |= (1 << 9) | (1 << 10) | (1 << 18);
        core::arch::asm!("mov cr4, {}", in(reg) cr4);

        // XCR0: enable SSE (bit 1) and AVX (bit 2) state saving
        // Only do this if OSXSAVE is supported
        let (_, _, ecx, _) = cpuid(1);
        if ecx & (1 << 27) != 0 {
            // OSXSAVE supported
            let mut xcr0: u64;
            core::arch::asm!("xgetbv", in("ecx") 0u32, out("eax") xcr0, out("edx") _);
            xcr0 |= (1 << 1) | (1 << 2); // SSE + AVX
            let lo = xcr0 as u32;
            let hi = (xcr0 >> 32) as u32;
            core::arch::asm!("xsetbv", in("ecx") 0u32, in("eax") lo, in("edx") hi);
        }
    }

    // Detect extended features
    let (_, ebx7, _, _) = cpuid_count(7, 0);
    let has_avx2 = ebx7 & (1 << 5) != 0;
    let has_avx512f = ebx7 & (1 << 16) != 0;

    HAS_AVX2.store(has_avx2, Ordering::SeqCst);
    HAS_AVX512.store(has_avx512f, Ordering::SeqCst);

    // AMX detection (CPUID leaf 7, subleaf 0, EDX bit 22 = AMX-BF16, bit 24 = AMX-TILE)
    let (_, _, _, edx7) = cpuid_count(7, 0);
    let has_amx = edx7 & (1 << 24) != 0;
    HAS_AMX.store(has_amx, Ordering::SeqCst);

    crate::serial_println!("[simd] SSE=yes AVX2={} AVX-512={} AMX={}",
        if has_avx2 { "yes" } else { "no" },
        if has_avx512f { "yes" } else { "no" },
        if has_amx { "yes" } else { "no" },
    );
}

pub fn has_avx2() -> bool { HAS_AVX2.load(Ordering::Relaxed) }
pub fn has_avx512() -> bool { HAS_AVX512.load(Ordering::Relaxed) }
pub fn has_amx() -> bool { HAS_AMX.load(Ordering::Relaxed) }

fn cpuid(leaf: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") leaf => eax,
            ebx_out = out(reg) ebx,
            out("ecx") ecx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}

fn cpuid_count(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") leaf => eax,
            ebx_out = out(reg) ebx,
            inout("ecx") subleaf => ecx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}
