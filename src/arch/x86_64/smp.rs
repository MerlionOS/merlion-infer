/// SMP detection and CPU feature reporting.
/// Detects CPU via CPUID, reports capabilities.
/// APs are parked — BSP only for Phase 1.

use alloc::string::String;

/// CPU feature information from CPUID.
pub struct CpuFeatures {
    pub vendor: [u8; 12],
    pub brand: String,
    pub family: u8,
    pub model: u8,
    pub stepping: u8,
    pub logical_cores: u8,
    pub has_apic: bool,
    pub has_x2apic: bool,
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
}

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

pub fn apic_id() -> u8 {
    let (_, ebx, _, _) = cpuid(1);
    ((ebx >> 24) & 0xFF) as u8
}

pub fn detect_features() -> CpuFeatures {
    let (max_cpuid, ebx, ecx, edx) = cpuid(0);

    let mut vendor = [0u8; 12];
    vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&edx.to_le_bytes());
    vendor[8..12].copy_from_slice(&ecx.to_le_bytes());

    let (eax1, ebx1, ecx1, edx1) = if max_cpuid >= 1 {
        cpuid(1)
    } else {
        (0, 0, 0, 0)
    };

    let family = ((eax1 >> 8) & 0xF) as u8;
    let model = ((eax1 >> 4) & 0xF) as u8;
    let stepping = (eax1 & 0xF) as u8;
    let logical_cores = ((ebx1 >> 16) & 0xFF) as u8;

    // AVX2 from leaf 7
    let (_, ebx7, _, _) = if max_cpuid >= 7 {
        cpuid_count(7, 0)
    } else {
        (0, 0, 0, 0)
    };

    // Brand string
    let (max_ext, _, _, _) = cpuid(0x80000000);
    let brand = if max_ext >= 0x80000004 {
        let mut brand_bytes = [0u8; 48];
        for i in 0..3u32 {
            let (a, b, c, d) = cpuid(0x80000002 + i);
            let off = (i as usize) * 16;
            brand_bytes[off..off+4].copy_from_slice(&a.to_le_bytes());
            brand_bytes[off+4..off+8].copy_from_slice(&b.to_le_bytes());
            brand_bytes[off+8..off+12].copy_from_slice(&c.to_le_bytes());
            brand_bytes[off+12..off+16].copy_from_slice(&d.to_le_bytes());
        }
        let s = core::str::from_utf8(&brand_bytes).unwrap_or("").trim_end_matches('\0').trim();
        String::from(s)
    } else {
        String::from("Unknown")
    };

    CpuFeatures {
        vendor,
        brand,
        family,
        model,
        stepping,
        logical_cores: if logical_cores == 0 { 1 } else { logical_cores },
        has_apic: edx1 & (1 << 9) != 0,
        has_x2apic: ecx1 & (1 << 21) != 0,
        has_sse: edx1 & (1 << 25) != 0,
        has_sse2: edx1 & (1 << 26) != 0,
        has_avx: ecx1 & (1 << 28) != 0,
        has_avx2: ebx7 & (1 << 5) != 0,
    }
}

pub fn init() {
    let id = apic_id();
    let features = detect_features();
    let vendor = core::str::from_utf8(&features.vendor).unwrap_or("?");

    crate::serial_println!("[smp] BSP APIC ID: {}", id);
    crate::serial_println!("[smp] CPU: {} ({})", features.brand, vendor);
    crate::serial_println!("[smp] Cores: {} | APIC={} x2APIC={}",
        features.logical_cores,
        if features.has_apic { "yes" } else { "no" },
        if features.has_x2apic { "yes" } else { "no" },
    );
    crate::serial_println!("[smp] SSE={} SSE2={} AVX={} AVX2={}",
        if features.has_sse { "yes" } else { "no" },
        if features.has_sse2 { "yes" } else { "no" },
        if features.has_avx { "yes" } else { "no" },
        if features.has_avx2 { "yes" } else { "no" },
    );
}
