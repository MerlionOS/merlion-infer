/// AMD GPU discovery via PCIe.
/// Identifies AMD Radeon GPUs by vendor ID 0x1002 (AMD) with display class.

use crate::drivers::pci;
use crate::memory::phys;
use core::sync::atomic::{AtomicBool, Ordering};
use alloc::string::String;

const AMD_VENDOR_ID: u16 = 0x1002;
const DISPLAY_CLASS: u8 = 0x03;

static DETECTED: AtomicBool = AtomicBool::new(false);
static GCN_CAPABLE: AtomicBool = AtomicBool::new(false);

struct GpuInfo {
    device_id: u16,
    bar0_phys: u64,
    bar0_size: u64,
    bar2_phys: u64, // VRAM BAR (for resizable BAR)
    bar2_size: u64,
    vram_mb: u64,
}

static mut GPU: GpuInfo = GpuInfo {
    device_id: 0,
    bar0_phys: 0, bar0_size: 0,
    bar2_phys: 0, bar2_size: 0,
    vram_mb: 0,
};

/// Scan PCIe for an AMD GPU.
pub fn scan() {
    let devices = pci::scan();
    let gpu = devices.iter().find(|d| {
        d.vendor_id == AMD_VENDOR_ID && d.class == DISPLAY_CLASS
    });

    let dev = match gpu {
        Some(d) => d.clone(),
        None => { crate::serial_println!("[gpu] no AMD GPU found"); return; }
    };

    crate::serial_println!("[gpu] found AMD GPU: {}", dev.summary());

    // Enable bus-master + memory-space
    let cmd = pci::pci_read32(dev.bus, dev.device, dev.function, 0x04);
    pci::pci_write32(dev.bus, dev.device, dev.function, 0x04, cmd | 0x06);

    // Read BAR0 (MMIO registers, 64-bit)
    let bar0_lo = pci::pci_read32(dev.bus, dev.device, dev.function, 0x10);
    let bar0_hi = pci::pci_read32(dev.bus, dev.device, dev.function, 0x14);
    let bar0_phys = ((bar0_hi as u64) << 32) | ((bar0_lo & 0xFFFF_FFF0) as u64);

    // Read BAR2 (VRAM, 64-bit — for resizable BAR)
    let bar2_lo = pci::pci_read32(dev.bus, dev.device, dev.function, 0x18);
    let bar2_hi = pci::pci_read32(dev.bus, dev.device, dev.function, 0x1C);
    let bar2_phys = ((bar2_hi as u64) << 32) | ((bar2_lo & 0xFFFF_FFF0) as u64);

    unsafe {
        GPU.device_id = dev.device_id;
        GPU.bar0_phys = bar0_phys;
        GPU.bar2_phys = bar2_phys;
    }

    crate::serial_println!("[gpu] BAR0 (MMIO): {:#x}", bar0_phys);
    crate::serial_println!("[gpu] BAR2 (VRAM): {:#x}", bar2_phys);

    let chip_name = match dev.device_id {
        // RDNA3 (Navi 3x)
        0x744C => "Navi 31 (RX 7900 XT/XTX)",
        0x7480 => "Navi 31 (RX 7900 GRE)",
        0x7470..=0x747F => "Navi 32 (RX 7800/7700)",
        0x7460..=0x746F => "Navi 33 (RX 7600)",
        // RDNA2 (Navi 2x)
        0x73DF => "Navi 21 (RX 6800/6900)",
        0x73BF => "Navi 21 (RX 6900 XT)",
        0x73EF => "Navi 23 (RX 6600)",
        // RDNA1 (Navi 1x)
        0x7340 => "Navi 14 (RX 5500)",
        0x731F => "Navi 10 (RX 5700)",
        // GCN 5 — Vega
        0x6860..=0x687F => "Vega 10 (RX Vega 56/64)",
        0x69A0..=0x69AF => "Vega 12",
        0x66A0..=0x66AF => "Vega 20 (Radeon VII)",
        // GCN 4 — Polaris (MacBook Pro 2017)
        0x67C0..=0x67CF => "Polaris 10 (RX 480/470)",
        0x67DF => "Polaris 10 (RX 480/580)",
        0x67E0 => "Polaris 11 (RX 460)",
        0x67E1..=0x67E8 => "Polaris 11",
        0x67E9 => "Polaris 11 (Pro 560, MacBook Pro)",
        0x67EF => "Polaris 11 (Pro 555/560, MacBook Pro 2017)",
        0x67EA..=0x67EE => "Polaris 11",
        0x67FF => "Polaris 11 (RX 560/Pro 560)",
        0x6FDF => "Polaris 12 (RX 550)",
        // GCN 3 — Tonga/Fiji
        0x6920..=0x692F => "Tonga (R9 380)",
        0x7300 => "Fiji (R9 Fury/Nano)",
        _ => "Unknown AMD GPU",
    };

    let arch = match dev.device_id {
        0x7440..=0x74FF => "RDNA3",
        0x73A0..=0x73FF => "RDNA2",
        0x7310..=0x7340 => "RDNA1",
        0x66A0..=0x66AF | 0x6860..=0x687F | 0x69A0..=0x69AF => "GCN5 (Vega)",
        0x67C0..=0x67FF | 0x6FD0..=0x6FDF => "GCN4 (Polaris)",
        0x6920..=0x692F | 0x7300 => "GCN3",
        _ => "Unknown",
    };
    crate::serial_println!("[gpu] Chip: {} [{}] (device_id={:#06x})", chip_name, arch, dev.device_id);

    match arch {
        "RDNA3" => {
            crate::serial_println!("[gpu] Compute: RDNA3 supported (Phase 6)");
        }
        "GCN4 (Polaris)" => {
            let (cus, vram_gb) = match dev.device_id {
                0x67EF => (12, 2), // Polaris 11: MacBook Pro 2017 (Pro 555=12CU/2GB, Pro 560=16CU/4GB)
                0x67E9 => (16, 4), // Pro 560
                0x67FF => (16, 4), // RX 560
                0x67DF => (36, 8), // RX 580
                _ => (16, 4),
            };
            crate::serial_println!("[gpu] Compute: GCN4/Polaris — gfx803 ISA");
            crate::serial_println!("[gpu] CUs: {} | VRAM: ~{} GB GDDR5 | Wavefront: 64", cus, vram_gb);
            crate::serial_println!("[gpu] FP32 capable, usable for small GEMM offload");
        }
        "GCN5 (Vega)" => {
            crate::serial_println!("[gpu] Compute: GCN5/Vega — gfx900 ISA");
        }
        _ => {
            crate::serial_println!("[gpu] Compute: not yet supported for {}", arch);
        }
    }

    // Init MMIO for GCN3+ GPUs only
    let is_gcn = matches!(arch, "GCN3" | "GCN4 (Polaris)" | "GCN5 (Vega)" | "RDNA1" | "RDNA2" | "RDNA3");
    GCN_CAPABLE.store(is_gcn, Ordering::SeqCst);
    DETECTED.store(true, Ordering::SeqCst);

    if bar0_phys != 0 && is_gcn {
        let mmio_virt = phys::phys_to_virt(x86_64::PhysAddr::new(bar0_phys));
        super::mmio::init(mmio_virt.as_u64());
        crate::serial_println!("[gpu] MMIO mapped at {:#x}", mmio_virt.as_u64());

        // Initialize VRAM allocator from BAR2 if available
        if bar2_phys != 0 {
            // Probe VRAM size from CONFIG_MEMSIZE register
            let memsize_raw = super::mmio::read32(super::regs::CONFIG_MEMSIZE);
            let vram_bytes = if memsize_raw > 0 && memsize_raw < 65536 {
                memsize_raw as u64 * 1024 * 1024 // value is in MiB
            } else if memsize_raw > 0 {
                memsize_raw as u64 // value is in bytes
            } else {
                0
            };
            if vram_bytes > 0 {
                super::vram::init(bar2_phys, vram_bytes);
            }
        }
    } else if !is_gcn {
        crate::serial_println!("[gpu] Pre-GCN — MMIO skipped");
    }
}

pub fn is_detected() -> bool { DETECTED.load(Ordering::SeqCst) }
pub fn is_gcn_capable() -> bool { GCN_CAPABLE.load(Ordering::SeqCst) }

pub fn info() -> String {
    if !is_detected() { return String::from("gpu: not detected"); }
    unsafe {
        let did = core::ptr::read_volatile(&raw const GPU.device_id);
        let bar0 = core::ptr::read_volatile(&raw const GPU.bar0_phys);
        let bar2 = core::ptr::read_volatile(&raw const GPU.bar2_phys);
        alloc::format!("gpu: AMD device_id={:#06x} BAR0={:#x} BAR2={:#x}", did, bar0, bar2)
    }
}
