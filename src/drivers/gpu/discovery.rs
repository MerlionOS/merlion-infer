/// AMD GPU discovery via PCIe.
/// Identifies AMD Radeon GPUs by vendor ID 0x1002 (AMD) with display class.

use crate::drivers::pci;
use crate::memory::phys;
use core::sync::atomic::{AtomicBool, Ordering};
use alloc::string::String;

const AMD_VENDOR_ID: u16 = 0x1002;
const DISPLAY_CLASS: u8 = 0x03;

static DETECTED: AtomicBool = AtomicBool::new(false);

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

    // Try to read GPU identification from MMIO
    if bar0_phys != 0 {
        let mmio = phys::phys_to_virt(x86_64::PhysAddr::new(bar0_phys));
        unsafe {
            // Read MM_INDEX/MM_DATA to probe GPU family
            // Register 0x00 is typically the GPU family identifier
            let reg0 = core::ptr::read_volatile(mmio.as_ptr() as *const u32);
            crate::serial_println!("[gpu] MMIO[0x00] = {:#010x}", reg0);
        }
        DETECTED.store(true, Ordering::SeqCst);
    }

    let chip_name = match dev.device_id {
        0x744C => "Navi 31 (RX 7900 XT/XTX)",
        0x7480 => "Navi 31 (RX 7900 GRE)",
        0x73DF => "Navi 21 (RX 6800/6900)",
        0x73BF => "Navi 21 (RX 6900 XT)",
        0x7340 => "Navi 14 (RX 5500)",
        0x731F => "Navi 10 (RX 5700)",
        _ => "Unknown AMD GPU",
    };
    crate::serial_println!("[gpu] Chip: {} (device_id={:#06x})", chip_name, dev.device_id);
    crate::serial_println!("[gpu] Status: discovered (driver not yet active)");
    crate::serial_println!("[gpu] Phase 6 TODO: firmware load, engine init, compute queue");
}

pub fn is_detected() -> bool { DETECTED.load(Ordering::SeqCst) }

pub fn info() -> String {
    if !is_detected() { return String::from("gpu: not detected"); }
    unsafe {
        let did = core::ptr::read_volatile(&raw const GPU.device_id);
        let bar0 = core::ptr::read_volatile(&raw const GPU.bar0_phys);
        let bar2 = core::ptr::read_volatile(&raw const GPU.bar2_phys);
        alloc::format!("gpu: AMD device_id={:#06x} BAR0={:#x} BAR2={:#x}", did, bar0, bar2)
    }
}
