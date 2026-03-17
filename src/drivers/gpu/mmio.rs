/// AMD GPU MMIO register access.
/// Direct access for registers in BAR0 range (< 256KB),
/// indirect access via MM_INDEX/MM_DATA for registers beyond.

use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};

static MMIO_BASE: AtomicU64 = AtomicU64::new(0);

/// MM_INDEX and MM_DATA registers for indirect access.
const MM_INDEX: usize = 0x0000;
const MM_DATA: usize = 0x0004;

/// Initialize MMIO with BAR0 virtual address.
pub fn init(virt_base: u64) {
    MMIO_BASE.store(virt_base, Ordering::SeqCst);
}

fn base() -> *mut u8 {
    MMIO_BASE.load(Ordering::SeqCst) as *mut u8
}

/// Read a 32-bit MMIO register (direct access, offset < 256KB).
pub fn read32(offset: usize) -> u32 {
    unsafe {
        ptr::read_volatile(base().add(offset) as *const u32)
    }
}

/// Write a 32-bit MMIO register (direct access).
pub fn write32(offset: usize, val: u32) {
    unsafe {
        ptr::write_volatile(base().add(offset) as *mut u32, val);
    }
}

/// Read a register via indirect MM_INDEX/MM_DATA access.
/// Required for registers at offsets >= 256KB.
pub fn read_indirect(reg: u32) -> u32 {
    unsafe {
        let b = base();
        ptr::write_volatile(b.add(MM_INDEX) as *mut u32, reg);
        ptr::read_volatile(b.add(MM_DATA) as *const u32)
    }
}

/// Write a register via indirect MM_INDEX/MM_DATA access.
pub fn write_indirect(reg: u32, val: u32) {
    unsafe {
        let b = base();
        ptr::write_volatile(b.add(MM_INDEX) as *mut u32, reg);
        ptr::write_volatile(b.add(MM_DATA) as *mut u32, val);
    }
}

/// Read a SMC (System Management Controller) register.
/// Used for clocks, temperature, power on GCN GPUs.
/// Access via SMC_IND_INDEX (0x200) / SMC_IND_DATA (0x204).
pub fn read_smc(reg: u32) -> u32 {
    const SMC_IND_INDEX: usize = 0x0200;
    const SMC_IND_DATA: usize = 0x0204;
    unsafe {
        let b = base();
        ptr::write_volatile(b.add(SMC_IND_INDEX) as *mut u32, reg);
        ptr::read_volatile(b.add(SMC_IND_DATA) as *const u32)
    }
}

pub fn is_initialized() -> bool {
    MMIO_BASE.load(Ordering::SeqCst) != 0
}
