/// Physical frame allocator for Limine boot.
/// Bump allocator over the largest usable memory region.
/// Frames are never freed — acceptable for early boot.

use x86_64::structures::paging::{FrameAllocator, PageTable, PhysFrame, Size4KiB, OffsetPageTable};
use x86_64::{PhysAddr, VirtAddr};
use core::sync::atomic::{AtomicU64, Ordering};

static USABLE_START: AtomicU64 = AtomicU64::new(0);
static USABLE_END: AtomicU64 = AtomicU64::new(0);
static NEXT_FRAME: AtomicU64 = AtomicU64::new(0);
static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Set up the physical memory region for allocation.
pub fn init(start: u64, end: u64, hhdm: u64) {
    USABLE_START.store(start, Ordering::SeqCst);
    USABLE_END.store(end, Ordering::SeqCst);
    NEXT_FRAME.store(start, Ordering::SeqCst);
    HHDM_OFFSET.store(hhdm, Ordering::SeqCst);
}

pub fn hhdm_offset() -> u64 {
    HHDM_OFFSET.load(Ordering::SeqCst)
}

/// Convert physical address to virtual via HHDM.
pub fn phys_to_virt(phys: PhysAddr) -> VirtAddr {
    VirtAddr::new(HHDM_OFFSET.load(Ordering::SeqCst) + phys.as_u64())
}

/// Get an OffsetPageTable from the currently active CR3.
///
/// # Safety
/// HHDM must be initialized.
pub unsafe fn active_page_table() -> OffsetPageTable<'static> {
    use x86_64::registers::control::Cr3;
    let offset = VirtAddr::new(HHDM_OFFSET.load(Ordering::SeqCst));
    let (frame, _) = Cr3::read();
    let phys = frame.start_address();
    let virt = offset + phys.as_u64();
    let table: &mut PageTable = unsafe { &mut *virt.as_mut_ptr() };
    unsafe { OffsetPageTable::new(table, offset) }
}

/// Total usable memory in bytes.
pub fn total_usable() -> u64 {
    USABLE_END.load(Ordering::SeqCst) - USABLE_START.load(Ordering::SeqCst)
}

/// Bytes allocated so far.
pub fn allocated_bytes() -> u64 {
    NEXT_FRAME.load(Ordering::SeqCst) - USABLE_START.load(Ordering::SeqCst)
}

/// Bump frame allocator.
pub struct BumpAllocator;

unsafe impl FrameAllocator<Size4KiB> for BumpAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        let end = USABLE_END.load(Ordering::SeqCst);
        loop {
            let addr = NEXT_FRAME.load(Ordering::SeqCst);
            if addr >= end { return None; }
            let next = addr + 4096;
            if NEXT_FRAME.compare_exchange(addr, next, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                return Some(PhysFrame::containing_address(PhysAddr::new(addr)));
            }
        }
    }
}
