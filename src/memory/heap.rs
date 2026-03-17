/// Kernel heap allocator.
/// Maps a virtual range and uses linked-list allocator.
/// Size is determined at boot based on available physical memory.

use x86_64::structures::paging::{
    FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
};
use x86_64::VirtAddr;
use linked_list_allocator::LockedHeap;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Heap at a fixed virtual address.
pub const HEAP_START: usize = 0x4444_4444_0000;

/// Minimum heap: 4 MiB (enough for test model).
pub const HEAP_SIZE_MIN: usize = 4 * 1024 * 1024;
/// Maximum heap: 768 MiB.
pub const HEAP_SIZE_MAX: usize = 768 * 1024 * 1024;

/// Actual heap size (set at boot).
static ACTUAL_HEAP_SIZE: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Compute heap size from total usable physical memory.
/// Allocates ~75% of usable RAM for heap, clamped to [4 MiB, 768 MiB].
pub fn compute_heap_size(total_usable_bytes: u64) -> usize {
    let target = (total_usable_bytes as usize * 3) / 4;
    target.clamp(HEAP_SIZE_MIN, HEAP_SIZE_MAX)
}

pub fn init_with_size(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    heap_size: usize,
) -> Result<(), &'static str> {
    let heap_start = VirtAddr::new(HEAP_START as u64);
    let heap_end = heap_start + heap_size as u64 - 1u64;
    let page_range = {
        let start_page = Page::containing_address(heap_start);
        let end_page = Page::containing_address(heap_end);
        Page::range_inclusive(start_page, end_page)
    };

    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or("out of memory for heap")?;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe {
            mapper
                .map_to(page, frame, flags, frame_allocator)
                .map_err(|_| "failed to map heap page")?
                .flush();
        }
    }

    unsafe {
        ALLOCATOR.lock().init(HEAP_START as *mut u8, heap_size);
    }
    ACTUAL_HEAP_SIZE.store(heap_size, Ordering::SeqCst);

    Ok(())
}

/// Backward-compatible init with minimum heap (4 MiB).
pub fn init(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), &'static str> {
    init_with_size(mapper, frame_allocator, HEAP_SIZE_MIN)
}

pub fn heap_size() -> usize {
    ACTUAL_HEAP_SIZE.load(Ordering::SeqCst)
}

pub fn used() -> usize { ALLOCATOR.lock().used() }
pub fn free() -> usize { ALLOCATOR.lock().free() }
