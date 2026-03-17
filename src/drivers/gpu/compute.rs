/// GPU compute queue for dispatching shader kernels.
///
/// GCN4 (Polaris) compute architecture:
/// - MEC (Micro Engine Compute): manages compute queues
/// - Each MEC has 4 pipes, each pipe has 8 queues
/// - Queues are ring buffers of PM4 packets or AQL packets
/// - Doorbell write triggers queue processing
///
/// For inference GEMM, we'd submit pre-compiled GCN shader binaries
/// (.hsaco format) via AQL dispatch packets.

use super::{mmio, regs};
use crate::memory::phys;
use core::sync::atomic::{AtomicBool, Ordering};

static COMPUTE_READY: AtomicBool = AtomicBool::new(false);

/// AQL (Architected Queuing Language) dispatch packet.
/// Used to submit compute kernel launches to the GPU.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AqlDispatchPacket {
    /// Header: packet type (dispatch=2) + barrier + acquire/release fence
    pub header: u16,
    /// Setup: number of dimensions (1, 2, or 3)
    pub setup: u16,
    /// Workgroup size X
    pub workgroup_size_x: u16,
    /// Workgroup size Y
    pub workgroup_size_y: u16,
    /// Workgroup size Z
    pub workgroup_size_z: u16,
    _reserved0: u16,
    /// Grid size X (total work-items)
    pub grid_size_x: u32,
    /// Grid size Y
    pub grid_size_y: u32,
    /// Grid size Z
    pub grid_size_z: u32,
    /// Private segment size per work-item
    pub private_segment_size: u32,
    /// Group segment size (LDS) per workgroup
    pub group_segment_size: u32,
    /// Kernel object address (pointer to ISA binary)
    pub kernel_object: u64,
    /// Kernarg address (pointer to kernel arguments)
    pub kernarg_address: u64,
    _reserved1: u64,
    /// Completion signal (written when dispatch finishes)
    pub completion_signal: u64,
}

impl AqlDispatchPacket {
    pub const fn zeroed() -> Self {
        Self {
            header: 0, setup: 0,
            workgroup_size_x: 0, workgroup_size_y: 0, workgroup_size_z: 0,
            _reserved0: 0,
            grid_size_x: 0, grid_size_y: 0, grid_size_z: 0,
            private_segment_size: 0, group_segment_size: 0,
            kernel_object: 0, kernarg_address: 0,
            _reserved1: 0, completion_signal: 0,
        }
    }

    /// Create a 1D dispatch packet.
    pub fn dispatch_1d(
        kernel_addr: u64,
        kernarg_addr: u64,
        grid_size: u32,
        workgroup_size: u16,
    ) -> Self {
        Self {
            // header: type=dispatch(2), barrier=0, acquire_fence=system(2), release_fence=system(2)
            header: (2 << 0) | (2 << 8) | (2 << 11),
            setup: 1, // 1 dimension
            workgroup_size_x: workgroup_size,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            _reserved0: 0,
            grid_size_x: grid_size,
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: 0,
            group_segment_size: 0,
            kernel_object: kernel_addr,
            kernarg_address: kernarg_addr,
            _reserved1: 0,
            completion_signal: 0,
        }
    }
}

/// Compute queue state.
struct ComputeQueue {
    /// Physical address of the queue ring buffer.
    ring_phys: u64,
    /// Virtual address of the queue ring buffer.
    ring_virt: *mut u8,
    /// Queue size in bytes (must be power of 2).
    ring_size: usize,
    /// Write pointer (byte offset into ring).
    wptr: u64,
    /// Read pointer (byte offset, updated by hardware).
    rptr: u64,
    /// Doorbell index for this queue.
    doorbell_idx: u32,
}

static mut QUEUE: ComputeQueue = ComputeQueue {
    ring_phys: 0, ring_virt: core::ptr::null_mut(),
    ring_size: 0, wptr: 0, rptr: 0, doorbell_idx: 0,
};

/// Initialize one compute queue on MEC pipe 0, queue 0.
///
/// This is a simplified init — a full driver would:
/// 1. Load MEC firmware
/// 2. Initialize RLC (Run List Controller)
/// 3. Program HQD (Hardware Queue Descriptor) registers
/// 4. Map doorbell pages
pub fn init() {
    if !mmio::is_initialized() {
        crate::serial_println!("[gpu-compute] MMIO not ready");
        return;
    }

    // Allocate ring buffer (4 KiB = 64 AQL packets of 64 bytes each)
    use x86_64::structures::paging::FrameAllocator;
    let frame = match phys::BumpAllocator.allocate_frame() {
        Some(f) => f,
        None => { crate::serial_println!("[gpu-compute] alloc failed"); return; }
    };
    let ring_phys = frame.start_address().as_u64();
    let ring_virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr::<u8>();

    unsafe {
        core::ptr::write_bytes(ring_virt, 0, 4096);

        QUEUE.ring_phys = ring_phys;
        QUEUE.ring_virt = ring_virt;
        QUEUE.ring_size = 4096;
        QUEUE.wptr = 0;
        QUEUE.rptr = 0;
        QUEUE.doorbell_idx = 0;
    }

    // Check if MEC is alive by reading CP_MEC_CNTL
    let mec_cntl = mmio::read32(regs::CP_MEC_CNTL);
    crate::serial_println!("[gpu-compute] CP_MEC_CNTL: {:#010x}", mec_cntl);

    // Read HQD active status
    let hqd_active = mmio::read32(regs::CP_HQD_ACTIVE);
    crate::serial_println!("[gpu-compute] CP_HQD_ACTIVE: {:#010x}", hqd_active);

    // Program the HQD for our queue
    // Note: on real hardware, this requires RLC to be running and MEC firmware
    // to be loaded. Without firmware, the HQD won't actually process packets.
    // This sets up the data structures so we're ready when firmware is available.

    // Set queue base address
    mmio::write32(regs::CP_HQD_PQ_BASE, (ring_phys >> 8) as u32);
    mmio::write32(regs::CP_HQD_PQ_BASE_HI, (ring_phys >> 40) as u32);

    // Set queue size: log2(4096/4) = 10, plus enable bit
    // PQ_CONTROL: bits [5:0] = log2(size_in_dwords), bit 31 = enable
    let pq_control = (10u32) | (1 << 31);
    mmio::write32(regs::CP_HQD_PQ_CONTROL, pq_control);

    // Initialize pointers
    mmio::write32(regs::CP_HQD_PQ_RPTR, 0);
    mmio::write32(regs::CP_HQD_PQ_WPTR, 0);

    COMPUTE_READY.store(true, Ordering::SeqCst);
    crate::serial_println!("[gpu-compute] Queue initialized: ring={:#x} size=4096", ring_phys);
    crate::serial_println!("[gpu-compute] Note: requires MEC firmware for actual dispatch");
}

/// Submit an AQL dispatch packet to the compute queue.
/// Returns true if the packet was enqueued.
pub fn submit_dispatch(packet: &AqlDispatchPacket) -> bool {
    if !COMPUTE_READY.load(Ordering::SeqCst) { return false; }

    unsafe {
        let wptr = QUEUE.wptr as usize;
        let ring_mask = QUEUE.ring_size - 1;
        let offset = wptr & ring_mask;

        // Copy 64-byte AQL packet to ring
        let src = packet as *const AqlDispatchPacket as *const u8;
        let dst = QUEUE.ring_virt.add(offset);
        core::ptr::copy_nonoverlapping(src, dst, 64);

        // Advance write pointer
        QUEUE.wptr += 64;

        // Write doorbell to notify GPU
        // On real hardware, this would write to the doorbell BAR.
        // For now, update the WPTR register directly.
        mmio::write32(regs::CP_HQD_PQ_WPTR, (QUEUE.wptr & 0xFFFFFFFF) as u32);

        core::sync::atomic::fence(Ordering::SeqCst);
    }

    true
}

/// Poll for dispatch completion by checking the completion signal.
/// Returns true if the dispatch has finished.
pub fn poll_completion(signal_addr: u64) -> bool {
    if signal_addr == 0 { return true; }
    unsafe {
        let ptr = signal_addr as *const u64;
        core::ptr::read_volatile(ptr) != 0
    }
}

pub fn is_ready() -> bool { COMPUTE_READY.load(Ordering::SeqCst) }
