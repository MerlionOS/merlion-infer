/// GPU compute queue for dispatching shader kernels.
///
/// GCN4 (Polaris / gfx803) compute architecture:
/// - MEC (Micro Engine Compute): manages compute queues
/// - Each MEC has 4 pipes, each pipe has 8 queues
/// - Queues are ring buffers of PM4 packets or AQL packets
/// - Doorbell write triggers queue processing
///
/// Initialization sequence (based on Linux amdgpu gfx_v8_0):
/// 1. Halt MEC via CP_MEC_CNTL
/// 2. Upload MEC microcode (if available from disk)
/// 3. Unhalt MEC
/// 4. Select pipe/queue via SRBM_GFX_CNTL
/// 5. Deactivate any existing HQD
/// 6. Program HQD registers (base, size, doorbell, VMID)
/// 7. Activate HQD
///
/// For inference GEMM, we'd submit pre-compiled GCN shader binaries
/// (.hsaco format) via AQL dispatch packets.

use super::{mmio, regs};
use crate::memory::phys;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

static COMPUTE_READY: AtomicBool = AtomicBool::new(false);
static MEC_FW_LOADED: AtomicBool = AtomicBool::new(false);

/// Initialization phase tracking for diagnostics.
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum InitPhase {
    None = 0,
    QueueAllocated = 1,
    MecHalted = 2,
    MecFirmware = 3,
    MecRunning = 4,
    HqdProgrammed = 5,
    Ready = 6,
}
static INIT_PHASE: AtomicU8 = AtomicU8::new(0);

fn set_phase(p: InitPhase) { INIT_PHASE.store(p as u8, Ordering::SeqCst); }

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
    /// EOP (End Of Pipe) signal buffer physical address.
    eop_phys: u64,
    /// EOP virtual address.
    eop_virt: *mut u8,
}

static mut QUEUE: ComputeQueue = ComputeQueue {
    ring_phys: 0, ring_virt: core::ptr::null_mut(),
    ring_size: 0, wptr: 0, rptr: 0, doorbell_idx: 0,
    eop_phys: 0, eop_virt: core::ptr::null_mut(),
};

/// Select a specific MEC/pipe/queue for HQD register access.
/// On GFX v8.0: SRBM_GFX_CNTL bits [3:0]=ME, [5:4]=pipe, [9:6]=queue
fn srbm_select(me: u32, pipe: u32, queue: u32) {
    let val = (me & 0xF) | ((pipe & 0x3) << 4) | ((queue & 0xF) << 6);
    mmio::write32(regs::SRBM_GFX_CNTL, val);
    // Read back to flush the write
    let _ = mmio::read32(regs::SRBM_GFX_CNTL);
}

/// Reset SRBM selection to default.
fn srbm_select_default() {
    mmio::write32(regs::SRBM_GFX_CNTL, 0);
    let _ = mmio::read32(regs::SRBM_GFX_CNTL);
}

/// Deactivate the currently selected HQD.
/// Returns true if successfully deactivated.
fn deactivate_hqd() -> bool {
    let active = mmio::read32(regs::CP_HQD_ACTIVE);
    if active & 1 == 0 {
        return true; // Already inactive
    }

    // Request dequeue
    mmio::write32(regs::CP_HQD_DEQUEUE_REQUEST, 1);

    // Poll for deactivation (timeout after ~10ms at boot)
    for _ in 0..100_000u32 {
        if mmio::read32(regs::CP_HQD_ACTIVE) & 1 == 0 {
            mmio::write32(regs::CP_HQD_DEQUEUE_REQUEST, 0);
            return true;
        }
        core::hint::spin_loop();
    }

    crate::serial_println!("[gpu-compute] Warning: HQD deactivation timeout");
    mmio::write32(regs::CP_HQD_DEQUEUE_REQUEST, 0);
    false
}

/// Halt the MEC engines.
fn halt_mec() {
    // CP_MEC_CNTL: bit 28 = MEC_ME1_HALT, bit 30 = MEC_ME2_HALT
    let cntl = mmio::read32(regs::CP_MEC_CNTL);
    mmio::write32(regs::CP_MEC_CNTL, cntl | (1 << 28) | (1 << 30));
}

/// Unhalt the MEC engines.
fn unhalt_mec() {
    let cntl = mmio::read32(regs::CP_MEC_CNTL);
    mmio::write32(regs::CP_MEC_CNTL, cntl & !((1 << 28) | (1 << 30)));
}

/// Upload MEC firmware from a raw u32 slice.
/// The firmware is loaded via CP_MEC_ME1_UCODE_ADDR/DATA registers.
/// Returns true if firmware was uploaded.
pub fn upload_mec_firmware(fw_data: &[u32]) -> bool {
    if fw_data.is_empty() { return false; }

    crate::serial_println!("[gpu-compute] Uploading MEC firmware ({} dwords)...", fw_data.len());

    halt_mec();
    set_phase(InitPhase::MecHalted);

    // Reset MEC program counter to 0
    mmio::write32(regs::CP_MEC_ME1_UCODE_ADDR, 0);

    // Write firmware dwords
    for &dword in fw_data {
        mmio::write32(regs::CP_MEC_ME1_UCODE_DATA, dword);
    }

    // Also load MEC2 with the same firmware
    mmio::write32(regs::CP_MEC_ME2_UCODE_ADDR, 0);
    for &dword in fw_data {
        mmio::write32(regs::CP_MEC_ME2_UCODE_DATA, dword);
    }

    unhalt_mec();
    MEC_FW_LOADED.store(true, Ordering::SeqCst);
    set_phase(InitPhase::MecFirmware);
    crate::serial_println!("[gpu-compute] MEC firmware uploaded, engines unhalted");
    true
}

/// Initialize one compute queue on MEC ME1, pipe 0, queue 0.
///
/// This performs the full GFX v8.0 HQD initialization:
/// 1. Allocate ring buffer + EOP buffer
/// 2. Select pipe/queue via SRBM
/// 3. Deactivate any existing HQD
/// 4. Program all HQD registers
/// 5. Activate the HQD
pub fn init() {
    if !mmio::is_initialized() {
        crate::serial_println!("[gpu-compute] MMIO not ready");
        return;
    }

    // Allocate ring buffer (4 KiB = 64 AQL packets of 64 bytes each)
    use x86_64::structures::paging::FrameAllocator;
    let ring_frame = match phys::BumpAllocator.allocate_frame() {
        Some(f) => f,
        None => { crate::serial_println!("[gpu-compute] ring alloc failed"); return; }
    };
    let ring_phys = ring_frame.start_address().as_u64();
    let ring_virt = phys::phys_to_virt(ring_frame.start_address()).as_mut_ptr::<u8>();

    // Allocate EOP (End Of Pipe) signal buffer (4 KiB)
    let eop_frame = match phys::BumpAllocator.allocate_frame() {
        Some(f) => f,
        None => { crate::serial_println!("[gpu-compute] EOP alloc failed"); return; }
    };
    let eop_phys = eop_frame.start_address().as_u64();
    let eop_virt = phys::phys_to_virt(eop_frame.start_address()).as_mut_ptr::<u8>();

    unsafe {
        core::ptr::write_bytes(ring_virt, 0, 4096);
        core::ptr::write_bytes(eop_virt, 0, 4096);

        QUEUE.ring_phys = ring_phys;
        QUEUE.ring_virt = ring_virt;
        QUEUE.ring_size = 4096;
        QUEUE.wptr = 0;
        QUEUE.rptr = 0;
        QUEUE.doorbell_idx = 0;
        QUEUE.eop_phys = eop_phys;
        QUEUE.eop_virt = eop_virt;
    }
    set_phase(InitPhase::QueueAllocated);

    // Check MEC status
    let mec_cntl = mmio::read32(regs::CP_MEC_CNTL);
    let mec_halted = (mec_cntl >> 28) & 1 != 0;
    crate::serial_println!("[gpu-compute] CP_MEC_CNTL: {:#010x} (halted={})", mec_cntl, mec_halted);

    // Select MEC ME1, pipe 0, queue 0
    srbm_select(1, 0, 0);

    // Deactivate any existing HQD on this pipe/queue
    deactivate_hqd();

    // Program HQD registers
    // VMID 0 (kernel mode)
    mmio::write32(regs::CP_HQD_VMID, 0);

    // Queue base address (256-byte aligned, register takes addr >> 8)
    mmio::write32(regs::CP_HQD_PQ_BASE, (ring_phys >> 8) as u32);
    mmio::write32(regs::CP_HQD_PQ_BASE_HI, (ring_phys >> 40) as u32);

    // Queue size: log2(size_in_dwords) in bits [5:0], enable in bit 31
    // 4096 bytes = 1024 dwords, log2(1024) = 10
    let pq_control = 10u32 | (1 << 31);
    mmio::write32(regs::CP_HQD_PQ_CONTROL, pq_control);

    // Initialize read/write pointers
    mmio::write32(regs::CP_HQD_PQ_RPTR, 0);
    mmio::write32(regs::CP_HQD_PQ_WPTR, 0);

    // Disable IB (Indirect Buffer) for now
    mmio::write32(regs::CP_HQD_IB_CONTROL, 0);

    // EOP buffer for completion signals
    mmio::write32(regs::CP_HQD_EOP_BASE_ADDR, (eop_phys >> 8) as u32);
    mmio::write32(regs::CP_HQD_EOP_BASE_ADDR_HI, (eop_phys >> 40) as u32);
    // EOP control: log2(size_in_dwords) = 10 for 4KiB
    mmio::write32(regs::CP_HQD_EOP_CONTROL, 10);
    mmio::write32(regs::CP_HQD_EOP_RPTR, 0);
    mmio::write32(regs::CP_HQD_EOP_WPTR, 0);

    // Doorbell: offset 0, enable
    mmio::write32(regs::CP_HQD_PQ_DOORBELL, (0 << 2) | (1 << 30));

    // Activate the HQD
    mmio::write32(regs::CP_HQD_ACTIVE, 1);

    // Read back active status
    let hqd_active = mmio::read32(regs::CP_HQD_ACTIVE);
    set_phase(InitPhase::HqdProgrammed);

    // Restore SRBM to default
    srbm_select_default();

    let fw_loaded = MEC_FW_LOADED.load(Ordering::SeqCst);
    if hqd_active & 1 != 0 {
        COMPUTE_READY.store(true, Ordering::SeqCst);
        set_phase(InitPhase::Ready);
        crate::serial_println!("[gpu-compute] HQD active, ring={:#x} eop={:#x}", ring_phys, eop_phys);
    } else {
        crate::serial_println!("[gpu-compute] HQD not active (firmware needed)");
    }

    if !fw_loaded {
        crate::serial_println!("[gpu-compute] MEC firmware not loaded — dispatch unavailable");
        crate::serial_println!("[gpu-compute] To load firmware: place polaris11_mec.bin on disk");
    }
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

        // Memory barrier before updating write pointer
        core::sync::atomic::fence(Ordering::SeqCst);

        // Advance write pointer
        QUEUE.wptr += 64;

        // Write WPTR register to notify MEC
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

/// Read the current HQD read pointer (how far GPU has consumed).
pub fn read_rptr() -> u64 {
    srbm_select(1, 0, 0);
    let rptr = mmio::read32(regs::CP_HQD_PQ_RPTR) as u64;
    srbm_select_default();
    rptr
}

/// Print compute queue diagnostic info.
pub fn print_status() {
    let phase = INIT_PHASE.load(Ordering::SeqCst);
    let phase_name = match phase {
        0 => "none",
        1 => "queue_allocated",
        2 => "mec_halted",
        3 => "mec_firmware",
        4 => "mec_running",
        5 => "hqd_programmed",
        6 => "ready",
        _ => "unknown",
    };

    crate::serial_println!("[gpu-compute] Phase: {} ({})", phase, phase_name);
    crate::serial_println!("[gpu-compute] MEC firmware: {}", if MEC_FW_LOADED.load(Ordering::SeqCst) { "loaded" } else { "not loaded" });
    crate::serial_println!("[gpu-compute] Dispatch ready: {}", COMPUTE_READY.load(Ordering::SeqCst));

    if mmio::is_initialized() {
        srbm_select(1, 0, 0);
        let active = mmio::read32(regs::CP_HQD_ACTIVE);
        let rptr = mmio::read32(regs::CP_HQD_PQ_RPTR);
        let wptr = mmio::read32(regs::CP_HQD_PQ_WPTR);
        let ctrl = mmio::read32(regs::CP_HQD_PQ_CONTROL);
        srbm_select_default();

        crate::serial_println!("[gpu-compute] HQD: active={} rptr={} wptr={} ctrl={:#010x}",
            active & 1, rptr, wptr, ctrl);
    }

    unsafe {
        let ring_virt = core::ptr::read_volatile(&raw const QUEUE.ring_virt);
        if !ring_virt.is_null() {
            let ring_phys = core::ptr::read_volatile(&raw const QUEUE.ring_phys);
            let wptr = core::ptr::read_volatile(&raw const QUEUE.wptr);
            let rptr = core::ptr::read_volatile(&raw const QUEUE.rptr);
            crate::serial_println!("[gpu-compute] Ring: phys={:#x} wptr={} rptr={}",
                ring_phys, wptr, rptr);
        }
    }
}

pub fn is_ready() -> bool { COMPUTE_READY.load(Ordering::SeqCst) }
pub fn has_firmware() -> bool { MEC_FW_LOADED.load(Ordering::SeqCst) }
