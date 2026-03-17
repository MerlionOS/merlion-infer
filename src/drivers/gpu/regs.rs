/// AMD GPU register definitions for GCN4 (Polaris) and later.
/// References: Linux amdgpu driver, AMD GPU programming guides.

// ============================================================
// GRBM (Graphics Register Bus Manager) — GPU engine status
// ============================================================

/// GRBM_STATUS: shows which engines are busy.
pub const GRBM_STATUS: usize = 0xD00;
/// GRBM_STATUS2: additional engine status.
pub const GRBM_STATUS2: usize = 0xD04;

// GRBM_STATUS bit definitions
pub const GRBM_GUI_ACTIVE: u32 = 1 << 31;  // Any engine active
pub const GRBM_CP_BUSY: u32 = 1 << 29;     // Command Processor busy
pub const GRBM_CP_COHERENCY_BUSY: u32 = 1 << 28;
pub const GRBM_GFX_BUSY: u32 = 1 << 26;    // GFX pipe busy
pub const GRBM_ME0_PIPE0_BUSY: u32 = 1 << 12;

// ============================================================
// CP (Command Processor)
// ============================================================

pub const CP_STAT: usize = 0xD048;
pub const CP_ME_CNTL: usize = 0xD414;
pub const CP_RB_WPTR: usize = 0xD06C;

// ============================================================
// MC (Memory Controller) — VRAM configuration
// ============================================================

/// MC_VM_FB_LOCATION: framebuffer base address in GPU address space.
pub const MC_VM_FB_LOCATION: u32 = 0x809;
/// MC_VM_FB_OFFSET: offset to add when converting MC address to system address.
pub const MC_VM_FB_OFFSET: u32 = 0x80A;

// For Polaris, VRAM size is typically read from CONFIG_MEMSIZE
pub const CONFIG_MEMSIZE: usize = 0x150;

// ============================================================
// HDP (Host Data Path)
// ============================================================

pub const HDP_HOST_PATH_CNTL: usize = 0x5428;
pub const HDP_NONSURFACE_BASE: usize = 0x5440;
pub const HDP_NONSURFACE_SIZE: usize = 0x5444;

// ============================================================
// GC (Graphics Controller) — for GCN identification
// ============================================================

/// RLC (Run List Controller) status and control.
pub const RLC_STAT: usize = 0xC034;
pub const RLC_CNTL: usize = 0xC030;
pub const RLC_GPU_CLOCK_32_RES_SEL: usize = 0xC038;
pub const RLC_GPU_CLOCK_32: usize = 0xC03C;

// ============================================================
// SMU (System Management Unit) — power/thermal
// ============================================================

/// Current GPU temperature (SMC register, Polaris).
pub const SMC_TEMP_STATUS: u32 = 0xC0300E0C;
/// Current SCLK (shader clock) frequency.
pub const SMC_SCLK_STATUS: u32 = 0xC0300028;
/// Current MCLK (memory clock) frequency.
pub const SMC_MCLK_STATUS: u32 = 0xC0300030;

// ============================================================
// Compute-related registers (MEC / Compute Queue)
// ============================================================

/// MEC engine control.
pub const CP_MEC_CNTL: usize = 0xD420;
/// Compute queue doorbell range.
pub const CP_HQD_PQ_DOORBELL_CONTROL: usize = 0xDD24;
/// Compute queue PQ (Pipeline Queue) base address.
pub const CP_HQD_PQ_BASE: usize = 0xDD30;
pub const CP_HQD_PQ_BASE_HI: usize = 0xDD34;
/// Compute queue PQ read/write pointers.
pub const CP_HQD_PQ_RPTR: usize = 0xDD38;
pub const CP_HQD_PQ_WPTR: usize = 0xDD3C;
/// Compute queue PQ control (size, enable).
pub const CP_HQD_PQ_CONTROL: usize = 0xDD40;
/// Compute queue active status.
pub const CP_HQD_ACTIVE: usize = 0xDD14;
/// Number of MEC pipes and queues.
pub const CP_MEC_ME1_PIPE0_INT_CNTL: usize = 0xDB00;

// ============================================================
// MEC firmware upload registers (GFX v8.0 / Polaris)
// ============================================================

/// MEC microcode address (write shader program counter).
pub const CP_MEC_ME1_UCODE_ADDR: usize = 0xDE40;
/// MEC microcode data (write instruction dwords).
pub const CP_MEC_ME1_UCODE_DATA: usize = 0xDE44;
/// MEC2 microcode address.
pub const CP_MEC_ME2_UCODE_ADDR: usize = 0xDE48;
/// MEC2 microcode data.
pub const CP_MEC_ME2_UCODE_DATA: usize = 0xDE4C;

// ============================================================
// HQD (Hardware Queue Descriptor) additional registers
// ============================================================

/// HQD VMID — which VMID this queue uses.
pub const CP_HQD_VMID: usize = 0xDD18;
/// HQD IB (Indirect Buffer) control.
pub const CP_HQD_IB_CONTROL: usize = 0xDD60;
/// HQD dequeue request (to deactivate queue).
pub const CP_HQD_DEQUEUE_REQUEST: usize = 0xDD68;
/// HQD SEMA_CMD (queue semaphore).
pub const CP_HQD_SEMA_CMD: usize = 0xDD20;
/// HQD EOP (End Of Pipe) base address.
pub const CP_HQD_EOP_BASE_ADDR: usize = 0xDD80;
pub const CP_HQD_EOP_BASE_ADDR_HI: usize = 0xDD84;
/// HQD EOP control.
pub const CP_HQD_EOP_CONTROL: usize = 0xDD88;
/// HQD EOP read/write pointer.
pub const CP_HQD_EOP_RPTR: usize = 0xDD8C;
pub const CP_HQD_EOP_WPTR: usize = 0xDD90;

// ============================================================
// Doorbell registers
// ============================================================

/// Doorbell range for the BIF (Bus Interface).
pub const BIF_DOORBELL_RANGE: usize = 0x5A00;
/// Doorbell enable.
pub const BIF_DOORBELL_ENABLE: usize = 0x5A04;
/// HQD doorbell offset.
pub const CP_HQD_PQ_DOORBELL: usize = 0xDD28;

// ============================================================
// SRBM (System Register Bus Manager) — per-pipe selection
// ============================================================

/// Select which MEC/pipe/queue the HQD registers refer to.
pub const SRBM_GFX_CNTL: usize = 0x30;
/// SRBM status (pipe busy bits).
pub const SRBM_STATUS: usize = 0x34;
/// SRBM status 2.
pub const SRBM_STATUS2: usize = 0x38;
