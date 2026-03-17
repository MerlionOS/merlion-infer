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
