/// GPU information readout: identification, VRAM, clocks, temperature.
/// Reads hardware registers to report GPU status.

use super::{mmio, regs};

/// GPU family identification from register probing.
pub struct GpuInfo {
    pub vram_mb: u32,
    pub gfx_busy: bool,
    pub cp_busy: bool,
    pub gui_active: bool,
}

/// Read GPU engine status from GRBM_STATUS.
pub fn read_status() -> GpuInfo {
    if !mmio::is_initialized() {
        return GpuInfo { vram_mb: 0, gfx_busy: false, cp_busy: false, gui_active: false };
    }

    let grbm = mmio::read32(regs::GRBM_STATUS);

    // VRAM size from CONFIG_MEMSIZE register (in bytes, but often in MB units)
    let memsize_raw = mmio::read32(regs::CONFIG_MEMSIZE);
    // CONFIG_MEMSIZE is typically in MB for GCN GPUs
    let vram_mb = if memsize_raw > 0 && memsize_raw < 65536 {
        memsize_raw
    } else {
        // Might be in bytes
        memsize_raw / (1024 * 1024)
    };

    GpuInfo {
        vram_mb,
        gfx_busy: grbm & regs::GRBM_GFX_BUSY != 0,
        cp_busy: grbm & regs::GRBM_CP_BUSY != 0,
        gui_active: grbm & regs::GRBM_GUI_ACTIVE != 0,
    }
}

/// Read GPU temperature (Polaris/GCN4 via SMC).
/// Returns temperature in degrees Celsius, or 0 if unavailable.
pub fn read_temperature() -> u32 {
    if !mmio::is_initialized() { return 0; }
    let raw = mmio::read_smc(regs::SMC_TEMP_STATUS);
    // Temperature is typically in bits [23:8] as a fixed-point value
    // Format varies by GPU family; for Polaris it's (raw >> 8) & 0xFF
    let temp = (raw >> 8) & 0xFFF;
    if temp > 0 && temp < 200 { temp } else { 0 }
}

/// Read current shader clock (SCLK) in MHz.
pub fn read_sclk_mhz() -> u32 {
    if !mmio::is_initialized() { return 0; }
    // Read from RLC GPU clock counter
    let _clk = mmio::read32(regs::RLC_GPU_CLOCK_32);
    // Free-running counter, not actual frequency.
    // Actual frequency requires SMU communication.
    0
}

/// Print full GPU diagnostic info to serial.
pub fn print_diagnostics() {
    if !mmio::is_initialized() {
        crate::serial_println!("[gpu] MMIO not initialized");
        return;
    }

    let info = read_status();
    let temp = read_temperature();

    crate::serial_println!("[gpu] VRAM: {} MiB", info.vram_mb);
    crate::serial_println!("[gpu] GRBM: GUI_ACTIVE={} GFX_BUSY={} CP_BUSY={}",
        info.gui_active, info.gfx_busy, info.cp_busy);
    crate::serial_println!("[gpu] Temperature: {}°C", temp);

    // Raw register dumps for debugging
    let grbm = mmio::read32(regs::GRBM_STATUS);
    let grbm2 = mmio::read32(regs::GRBM_STATUS2);
    let cp_stat = mmio::read32(regs::CP_STAT);
    let rlc_stat = mmio::read32(regs::RLC_STAT);

    crate::serial_println!("[gpu] GRBM_STATUS:  {:#010x}", grbm);
    crate::serial_println!("[gpu] GRBM_STATUS2: {:#010x}", grbm2);
    crate::serial_println!("[gpu] CP_STAT:      {:#010x}", cp_stat);
    crate::serial_println!("[gpu] RLC_STAT:     {:#010x}", rlc_stat);
}
