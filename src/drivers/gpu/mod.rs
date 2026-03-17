pub mod discovery;
pub mod mmio;
pub mod regs;
pub mod info;
pub mod vram;
pub mod compute;

/// Initialize GPU subsystem: discover, map MMIO, read status.
pub fn init() {
    discovery::scan();

    if discovery::is_detected() && discovery::is_gcn_capable() {
        info::print_diagnostics();
        compute::init();
    }
}
