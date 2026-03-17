pub mod discovery;
pub mod vram;

/// Initialize GPU subsystem: discover AMD GPU on PCIe, map BARs.
pub fn init() {
    discovery::scan();
}
