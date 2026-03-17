/// Apple NVMe quirks for MacBook Pro 2017.
///
/// Apple's NVMe controller (vendor 0x106B) has several non-standard behaviors:
///
/// 1. **Custom Admin Commands**: Apple uses vendor-specific admin opcodes
///    for features like S3 resume and thermal management.
///
/// 2. **Namespace ID**: Apple sometimes uses namespace ID 0 where the
///    spec requires 1. Our driver should try both.
///
/// 3. **Shutdown Notification**: Apple NVMe requires a proper shutdown
///    notification (CC.SHN) or it may refuse to initialize on next boot.
///
/// 4. **Queue Depth**: Apple's controller may report a smaller max queue
///    depth than standard controllers. Respect MQES from CAP register.
///
/// 5. **Identify quirk**: Some Apple NVMe controllers return slightly
///    different Identify Controller data layout.
///
/// PCI IDs:
///   - Vendor 0x106B (Apple) with class 01:08:02 (NVMe)
///   - Apple ANS2 controller: device ID varies by model year
///
/// References:
///   - Linux kernel: drivers/nvme/host/apple.c
///   - t2linux project: Apple NVMe driver patches

use crate::drivers::pci;

const APPLE_VENDOR_ID: u16 = 0x106B;
const NVME_CLASS: u8 = 0x01;
const NVME_SUBCLASS: u8 = 0x08;

/// Quirk flags for Apple NVMe.
pub struct AppleNvmeQuirks {
    /// Apple NVMe controller detected.
    pub is_apple: bool,
    /// Device ID for diagnostics.
    pub device_id: u16,
    /// Maximum queue entries (from CAP.MQES, Apple may limit this).
    pub max_queue_entries: u16,
    /// Whether to use namespace ID 0 for some commands.
    pub ns_id_zero_hack: bool,
    /// Whether to send shutdown notification before reset.
    pub needs_shutdown_notify: bool,
}

impl AppleNvmeQuirks {
    pub const fn none() -> Self {
        Self {
            is_apple: false,
            device_id: 0,
            max_queue_entries: 64,
            ns_id_zero_hack: false,
            needs_shutdown_notify: false,
        }
    }
}

/// Detect Apple NVMe controller and determine required quirks.
pub fn detect() -> AppleNvmeQuirks {
    let devices = pci::scan();
    let apple_nvme = devices.iter().find(|d| {
        d.vendor_id == APPLE_VENDOR_ID && d.class == NVME_CLASS && d.subclass == NVME_SUBCLASS
    });

    match apple_nvme {
        Some(dev) => {
            crate::serial_println!("[apple-nvme] Detected Apple NVMe controller: device_id={:#06x}",
                dev.device_id);

            let quirks = AppleNvmeQuirks {
                is_apple: true,
                device_id: dev.device_id,
                // Apple controllers typically support fewer queue entries
                max_queue_entries: 32,
                // Some Apple controllers need NS ID 0 for Identify Controller
                ns_id_zero_hack: true,
                // Always send shutdown notification
                needs_shutdown_notify: true,
            };

            crate::serial_println!("[apple-nvme] Quirks: max_qe={}, ns0_hack={}, shutdown_notify={}",
                quirks.max_queue_entries, quirks.ns_id_zero_hack, quirks.needs_shutdown_notify);

            quirks
        }
        None => AppleNvmeQuirks::none(),
    }
}

/// Send NVMe shutdown notification (CC.SHN = 01b: Normal shutdown).
/// Must be called before resetting the controller on Apple hardware,
/// or the controller may not initialize properly on next boot.
pub fn shutdown_notify(regs_base: *mut u8) {
    unsafe {
        // CC register is at offset 0x14
        let cc_ptr = regs_base.add(0x14) as *mut u32;
        let mut cc = core::ptr::read_volatile(cc_ptr);

        // Clear SHN bits [15:14], then set to 01 (normal shutdown)
        cc &= !(0x3 << 14);
        cc |= 1 << 14;
        core::ptr::write_volatile(cc_ptr, cc);

        // Wait for CSTS.SHST to become 10b (shutdown complete)
        let csts_ptr = regs_base.add(0x1C) as *const u32;
        for _ in 0..10_000_000u32 {
            let csts = core::ptr::read_volatile(csts_ptr);
            let shst = (csts >> 2) & 0x3;
            if shst == 0x2 {
                crate::serial_println!("[apple-nvme] Shutdown complete");
                return;
            }
            core::hint::spin_loop();
        }
        crate::serial_println!("[apple-nvme] Shutdown notification timeout");
    }
}
