/// xHCI Transfer Ring and TRB (Transfer Request Block) support.
///
/// Transfer rings are circular buffers of TRBs that the host controller
/// uses to transfer data to/from USB endpoints.
///
/// For USB Ethernet, we need:
/// - Bulk OUT ring: host → device (send Ethernet frames)
/// - Bulk IN ring: device → host (receive Ethernet frames)

use crate::memory::phys;

/// Transfer Request Block — the fundamental unit of xHCI I/O.
/// Each TRB is 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Trb {
    pub param_lo: u32,
    pub param_hi: u32,
    pub status: u32,
    pub control: u32,
}

impl Trb {
    pub const fn zeroed() -> Self {
        Self { param_lo: 0, param_hi: 0, status: 0, control: 0 }
    }
}

// TRB type codes (bits [15:10] of control field)
pub const TRB_TYPE_NORMAL: u32 = 1 << 10;
pub const TRB_TYPE_SETUP: u32 = 2 << 10;
pub const TRB_TYPE_DATA: u32 = 3 << 10;
pub const TRB_TYPE_STATUS: u32 = 4 << 10;
pub const TRB_TYPE_LINK: u32 = 6 << 10;
pub const TRB_TYPE_NOOP: u32 = 8 << 10;

// TRB control flags
pub const TRB_CYCLE: u32 = 1 << 0;
pub const TRB_IOC: u32 = 1 << 5;   // Interrupt On Completion
pub const TRB_IDT: u32 = 1 << 6;   // Immediate Data

const RING_SIZE: usize = 256; // TRBs per ring (256 * 16 = 4096 bytes = 1 page)

/// A transfer ring for one endpoint.
pub struct TransferRing {
    /// Physical address of the ring buffer.
    pub phys: u64,
    /// Virtual pointer to the ring buffer.
    pub virt: *mut Trb,
    /// Current enqueue index.
    pub enqueue_idx: usize,
    /// Producer Cycle State.
    pub cycle: bool,
    /// Whether this ring is initialized.
    pub initialized: bool,
}

unsafe impl Send for TransferRing {}
unsafe impl Sync for TransferRing {}

impl TransferRing {
    pub const fn empty() -> Self {
        Self {
            phys: 0,
            virt: core::ptr::null_mut(),
            enqueue_idx: 0,
            cycle: true,
            initialized: false,
        }
    }

    /// Allocate and initialize a transfer ring.
    pub fn init(&mut self) -> bool {
        use x86_64::structures::paging::FrameAllocator;
        let frame = match phys::BumpAllocator.allocate_frame() {
            Some(f) => f,
            None => return false,
        };
        self.phys = frame.start_address().as_u64();
        self.virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr() as *mut Trb;
        unsafe { core::ptr::write_bytes(self.virt, 0, 4096); }

        // Last TRB is a Link TRB pointing back to start
        let link_idx = RING_SIZE - 1;
        unsafe {
            let link = &mut *self.virt.add(link_idx);
            link.param_lo = self.phys as u32;
            link.param_hi = (self.phys >> 32) as u32;
            link.control = TRB_TYPE_LINK | TRB_CYCLE; // toggle cycle bit
        }

        self.enqueue_idx = 0;
        self.cycle = true;
        self.initialized = true;
        true
    }

    /// Enqueue a Normal TRB for a bulk transfer.
    /// Returns the physical address of the TRB for doorbell notification.
    pub fn enqueue_bulk(&mut self, data_phys: u64, length: u32, ioc: bool) -> Option<u64> {
        if !self.initialized { return None; }
        if self.enqueue_idx >= RING_SIZE - 1 {
            // Wrap around via Link TRB
            self.enqueue_idx = 0;
            self.cycle = !self.cycle;
        }

        let idx = self.enqueue_idx;
        let cycle_bit = if self.cycle { TRB_CYCLE } else { 0 };
        let ioc_bit = if ioc { TRB_IOC } else { 0 };

        unsafe {
            let trb = &mut *self.virt.add(idx);
            trb.param_lo = data_phys as u32;
            trb.param_hi = (data_phys >> 32) as u32;
            trb.status = length;
            trb.control = TRB_TYPE_NORMAL | cycle_bit | ioc_bit;
        }

        self.enqueue_idx += 1;
        Some(self.phys + (idx * 16) as u64)
    }

    /// Enqueue a No-Op TRB (for testing the ring).
    pub fn enqueue_noop(&mut self) -> Option<u64> {
        if !self.initialized { return None; }
        if self.enqueue_idx >= RING_SIZE - 1 {
            self.enqueue_idx = 0;
            self.cycle = !self.cycle;
        }

        let idx = self.enqueue_idx;
        let cycle_bit = if self.cycle { TRB_CYCLE } else { 0 };

        unsafe {
            let trb = &mut *self.virt.add(idx);
            *trb = Trb::zeroed();
            trb.control = TRB_TYPE_NOOP | cycle_bit | TRB_IOC;
        }

        self.enqueue_idx += 1;
        Some(self.phys + (idx * 16) as u64)
    }
}

/// Bulk OUT transfer ring (for sending Ethernet frames).
static mut BULK_OUT: TransferRing = TransferRing::empty();
/// Bulk IN transfer ring (for receiving Ethernet frames).
static mut BULK_IN: TransferRing = TransferRing::empty();

/// Initialize transfer rings for USB Ethernet.
pub fn init_ethernet_rings() -> bool {
    unsafe {
        let out_ok = (*core::ptr::addr_of_mut!(BULK_OUT)).init();
        let in_ok = (*core::ptr::addr_of_mut!(BULK_IN)).init();
        if out_ok && in_ok {
            let out_phys = core::ptr::read_volatile(&raw const BULK_OUT.phys);
            let in_phys = core::ptr::read_volatile(&raw const BULK_IN.phys);
            crate::serial_println!("[xhci-xfer] Bulk OUT ring: phys={:#x}", out_phys);
            crate::serial_println!("[xhci-xfer] Bulk IN ring: phys={:#x}", in_phys);
            true
        } else {
            crate::serial_println!("[xhci-xfer] Failed to allocate transfer rings");
            false
        }
    }
}

/// Queue an Ethernet frame for transmission via bulk OUT.
pub fn queue_tx_frame(frame_phys: u64, length: u32) -> bool {
    unsafe { (*core::ptr::addr_of_mut!(BULK_OUT)).enqueue_bulk(frame_phys, length, true).is_some() }
}

/// Queue a receive buffer on the bulk IN ring.
pub fn queue_rx_buffer(buf_phys: u64, length: u32) -> bool {
    unsafe { (*core::ptr::addr_of_mut!(BULK_IN)).enqueue_bulk(buf_phys, length, true).is_some() }
}
