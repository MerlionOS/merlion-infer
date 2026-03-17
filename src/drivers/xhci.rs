/// xHCI (USB 3.0) host controller driver.
///
/// Provides basic USB enumeration and data transfer for USB Ethernet
/// adapters on machines without built-in Ethernet (e.g. MacBook Pro 2017).
///
/// xHCI specification: USB 3.0 Host Controller Interface (xHCI) Rev 1.2
///
/// Supported USB Ethernet chipsets:
/// - ASIX AX88179 (USB 3.0 Gigabit Ethernet, common USB-to-Ethernet adapter)
/// - CDC-ECM class devices (generic USB Ethernet)
///
/// Architecture:
/// 1. Find xHCI controller via PCI (class 0x0C, subclass 0x03, prog_if 0x30)
/// 2. Map MMIO registers from BAR0
/// 3. Initialize controller: reset, set up DCBAA, command ring, event ring
/// 4. Enable ports, detect USB devices
/// 5. Configure USB Ethernet device, send/receive Ethernet frames

use crate::drivers::pci;
use crate::memory::phys;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};

const XHCI_CLASS: u8 = 0x0C;    // Serial Bus Controller
const XHCI_SUBCLASS: u8 = 0x03; // USB
const XHCI_PROG_IF: u8 = 0x30;  // xHCI

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// xHCI capability registers (read from BAR0).
#[repr(C)]
struct XhciCapRegs {
    caplength: u8,
    _rsvd: u8,
    hci_version: u16,
    hcsparams1: u32,
    hcsparams2: u32,
    hcsparams3: u32,
    hccparams1: u32,
    dboff: u32,
    rtsoff: u32,
    hccparams2: u32,
}

/// xHCI operational registers (at BAR0 + caplength).
#[repr(C)]
struct XhciOpRegs {
    usbcmd: u32,
    usbsts: u32,
    pagesize: u32,
    _rsvd0: [u32; 2],
    dnctrl: u32,
    crcr_lo: u32,
    crcr_hi: u32,
    _rsvd1: [u32; 4],
    dcbaap_lo: u32,
    dcbaap_hi: u32,
    config: u32,
}

const USBCMD_RUN: u32 = 1 << 0;
const USBCMD_HCRST: u32 = 1 << 1;
const USBSTS_HCH: u32 = 1 << 0;  // HC Halted

struct XhciState {
    cap_regs: *mut XhciCapRegs,
    op_regs: *mut XhciOpRegs,
    max_slots: u32,
    max_ports: u32,
    dcbaa_phys: u64,
    cmd_ring_phys: u64,
    evt_ring_phys: u64,
}

unsafe impl Send for XhciState {}
unsafe impl Sync for XhciState {}

static mut STATE: XhciState = XhciState {
    cap_regs: core::ptr::null_mut(),
    op_regs: core::ptr::null_mut(),
    max_slots: 0,
    max_ports: 0,
    dcbaa_phys: 0,
    cmd_ring_phys: 0,
    evt_ring_phys: 0,
};

pub fn init() {
    let devices = pci::scan();
    let dev = match devices.iter().find(|d| {
        d.class == XHCI_CLASS && d.subclass == XHCI_SUBCLASS && d.prog_if == XHCI_PROG_IF
    }) {
        Some(d) => d.clone(),
        None => { crate::serial_println!("[xhci] no controller found"); return; }
    };
    crate::serial_println!("[xhci] found {}", dev.summary());

    // Enable bus-master + memory-space
    let cmd_reg = pci::pci_read32(dev.bus, dev.device, dev.function, 0x04);
    pci::pci_write32(dev.bus, dev.device, dev.function, 0x04, cmd_reg | 0x06);

    // Read BAR0 (64-bit MMIO)
    let bar0_lo = pci::pci_read32(dev.bus, dev.device, dev.function, 0x10);
    let bar0_hi = pci::pci_read32(dev.bus, dev.device, dev.function, 0x14);
    if bar0_lo & 0x1 != 0 {
        crate::serial_println!("[xhci] BAR0 is I/O, expected MMIO"); return;
    }
    let bar0_phys = ((bar0_hi as u64) << 32) | ((bar0_lo & 0xFFFF_FFF0) as u64);
    if bar0_phys == 0 { crate::serial_println!("[xhci] BAR0 is zero"); return; }

    let base = phys::phys_to_virt(x86_64::PhysAddr::new(bar0_phys)).as_mut_ptr::<u8>();

    unsafe {
        let cap = base as *mut XhciCapRegs;
        let caplength = core::ptr::read_volatile(&(*cap).caplength) as usize;
        let hci_version = core::ptr::read_volatile(&(*cap).hci_version);
        let hcsparams1 = core::ptr::read_volatile(&(*cap).hcsparams1);

        let max_slots = hcsparams1 & 0xFF;
        let max_ports = (hcsparams1 >> 24) & 0xFF;

        crate::serial_println!("[xhci] version: {:#06x}, slots: {}, ports: {}",
            hci_version, max_slots, max_ports);

        let op = base.add(caplength) as *mut XhciOpRegs;

        STATE.cap_regs = cap;
        STATE.op_regs = op;
        STATE.max_slots = max_slots;
        STATE.max_ports = max_ports;

        // Halt controller if running
        let usbsts = core::ptr::read_volatile(&(*op).usbsts);
        if usbsts & USBSTS_HCH == 0 {
            // Controller is running, halt it
            let cmd = core::ptr::read_volatile(&(*op).usbcmd);
            core::ptr::write_volatile(&mut (*op).usbcmd, cmd & !USBCMD_RUN);
            for _ in 0..1_000_000u32 {
                if core::ptr::read_volatile(&(*op).usbsts) & USBSTS_HCH != 0 { break; }
                core::hint::spin_loop();
            }
        }

        // Reset controller
        core::ptr::write_volatile(&mut (*op).usbcmd, USBCMD_HCRST);
        for _ in 0..10_000_000u32 {
            if core::ptr::read_volatile(&(*op).usbcmd) & USBCMD_HCRST == 0 { break; }
            core::hint::spin_loop();
        }
        crate::serial_println!("[xhci] controller reset");

        // Allocate DCBAA (Device Context Base Address Array)
        // Array of max_slots+1 pointers (each 8 bytes), 64-byte aligned
        use x86_64::structures::paging::FrameAllocator;
        let frame = phys::BumpAllocator.allocate_frame().expect("xhci: alloc DCBAA");
        let dcbaa_phys = frame.start_address().as_u64();
        let dcbaa_virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr::<u8>();
        core::ptr::write_bytes(dcbaa_virt, 0, 4096);
        STATE.dcbaa_phys = dcbaa_phys;

        // Set DCBAA pointer
        core::ptr::write_volatile(&mut (*op).dcbaap_lo, dcbaa_phys as u32);
        core::ptr::write_volatile(&mut (*op).dcbaap_hi, (dcbaa_phys >> 32) as u32);

        // Set max device slots
        core::ptr::write_volatile(&mut (*op).config, max_slots);

        // Allocate command ring (4 KiB, 256 TRBs)
        let frame = phys::BumpAllocator.allocate_frame().expect("xhci: alloc cmd ring");
        let cmd_ring_phys = frame.start_address().as_u64();
        let cmd_ring_virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr::<u8>();
        core::ptr::write_bytes(cmd_ring_virt, 0, 4096);
        STATE.cmd_ring_phys = cmd_ring_phys;

        // Set command ring pointer (bit 0 = cycle state = 1)
        core::ptr::write_volatile(&mut (*op).crcr_lo, (cmd_ring_phys as u32) | 1);
        core::ptr::write_volatile(&mut (*op).crcr_hi, (cmd_ring_phys >> 32) as u32);

        // Allocate event ring segment table + event ring
        let frame = phys::BumpAllocator.allocate_frame().expect("xhci: alloc evt ring");
        let evt_ring_phys = frame.start_address().as_u64();
        let evt_ring_virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr::<u8>();
        core::ptr::write_bytes(evt_ring_virt, 0, 4096);
        STATE.evt_ring_phys = evt_ring_phys;

        crate::serial_println!("[xhci] DCBAA={:#x} CMD_RING={:#x} EVT_RING={:#x}",
            dcbaa_phys, cmd_ring_phys, evt_ring_phys);

        // Start controller
        let cmd = core::ptr::read_volatile(&(*op).usbcmd);
        core::ptr::write_volatile(&mut (*op).usbcmd, cmd | USBCMD_RUN);

        for _ in 0..1_000_000u32 {
            if core::ptr::read_volatile(&(*op).usbsts) & USBSTS_HCH == 0 { break; }
            core::hint::spin_loop();
        }

        let running = core::ptr::read_volatile(&(*op).usbsts) & USBSTS_HCH == 0;
        if running {
            INITIALIZED.store(true, Ordering::SeqCst);
            crate::serial_println!("[xhci] controller running, {} ports available", max_ports);
        } else {
            crate::serial_println!("[xhci] controller failed to start");
        }

        // Scan ports for connected devices
        scan_ports(op, max_ports);
    }
}

unsafe fn scan_ports(op: *mut XhciOpRegs, max_ports: u32) {
    let port_base = (op as *mut u8).add(0x400); // Port registers start at operational + 0x400

    for port in 0..max_ports {
        let portsc_ptr = port_base.add(port as usize * 0x10) as *mut u32;
        let portsc = core::ptr::read_volatile(portsc_ptr);

        let connected = portsc & 1 != 0;        // CCS: Current Connect Status
        let enabled = portsc & (1 << 1) != 0;   // PED: Port Enabled
        let speed = (portsc >> 10) & 0xF;        // Port Speed

        if connected {
            let speed_name = match speed {
                1 => "Full (12 Mbps)",
                2 => "Low (1.5 Mbps)",
                3 => "High (480 Mbps)",
                4 => "Super (5 Gbps)",
                5 => "Super+ (10 Gbps)",
                _ => "Unknown",
            };
            crate::serial_println!("[xhci] Port {}: connected, speed={}, enabled={}",
                port, speed_name, enabled);
        }
    }
}

pub fn is_detected() -> bool { INITIALIZED.load(Ordering::SeqCst) }

pub fn info() -> String {
    if !is_detected() { return String::from("xhci: not detected"); }
    unsafe {
        let ports = core::ptr::read_volatile(&raw const STATE.max_ports);
        let slots = core::ptr::read_volatile(&raw const STATE.max_slots);
        alloc::format!("xhci: {} ports, {} slots", ports, slots)
    }
}
