/// Virtio-net driver for QEMU Ethernet.
/// Legacy PCI I/O transport, single RX + TX queue pair.
/// Requires: -netdev user,id=n0 -device virtio-net-pci,netdev=n0

use crate::drivers::{pci, virtio};
use crate::memory::phys;
use alloc::vec;
use alloc::vec::Vec;
use x86_64::instructions::port::Port;
use core::sync::atomic::{AtomicBool, Ordering};

const REG_DEVICE_FEATURES: u16 = 0;
const REG_GUEST_FEATURES: u16 = 4;
const REG_QUEUE_ADDR: u16 = 8;
const REG_QUEUE_SIZE: u16 = 12;
const REG_QUEUE_SELECT: u16 = 14;
const REG_QUEUE_NOTIFY: u16 = 16;
const REG_DEVICE_STATUS: u16 = 18;
const REG_MAC: u16 = 0x14;

const QUEUE_SIZE: usize = 16;
const RX_BUF_SIZE: usize = 2048;
const NET_HDR_SIZE: usize = 10; // virtio_net_hdr

#[repr(C)]
#[derive(Clone, Copy)]
struct VqDesc { addr: u64, len: u32, flags: u16, next: u16 }

#[repr(C)]
struct VqAvail { flags: u16, idx: u16, ring: [u16; QUEUE_SIZE] }

#[repr(C)]
#[derive(Clone, Copy)]
struct VqUsedElem { id: u32, len: u32 }

#[repr(C)]
struct VqUsed { flags: u16, idx: u16, ring: [VqUsedElem; QUEUE_SIZE] }

struct NetDevice {
    io_base: u16,
    mac: [u8; 6],
    // RX queue (0)
    rx_descs: *mut VqDesc,
    rx_avail: *mut VqAvail,
    rx_used: *mut VqUsed,
    rx_bufs: [[u8; RX_BUF_SIZE]; QUEUE_SIZE],
    rx_last_used: u16,
    // TX queue (1)
    tx_descs: *mut VqDesc,
    tx_avail: *mut VqAvail,
    tx_used: *mut VqUsed,
    tx_last_used: u16,
}

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static mut DEVICE: NetDevice = NetDevice {
    io_base: 0, mac: [0; 6],
    rx_descs: core::ptr::null_mut(), rx_avail: core::ptr::null_mut(),
    rx_used: core::ptr::null_mut(), rx_bufs: [[0; RX_BUF_SIZE]; QUEUE_SIZE],
    rx_last_used: 0,
    tx_descs: core::ptr::null_mut(), tx_avail: core::ptr::null_mut(),
    tx_used: core::ptr::null_mut(), tx_last_used: 0,
};

fn alloc_vq() -> (*mut VqDesc, *mut VqAvail, *mut VqUsed, u64) {
    let size = core::mem::size_of::<VqDesc>() * QUEUE_SIZE
        + core::mem::size_of::<VqAvail>()
        + core::mem::size_of::<VqUsed>() + 4096;
    let mem = vec![0u8; size + 4096];
    let ptr = mem.as_ptr() as usize;
    let aligned = (ptr + 4095) & !4095;
    core::mem::forget(mem);

    let descs = aligned as *mut VqDesc;
    let avail = (aligned + core::mem::size_of::<VqDesc>() * QUEUE_SIZE) as *mut VqAvail;
    let used_off = aligned + core::mem::size_of::<VqDesc>() * QUEUE_SIZE
        + core::mem::size_of::<VqAvail>();
    let used = ((used_off + 4095) & !4095) as *mut VqUsed;
    let phys = aligned as u64 - phys::hhdm_offset();
    (descs, avail, used, phys)
}

pub fn init() {
    let devices = virtio::scan();
    let dev = match devices.iter().find(|d| d.device_type == virtio::VirtioDeviceType::Network) {
        Some(d) => d,
        None => { crate::serial_println!("[virtio-net] no device found"); return; }
    };
    crate::serial_println!("[virtio-net] found {}", dev.summary());

    let bar0 = pci::pci_read32(dev.pci.bus, dev.pci.device, dev.pci.function, 0x10);
    if bar0 & 1 == 0 { crate::serial_println!("[virtio-net] BAR0 not I/O"); return; }
    let io_base = (bar0 & 0xFFFC) as u16;

    unsafe {
        DEVICE.io_base = io_base;

        // Reset + negotiate
        write8(io_base, REG_DEVICE_STATUS, 0);
        write8(io_base, REG_DEVICE_STATUS, virtio::VIRTIO_STATUS_ACKNOWLEDGE);
        write8(io_base, REG_DEVICE_STATUS,
            virtio::VIRTIO_STATUS_ACKNOWLEDGE | virtio::VIRTIO_STATUS_DRIVER);
        write32(io_base, REG_GUEST_FEATURES, 0);

        // Read MAC
        for i in 0..6 {
            DEVICE.mac[i] = read8(io_base, REG_MAC + i as u16);
        }
        crate::serial_println!("[virtio-net] MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            DEVICE.mac[0], DEVICE.mac[1], DEVICE.mac[2],
            DEVICE.mac[3], DEVICE.mac[4], DEVICE.mac[5]);

        // Set up RX queue (0)
        write16(io_base, REG_QUEUE_SELECT, 0);
        let (rd, ra, ru, rp) = alloc_vq();
        DEVICE.rx_descs = rd; DEVICE.rx_avail = ra; DEVICE.rx_used = ru;
        write32(io_base, REG_QUEUE_ADDR, (rp / 4096) as u32);

        // Pre-populate RX descriptors
        for i in 0..QUEUE_SIZE {
            let buf_phys = (&DEVICE.rx_bufs[i] as *const _ as u64) - phys::hhdm_offset();
            (*rd.add(i)) = VqDesc {
                addr: buf_phys, len: RX_BUF_SIZE as u32,
                flags: virtio::VIRTQ_DESC_F_WRITE, next: 0,
            };
            (*ra).ring[i] = i as u16;
        }
        (*ra).idx = QUEUE_SIZE as u16;

        // Set up TX queue (1)
        write16(io_base, REG_QUEUE_SELECT, 1);
        let (td, ta, tu, tp) = alloc_vq();
        DEVICE.tx_descs = td; DEVICE.tx_avail = ta; DEVICE.tx_used = tu;
        write32(io_base, REG_QUEUE_ADDR, (tp / 4096) as u32);

        // Mark ready
        write8(io_base, REG_DEVICE_STATUS,
            virtio::VIRTIO_STATUS_ACKNOWLEDGE
            | virtio::VIRTIO_STATUS_DRIVER
            | virtio::VIRTIO_STATUS_DRIVER_OK);

        // Update NET state
        crate::net::NET.lock().mac = DEVICE.mac;

        INITIALIZED.store(true, Ordering::SeqCst);
        crate::serial_println!("[virtio-net] ready");
    }
}

pub fn send_frame(frame: &[u8]) -> Result<(), &'static str> {
    if !INITIALIZED.load(Ordering::SeqCst) { return Err("not init"); }
    unsafe {
        // Prepend virtio net header (10 bytes of zeros)
        let total = NET_HDR_SIZE + frame.len();
        let mut buf = vec![0u8; total];
        buf[NET_HDR_SIZE..].copy_from_slice(frame);

        let buf_phys = buf.as_ptr() as u64 - phys::hhdm_offset();
        let idx = (*DEVICE.tx_avail).idx as usize % QUEUE_SIZE;

        (*DEVICE.tx_descs.add(idx)) = VqDesc {
            addr: buf_phys, len: total as u32, flags: 0, next: 0,
        };
        (*DEVICE.tx_avail).ring[idx] = idx as u16;
        core::sync::atomic::fence(Ordering::SeqCst);
        (*DEVICE.tx_avail).idx = (*DEVICE.tx_avail).idx.wrapping_add(1);
        core::sync::atomic::fence(Ordering::SeqCst);

        write16(DEVICE.io_base, REG_QUEUE_NOTIFY, 1);
        core::mem::forget(buf); // keep buffer alive for DMA
    }
    Ok(())
}

pub fn recv_frame() -> Option<Vec<u8>> {
    if !INITIALIZED.load(Ordering::SeqCst) { return None; }
    unsafe {
        let used_idx = (*DEVICE.rx_used).idx;
        if used_idx == DEVICE.rx_last_used { return None; }

        let elem = (*DEVICE.rx_used).ring[DEVICE.rx_last_used as usize % QUEUE_SIZE];
        let desc_idx = elem.id as usize;
        let len = elem.len as usize;

        DEVICE.rx_last_used = DEVICE.rx_last_used.wrapping_add(1);

        // Skip the 10-byte virtio-net header
        if len <= NET_HDR_SIZE { return None; }
        let frame = DEVICE.rx_bufs[desc_idx][NET_HDR_SIZE..len].to_vec();

        // Re-arm the descriptor
        let buf_phys = (&DEVICE.rx_bufs[desc_idx] as *const _ as u64) - phys::hhdm_offset();
        (*DEVICE.rx_descs.add(desc_idx)) = VqDesc {
            addr: buf_phys, len: RX_BUF_SIZE as u32,
            flags: virtio::VIRTQ_DESC_F_WRITE, next: 0,
        };
        let avail_idx = (*DEVICE.rx_avail).idx as usize % QUEUE_SIZE;
        (*DEVICE.rx_avail).ring[avail_idx] = desc_idx as u16;
        (*DEVICE.rx_avail).idx = (*DEVICE.rx_avail).idx.wrapping_add(1);
        write16(DEVICE.io_base, REG_QUEUE_NOTIFY, 0);

        Some(frame)
    }
}

pub fn is_detected() -> bool { INITIALIZED.load(Ordering::SeqCst) }

unsafe fn read8(base: u16, off: u16) -> u8 { Port::<u8>::new(base + off).read() }
unsafe fn write8(base: u16, off: u16, v: u8) { Port::<u8>::new(base + off).write(v); }
unsafe fn write16(base: u16, off: u16, v: u16) { Port::<u16>::new(base + off).write(v); }
unsafe fn write32(base: u16, off: u16, v: u32) { Port::<u32>::new(base + off).write(v); }
