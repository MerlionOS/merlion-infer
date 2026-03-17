/// Virtio-blk driver for QEMU development.
/// Uses legacy PCI interface with BAR0 I/O ports.
/// Requires: -drive file=disk.img,format=raw,if=virtio

use crate::drivers::{pci, virtio};
use crate::memory::phys;
use alloc::string::String;
use alloc::vec;
use x86_64::instructions::port::Port;
use core::sync::atomic::{AtomicBool, Ordering};

const REG_DEVICE_FEATURES: u16 = 0;
const REG_GUEST_FEATURES: u16 = 4;
const REG_QUEUE_ADDR: u16 = 8;
const REG_QUEUE_SIZE: u16 = 12;
const REG_QUEUE_SELECT: u16 = 14;
const REG_QUEUE_NOTIFY: u16 = 16;
const REG_DEVICE_STATUS: u16 = 18;
const REG_BLK_CAPACITY: u16 = 0x14;

const QUEUE_SIZE: usize = 16;
const VIRTIO_BLK_T_IN: u32 = 0;

#[repr(C)]
struct VirtioBlkReqHeader {
    type_: u32,
    reserved: u32,
    sector: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VqDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

#[repr(C)]
struct VqAvail {
    flags: u16,
    idx: u16,
    ring: [u16; QUEUE_SIZE],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VqUsedElem {
    id: u32,
    len: u32,
}

#[repr(C)]
struct VqUsed {
    flags: u16,
    idx: u16,
    ring: [VqUsedElem; QUEUE_SIZE],
}

struct BlkDevice {
    io_base: u16,
    capacity: u64,
    descs: *mut VqDesc,
    avail: *mut VqAvail,
    used: *mut VqUsed,
    last_used_idx: u16,
}

static INITIALIZED: AtomicBool = AtomicBool::new(false);

static mut DEVICE: BlkDevice = BlkDevice {
    io_base: 0,
    capacity: 0,
    descs: core::ptr::null_mut(),
    avail: core::ptr::null_mut(),
    used: core::ptr::null_mut(),
    last_used_idx: 0,
};

pub fn init() {
    let devices = virtio::scan();
    let dev = match devices.iter().find(|d| d.device_type == virtio::VirtioDeviceType::Block) {
        Some(d) => d,
        None => { crate::serial_println!("[virtio-blk] no device found"); return; }
    };
    crate::serial_println!("[virtio-blk] found {}", dev.summary());

    let bar0 = pci::pci_read32(dev.pci.bus, dev.pci.device, dev.pci.function, 0x10);
    if bar0 & 1 == 0 {
        crate::serial_println!("[virtio-blk] BAR0 is MMIO, need I/O port"); return;
    }
    let io_base = (bar0 & 0xFFFC) as u16;

    unsafe {
        DEVICE.io_base = io_base;

        write_reg8(io_base, REG_DEVICE_STATUS, 0);
        write_reg8(io_base, REG_DEVICE_STATUS, virtio::VIRTIO_STATUS_ACKNOWLEDGE);
        write_reg8(io_base, REG_DEVICE_STATUS,
            virtio::VIRTIO_STATUS_ACKNOWLEDGE | virtio::VIRTIO_STATUS_DRIVER);

        let _features = read_reg32(io_base, REG_DEVICE_FEATURES);
        write_reg32(io_base, REG_GUEST_FEATURES, 0);

        let cap_lo = read_reg32(io_base, REG_BLK_CAPACITY) as u64;
        let cap_hi = read_reg32(io_base, REG_BLK_CAPACITY + 4) as u64;
        DEVICE.capacity = cap_lo | (cap_hi << 32);
        let cap = core::ptr::read_volatile(&raw const DEVICE.capacity);
        crate::serial_println!("[virtio-blk] capacity: {} sectors ({} KiB)", cap, cap / 2);

        write_reg16(io_base, REG_QUEUE_SELECT, 0);
        let queue_size_max = read_reg16(io_base, REG_QUEUE_SIZE);
        if queue_size_max == 0 {
            crate::serial_println!("[virtio-blk] queue not available"); return;
        }

        // Allocate virtqueue memory
        let vq_size = core::mem::size_of::<VqDesc>() * QUEUE_SIZE
            + core::mem::size_of::<VqAvail>()
            + core::mem::size_of::<VqUsed>();
        let vq_mem = vec![0u8; vq_size + 4096];
        let vq_ptr = vq_mem.as_ptr() as usize;
        let vq_aligned = (vq_ptr + 4095) & !4095;
        core::mem::forget(vq_mem);

        DEVICE.descs = vq_aligned as *mut VqDesc;
        DEVICE.avail = (vq_aligned + core::mem::size_of::<VqDesc>() * QUEUE_SIZE) as *mut VqAvail;
        let used_offset = vq_aligned + core::mem::size_of::<VqDesc>() * QUEUE_SIZE
            + core::mem::size_of::<VqAvail>();
        let used_aligned = (used_offset + 4095) & !4095;
        DEVICE.used = used_aligned as *mut VqUsed;

        let phys_addr = vq_aligned as u64 - phys::hhdm_offset();
        write_reg32(io_base, REG_QUEUE_ADDR, (phys_addr / 4096) as u32);

        write_reg8(io_base, REG_DEVICE_STATUS,
            virtio::VIRTIO_STATUS_ACKNOWLEDGE
            | virtio::VIRTIO_STATUS_DRIVER
            | virtio::VIRTIO_STATUS_DRIVER_OK);

        INITIALIZED.store(true, Ordering::SeqCst);
    }

    crate::serial_println!("[virtio-blk] ready");
}

pub fn read_sector(sector: u64, buf: &mut [u8; 512]) -> Result<(), &'static str> {
    if !INITIALIZED.load(Ordering::SeqCst) { return Err("virtio-blk: not initialized"); }

    unsafe {
        let io_base = DEVICE.io_base;
        let header = VirtioBlkReqHeader { type_: VIRTIO_BLK_T_IN, reserved: 0, sector };
        let status: u8 = 0xFF;

        let header_phys = &header as *const _ as u64 - phys::hhdm_offset();
        let data_phys = buf.as_ptr() as u64 - phys::hhdm_offset();
        let status_phys = &status as *const _ as u64 - phys::hhdm_offset();

        (*DEVICE.descs.add(0)) = VqDesc {
            addr: header_phys,
            len: core::mem::size_of::<VirtioBlkReqHeader>() as u32,
            flags: virtio::VIRTQ_DESC_F_NEXT, next: 1,
        };
        (*DEVICE.descs.add(1)) = VqDesc {
            addr: data_phys, len: 512,
            flags: virtio::VIRTQ_DESC_F_NEXT | virtio::VIRTQ_DESC_F_WRITE, next: 2,
        };
        (*DEVICE.descs.add(2)) = VqDesc {
            addr: status_phys, len: 1,
            flags: virtio::VIRTQ_DESC_F_WRITE, next: 0,
        };

        let avail = &mut *DEVICE.avail;
        let avail_idx = avail.idx;
        avail.ring[(avail_idx as usize) % QUEUE_SIZE] = 0;
        core::sync::atomic::fence(Ordering::SeqCst);
        avail.idx = avail_idx.wrapping_add(1);
        core::sync::atomic::fence(Ordering::SeqCst);

        write_reg16(io_base, REG_QUEUE_NOTIFY, 0);

        // Poll for completion
        for _ in 0..10_000_000u32 {
            let used = &*DEVICE.used;
            if used.idx != DEVICE.last_used_idx {
                DEVICE.last_used_idx = used.idx;
                return if status == 0 { Ok(()) } else { Err("virtio-blk: I/O error") };
            }
            core::hint::spin_loop();
        }
        Err("virtio-blk: timeout")
    }
}

/// Read multiple sectors.
pub fn read_sectors(start_sector: u64, buf: &mut [u8]) -> Result<usize, &'static str> {
    let total = buf.len() / 512;
    let mut sector_buf = [0u8; 512];
    for i in 0..total {
        read_sector(start_sector + i as u64, &mut sector_buf)?;
        let off = i * 512;
        buf[off..off + 512].copy_from_slice(&sector_buf);
    }
    Ok(total * 512)
}

pub fn is_detected() -> bool { INITIALIZED.load(Ordering::SeqCst) }
pub fn capacity() -> u64 { unsafe { DEVICE.capacity } }

pub fn info() -> String {
    if is_detected() {
        alloc::format!("virtio-blk: {} sectors ({} KiB)", capacity(), capacity() / 2)
    } else {
        String::from("virtio-blk: not detected")
    }
}

unsafe fn read_reg32(base: u16, offset: u16) -> u32 { Port::<u32>::new(base + offset).read() }
unsafe fn read_reg16(base: u16, offset: u16) -> u16 { Port::<u16>::new(base + offset).read() }
unsafe fn write_reg32(base: u16, offset: u16, val: u32) { Port::<u32>::new(base + offset).write(val); }
unsafe fn write_reg16(base: u16, offset: u16, val: u16) { Port::<u16>::new(base + offset).write(val); }
unsafe fn write_reg8(base: u16, offset: u16, val: u8) { Port::<u8>::new(base + offset).write(val); }
