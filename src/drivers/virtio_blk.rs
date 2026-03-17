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

        // Allocate VQ memory from the frame allocator. We need the physical address
        // known, and the layout must match what virtio legacy expects:
        //   PFN * 4096 + 0:     descriptor table (16 * queue_size bytes)
        //   PFN * 4096 + desc:  available ring (6 + 2 * queue_size bytes)
        //   align(above, 4096): used ring
        // With QUEUE_SIZE=16: desc=256, avail=38, total=294 → used at offset 4096
        // So we need 2 contiguous pages.
        //
        // The bump allocator gives sequential frames from a contiguous region,
        // so allocating 2 frames in sequence gives contiguous physical pages.
        use x86_64::structures::paging::FrameAllocator;
        let f0 = phys::BumpAllocator.allocate_frame().expect("vq alloc 0");
        let f1 = phys::BumpAllocator.allocate_frame().expect("vq alloc 1");
        let phys_addr = f0.start_address().as_u64();
        let virt0 = phys::phys_to_virt(f0.start_address()).as_u64() as usize;
        let virt1 = phys::phys_to_virt(f1.start_address()).as_u64() as usize;
        core::ptr::write_bytes(virt0 as *mut u8, 0, 4096);
        core::ptr::write_bytes(virt1 as *mut u8, 0, 4096);

        DEVICE.descs = virt0 as *mut VqDesc;
        DEVICE.avail = (virt0 + 256) as *mut VqAvail; // after 16 descriptors * 16 bytes
        DEVICE.used = virt1 as *mut VqUsed; // page-aligned at next page

        // Tell the device: queue size and PFN
        write_reg16(io_base, REG_QUEUE_SIZE, QUEUE_SIZE as u16);
        write_reg32(io_base, REG_QUEUE_ADDR, (phys_addr / 4096) as u32);

        // Set DRIVER_OK to indicate we're ready
        write_reg8(io_base, REG_DEVICE_STATUS,
            virtio::VIRTIO_STATUS_ACKNOWLEDGE
            | virtio::VIRTIO_STATUS_DRIVER
            | virtio::VIRTIO_STATUS_DRIVER_OK);

        let dev_status = read_reg8(io_base, REG_DEVICE_STATUS);
        crate::serial_println!("[virtio-blk] VQ phys={:#x} PFN={:#x} status={:#x}",
            phys_addr, phys_addr / 4096, dev_status);

        INITIALIZED.store(true, Ordering::SeqCst);
    }

    crate::serial_println!("[virtio-blk] ready");
}

pub fn read_sector(sector: u64, buf: &mut [u8; 512]) -> Result<(), &'static str> {
    if !INITIALIZED.load(Ordering::SeqCst) { return Err("virtio-blk: not initialized"); }

    // Use a physical frame for DMA bounce buffer.
    use x86_64::structures::paging::FrameAllocator;
    let frame = phys::BumpAllocator.allocate_frame().ok_or("virtio-blk: alloc failed")?;
    let page_phys = frame.start_address().as_u64();
    let page_virt = phys::phys_to_virt(frame.start_address()).as_mut_ptr::<u8>();

    unsafe {
        core::ptr::write_bytes(page_virt, 0, 4096);

        // Write request header at offset 0
        let header_ptr = page_virt as *mut VirtioBlkReqHeader;
        (*header_ptr).type_ = VIRTIO_BLK_T_IN;
        (*header_ptr).reserved = 0;
        (*header_ptr).sector = sector;

        let io_base = DEVICE.io_base;

        // Descriptor 0: header (device reads)
        (*DEVICE.descs.add(0)) = VqDesc {
            addr: page_phys,
            len: 16,
            flags: virtio::VIRTQ_DESC_F_NEXT, next: 1,
        };
        // Descriptor 1: data (device writes)
        (*DEVICE.descs.add(1)) = VqDesc {
            addr: page_phys + 0x200,
            len: 512,
            flags: virtio::VIRTQ_DESC_F_NEXT | virtio::VIRTQ_DESC_F_WRITE, next: 2,
        };
        // Descriptor 2: status (device writes)
        (*DEVICE.descs.add(2)) = VqDesc {
            addr: page_phys + 0x400,
            len: 1,
            flags: virtio::VIRTQ_DESC_F_WRITE, next: 0,
        };

        // Submit to available ring
        let avail = &mut *DEVICE.avail;
        let avail_idx = avail.idx;
        avail.ring[(avail_idx as usize) % QUEUE_SIZE] = 0;
        core::sync::atomic::fence(Ordering::SeqCst);
        avail.idx = avail_idx.wrapping_add(1);
        core::sync::atomic::fence(Ordering::SeqCst);

        // Notify device
        write_reg16(io_base, REG_QUEUE_NOTIFY, 0);

        let before_used = core::ptr::read_volatile(&(*DEVICE.used).idx);
        crate::serial_println!("[virtio-blk] submitted read sector {}, avail.idx={}, used.idx={}, last_used={}",
            sector, (*DEVICE.avail).idx, before_used, DEVICE.last_used_idx);

        // Poll for completion
        for _ in 0..50_000_000u32 {
            core::sync::atomic::fence(Ordering::SeqCst);
            let used_idx = core::ptr::read_volatile(&(*DEVICE.used).idx);
            if used_idx != DEVICE.last_used_idx {
                DEVICE.last_used_idx = used_idx;
                let status_byte = core::ptr::read_volatile(page_virt.add(0x400));
                core::ptr::copy_nonoverlapping(page_virt.add(0x200), buf.as_mut_ptr(), 512);
                return if status_byte == 0 { Ok(()) } else { Err("virtio-blk: I/O error") };
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

unsafe fn read_reg8(base: u16, offset: u16) -> u8 { Port::<u8>::new(base + offset).read() }
unsafe fn read_reg32(base: u16, offset: u16) -> u32 { Port::<u32>::new(base + offset).read() }
unsafe fn read_reg16(base: u16, offset: u16) -> u16 { Port::<u16>::new(base + offset).read() }
unsafe fn write_reg32(base: u16, offset: u16, val: u32) { Port::<u32>::new(base + offset).write(val); }
unsafe fn write_reg16(base: u16, offset: u16, val: u16) { Port::<u16>::new(base + offset).write(val); }
unsafe fn write_reg8(base: u16, offset: u16, val: u8) { Port::<u8>::new(base + offset).write(val); }
