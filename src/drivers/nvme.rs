/// NVMe storage driver.
/// Discovers NVMe controller via PCI, creates admin + I/O queue pairs,
/// provides sector-level and multi-sector read for model loading.

use crate::drivers::pci;
use crate::memory::phys;
use alloc::string::String;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};

const NVME_CLASS: u8 = 0x01;
const NVME_SUBCLASS: u8 = 0x08;
const NVME_PROG_IF: u8 = 0x02;

const SECTOR_SIZE: usize = 512;
const DEFAULT_ADMIN_QUEUE_DEPTH: usize = 16;
const DEFAULT_IO_QUEUE_DEPTH: usize = 64;
const PAGE_SIZE: usize = 4096;

static ADMIN_QUEUE_DEPTH: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(DEFAULT_ADMIN_QUEUE_DEPTH);
static IO_QUEUE_DEPTH: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(DEFAULT_IO_QUEUE_DEPTH);
static IS_APPLE: AtomicBool = AtomicBool::new(false);

const ADMIN_OPC_IDENTIFY: u8 = 0x06;
const ADMIN_OPC_CREATE_IO_CQ: u8 = 0x05;
const ADMIN_OPC_CREATE_IO_SQ: u8 = 0x01;

const IO_OPC_READ: u8 = 0x02;

const CC_EN: u32 = 1 << 0;
const CSTS_RDY: u32 = 1 << 0;

#[repr(C)]
pub struct NvmeRegs {
    pub cap: u64,
    pub vs: u32,
    pub intms: u32,
    pub intmc: u32,
    pub cc: u32,
    _rsvd0: u32,
    pub csts: u32,
    pub nssr: u32,
    pub aqa: u32,
    pub asq: u64,
    pub acq: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvmeSqe {
    pub cdw0: u32,
    pub nsid: u32,
    pub cdw2: u32,
    pub cdw3: u32,
    pub mptr: u64,
    pub prp1: u64,
    pub prp2: u64,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

impl NvmeSqe {
    const fn zeroed() -> Self {
        Self {
            cdw0: 0, nsid: 0, cdw2: 0, cdw3: 0,
            mptr: 0, prp1: 0, prp2: 0,
            cdw10: 0, cdw11: 0, cdw12: 0,
            cdw13: 0, cdw14: 0, cdw15: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvmeCqe {
    pub dw0: u32,
    pub dw1: u32,
    pub sq_head_sqid: u32,
    pub status_cid: u32,
}

impl NvmeCqe {
    pub fn phase(&self) -> bool { self.status_cid & (1 << 16) != 0 }
    pub fn status_code(&self) -> u16 { ((self.status_cid >> 17) & 0x7FF) as u16 }
}

static INITIALIZED: AtomicBool = AtomicBool::new(false);

struct NvmeState {
    regs: *mut NvmeRegs,
    dstrd: usize,
    admin_sq: *mut NvmeSqe,
    admin_cq: *mut NvmeCqe,
    admin_sq_tail: u16,
    admin_cq_head: u16,
    admin_cq_phase: bool,
    io_sq: *mut NvmeSqe,
    io_cq: *mut NvmeCqe,
    io_sq_tail: u16,
    io_cq_head: u16,
    io_cq_phase: bool,
    next_cid: u16,
    ns_blocks: u64,
    model: [u8; 40],
}

unsafe impl Send for NvmeState {}
unsafe impl Sync for NvmeState {}

static mut STATE: NvmeState = NvmeState {
    regs: core::ptr::null_mut(),
    dstrd: 0,
    admin_sq: core::ptr::null_mut(),
    admin_cq: core::ptr::null_mut(),
    admin_sq_tail: 0,
    admin_cq_head: 0,
    admin_cq_phase: true,
    io_sq: core::ptr::null_mut(),
    io_cq: core::ptr::null_mut(),
    io_sq_tail: 0,
    io_cq_head: 0,
    io_cq_phase: true,
    next_cid: 1,
    ns_blocks: 0,
    model: [0u8; 40],
};

unsafe fn ring_sq_doorbell(qid: u16, tail: u16) {
    let base = STATE.regs as *mut u8;
    let offset = 0x1000 + (2 * qid as usize) * STATE.dstrd;
    ptr::write_volatile(base.add(offset) as *mut u32, tail as u32);
}

unsafe fn ring_cq_doorbell(qid: u16, head: u16) {
    let base = STATE.regs as *mut u8;
    let offset = 0x1000 + (2 * qid as usize + 1) * STATE.dstrd;
    ptr::write_volatile(base.add(offset) as *mut u32, head as u32);
}

unsafe fn alloc_cid() -> u16 {
    let cid = STATE.next_cid;
    STATE.next_cid = STATE.next_cid.wrapping_add(1);
    if STATE.next_cid == 0 { STATE.next_cid = 1; }
    cid
}

fn admin_qd() -> usize { ADMIN_QUEUE_DEPTH.load(Ordering::Relaxed) }
fn io_qd() -> usize { IO_QUEUE_DEPTH.load(Ordering::Relaxed) }

unsafe fn admin_submit_and_wait(cmd: &NvmeSqe) -> Result<NvmeCqe, &'static str> {
    let qd = admin_qd();
    let idx = STATE.admin_sq_tail as usize;
    ptr::write_volatile(STATE.admin_sq.add(idx), *cmd);
    STATE.admin_sq_tail = ((idx + 1) % qd) as u16;
    ring_sq_doorbell(0, STATE.admin_sq_tail);

    for _ in 0..10_000_000u32 {
        let cqe = ptr::read_volatile(STATE.admin_cq.add(STATE.admin_cq_head as usize));
        if cqe.phase() == STATE.admin_cq_phase {
            STATE.admin_cq_head += 1;
            if STATE.admin_cq_head as usize >= qd {
                STATE.admin_cq_head = 0;
                STATE.admin_cq_phase = !STATE.admin_cq_phase;
            }
            ring_cq_doorbell(0, STATE.admin_cq_head);
            if cqe.status_code() != 0 { return Err("nvme: admin command failed"); }
            return Ok(cqe);
        }
        core::hint::spin_loop();
    }
    Err("nvme: admin command timeout")
}

unsafe fn io_submit_and_wait(cmd: &NvmeSqe) -> Result<NvmeCqe, &'static str> {
    let qd = io_qd();
    let idx = STATE.io_sq_tail as usize;
    ptr::write_volatile(STATE.io_sq.add(idx), *cmd);
    STATE.io_sq_tail = ((idx + 1) % qd) as u16;
    ring_sq_doorbell(1, STATE.io_sq_tail);

    for _ in 0..10_000_000u32 {
        let cqe = ptr::read_volatile(STATE.io_cq.add(STATE.io_cq_head as usize));
        if cqe.phase() == STATE.io_cq_phase {
            STATE.io_cq_head += 1;
            if STATE.io_cq_head as usize >= qd {
                STATE.io_cq_head = 0;
                STATE.io_cq_phase = !STATE.io_cq_phase;
            }
            ring_cq_doorbell(1, STATE.io_cq_head);
            if cqe.status_code() != 0 { return Err("nvme: I/O command failed"); }
            return Ok(cqe);
        }
        core::hint::spin_loop();
    }
    Err("nvme: I/O command timeout")
}

fn alloc_zeroed_frame() -> Option<(*mut u8, u64)> {
    use x86_64::structures::paging::FrameAllocator;
    let frame = phys::BumpAllocator.allocate_frame()?;
    let phys_addr = frame.start_address().as_u64();
    let virt = phys::phys_to_virt(frame.start_address());
    unsafe { ptr::write_bytes(virt.as_mut_ptr::<u8>(), 0, PAGE_SIZE); }
    Some((virt.as_mut_ptr(), phys_addr))
}

/// Initialize with Apple NVMe quirks applied.
pub fn init_with_quirks(quirks: &crate::drivers::apple_nvme::AppleNvmeQuirks) {
    if quirks.is_apple {
        IS_APPLE.store(true, Ordering::SeqCst);
        let max_qe = quirks.max_queue_entries as usize;
        if max_qe < DEFAULT_IO_QUEUE_DEPTH {
            IO_QUEUE_DEPTH.store(max_qe, Ordering::SeqCst);
            crate::serial_println!("[nvme] Apple quirk: IO queue depth reduced to {}", max_qe);
        }
    }
    init();
}

pub fn init() {
    let devices = pci::scan();
    let dev = match devices.iter().find(|d| {
        d.class == NVME_CLASS && d.subclass == NVME_SUBCLASS && d.prog_if == NVME_PROG_IF
    }) {
        Some(d) => d.clone(),
        None => { crate::serial_println!("[nvme] no controller found"); return; }
    };
    crate::serial_println!("[nvme] found {}", dev.summary());

    // Enable bus-master + memory-space
    let cmd_reg = pci::pci_read32(dev.bus, dev.device, dev.function, 0x04);
    pci::pci_write32(dev.bus, dev.device, dev.function, 0x04, cmd_reg | 0x06);

    // Read BAR0 (64-bit MMIO)
    let bar0_lo = pci::pci_read32(dev.bus, dev.device, dev.function, 0x10);
    let bar0_hi = pci::pci_read32(dev.bus, dev.device, dev.function, 0x14);
    if bar0_lo & 0x1 != 0 {
        crate::serial_println!("[nvme] BAR0 is I/O, expected MMIO"); return;
    }
    let bar0_phys = ((bar0_hi as u64) << 32) | ((bar0_lo & 0xFFFF_FFF0) as u64);
    if bar0_phys == 0 { crate::serial_println!("[nvme] BAR0 is zero"); return; }

    let regs = phys::phys_to_virt(x86_64::PhysAddr::new(bar0_phys)).as_mut_ptr() as *mut NvmeRegs;

    unsafe {
        let cap = ptr::read_volatile(&(*regs).cap);
        let dstrd = 4usize << ((cap >> 32) & 0xF);
        let vs = ptr::read_volatile(&(*regs).vs);
        crate::serial_println!("[nvme] v{}.{}.{}, dstrd={}",
            (vs >> 16) & 0xFF, (vs >> 8) & 0xFF, vs & 0xFF, dstrd);

        STATE.regs = regs;
        STATE.dstrd = dstrd;

        // Disable controller
        let mut cc = ptr::read_volatile(&(*regs).cc);
        cc &= !CC_EN;
        ptr::write_volatile(&mut (*regs).cc, cc);
        for _ in 0..10_000_000u32 {
            if ptr::read_volatile(&(*regs).csts) & CSTS_RDY == 0 { break; }
            core::hint::spin_loop();
        }

        // Admin queues
        let (asq_virt, asq_phys) = alloc_zeroed_frame().expect("nvme: alloc admin SQ");
        let (acq_virt, acq_phys) = alloc_zeroed_frame().expect("nvme: alloc admin CQ");
        STATE.admin_sq = asq_virt as *mut NvmeSqe;
        STATE.admin_cq = acq_virt as *mut NvmeCqe;
        STATE.admin_sq_tail = 0;
        STATE.admin_cq_head = 0;
        STATE.admin_cq_phase = true;

        let aqd = admin_qd();
        let aqa = ((aqd as u32 - 1) << 16) | (aqd as u32 - 1);
        ptr::write_volatile(&mut (*regs).aqa, aqa);
        ptr::write_volatile(&mut (*regs).asq, asq_phys);
        ptr::write_volatile(&mut (*regs).acq, acq_phys);

        // Enable controller
        let cc_val: u32 = (6 << 16) | (4 << 20) | CC_EN;
        ptr::write_volatile(&mut (*regs).cc, cc_val);
        for _ in 0..10_000_000u32 {
            if ptr::read_volatile(&(*regs).csts) & CSTS_RDY != 0 { break; }
            core::hint::spin_loop();
        }
        if ptr::read_volatile(&(*regs).csts) & CSTS_RDY == 0 {
            crate::serial_println!("[nvme] controller failed to become ready"); return;
        }
        crate::serial_println!("[nvme] controller ready");

        // Identify Controller
        let (id_virt, id_phys) = alloc_zeroed_frame().expect("nvme: alloc identify");
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (ADMIN_OPC_IDENTIFY as u32) | ((alloc_cid() as u32) << 16);
        cmd.prp1 = id_phys;
        cmd.cdw10 = 1;
        if admin_submit_and_wait(&cmd).is_err() {
            crate::serial_println!("[nvme] Identify Controller failed"); return;
        }
        ptr::copy_nonoverlapping(id_virt.add(24), (&raw mut STATE.model).cast::<u8>(), 40);
        let model_str = core::str::from_utf8(core::slice::from_raw_parts(
            (&raw const STATE.model).cast::<u8>(), 40)).unwrap_or("?").trim();
        crate::serial_println!("[nvme] model: {}", model_str);

        // Identify Namespace 1
        ptr::write_bytes(id_virt, 0, PAGE_SIZE);
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (ADMIN_OPC_IDENTIFY as u32) | ((alloc_cid() as u32) << 16);
        cmd.nsid = 1;
        cmd.prp1 = id_phys;
        cmd.cdw10 = 0;
        if admin_submit_and_wait(&cmd).is_err() {
            crate::serial_println!("[nvme] Identify Namespace failed"); return;
        }
        let nsze = ptr::read_unaligned(id_virt as *const u64);
        STATE.ns_blocks = nsze;
        crate::serial_println!("[nvme] ns1: {} sectors ({} MiB)",
            nsze, nsze * SECTOR_SIZE as u64 / (1024 * 1024));

        // Create I/O CQ
        let ioqd = io_qd();
        let (iocq_virt, iocq_phys) = alloc_zeroed_frame().expect("nvme: alloc I/O CQ");
        STATE.io_cq = iocq_virt as *mut NvmeCqe;
        STATE.io_cq_head = 0;
        STATE.io_cq_phase = true;
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (ADMIN_OPC_CREATE_IO_CQ as u32) | ((alloc_cid() as u32) << 16);
        cmd.prp1 = iocq_phys;
        cmd.cdw10 = ((ioqd as u32 - 1) << 16) | 1;
        cmd.cdw11 = 1;
        if admin_submit_and_wait(&cmd).is_err() {
            crate::serial_println!("[nvme] Create I/O CQ failed"); return;
        }

        // Create I/O SQ
        let (iosq_virt, iosq_phys) = alloc_zeroed_frame().expect("nvme: alloc I/O SQ");
        STATE.io_sq = iosq_virt as *mut NvmeSqe;
        STATE.io_sq_tail = 0;
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (ADMIN_OPC_CREATE_IO_SQ as u32) | ((alloc_cid() as u32) << 16);
        cmd.prp1 = iosq_phys;
        cmd.cdw10 = ((ioqd as u32 - 1) << 16) | 1;
        cmd.cdw11 = (1 << 16) | 1;
        if admin_submit_and_wait(&cmd).is_err() {
            crate::serial_println!("[nvme] Create I/O SQ failed"); return;
        }

        crate::serial_println!("[nvme] I/O queues ready (depth {})", ioqd);
        INITIALIZED.store(true, Ordering::SeqCst);
    }
}

/// Maximum sectors per NVMe read command (limited by 4 KiB bounce buffer).
const MAX_SECTORS_PER_READ: usize = PAGE_SIZE / SECTOR_SIZE; // 8

/// Read a single 512-byte sector.
pub fn read_sector(lba: u64, buf: &mut [u8; 512]) -> Result<(), &'static str> {
    if !INITIALIZED.load(Ordering::SeqCst) { return Err("nvme: not initialized"); }
    let (bounce_virt, bounce_phys) = alloc_zeroed_frame().ok_or("nvme: alloc failed")?;
    unsafe {
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (IO_OPC_READ as u32) | ((alloc_cid() as u32) << 16);
        cmd.nsid = 1;
        cmd.prp1 = bounce_phys;
        cmd.cdw10 = lba as u32;
        cmd.cdw11 = (lba >> 32) as u32;
        cmd.cdw12 = 0; // 1 sector (0-based count)
        io_submit_and_wait(&cmd)?;
        ptr::copy_nonoverlapping(bounce_virt, buf.as_mut_ptr(), SECTOR_SIZE);
    }
    Ok(())
}

/// Read multiple sectors (up to 8) in a single NVMe command using a 4 KiB bounce buffer.
/// Returns bytes read.
fn read_multi(start_lba: u64, buf: &mut [u8], n_sectors: usize) -> Result<usize, &'static str> {
    if n_sectors == 0 || n_sectors > MAX_SECTORS_PER_READ {
        return Err("nvme: invalid sector count");
    }
    let (bounce_virt, bounce_phys) = alloc_zeroed_frame().ok_or("nvme: alloc failed")?;
    let bytes = n_sectors * SECTOR_SIZE;
    unsafe {
        let mut cmd = NvmeSqe::zeroed();
        cmd.cdw0 = (IO_OPC_READ as u32) | ((alloc_cid() as u32) << 16);
        cmd.nsid = 1;
        cmd.prp1 = bounce_phys;
        cmd.cdw10 = start_lba as u32;
        cmd.cdw11 = (start_lba >> 32) as u32;
        cmd.cdw12 = (n_sectors as u32) - 1; // 0-based count
        io_submit_and_wait(&cmd)?;
        ptr::copy_nonoverlapping(bounce_virt, buf.as_mut_ptr(), bytes);
    }
    Ok(bytes)
}

/// Read multiple sectors into a buffer using multi-sector DMA (8x faster).
/// Falls back to single-sector reads for the remainder.
pub fn read_sectors(start_lba: u64, buf: &mut [u8]) -> Result<usize, &'static str> {
    if !INITIALIZED.load(Ordering::SeqCst) { return Err("nvme: not initialized"); }
    let total_sectors = buf.len() / SECTOR_SIZE;
    let mut offset = 0usize;
    let mut lba = start_lba;
    let mut remaining = total_sectors;

    // Read in chunks of MAX_SECTORS_PER_READ (8 sectors = 4 KiB per command)
    while remaining >= MAX_SECTORS_PER_READ {
        let bytes = read_multi(lba, &mut buf[offset..], MAX_SECTORS_PER_READ)?;
        offset += bytes;
        lba += MAX_SECTORS_PER_READ as u64;
        remaining -= MAX_SECTORS_PER_READ;
    }

    // Read remaining sectors one chunk at a time
    if remaining > 0 {
        let bytes = read_multi(lba, &mut buf[offset..], remaining)?;
        offset += bytes;
    }

    Ok(offset)
}

pub fn is_detected() -> bool { INITIALIZED.load(Ordering::SeqCst) }
pub fn is_apple() -> bool { IS_APPLE.load(Ordering::SeqCst) }

pub fn regs_base() -> *mut u8 {
    unsafe { core::ptr::read_volatile(&raw const STATE.regs) as *mut u8 }
}

pub fn capacity_sectors() -> u64 {
    unsafe { ptr::read_volatile(&raw const STATE.ns_blocks) }
}

pub fn info() -> String {
    if !is_detected() { return String::from("nvme: not detected"); }
    unsafe {
        let model = core::str::from_utf8(core::slice::from_raw_parts(
            (&raw const STATE.model).cast::<u8>(), 40)).unwrap_or("?").trim();
        let nsze = ptr::read_volatile(&raw const STATE.ns_blocks);
        alloc::format!("nvme: \"{}\", {} MiB", model, nsze * SECTOR_SIZE as u64 / (1024 * 1024))
    }
}
