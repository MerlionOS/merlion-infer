#![no_std]
#![no_main]

extern crate alloc;

use core::panic::PanicInfo;
use merlion_infer::boot::limine::*;

// ---------------------------------------------------------------------------
// Limine request markers and requests
// ---------------------------------------------------------------------------

#[used]
#[link_section = ".limine_requests_start"]
static mut REQUESTS_START_MARKER: [u64; 4] = [
    0xf6b8f4b39de7d1ae, 0xfab91a6940fcb9cf,
    0x785c6ed015d3e316, 0x181e920a7852b9d9,
];

#[used]
#[link_section = ".limine_requests"]
static mut BASE_REVISION: [u64; 3] = [0xf9562b2d5c95a6c8, 0x6a7b384944536bdc, 2];

#[used]
#[link_section = ".limine_requests"]
static mut MEMMAP_REQUEST: LimineMemmapRequest = LimineMemmapRequest {
    id: [0xc7b1dd30df4c8b88, 0x0a82e883a194f07b, 0x67cf3d9d378a806f, 0xe304acdfc50c3c62],
    revision: 0,
    response: core::ptr::null(),
};

#[used]
#[link_section = ".limine_requests"]
static mut HHDM_REQUEST: LimineHhdmRequest = LimineHhdmRequest {
    id: [0xc7b1dd30df4c8b88, 0x0a82e883a194f07b, 0x48dcf1cb8ad2b852, 0x63984e959a98244b],
    revision: 0,
    response: core::ptr::null(),
};

#[used]
#[link_section = ".limine_requests_end"]
static mut REQUESTS_END_MARKER: [u64; 2] = [0xadc0e0531bb10d03, 0x9572709f31764c62];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[no_mangle]
extern "C" fn _start() -> ! {
    // Ultra-early serial: write directly to COM1 before any init
    unsafe {
        use x86_64::instructions::port::Port;
        Port::<u8>::new(0x3F9).write(0x00);
        Port::<u8>::new(0x3FB).write(0x80);
        Port::<u8>::new(0x3F8).write(0x01); // 115200 baud
        Port::<u8>::new(0x3F9).write(0x00);
        Port::<u8>::new(0x3FB).write(0x03); // 8N1
        Port::<u8>::new(0x3FA).write(0xC7); // FIFO
        for &b in b"[boot] _start reached\r\n" {
            while Port::<u8>::new(0x3FD).read() & 0x20 == 0 {}
            Port::<u8>::new(0x3F8).write(b);
        }
    }

    // Phase 1: Serial
    merlion_infer::arch::x86_64::serial::SERIAL1.lock().init();
    merlion_infer::serial_println!("MerlionOS Inference v0.1.0");
    merlion_infer::serial_println!("Zero overhead. Maximum throughput.");
    merlion_infer::serial_println!("[boot] Limine UEFI boot path");

    // Phase 2: HHDM offset
    let hhdm_offset = unsafe {
        let resp = (*(&raw const HHDM_REQUEST)).response;
        if resp.is_null() {
            merlion_infer::serial_println!("[boot] ERROR: no HHDM response");
            halt();
        }
        let offset = (*resp).offset;
        merlion_infer::serial_println!("[boot] HHDM offset: {:#x}", offset);
        offset
    };

    // Phase 3: Memory map — find largest usable region
    unsafe {
        let resp = (*(&raw const MEMMAP_REQUEST)).response;
        if resp.is_null() {
            merlion_infer::serial_println!("[boot] ERROR: no memory map response");
            halt();
        }
        let count = (*resp).entry_count as usize;
        let entries = (*resp).entries;
        merlion_infer::serial_println!("[boot] Memory map: {} entries", count);

        let mut best_base: u64 = 0;
        let mut best_len: u64 = 0;
        let mut total_usable: u64 = 0;

        for i in 0..count {
            let entry = *entries.add(i);
            let base = (*entry).base;
            let len = (*entry).length;
            let etype = (*entry).entry_type;

            if etype == LIMINE_MEMMAP_USABLE {
                total_usable += len;
                if len > best_len && base >= 0x10_0000 {
                    best_base = base;
                    best_len = len;
                }
            }
        }

        merlion_infer::serial_println!("[boot] Total usable: {} MiB", total_usable / (1024*1024));
        merlion_infer::serial_println!("[boot] Best region: {:#x}..{:#x} ({} MiB)",
            best_base, best_base + best_len, best_len / (1024*1024));

        merlion_infer::memory::phys::init(best_base, best_base + best_len, hhdm_offset);
    }

    // Phase 4: SIMD state (must be before GDT/IDT since they don't depend on it,
    // but SIMD init must happen early)
    merlion_infer::arch::x86_64::simd::init();

    // Phase 5: CPU tables
    merlion_infer::arch::x86_64::gdt::init();
    merlion_infer::serial_println!("[ok] GDT");

    merlion_infer::arch::x86_64::timer::init();
    merlion_infer::serial_println!("[ok] PIT @ 100 Hz");

    merlion_infer::arch::x86_64::idt::init();
    merlion_infer::serial_println!("[ok] IDT + interrupts");

    // Phase 6: Page table + heap
    unsafe {
        let mut mapper = merlion_infer::memory::phys::active_page_table();
        let mut fa = merlion_infer::memory::phys::BumpAllocator;
        merlion_infer::memory::heap::init(&mut mapper, &mut fa)
            .expect("heap init failed");
        merlion_infer::serial_println!("[ok] Heap ({} KiB)",
            merlion_infer::memory::heap::HEAP_SIZE / 1024);
    }

    // Phase 7: SMP detection
    merlion_infer::arch::x86_64::smp::init();

    // Phase 8: Storage drivers
    merlion_infer::drivers::nvme::init();
    merlion_infer::drivers::virtio_blk::init();

    // Phase 9: Network
    merlion_infer::drivers::virtio_net::init();
    merlion_infer::net::init();

    // Phase 10: Kernel dispatch (AVX2 vs scalar)
    merlion_infer::inference::kernels::dispatch::init();

    // Phase 11: GPU
    merlion_infer::drivers::gpu::init();

    // Phase 11: Ready
    merlion_infer::log::info("kernel init complete");
    merlion_infer::serial_println!();
    merlion_infer::serial_println!("Kernel initialization complete.");
    merlion_infer::serial_println!("Type 'help' for available commands.");
    merlion_infer::serial_println!();
    merlion_infer::shell::prompt();

    // Main loop: poll serial input as fallback (IRQ4 may not fire in all QEMU configs)
    loop {
        if merlion_infer::arch::x86_64::serial::SERIAL1.lock().data_available() {
            let byte = merlion_infer::arch::x86_64::serial::SERIAL1.lock().read_byte();
            merlion_infer::shell::handle_serial_byte(byte);
        }
        x86_64::instructions::hlt();
    }
}

fn halt() -> ! {
    loop { x86_64::instructions::hlt(); }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    x86_64::instructions::interrupts::disable();
    merlion_infer::serial_println!("\n══ KERNEL PANIC ══");
    merlion_infer::serial_println!("{}", info);
    halt()
}
