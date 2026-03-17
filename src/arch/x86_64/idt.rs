/// Interrupt Descriptor Table.
/// Handles CPU exceptions and hardware interrupts (PIT timer, keyboard, serial).
/// No syscall (no ring 3).

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use spin::Lazy;
use crate::arch::x86_64::gdt;

const PIC_OFFSET_PRIMARY: u8 = 32;
const PIC_OFFSET_SECONDARY: u8 = PIC_OFFSET_PRIMARY + 8;

#[derive(Clone, Copy)]
#[repr(u8)]
enum HardwareInterrupt {
    Timer = PIC_OFFSET_PRIMARY,
    Keyboard = PIC_OFFSET_PRIMARY + 1, // IRQ1 — PS/2 keyboard
    Serial = PIC_OFFSET_PRIMARY + 4, // COM1 is IRQ4
}

static PICS: spin::Mutex<pic8259::ChainedPics> = spin::Mutex::new(
    unsafe { pic8259::ChainedPics::new(PIC_OFFSET_PRIMARY, PIC_OFFSET_SECONDARY) }
);

static IDT: Lazy<InterruptDescriptorTable> = Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();

    // CPU exceptions
    idt.breakpoint.set_handler_fn(breakpoint_handler);
    idt.page_fault.set_handler_fn(page_fault_handler);
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault_handler)
            .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
    }

    // Hardware interrupts
    idt[HardwareInterrupt::Timer as u8 as usize].set_handler_fn(timer_handler);
    idt[HardwareInterrupt::Keyboard as u8 as usize].set_handler_fn(keyboard_handler);
    idt[HardwareInterrupt::Serial as u8 as usize].set_handler_fn(serial_handler);

    idt
});

pub fn init() {
    IDT.load();
    unsafe {
        let mut pics = PICS.lock();
        pics.initialize();
        // Unmask timer (IRQ0), keyboard (IRQ1), and serial (IRQ4), mask rest
        // Primary PIC mask: bit=1 means masked
        x86_64::instructions::port::Port::<u8>::new(0x21).write(0b1110_1100);
        // Secondary PIC: mask all
        x86_64::instructions::port::Port::<u8>::new(0xA1).write(0xFF);
    }
    x86_64::instructions::interrupts::enable();
}

// --- Exception handlers ---

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    crate::serial_println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;
    let fault_addr = Cr2::read();

    crate::serial_println!("EXCEPTION: PAGE FAULT");
    crate::serial_println!("  Address: {:?}", fault_addr);
    crate::serial_println!("  Error: {:?}", error_code);
    crate::serial_println!("{:#?}", stack_frame);

    panic!("page fault at {:?}", fault_addr);
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    crate::serial_println!("EXCEPTION: DOUBLE FAULT\n{:#?}", stack_frame);
    panic!("double fault");
}

// --- Hardware interrupt handlers ---

extern "x86-interrupt" fn timer_handler(_stack_frame: InterruptStackFrame) {
    crate::arch::x86_64::timer::tick();

    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(HardwareInterrupt::Timer as u8);
    }
}

extern "x86-interrupt" fn keyboard_handler(_stack_frame: InterruptStackFrame) {
    let scancode = crate::arch::x86_64::keyboard::read_scancode();
    if let Some(byte) = crate::arch::x86_64::keyboard::handle_scancode(scancode) {
        crate::shell::handle_serial_byte(byte);
    }

    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(HardwareInterrupt::Keyboard as u8);
    }
}

extern "x86-interrupt" fn serial_handler(_stack_frame: InterruptStackFrame) {
    // Read incoming byte and feed to shell
    let byte = unsafe { x86_64::instructions::port::Port::<u8>::new(0x3F8).read() };
    crate::shell::handle_serial_byte(byte);

    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(HardwareInterrupt::Serial as u8);
    }
}
