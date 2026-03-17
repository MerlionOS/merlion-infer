/// ACPI power management (QEMU-specific).
/// Shutdown and reboot via well-known I/O ports.

use x86_64::instructions::port::Port;

pub fn shutdown() -> ! {
    crate::serial_println!("[acpi] shutting down...");
    unsafe {
        Port::<u16>::new(0x604).write(0x2000);
        Port::<u16>::new(0xB004).write(0x2000);
    }
    loop { x86_64::instructions::hlt(); }
}

pub fn reboot() -> ! {
    crate::serial_println!("[acpi] rebooting...");
    unsafe {
        x86_64::instructions::interrupts::disable();
        let mut status = Port::<u8>::new(0x64);
        let mut cmd = Port::<u8>::new(0x64);
        loop {
            if status.read() & 0x02 == 0 { break; }
        }
        cmd.write(0xFE);
    }
    loop { x86_64::instructions::hlt(); }
}
