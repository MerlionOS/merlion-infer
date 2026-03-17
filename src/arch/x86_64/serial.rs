/// Serial port (UART 16550) driver for COM1.
/// All kernel output goes here — no VGA, no framebuffer.

use core::fmt;
use spin::Mutex;
use x86_64::instructions::port::Port;

const COM1_PORT: u16 = 0x3F8;

pub static SERIAL1: Mutex<SerialPort> = Mutex::new(SerialPort::new(COM1_PORT));

pub struct SerialPort {
    base: u16,
}

impl SerialPort {
    const fn new(base: u16) -> Self {
        Self { base }
    }

    /// Initialize UART: 115200 baud, 8N1.
    pub fn init(&mut self) {
        unsafe {
            let mut ier = Port::<u8>::new(self.base + 1);
            let mut lcr = Port::<u8>::new(self.base + 3);
            let mut data = Port::<u8>::new(self.base);
            let mut fifo = Port::<u8>::new(self.base + 2);
            let mut mcr = Port::<u8>::new(self.base + 4);

            ier.write(0x00);  // Disable interrupts
            lcr.write(0x80);  // DLAB on
            data.write(0x01); // Divisor lo: 115200 baud
            ier.write(0x00);  // Divisor hi
            lcr.write(0x03);  // 8N1
            fifo.write(0xC7); // Enable FIFO, clear, 14-byte threshold
            mcr.write(0x0B);  // IRQs enabled, RTS/DSR set
        }
    }

    /// Read a byte from serial (blocking).
    pub fn read_byte(&mut self) -> u8 {
        unsafe {
            let mut line_status = Port::<u8>::new(self.base + 5);
            while line_status.read() & 0x01 == 0 {}
            Port::<u8>::new(self.base).read()
        }
    }

    /// Check if data is available to read.
    pub fn data_available(&self) -> bool {
        unsafe {
            Port::<u8>::new(self.base + 5).read() & 0x01 != 0
        }
    }

    fn write_byte(&mut self, byte: u8) {
        unsafe {
            let mut line_status = Port::<u8>::new(self.base + 5);
            while line_status.read() & 0x20 == 0 {}
            Port::new(self.base).write(byte);
        }
    }
}

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::arch::x86_64::serial::_serial_print(format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! serial_println {
    ()            => { $crate::serial_print!("\n") };
    ($($arg:tt)*) => { $crate::serial_print!("{}\n", format_args!($($arg)*)) };
}

#[doc(hidden)]
pub fn _serial_print(args: fmt::Arguments) {
    use fmt::Write;
    use x86_64::instructions::interrupts;

    interrupts::without_interrupts(|| {
        SERIAL1.lock().write_fmt(args).unwrap();
        // Mirror output to framebuffer console (if available)
        if super::framebuffer::is_ready() {
            super::framebuffer::CONSOLE.lock().write_fmt(args).unwrap();
        }
    });
}
