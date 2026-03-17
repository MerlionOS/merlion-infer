/// Framebuffer console driver.
/// Renders text to a GOP/Limine linear framebuffer using an 8x16 bitmap font.
/// Required for bare-metal output on machines without serial ports (e.g. MacBook).

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::font;

/// Whether the framebuffer has been initialized.
static FB_READY: AtomicBool = AtomicBool::new(false);

/// Global framebuffer console.
pub static CONSOLE: Mutex<FramebufferConsole> = Mutex::new(FramebufferConsole::empty());

/// ARGB color constants.
const FG_COLOR: u32 = 0x00CCCCCC; // light gray text
const BG_COLOR: u32 = 0x00000000; // black background

pub struct FramebufferConsole {
    /// Pointer to the linear framebuffer memory.
    fb: *mut u8,
    /// Width in pixels.
    width: usize,
    /// Height in pixels.
    height: usize,
    /// Bytes per scanline.
    pitch: usize,
    /// Bytes per pixel (typically 4 for 32-bit ARGB).
    bpp: usize,
    /// Current cursor column (in character cells).
    col: usize,
    /// Current cursor row (in character cells).
    row: usize,
    /// Number of character columns.
    cols: usize,
    /// Number of character rows.
    rows: usize,
}

unsafe impl Send for FramebufferConsole {}

impl FramebufferConsole {
    const fn empty() -> Self {
        Self {
            fb: core::ptr::null_mut(),
            width: 0,
            height: 0,
            pitch: 0,
            bpp: 0,
            col: 0,
            row: 0,
            cols: 0,
            rows: 0,
        }
    }

    /// Initialize with framebuffer parameters from Limine.
    pub fn init(&mut self, addr: *mut u8, width: u64, height: u64, pitch: u64, bpp: u16) {
        self.fb = addr;
        self.width = width as usize;
        self.height = height as usize;
        self.pitch = pitch as usize;
        self.bpp = (bpp as usize) / 8;
        self.col = 0;
        self.row = 0;
        self.cols = self.width / font::CHAR_WIDTH;
        self.rows = self.height / font::CHAR_HEIGHT;
        self.clear();
    }

    /// Clear the entire screen to the background color.
    pub fn clear(&mut self) {
        if self.fb.is_null() {
            return;
        }
        for y in 0..self.height {
            let row_ptr = unsafe { self.fb.add(y * self.pitch) };
            for x in 0..self.width {
                unsafe {
                    let pixel = row_ptr.add(x * self.bpp) as *mut u32;
                    pixel.write_volatile(BG_COLOR);
                }
            }
        }
        self.col = 0;
        self.row = 0;
    }

    /// Write a single character at the current cursor position.
    pub fn write_char(&mut self, c: u8) {
        if self.fb.is_null() {
            return;
        }

        match c {
            b'\n' => {
                self.col = 0;
                self.row += 1;
                if self.row >= self.rows {
                    self.scroll();
                }
            }
            b'\r' => {
                self.col = 0;
            }
            // Backspace
            0x08 => {
                if self.col > 0 {
                    self.col -= 1;
                    // Clear the character cell
                    self.draw_glyph(self.col, self.row, &[0; 16], FG_COLOR, BG_COLOR);
                }
            }
            // Tab
            b'\t' => {
                let next = (self.col + 8) & !7;
                self.col = if next < self.cols { next } else { self.cols - 1 };
            }
            // Printable character
            _ => {
                let glyph = font::glyph(c);
                self.draw_glyph(self.col, self.row, glyph, FG_COLOR, BG_COLOR);
                self.col += 1;
                if self.col >= self.cols {
                    self.col = 0;
                    self.row += 1;
                    if self.row >= self.rows {
                        self.scroll();
                    }
                }
            }
        }
    }

    /// Draw an 8x16 glyph at character cell (col, row).
    fn draw_glyph(&self, col: usize, row: usize, glyph: &[u8; 16], fg: u32, bg: u32) {
        let px = col * font::CHAR_WIDTH;
        let py = row * font::CHAR_HEIGHT;

        for (gy, &glyph_row) in glyph.iter().enumerate() {
            let y = py + gy;
            if y >= self.height {
                break;
            }
            let row_ptr = unsafe { self.fb.add(y * self.pitch) };
            for gx in 0..font::CHAR_WIDTH {
                let x = px + gx;
                if x >= self.width {
                    break;
                }
                let bit = (glyph_row >> (7 - gx)) & 1;
                let color = if bit != 0 { fg } else { bg };
                unsafe {
                    let pixel = row_ptr.add(x * self.bpp) as *mut u32;
                    pixel.write_volatile(color);
                }
            }
        }
    }

    /// Scroll the screen up by one character row.
    fn scroll(&mut self) {
        let total_rows = self.rows;

        // Copy each row up by one character row height
        for y in 0..(total_rows - 1) * font::CHAR_HEIGHT {
            let dst = unsafe { self.fb.add(y * self.pitch) };
            let src = unsafe { self.fb.add((y + font::CHAR_HEIGHT) * self.pitch) };
            unsafe {
                core::ptr::copy(src, dst, self.width * self.bpp);
            }
        }

        // Clear the last row
        let last_row_start = (total_rows - 1) * font::CHAR_HEIGHT;
        for y in last_row_start..last_row_start + font::CHAR_HEIGHT {
            if y >= self.height {
                break;
            }
            let row_ptr = unsafe { self.fb.add(y * self.pitch) };
            for x in 0..self.width {
                unsafe {
                    let pixel = row_ptr.add(x * self.bpp) as *mut u32;
                    pixel.write_volatile(BG_COLOR);
                }
            }
        }

        self.row = total_rows - 1;
        self.col = 0;
    }
}

impl fmt::Write for FramebufferConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_char(byte);
        }
        Ok(())
    }
}

/// Initialize the framebuffer console. Called from main after Limine provides the framebuffer.
pub fn init(addr: *mut u8, width: u64, height: u64, pitch: u64, bpp: u16) {
    CONSOLE.lock().init(addr, width, height, pitch, bpp);
    FB_READY.store(true, Ordering::Release);
}

/// Returns true if the framebuffer console is available.
pub fn is_ready() -> bool {
    FB_READY.load(Ordering::Acquire)
}

/// Write formatted text to the framebuffer (if available).
pub fn _fb_print(args: fmt::Arguments) {
    if is_ready() {
        use fmt::Write;
        x86_64::instructions::interrupts::without_interrupts(|| {
            CONSOLE.lock().write_fmt(args).unwrap();
        });
    }
}
