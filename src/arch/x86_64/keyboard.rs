/// PS/2 keyboard driver.
/// Translates scancode set 1 (make codes) to ASCII bytes.
/// UEFI firmware provides PS/2 emulation for USB keyboards,
/// so this works on machines like MacBook Pro 2017.

use x86_64::instructions::port::Port;
use core::sync::atomic::{AtomicBool, Ordering};

const PS2_DATA_PORT: u16 = 0x60;
const PS2_STATUS_PORT: u16 = 0x64;

/// Shift key state.
static LEFT_SHIFT: AtomicBool = AtomicBool::new(false);
static RIGHT_SHIFT: AtomicBool = AtomicBool::new(false);
static CAPS_LOCK: AtomicBool = AtomicBool::new(false);
static CTRL: AtomicBool = AtomicBool::new(false);

/// Check if a PS/2 keyboard byte is available.
pub fn data_available() -> bool {
    unsafe { Port::<u8>::new(PS2_STATUS_PORT).read() & 0x01 != 0 }
}

/// Read a raw scancode from the PS/2 data port.
pub fn read_scancode() -> u8 {
    unsafe { Port::<u8>::new(PS2_DATA_PORT).read() }
}

/// Handle a scancode and return an optional ASCII byte.
/// Tracks shift/caps/ctrl state internally.
pub fn handle_scancode(scancode: u8) -> Option<u8> {
    // Break codes (key release) have bit 7 set
    let is_release = scancode & 0x80 != 0;
    let code = scancode & 0x7F;

    match code {
        // Left Shift
        0x2A => { LEFT_SHIFT.store(!is_release, Ordering::Relaxed); None }
        // Right Shift
        0x36 => { RIGHT_SHIFT.store(!is_release, Ordering::Relaxed); None }
        // Left Ctrl
        0x1D => { CTRL.store(!is_release, Ordering::Relaxed); None }
        // Caps Lock (toggle on press only)
        0x3A if !is_release => {
            let cur = CAPS_LOCK.load(Ordering::Relaxed);
            CAPS_LOCK.store(!cur, Ordering::Relaxed);
            None
        }
        // Ignore release events for regular keys
        _ if is_release => None,
        // Translate make codes to ASCII
        _ => {
            let shifted = LEFT_SHIFT.load(Ordering::Relaxed)
                || RIGHT_SHIFT.load(Ordering::Relaxed);
            let caps = CAPS_LOCK.load(Ordering::Relaxed);
            let ctrl = CTRL.load(Ordering::Relaxed);

            let ch = if shifted {
                SCANCODE_SHIFT[code as usize]
            } else {
                SCANCODE_NORMAL[code as usize]
            };

            if ch == 0 { return None; }

            // Caps lock toggles letter case
            let ch = if caps && ch >= b'a' && ch <= b'z' {
                ch - 32 // to uppercase
            } else if caps && ch >= b'A' && ch <= b'Z' {
                ch + 32 // to lowercase (shift+caps = lower)
            } else {
                ch
            };

            // Ctrl+key produces control characters
            if ctrl && ch >= b'a' && ch <= b'z' {
                return Some(ch - b'a' + 1); // Ctrl+A = 0x01, etc.
            }
            if ctrl && ch >= b'A' && ch <= b'Z' {
                return Some(ch - b'A' + 1);
            }

            Some(ch)
        }
    }
}

/// Scancode set 1 → ASCII (unshifted).
/// Index is the make code (0x00-0x7F), value is ASCII byte (0 = no mapping).
static SCANCODE_NORMAL: [u8; 128] = {
    let mut t = [0u8; 128];
    t[0x01] = 0x1B; // Escape
    t[0x02] = b'1';
    t[0x03] = b'2';
    t[0x04] = b'3';
    t[0x05] = b'4';
    t[0x06] = b'5';
    t[0x07] = b'6';
    t[0x08] = b'7';
    t[0x09] = b'8';
    t[0x0A] = b'9';
    t[0x0B] = b'0';
    t[0x0C] = b'-';
    t[0x0D] = b'=';
    t[0x0E] = 0x08; // Backspace
    t[0x0F] = b'\t';
    t[0x10] = b'q';
    t[0x11] = b'w';
    t[0x12] = b'e';
    t[0x13] = b'r';
    t[0x14] = b't';
    t[0x15] = b'y';
    t[0x16] = b'u';
    t[0x17] = b'i';
    t[0x18] = b'o';
    t[0x19] = b'p';
    t[0x1A] = b'[';
    t[0x1B] = b']';
    t[0x1C] = b'\r'; // Enter
    // 0x1D = Ctrl (handled separately)
    t[0x1E] = b'a';
    t[0x1F] = b's';
    t[0x20] = b'd';
    t[0x21] = b'f';
    t[0x22] = b'g';
    t[0x23] = b'h';
    t[0x24] = b'j';
    t[0x25] = b'k';
    t[0x26] = b'l';
    t[0x27] = b';';
    t[0x28] = b'\'';
    t[0x29] = b'`';
    // 0x2A = Left Shift (handled separately)
    t[0x2B] = b'\\';
    t[0x2C] = b'z';
    t[0x2D] = b'x';
    t[0x2E] = b'c';
    t[0x2F] = b'v';
    t[0x30] = b'b';
    t[0x31] = b'n';
    t[0x32] = b'm';
    t[0x33] = b',';
    t[0x34] = b'.';
    t[0x35] = b'/';
    // 0x36 = Right Shift (handled separately)
    t[0x37] = b'*'; // Keypad *
    // 0x38 = Alt
    t[0x39] = b' '; // Space
    // 0x3A = Caps Lock (handled separately)
    t
};

/// Scancode set 1 → ASCII (shifted).
static SCANCODE_SHIFT: [u8; 128] = {
    let mut t = [0u8; 128];
    t[0x01] = 0x1B; // Escape
    t[0x02] = b'!';
    t[0x03] = b'@';
    t[0x04] = b'#';
    t[0x05] = b'$';
    t[0x06] = b'%';
    t[0x07] = b'^';
    t[0x08] = b'&';
    t[0x09] = b'*';
    t[0x0A] = b'(';
    t[0x0B] = b')';
    t[0x0C] = b'_';
    t[0x0D] = b'+';
    t[0x0E] = 0x08; // Backspace
    t[0x0F] = b'\t';
    t[0x10] = b'Q';
    t[0x11] = b'W';
    t[0x12] = b'E';
    t[0x13] = b'R';
    t[0x14] = b'T';
    t[0x15] = b'Y';
    t[0x16] = b'U';
    t[0x17] = b'I';
    t[0x18] = b'O';
    t[0x19] = b'P';
    t[0x1A] = b'{';
    t[0x1B] = b'}';
    t[0x1C] = b'\r'; // Enter
    t[0x1E] = b'A';
    t[0x1F] = b'S';
    t[0x20] = b'D';
    t[0x21] = b'F';
    t[0x22] = b'G';
    t[0x23] = b'H';
    t[0x24] = b'J';
    t[0x25] = b'K';
    t[0x26] = b'L';
    t[0x27] = b':';
    t[0x28] = b'"';
    t[0x29] = b'~';
    t[0x2B] = b'|';
    t[0x2C] = b'Z';
    t[0x2D] = b'X';
    t[0x2E] = b'C';
    t[0x2F] = b'V';
    t[0x30] = b'B';
    t[0x31] = b'N';
    t[0x32] = b'M';
    t[0x33] = b'<';
    t[0x34] = b'>';
    t[0x35] = b'?';
    t[0x37] = b'*';
    t[0x39] = b' ';
    t
};
