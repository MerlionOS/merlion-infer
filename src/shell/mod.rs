/// Minimal debug shell over serial console.
/// ~30 essential commands for system diagnostics and inference control.

use alloc::string::String;
use alloc::vec::Vec;

const MAX_INPUT: usize = 256;
const MAX_HISTORY: usize = 16;

static SHELL: spin::Mutex<Shell> = spin::Mutex::new(Shell::new());

struct Shell {
    buf: [u8; MAX_INPUT],
    pos: usize,
    history: [[u8; MAX_INPUT]; MAX_HISTORY],
    history_len: usize,
}

impl Shell {
    const fn new() -> Self {
        Self {
            buf: [0; MAX_INPUT],
            pos: 0,
            history: [[0; MAX_INPUT]; MAX_HISTORY],
            history_len: 0,
        }
    }
}

/// Print the shell prompt.
pub fn prompt() {
    crate::serial_print!("merlion> ");
}

/// Handle a byte from the serial port (called from IRQ handler).
pub fn handle_serial_byte(byte: u8) {
    let mut shell = SHELL.lock();

    match byte {
        // Enter
        b'\r' | b'\n' => {
            crate::serial_println!();
            let len = shell.pos;
            if len > 0 {
                let cmd = core::str::from_utf8(&shell.buf[..len]).unwrap_or("");
                let cmd = String::from(cmd);

                // Save to history
                if shell.history_len < MAX_HISTORY {
                    let idx = shell.history_len;
                    let mut tmp = [0u8; MAX_INPUT];
                    tmp[..len].copy_from_slice(&shell.buf[..len]);
                    shell.history[idx] = tmp;
                    shell.history_len += 1;
                }

                shell.pos = 0;
                shell.buf = [0; MAX_INPUT];

                // Release lock before dispatch
                drop(shell);
                dispatch(&cmd);
            } else {
                drop(shell);
            }
            prompt();
        }
        // Backspace
        0x7F | 0x08 => {
            if shell.pos > 0 {
                shell.pos -= 1;
                crate::serial_print!("\x08 \x08");
            }
        }
        // Printable ASCII
        0x20..=0x7E => {
            if shell.pos < MAX_INPUT - 1 {
                let pos = shell.pos;
                shell.buf[pos] = byte;
                shell.pos += 1;
                crate::serial_print!("{}", byte as char);
            }
        }
        _ => {}
    }
}

fn dispatch(input: &str) {
    let parts: Vec<&str> = input.trim().splitn(2, ' ').collect();
    let cmd = parts[0];
    let _args = if parts.len() > 1 { parts[1] } else { "" };

    match cmd {
        "help" => cmd_help(),
        "info" => cmd_info(),
        "free" => cmd_free(),
        "uptime" => cmd_uptime(),
        "cpuid" => cmd_cpuid(),
        "memmap" => cmd_memmap(),
        "reboot" => crate::arch::x86_64::acpi::reboot(),
        "shutdown" => crate::arch::x86_64::acpi::shutdown(),
        "clear" => crate::serial_print!("\x1b[2J\x1b[H"),
        "" => {}
        _ => crate::serial_println!("unknown command: '{}'. Type 'help' for list.", cmd),
    }
}

fn cmd_help() {
    crate::serial_println!("MerlionOS Inference Shell");
    crate::serial_println!("========================");
    crate::serial_println!("System:");
    crate::serial_println!("  info       — system overview");
    crate::serial_println!("  free       — memory usage");
    crate::serial_println!("  uptime     — time since boot");
    crate::serial_println!("  cpuid      — CPU feature details");
    crate::serial_println!("  memmap     — physical memory map");
    crate::serial_println!("  reboot     — ACPI reboot");
    crate::serial_println!("  shutdown   — ACPI shutdown");
    crate::serial_println!("  clear      — clear screen");
    crate::serial_println!("  help       — this message");
}

fn cmd_info() {
    let features = crate::arch::x86_64::smp::detect_features();
    let (h, m, s) = crate::arch::x86_64::timer::uptime_hms();

    crate::serial_println!("MerlionOS Inference v0.1.0");
    crate::serial_println!("CPU: {}", features.brand);
    crate::serial_println!("Cores: {} | AVX2={} AVX-512={} AMX={}",
        features.logical_cores,
        if features.has_avx2 { "yes" } else { "no" },
        if crate::arch::x86_64::simd::has_avx512() { "yes" } else { "no" },
        if crate::arch::x86_64::simd::has_amx() { "yes" } else { "no" },
    );
    crate::serial_println!("RAM: {} MiB usable",
        crate::memory::phys::total_usable() / (1024 * 1024));
    crate::serial_println!("Uptime: {}h {}m {}s", h, m, s);
    crate::serial_println!("Model: not loaded");
    crate::serial_println!("API: not running");
}

fn cmd_free() {
    let heap_used = crate::memory::heap::used();
    let heap_free = crate::memory::heap::free();
    let phys_alloc = crate::memory::phys::allocated_bytes();
    let phys_total = crate::memory::phys::total_usable();

    crate::serial_println!("Physical: {} KiB allocated / {} MiB total",
        phys_alloc / 1024, phys_total / (1024 * 1024));
    crate::serial_println!("Heap:     {} KiB used / {} KiB free / {} KiB total",
        heap_used / 1024, heap_free / 1024,
        crate::memory::heap::HEAP_SIZE / 1024);
}

fn cmd_uptime() {
    let (h, m, s) = crate::arch::x86_64::timer::uptime_hms();
    crate::serial_println!("up {}h {}m {}s ({} ticks)",
        h, m, s, crate::arch::x86_64::timer::ticks());
}

fn cmd_cpuid() {
    let features = crate::arch::x86_64::smp::detect_features();
    let vendor = core::str::from_utf8(&features.vendor).unwrap_or("?");

    crate::serial_println!("CPU: {}", features.brand);
    crate::serial_println!("Vendor: {}", vendor);
    crate::serial_println!("Family: {} Model: {} Stepping: {}",
        features.family, features.model, features.stepping);
    crate::serial_println!("Cores: {}", features.logical_cores);
    crate::serial_println!("APIC: {} x2APIC: {}",
        if features.has_apic { "yes" } else { "no" },
        if features.has_x2apic { "yes" } else { "no" });
    crate::serial_println!("SSE: {} SSE2: {} AVX: {} AVX2: {}",
        if features.has_sse { "yes" } else { "no" },
        if features.has_sse2 { "yes" } else { "no" },
        if features.has_avx { "yes" } else { "no" },
        if features.has_avx2 { "yes" } else { "no" });
    crate::serial_println!("AVX-512: {} AMX: {}",
        if crate::arch::x86_64::simd::has_avx512() { "yes" } else { "no" },
        if crate::arch::x86_64::simd::has_amx() { "yes" } else { "no" });
}

fn cmd_memmap() {
    let total = crate::memory::phys::total_usable();
    let alloc = crate::memory::phys::allocated_bytes();
    crate::serial_println!("Physical memory: {} MiB usable, {} KiB allocated",
        total / (1024 * 1024), alloc / 1024);
}
