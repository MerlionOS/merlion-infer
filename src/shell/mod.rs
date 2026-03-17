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
    let args = if parts.len() > 1 { parts[1] } else { "" };

    match cmd {
        "help" => cmd_help(),
        "info" => cmd_info(),
        "free" => cmd_free(),
        "uptime" => cmd_uptime(),
        "cpuid" => cmd_cpuid(),
        "memmap" => cmd_memmap(),
        "lspci" => cmd_lspci(),
        "lsblk" => cmd_lsblk(),
        "ai-load" => cmd_ai_load(),
        "disk-test" => cmd_disk_test(),
        "ai-info" => cmd_ai_info(),
        "ai" => cmd_ai(args),
        "ai-bench" => cmd_ai_bench(),
        "ai-serve" => cmd_ai_serve(args),
        "ip" => cmd_ip(),
        "ss" => cmd_ss(),
        "gpu-info" => cmd_gpu_info(),
        "dmesg" => crate::log::dmesg(),
        "config" => crate::config::show(),
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
    crate::serial_println!("  lspci      — PCI devices");
    crate::serial_println!("  lsblk      — block devices");
    crate::serial_println!("Inference:");
    crate::serial_println!("  ai-load    — load GGUF model from disk");
    crate::serial_println!("  ai-info    — current model info");
    crate::serial_println!("  ai <text>  — generate text");
    crate::serial_println!("  ai-bench   — benchmark inference speed");
    crate::serial_println!("  ai-serve   — start OpenAI-compatible API server");
    crate::serial_println!("Network:");
    crate::serial_println!("  ip         — show IP address");
    crate::serial_println!("  ss         — show TCP connections");
    crate::serial_println!("GPU:");
    crate::serial_println!("  gpu-info   — GPU status");
    crate::serial_println!("Debug:");
    crate::serial_println!("  dmesg      — kernel log");
    crate::serial_println!("  config     — show configuration");
    crate::serial_println!("Control:");
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

fn cmd_lspci() {
    let devices = crate::drivers::pci::scan();
    for dev in &devices {
        crate::serial_println!("  {}", dev.summary());
    }
    crate::serial_println!("{} devices", devices.len());
}

fn cmd_lsblk() {
    if crate::drivers::nvme::is_detected() {
        crate::serial_println!("  {}", crate::drivers::nvme::info());
    }
    if crate::drivers::virtio_blk::is_detected() {
        crate::serial_println!("  {}", crate::drivers::virtio_blk::info());
    }
    if !crate::drivers::nvme::is_detected() && !crate::drivers::virtio_blk::is_detected() {
        crate::serial_println!("  no block devices");
    }
}

fn cmd_disk_test() {
    if !crate::drivers::virtio_blk::is_detected() && !crate::drivers::nvme::is_detected() {
        crate::serial_println!("[disk-test] No block device");
        return;
    }

    // Read 16 KiB (32 sectors) using multi-sector DMA — should be ~4 requests
    let size = 16 * 1024;
    let mut buf = alloc::vec![0u8; size];

    let start = crate::arch::x86_64::timer::ticks();
    let result = if crate::drivers::virtio_blk::is_detected() {
        crate::drivers::virtio_blk::read_sectors(0, &mut buf)
    } else {
        crate::drivers::nvme::read_sectors(0, &mut buf)
    };
    let elapsed = crate::arch::x86_64::timer::ticks() - start;

    match result {
        Ok(bytes) => {
            crate::serial_println!("[disk-test] Read {} bytes in {} ticks", bytes, elapsed);
            // Check GGUF magic
            if buf[0..4] == [0x47, 0x47, 0x55, 0x46] {
                crate::serial_println!("[disk-test] GGUF magic OK");
            }
            crate::serial_println!("[disk-test] First 16 bytes: {:02x?}", &buf[..16]);
        }
        Err(e) => crate::serial_println!("[disk-test] Error: {}", e),
    }
}

fn cmd_ai_load() {
    if crate::inference::state::is_loaded() {
        crate::serial_println!("[ai-load] Model already loaded. Use 'ai-info' to see details.");
        return;
    }

    // Load built-in test model (tiny Llama, dim=32, 1 layer, vocab=256)
    crate::serial_println!("[ai-load] Loading built-in test model...");
    crate::inference::test_model::setup_byte_tokenizer();
    let engine = crate::inference::test_model::create_test_engine();
    crate::inference::state::load(engine);
    crate::serial_println!("[ai-load] Ready. Try: ai Hello");
}

fn cmd_ai_info() {
    crate::serial_println!("{}", crate::inference::state::model_info());
}

fn cmd_ai(prompt: &str) {
    if prompt.is_empty() {
        crate::serial_println!("Usage: ai <prompt text>");
        return;
    }
    if !crate::inference::state::is_loaded() {
        crate::serial_println!("[ai] No model loaded. Run 'ai-load' first.");
        return;
    }

    let max_tokens = 64;
    let sampler = crate::inference::sampler::Sampler::new(0.8, 0.9);

    let result = crate::inference::state::with_engine(|engine| {
        crate::inference::generate::generate(engine, prompt, max_tokens, &sampler)
    });

    if let Some((_text, n_tokens, elapsed_ticks)) = result {
        let elapsed_secs = elapsed_ticks as f32 / crate::arch::x86_64::timer::PIT_FREQUENCY_HZ as f32;
        let tok_per_sec = if elapsed_secs > 0.0 { n_tokens as f32 / elapsed_secs } else { 0.0 };
        crate::serial_println!("[Generated {} tokens in {:.1}s | {:.1} tok/s | scalar]",
            n_tokens, elapsed_secs, tok_per_sec);
    }
}

fn cmd_ai_bench() {
    if !crate::inference::state::is_loaded() {
        crate::serial_println!("[ai-bench] No model loaded. Run 'ai-load' first.");
        return;
    }

    crate::serial_println!("[ai-bench] Running benchmark...");

    let sampler = crate::inference::sampler::Sampler::greedy();

    // Prefill benchmark: process a 16-token prompt
    let prefill_prompt = "The quick brown fox";
    let (prefill_result, prefill_ticks) = crate::inference::bench::measure(|| {
        crate::inference::state::with_engine(|engine| {
            let tok = crate::inference::tokenizer::global();
            let tokens = tok.encode(prefill_prompt);
            let n = tokens.len();
            drop(tok);
            // Run forward pass for each prompt token
            for (i, &t) in tokens.iter().enumerate() {
                engine.forward(t, i);
            }
            n
        })
    });

    let prefill_tokens = prefill_result.unwrap_or(0);
    let _prefill_secs = prefill_ticks as f32 / crate::arch::x86_64::timer::PIT_FREQUENCY_HZ as f32;

    // Decode benchmark: generate 32 tokens
    let decode_tokens = 32;
    let (_, decode_ticks) = crate::inference::bench::measure(|| {
        crate::inference::state::with_engine(|engine| {
            crate::inference::generate::generate(engine, "Hello", decode_tokens, &sampler)
        })
    });

    let _decode_secs = decode_ticks as f32 / crate::arch::x86_64::timer::PIT_FREQUENCY_HZ as f32;

    let result = crate::inference::bench::BenchResult {
        prefill_tokens,
        prefill_ticks,
        decode_tokens,
        decode_ticks,
        peak_memory_bytes: crate::memory::heap::used(),
    };

    crate::serial_println!();
    result.report();
}

fn cmd_ai_serve(args: &str) {
    let port: u16 = if args.is_empty() {
        8080
    } else {
        args.parse().unwrap_or(8080)
    };

    if !crate::drivers::virtio_net::is_detected() {
        crate::serial_println!("[ai-serve] No network device. Add QEMU flags:");
        crate::serial_println!("  -netdev user,id=n0,hostfwd=tcp::8080-:8080 -device virtio-net-pci,netdev=n0");
        return;
    }

    crate::serial_println!("[ai-serve] Starting OpenAI-compatible API on port {}", port);
    crate::serial_println!("[ai-serve] Endpoints:");
    crate::serial_println!("  POST /v1/chat/completions");
    crate::serial_println!("  GET  /v1/models");
    crate::serial_println!("  GET  /health");
    crate::serial_println!("  GET  /metrics");

    // This blocks — runs the HTTP server loop
    crate::serving::http::serve(port, crate::serving::openai_api::handle_request);
}

fn cmd_ip() {
    let net = crate::net::NET.lock();
    crate::serial_println!("eth0:");
    crate::serial_println!("  MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        net.mac[0], net.mac[1], net.mac[2], net.mac[3], net.mac[4], net.mac[5]);
    crate::serial_println!("  IP:  {}", net.ip);
    crate::serial_println!("  GW:  {}", net.gateway);
    crate::serial_println!("  RX:  {} packets", net.rx_packets);
    crate::serial_println!("  TX:  {} packets", net.tx_packets);
    if !crate::drivers::virtio_net::is_detected() {
        crate::serial_println!("  NIC: not detected");
    }
}

fn cmd_gpu_info() {
    crate::serial_println!("{}", crate::drivers::gpu::discovery::info());
    if crate::drivers::gpu::discovery::is_detected() {
        let used = crate::drivers::gpu::vram::used_bytes();
        let total = crate::drivers::gpu::vram::total_bytes();
        if total > 0 {
            crate::serial_println!("VRAM: {} MiB / {} MiB", used / (1024*1024), total / (1024*1024));
        }
    }
}

fn cmd_ss() {
    let sockets = crate::net::tcp::list_sockets();
    if sockets.is_empty() {
        crate::serial_println!("No active TCP connections");
    } else {
        for (id, lip, lp, rip, rp, state) in &sockets {
            crate::serial_println!("  sock {} | {}:{} <-> {}:{} | {:?}", id, lip, lp, rip, rp, state);
        }
    }
}
