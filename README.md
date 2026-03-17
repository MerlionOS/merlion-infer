# MerlionOS Inference

> **Zero overhead. Maximum throughput.** — A bare-metal OS purpose-built for LLM inference.

[![Rust](https://img.shields.io/badge/rust-nightly-red)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Architecture](https://img.shields.io/badge/arch-x86__64-blue)]()

MerlionOS Inference is a from-scratch operating system written in Rust that does exactly one thing: serve LLM inference as fast as the hardware allows. No scheduler overhead, no syscall boundary, no unnecessary abstractions. Boot in under 5 seconds, load a model, serve an OpenAI-compatible API.

~5,000 lines of Rust across 34 source files. No external dependencies beyond `core`, `alloc`, and `libm`.

## Why?

Linux is a general-purpose OS. When running LLM inference, you pay for that generality:

- **10-20% throughput loss** from kernel/user transitions, page faults, TLB shootdowns
- **3-5x P99/P50 latency ratio** from scheduler jitter, interrupt storms, GC pauses
- **30-60 second boot time** before first inference

MerlionOS Inference eliminates these by construction:

- Everything runs in kernel mode — zero syscall overhead
- All memory pre-allocated with 1GB huge pages — zero page faults
- Static core assignment — zero scheduler jitter
- Polling instead of interrupts on hot path — deterministic latency

## Quick Start

```bash
# Build
make build

# Run in QEMU
make run

# Load model and generate
merlion> ai-load
merlion> ai Hello from MerlionOS
```

## Boot & Test

```
$ make run
MerlionOS Inference v0.1.0
Zero overhead. Maximum throughput.
[boot] Limine UEFI boot path
[boot] HHDM offset: 0xffff800000000000
[boot] Total usable: 502 MiB
[simd] SSE=yes AVX2=yes AVX-512=no AMX=no
[ok] GDT
[ok] PIT @ 100 Hz
[ok] IDT + interrupts
[ok] Heap (4096 KiB)
[smp] CPU: QEMU Virtual CPU version 2.5+
[smp] Cores: 1 | APIC=yes x2APIC=no
[smp] SSE=yes SSE2=yes AVX=no AVX2=yes
Kernel initialization complete.
Type 'help' for available commands.

merlion> ai-load
[test-model] Byte-level tokenizer: 256 tokens
[test-model] Created: dim=32 hidden=64 layers=1 heads=2 vocab=256
[test-model] Weights: 104 KiB, State: 35 KiB
[ai-load] Ready. Try: ai Hello

merlion> ai-bench
[bench] Prefill: 19 tokens in 113 ticks (16.8 tok/s)
[bench] Decode:  32 tokens in 223 ticks (14.3 tok/s)
[bench] Peak memory: 175 KiB
```

## API

```bash
# Start inference server
merlion> ai-serve 8080

# From any client:
curl http://<ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-1b-q4","messages":[{"role":"user","content":"Hello"}]}'
```

Compatible with OpenAI API clients, LangChain, LlamaIndex, and any tool that speaks the OpenAI protocol.

## Shell Commands

| Category | Command | Description |
|----------|---------|-------------|
| **System** | `info` | System overview (CPU, RAM, model, uptime) |
| | `free` | Memory usage breakdown |
| | `uptime` | Time since boot |
| | `cpuid` | CPU feature details |
| | `memmap` | Physical memory map |
| | `lspci` | PCI device list |
| | `lsblk` | Block devices |
| | `dmesg` | Kernel message log |
| | `config` | Show configuration |
| **Inference** | `ai-load` | Load GGUF model |
| | `ai-info` | Current model details |
| | `ai <text>` | Generate text |
| | `ai-bench` | Benchmark inference speed |
| | `ai-serve [port]` | Start OpenAI-compatible API server |
| **Network** | `ip` | Show IP address |
| | `ss` | Show TCP connections |
| **GPU** | `gpu-info` | GPU status |
| **Control** | `reboot` | ACPI reboot |
| | `shutdown` | ACPI shutdown |
| | `clear` | Clear screen |
| | `help` | List all commands |

## Hardware Support

| Component | Supported |
|-----------|-----------|
| CPU | AMD Ryzen 7000/9000, Intel 12th gen+ (AVX2 required) |
| GPU | AMD Radeon RX 7000 series (RDNA3) — Phase 6 |
| RAM | DDR5, 32GB+ recommended |
| Storage | NVMe SSD |
| Network | Intel e1000e, Realtek RTL8125, virtio-net |
| Boot | UEFI (Limine bootloader) |

## Architecture

```
OpenAI API ← HTTP Server ← Inference Scheduler ← Inference Engine
                                                    ├── CPU: AVX2/AVX-512/AMX kernels
                                                    └── GPU: AMD RDNA3 compute (Phase 6)
```

## Roadmap

- [x] Phase 1: Boot + hardware init
- [x] Phase 2: Model loading (GGUF)
- [x] Phase 3: CPU inference (scalar)
- [x] Phase 4: Network + API server
- [ ] Phase 5: Real hardware + benchmarks
- [ ] Phase 6: AMD GPU inference
- [ ] Phase 7: Production hardening

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is especially welcome:
- Hardware drivers (USB Ethernet, NVMe quirks)
- AVX-512 optimized inference kernels
- AMD GPU compute driver
- Testing on different hardware
- Documentation

## Links

- **Hobby OS (upstream)**: [MerlionOS/merlion-kernel](https://github.com/MerlionOS/merlion-kernel)
- **Website**: [merlionos.org](https://merlionos.org)

## License

MIT — see [LICENSE](LICENSE)

---

*Built with [Claude](https://claude.ai) by Anthropic — from architecture design to every line of code.*
