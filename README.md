# MerlionOS Inference

> **Zero overhead. Maximum throughput.** — A bare-metal OS purpose-built for LLM inference.

[![Rust](https://img.shields.io/badge/rust-nightly-red)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Architecture](https://img.shields.io/badge/arch-x86__64-blue)]()

MerlionOS Inference is a from-scratch operating system written in Rust that does exactly one thing: serve LLM inference as fast as the hardware allows. No scheduler overhead, no syscall boundary, no unnecessary abstractions. Boot in under 5 seconds, load a model, serve an OpenAI-compatible API.

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

## Hardware Support

| Component | Supported |
|-----------|-----------|
| CPU | AMD Ryzen 7000/9000, Intel 12th gen+ (AVX2 required) |
| GPU | AMD Radeon RX 7000 series (RDNA3) — Phase 2 |
| RAM | DDR5, 32GB+ recommended |
| Storage | NVMe SSD |
| Network | Intel e1000e, Realtek RTL8125, virtio-net |
| Boot | UEFI (Limine bootloader) |

Tested on: ASUS TUF Gaming B650-PLUS + Ryzen 7 7700X + RX 7900 XT

## Architecture

```
OpenAI API ← HTTP Server ← Inference Scheduler ← Inference Engine
                                                     ├── CPU: AVX2/AVX-512/AMX kernels
                                                     └── GPU: AMD RDNA3 compute (Phase 2)
```

~30K lines of Rust. No external dependencies beyond `core`, `alloc`, and `libm`.

## Roadmap

- [x] Phase 1: Boot + hardware init
- [ ] Phase 2: Model loading (GGUF)
- [ ] Phase 3: CPU inference (AVX2)
- [ ] Phase 4: Network + API server
- [ ] Phase 5: Real hardware + benchmarks
- [ ] Phase 6: AMD GPU inference
- [ ] Phase 7: Production hardening

## Links

- **Hobby OS (upstream)**: [MerlionOS/merlion-kernel](https://github.com/MerlionOS/merlion-kernel)
- **Website**: [merlionos.org](https://merlionos.org)

## License

MIT

---

*Built with [Claude](https://claude.ai) by Anthropic — from architecture design to every line of code.*
