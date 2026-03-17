# CLAUDE.md — MerlionOS Inference

> **MerlionOS Inference** is a bare-metal operating system purpose-built for LLM inference on x86_64 + AMD GPU.
> Zero overhead. Maximum throughput. Every line of code serves inference.

---

## Project Identity

- **Name**: MerlionOS Inference (repo: `merlion-inference`)
- **Language**: Rust (nightly, `no_std`, `#![no_main]`)
- **Target**: `x86_64-unknown-none`
- **Boot**: UEFI via Limine bootloader
- **License**: MIT
- **Upstream**: Selectively forked from `merlion-kernel` (the hobby OS)

This is NOT a general-purpose OS. This is a **firmware** for LLM inference servers.
The mental model is: ESXi is to virtualization as MerlionOS Inference is to LLM inference.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   OpenAI-Compatible API                  │
│              POST /v1/chat/completions                   │
├─────────────────────────────────────────────────────────┤
│                  Inference Scheduler                     │
│         (continuous batching, prefill/decode split)      │
├──────────────────────┬──────────────────────────────────┤
│   CPU Inference      │       GPU Inference               │
│   (AVX2/AVX-512/AMX) │   (AMD RDNA3/CDNA compute)       │
├──────────────────────┴──────────────────────────────────┤
│              KV Cache Manager (paged)                    │
├─────────────────────────────────────────────────────────┤
│         Memory Manager (buddy + 1GB huge pages)          │
├──────────┬──────────┬──────────┬────────────────────────┤
│  NVMe    │  PCIe    │  Network │   AMD GPU Driver       │
│  Driver  │  Bus     │  Stack   │   (compute only)       │
├──────────┴──────────┴──────────┴────────────────────────┤
│              x86_64 Hardware Abstraction                  │
│        (GDT, IDT, APIC, HPET, SMP, SIMD state)          │
├─────────────────────────────────────────────────────────┤
│                  UEFI Boot (Limine)                       │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
merlion-inference/
├── .cargo/
│   └── config.toml            # target, rustflags
├── .github/
│   └── workflows/ci.yml       # build + QEMU smoke test
├── src/
│   ├── lib.rs                 # kernel entry, panic handler, global allocator
│   │
│   ├── boot/
│   │   ├── mod.rs
│   │   ├── limine.rs          # Limine protocol: memory map, framebuffer, RSDP
│   │   └── early_init.rs      # GDT, IDT, TSS, SIMD state (CR0/CR4/XCR0)
│   │
│   ├── arch/
│   │   └── x86_64/
│   │       ├── mod.rs
│   │       ├── gdt.rs
│   │       ├── idt.rs
│   │       ├── apic.rs        # Local APIC + I/O APIC
│   │       ├── hpet.rs        # High Precision Event Timer
│   │       ├── smp.rs         # AP startup, core parking
│   │       ├── simd.rs        # SSE/AVX/AVX-512/AMX state init + CPUID detection
│   │       └── serial.rs      # NS16550 UART for debug console
│   │
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── phys.rs            # physical frame allocator (buddy, supports 4K/2M/1G)
│   │   ├── virt.rs            # kernel page table management
│   │   ├── heap.rs            # small object heap (slab or linked-list, for kernel data structures)
│   │   ├── huge_page.rs       # 1GB huge page pool for model weights + KV cache
│   │   └── numa.rs            # NUMA topology detection (from ACPI SRAT)
│   │
│   ├── drivers/
│   │   ├── mod.rs
│   │   ├── pcie/
│   │   │   ├── mod.rs
│   │   │   ├── config.rs      # PCIe configuration space (ECAM)
│   │   │   ├── bar.rs         # BAR mapping, above-4G, resizable BAR
│   │   │   └── msi.rs         # MSI/MSI-X interrupt setup
│   │   ├── nvme/
│   │   │   ├── mod.rs
│   │   │   ├── queue.rs       # admin + I/O submission/completion queues
│   │   │   └── commands.rs    # read, identify
│   │   ├── net/
│   │   │   ├── mod.rs
│   │   │   ├── virtio_net.rs  # QEMU development
│   │   │   └── e1000e.rs      # real hardware fallback
│   │   ├── serial.rs          # UART (re-export from arch)
│   │   └── gpu/               # Phase 2 — AMD GPU compute driver
│   │       ├── mod.rs
│   │       ├── discovery.rs   # PCIe enumeration, identify AMD GPU by vendor/device ID
│   │       ├── firmware.rs    # load PSP/SDMA/GFX/SMU firmware blobs
│   │       ├── init.rs        # golden register sequence, engine startup
│   │       ├── mem.rs         # VRAM allocator (buddy over GPU local memory)
│   │       ├── queue.rs       # AQL compute queue, doorbell, fence
│   │       └── dispatch.rs    # submit pre-compiled compute kernels
│   │
│   ├── net/
│   │   ├── mod.rs
│   │   ├── ethernet.rs        # Ethernet frame TX/RX
│   │   ├── arp.rs
│   │   ├── ip.rs              # IPv4 only (IPv6 deferred)
│   │   ├── tcp.rs             # TCP with Cubic congestion control
│   │   ├── dhcp.rs            # DHCP client (auto-configure on boot)
│   │   └── dns.rs             # DNS resolver (for service discovery)
│   │
│   ├── tls/
│   │   ├── mod.rs
│   │   ├── record.rs          # TLS 1.3 record layer
│   │   ├── handshake.rs       # TLS handshake
│   │   ├── aes_gcm.rs         # AES-128-GCM / AES-256-GCM
│   │   ├── x25519.rs          # key exchange
│   │   └── x509.rs            # certificate parsing
│   │
│   ├── inference/             # THE CORE — LLM inference engine
│   │   ├── mod.rs
│   │   ├── tensor.rs          # BlockQ4_0, BlockQ8_0, TensorView, f16<->f32
│   │   ├── gguf.rs            # GGUF parser + model loader
│   │   ├── tokenizer.rs       # BPE tokenizer (encode/decode)
│   │   ├── sampler.rs         # temperature, top-p, top-k sampling
│   │   ├── engine.rs          # LlamaEngine: forward(), RunState, KV cache
│   │   ├── generate.rs        # text generation loop (single request)
│   │   ├── kv_cache.rs        # Paged KV cache manager (PagedAttention-style)
│   │   ├── scheduler.rs       # Continuous batching scheduler
│   │   │                      #   - dynamic batch assembly
│   │   │                      #   - prefill/decode phase scheduling
│   │   │                      #   - preemption & eviction policy
│   │   ├── kernels/
│   │   │   ├── mod.rs         # KernelDispatch: runtime selection based on CPUID
│   │   │   ├── scalar.rs      # fallback: pure Rust, no SIMD
│   │   │   ├── avx2.rs        # AVX2 intrinsics (gemv_q4, rmsnorm, softmax, rope, silu)
│   │   │   ├── avx512.rs      # AVX-512 (future)
│   │   │   ├── amx.rs         # Intel AMX BF16/INT8 matrix tiles (future)
│   │   │   └── gpu.rs         # GPU kernel dispatch (submit pre-compiled .hsaco)
│   │   └── models/
│   │       ├── mod.rs
│   │       ├── llama.rs       # Llama/Llama2/Llama3/SmolLM architecture
│   │       ├── qwen.rs        # Qwen2.5 architecture (future)
│   │       └── phi.rs         # Phi-3 architecture (future)
│   │
│   ├── serving/               # HTTP API server
│   │   ├── mod.rs
│   │   ├── http.rs            # minimal HTTP/1.1 server (request parsing, response writing)
│   │   ├── router.rs          # route dispatch
│   │   ├── openai_api.rs      # POST /v1/chat/completions (streaming SSE)
│   │   │                      # POST /v1/completions
│   │   │                      # GET  /v1/models
│   │   ├── health.rs          # GET /health (liveness/readiness probes)
│   │   └── metrics.rs         # GET /metrics (Prometheus format)
│   │                          #   - tokens_generated_total
│   │                          #   - request_duration_seconds
│   │                          #   - gpu_memory_used_bytes
│   │                          #   - kv_cache_utilization
│   │                          #   - batch_size_current
│   │
│   └── shell/                 # minimal debug shell (serial console only)
│       ├── mod.rs
│       └── commands.rs        # ~30 essential commands (listed below)
│
├── firmware/                  # AMD GPU firmware blobs (git-lfs or download script)
│   └── amdgpu/
│       └── navi31/            # RX 7900 series firmware
│
├── gpu_kernels/               # Pre-compiled GPU compute kernels
│   ├── build.sh              # script to compile on Linux + ROCm → .hsaco
│   ├── src/
│   │   ├── gemm_f16.hip      # FP16 GEMM
│   │   ├── gemm_q4.hip       # quantized GEMM
│   │   ├── flash_attn.hip    # FlashAttention-2
│   │   ├── rmsnorm.hip       # RMSNorm
│   │   ├── rope.hip          # RoPE
│   │   ├── softmax.hip       # Softmax
│   │   └── silu.hip          # SiLU activation
│   └── bin/
│       └── navi31/            # compiled .hsaco binaries for RX 7900
│
├── models/                    # model files (not in git, download script provided)
│   └── download.sh           # download SmolLM-135M, Llama-3.2-1B, etc.
│
├── tools/
│   ├── mkimage.sh            # build bootable disk image
│   └── flash_usb.sh          # write image to USB drive for real hardware
│
├── docs/
│   ├── architecture.md
│   ├── gpu_driver.md
│   ├── benchmarking.md
│   └── deployment.md
│
├── Cargo.toml
├── Makefile
├── limine.conf
├── linker.ld
├── rust-toolchain.toml
├── CLAUDE.md                  # this file
├── README.md
└── LICENSE
```

---

## Code Migration from merlion-kernel

The following modules should be migrated from the `merlion-kernel` repository.
For each module: copy the source, strip out unnecessary dependencies, adapt to the new structure.

### Migrate AS-IS (minimal changes)
| Source (merlion-kernel) | Destination | Notes |
|------------------------|-------------|-------|
| Boot / Limine protocol | `src/boot/` | Keep UEFI path only, remove legacy BIOS |
| GDT, IDT, TSS | `src/arch/x86_64/` | Direct copy |
| APIC (Local + I/O) | `src/arch/x86_64/apic.rs` | Direct copy |
| HPET timer | `src/arch/x86_64/hpet.rs` | Direct copy |
| SMP detection/startup | `src/arch/x86_64/smp.rs` | Direct copy |
| Serial UART (NS16550) | `src/arch/x86_64/serial.rs` | Direct copy |
| Physical frame allocator | `src/memory/phys.rs` | Will be extended with huge page support |
| Page table management | `src/memory/virt.rs` | Will be extended |
| NVMe driver | `src/drivers/nvme/` | Direct copy |
| PCIe enumeration | `src/drivers/pcie/` | May need enhancement for ECAM, resizable BAR |
| virtio-net driver | `src/drivers/net/virtio_net.rs` | For QEMU development |
| e1000e driver | `src/drivers/net/e1000e.rs` | For real hardware |
| TCP/IP stack | `src/net/` | Migrate TCP, IPv4, ARP, DHCP, DNS. Remove UDP (add back if needed for DNS) |
| ACPI parser | wherever applicable | RSDP, MADT (for APIC), SRAT (for NUMA) |

### Migrate with MAJOR REFACTORING
| Source | Destination | Changes |
|--------|-------------|---------|
| Heap allocator (64KB linked-list) | `src/memory/heap.rs` | Keep for small kernel objects, but add buddy allocator for large allocations |
| Shell | `src/shell/` | Strip from 358 commands to ~30 |
| Scheduler | (embedded in engine) | Replace preemptive round-robin with static core assignment |
| TLS | `src/tls/` | Keep crypto primitives, rewrite handshake for TLS 1.3 only |

### DO NOT MIGRATE
| Module | Reason |
|--------|--------|
| VGA framebuffer, graphics | Not needed for headless inference server |
| Audio engine | Not needed |
| PS/2 keyboard driver | Not needed (serial console + network API) |
| USB xHCI driver | Not needed initially (can add later for USB NIC/debug) |
| Bluetooth | Not needed |
| FAT16 / ext2 / ext4 filesystem | Not needed (direct block read for model files) |
| RAM disk filesystem | Not needed |
| Ring 3 user mode | Everything runs in kernel mode |
| ELF-64 loader | No user programs |
| Signal framework | No user processes |
| Process groups | No user processes |
| Capabilities, seccomp, ACLs | Single-purpose appliance, no multi-user security |
| Screen saver, snake game, calculator | Entertainment features |
| Forth interpreter, Lisp interpreter | Language runtimes |
| WASM/WASI runtime | Not needed |
| NL shell (natural language) | Replaced by API server |
| LLM proxy over COM2 | Replaced by native inference engine |
| Semantic VFS, AI file tags | Not needed |
| AI system monitor, self-healing | Replaced by structured metrics endpoint |
| Agent framework | Not needed |
| AI-powered man pages | Not needed |
| HTTP/SSH/DNS/MQTT servers | Replaced by purpose-built API server |
| HTTPS reverse proxy | Not needed |
| wget, netstat (user tools) | Minimal shell equivalents only |
| Fortune, benchmarks (demo) | Replaced by `ai-bench` |

---

## Implementation Phases

### Phase 1: Bootable Skeleton (Week 1-2)

**Goal**: Boot on QEMU, print to serial, detect hardware.

1. Set up new repo with `Cargo.toml`, `rust-toolchain.toml` (nightly), `.cargo/config.toml`
2. Migrate boot code: Limine protocol, GDT, IDT, TSS
3. Migrate serial UART driver — all early output goes here
4. Migrate physical memory allocator (from Limine memory map)
5. Migrate kernel page table setup
6. Set up small heap allocator (for Vec, BTreeMap, String etc.)
7. Initialize SIMD state: CR0.EM clear, CR4.OSFXSR, CR4.OSXSAVE, XCR0 (SSE+AVX)
8. CPUID detection: report AVX2, AVX-512, AMX capabilities
9. SMP: detect cores, but park all APs for now (BSP only)
10. Minimal shell on serial: just `help`, `info`, `reboot`

**Verification**:
```
$ make run
MerlionOS Inference v0.1.0
CPU: AMD Ryzen 7 7700X (8 cores) [QEMU: whatever QEMU reports]
SIMD: AVX2=yes AVX-512=no AMX=no
RAM: 512MB usable
NVMe: not yet initialized
GPU: not yet initialized
Ready.
merlion> info
  Boot time: 0.3s
  Kernel size: 256KB
  Heap used: 12KB / 64KB
```

### Phase 2: Storage + Model Loading (Week 3-4)

**Goal**: Read a GGUF model file from NVMe into memory.

1. Migrate NVMe driver
2. Migrate PCIe enumeration (needed for NVMe discovery)
3. Implement large memory allocator: buddy allocator managing 256MB+ region with 2MB huge page support
4. Implement GGUF parser: read header, metadata, tensor info
5. Load model weights into huge-page-backed memory region
6. Implement `ai-load <offset> <size>` shell command (read model from raw NVMe offset)
   - Alternative: implement minimal FAT16 read-only driver just for model loading
7. Print model config after loading

**Verification**:
```
merlion> ai-load
[NVMe] Detected: Samsung 980 PRO 1TB
[Model] Loading SmolLM-135M Q4_0 (80MB)...
[Model] Read 80MB in 0.4s (200 MB/s)
[Model] Config: dim=576, layers=30, heads=9, kv_heads=3, vocab=49152
[Model] Tensors: 182 loaded, all Q4_0
[Model] Tokenizer: 49152 tokens loaded
[Memory] Weights: 80MB (40 x 2MB huge pages)
[Memory] KV cache: 27MB pre-allocated
[Memory] Scratch: 8MB pre-allocated
Ready for inference.
```

### Phase 3: CPU Inference Engine (Week 5-8)

**Goal**: Generate text from a prompt using CPU (scalar first, then AVX2).

1. Implement tensor.rs: BlockQ4_0, f16_to_f32, TensorView
2. Implement scalar math kernels: gemv_q4_0, rmsnorm, softmax, rope, silu, elementwise_add
3. Implement LlamaEngine: RunState allocation, forward() single-token pass
4. Implement BPE tokenizer (simplified greedy matching)
5. Implement sampler (temperature + top-p + argmax)
6. Implement generate loop with streaming token output
7. Shell command: `ai <prompt>` for interactive generation
8. Shell command: `ai-bench` for performance measurement
9. Add AVX2-optimized kernels (using `#[target_feature(enable = "avx2")]`)
10. Implement KernelDispatch: runtime selection between scalar/AVX2 based on CPUID

**Verification**:
```
merlion> ai Hello, I am MerlionOS
Hello, I am MerlionOS, a bare-metal operating system designed for high-performance
inference. I run directly on the hardware without the overhead of a traditional
operating system...
[Generated 64 tokens in 12.8s | 5.0 tok/s | AVX2 | CPU only]

merlion> ai-bench
[Bench] Model: SmolLM-135M Q4_0
[Bench] Backend: CPU AVX2
[Bench] Prefill: 32 tokens in 0.6s (53.3 tok/s)
[Bench] Decode: 64 tokens in 12.8s (5.0 tok/s)
[Bench] Peak memory: 115MB
```

### Phase 4: Network + API Server (Week 9-12)

**Goal**: Serve inference over HTTP, OpenAI-compatible API.

1. Migrate TCP/IP stack (TCP, IPv4, ARP, DHCP)
2. Migrate network driver (virtio-net for QEMU, e1000e for real HW)
3. Implement minimal HTTP/1.1 server (parse request, write response, chunked transfer for SSE)
4. Implement OpenAI-compatible endpoints:
   - `POST /v1/chat/completions` (streaming via SSE + non-streaming)
   - `POST /v1/completions`
   - `GET /v1/models`
5. Implement health endpoint: `GET /health`
6. Implement metrics endpoint: `GET /metrics` (Prometheus text format)
7. Integrate with inference scheduler: continuous batching for concurrent requests
8. TLS support (can defer to Phase 5 if complex)

**Verification**:
```bash
# From another machine on the network:
$ curl http://192.168.1.100:8080/v1/models
{"data": [{"id": "smollm-135m-q4", "object": "model"}]}

$ curl http://192.168.1.100:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm-135m-q4","messages":[{"role":"user","content":"What is Singapore?"}]}'
{"choices": [{"message": {"content": "Singapore is a city-state..."}}]}

$ curl http://192.168.1.100:8080/metrics
# HELP inference_tokens_total Total tokens generated
inference_tokens_total 1247
# HELP inference_request_duration_seconds Request latency
inference_request_duration_seconds{quantile="0.5"} 2.1
inference_request_duration_seconds{quantile="0.99"} 3.8
```

### Phase 5: Real Hardware + Benchmarking (Week 13-16)

**Goal**: Boot on ASUS TUF Gaming B650 + Ryzen 7 + RX 7900, benchmark vs Linux.

1. Build bootable USB image
2. Test UEFI boot on real hardware (Limine should just work)
3. Debug any hardware-specific issues (ACPI tables, NVMe quirks, etc.)
4. Add Realtek RTL8125 network driver (if needed for this motherboard)
5. Run inference on real CPU: measure tok/s, latency, memory usage
6. Set up identical Linux comparison: same hardware, same model, llama.cpp / vllm
7. Produce benchmark report comparing:
   - Throughput (tokens/sec)
   - First token latency (TTFT)
   - P99 latency
   - Memory usage
   - Boot time to first inference
   - Power consumption (if measurable)

### Phase 6: AMD GPU Driver + GPU Inference (Month 5-10)

**Goal**: Offload inference GEMM to AMD GPU.

This is the largest phase. See `docs/gpu_driver.md` for detailed plan.

High-level steps:
1. PCIe: enumerate AMD GPU, map BARs (MMIO + VRAM)
2. Firmware: load PSP, SDMA, GFX, SMU microcode from disk
3. Init: golden register sequence for RDNA3 (reference: Linux amdgpu driver)
4. Memory: VRAM allocator, CPU→GPU DMA transfers
5. Compute queue: create AQL queue, configure doorbell
6. Kernel dispatch: load pre-compiled .hsaco, submit dispatch packets
7. Sync: fence-based CPU-GPU synchronization
8. Integration: GPU backend for inference engine (gemv, attention on GPU; small ops on CPU)

### Phase 7: Production Hardening (Month 10-12)

1. TLS 1.3 for HTTPS API
2. Watchdog timer: auto-reboot on hang
3. Error recovery: GPU reset on timeout
4. Multi-model support: load/unload models via API
5. Graceful shutdown: drain requests on SIGTERM equivalent
6. Configuration file: model path, port, max batch size, etc.
7. Logging: structured log output over serial and/or network

---

## Shell Commands (Complete List)

Only ~30 commands. No entertainment, no demos.

### System
- `info` — system overview (CPU, RAM, GPU, model, uptime)
- `top` — real-time resource usage (CPU%, memory, GPU memory)
- `free` — memory breakdown (kernel heap, model weights, KV cache, GPU VRAM)
- `lspci` — PCI device list
- `dmesg` — kernel message buffer
- `uptime` — system uptime
- `reboot` — ACPI reboot
- `shutdown` — ACPI shutdown

### Storage
- `lsblk` — list block devices
- `nvme-info` — NVMe device details

### Network
- `ip` — show IP address and interface status
- `ping <host>` — ICMP ping
- `ss` — show active TCP connections

### Inference
- `ai-load [path]` — load a GGUF model
- `ai-unload` — unload current model, free memory
- `ai-info` — current model details + memory usage
- `ai <prompt>` — interactive text generation
- `ai-bench` — run standard benchmark (prefill + decode speed)
- `ai-serve [port]` — start HTTP API server (default: 8080)
- `ai-stop` — stop HTTP API server

### Files (minimal)
- `ls [path]` — list directory (if filesystem mounted)
- `cat <path>` — print file contents

### Debug
- `cpuid` — detailed CPU features
- `memmap` — physical memory map from bootloader
- `gpu-info` — GPU status (Phase 2: clocks, temp, VRAM usage)
- `gpu-reset` — reset GPU (Phase 2)
- `help` — list all commands

---

## Build & Run

### Development (QEMU)
```bash
# Prerequisites
rustup component add rust-src llvm-tools --toolchain nightly
# QEMU with enough RAM for model loading
sudo apt install qemu-system-x86

# Build kernel
make build

# Run in QEMU (serial console, 1GB RAM, NVMe disk with model)
make run
# Equivalent to:
# qemu-system-x86_64 \
#   -machine q35 \
#   -cpu qemu64,+avx2 \
#   -m 1G \
#   -drive file=disk.img,format=raw,if=virtio \
#   -serial stdio \
#   -nographic

# Run with KVM for AVX2/AVX-512 testing
make run-kvm
# adds: -enable-kvm -cpu host
```

### Real Hardware
```bash
# Build bootable USB image
make image    # produces merlion-inference.img

# Write to USB drive
sudo dd if=merlion-inference.img of=/dev/sdX bs=4M status=progress

# Boot: plug USB into ASUS TUF Gaming, enter BIOS:
#   - Disable Secure Boot
#   - Disable CSM
#   - Enable Above 4G Decoding
#   - Enable Resizable BAR
#   - Boot from USB
```

### Disk Image Layout
```
disk.img (256MB - 2GB depending on model)
├── EFI/                    # UEFI boot partition (FAT32)
│   └── BOOT/
│       ├── BOOTX64.EFI    # Limine UEFI bootloader
│       └── limine.conf     # Limine configuration
├── kernel.elf              # MerlionOS Inference kernel
├── firmware/               # AMD GPU firmware blobs
│   └── amdgpu/
└── models/                 # GGUF model files
    └── smollm-135m-q4_0.gguf
```

---

## Performance Targets

### Phase 3 (CPU only, SmolLM-135M Q4_0)
| Metric | Target | Notes |
|--------|--------|-------|
| Decode throughput | ≥5 tok/s (scalar), ≥15 tok/s (AVX2) | Single request |
| Prefill throughput | ≥50 tok/s | 32-token prompt |
| Boot to ready | <3 seconds | Including model load |
| Kernel memory footprint | <2MB | Excluding model and caches |
| Total memory usage | <150MB | Including model + KV cache |

### Phase 5 (CPU, real hardware, Ryzen 7 7700X, Llama-3.2-1B Q4_0)
| Metric | Target | Notes |
|--------|--------|-------|
| Decode throughput | ≥30 tok/s (AVX2) | Single request |
| TTFT (time to first token) | <200ms | 128-token prompt |
| P99 latency per token | <50ms | Under load (4 concurrent) |
| Boot to serving | <5 seconds | Full API server ready |

### Phase 6 (GPU, RX 7900 XT, Llama-3.1-8B Q4_0)
| Metric | Target | Beat Linux by |
|--------|--------|--------------|
| Decode throughput | ≥60 tok/s | 10-20% |
| TTFT | <100ms | 30-50% |
| P99 latency | <30ms | 50-70% |
| GPU utilization | >90% | 5-10% |

---

## Coding Conventions

### General
- All code is `no_std`. Only `core` and `alloc` crates.
- No external crates unless absolutely necessary. Every dependency is a liability.
- Exception: `libm` crate is allowed for math functions (exp, sqrt, powf) not in core.
- All public APIs documented with `///` doc comments.
- Use `Result<T, E>` for fallible operations, never panic in production paths.
- `panic!` is acceptable only in early boot (before inference engine init).

### Memory
- All tensor buffers must be 64-byte aligned (x86 cache line).
- Model weights use 2MB or 1GB huge pages. Never 4KB pages for large allocations.
- Zero dynamic allocation during inference hot path. Everything pre-allocated.
- KV cache pages are pre-allocated in a pool and assigned/recycled, never malloc'd.

### SIMD
- Never enable AVX2/AVX-512 globally via rustflags (breaks early boot before SIMD init).
- Use `#[target_feature(enable = "avx2")]` on individual functions.
- Always provide scalar fallback for every SIMD kernel.
- CPUID detection at boot → populate `KernelDispatch` function pointer table.

### Concurrency
- BSP (core 0): runs network stack, API server, scheduler.
- Worker cores (1-N): run inference forward passes. One request per core, no preemption.
- Inter-core communication: lock-free SPSC ring buffers (scheduler → worker).
- No spinlocks on hot path. Use atomic operations and memory ordering.

### GPU
- All GPU compute kernels are pre-compiled offline (HIP → .hsaco).
- No JIT compilation, no shader compiler in the kernel.
- GPU memory layout is static: model weights pinned at fixed VRAM offset.
- CPU↔GPU sync via polling (not interrupts) for latency-sensitive paths.

---

## Key Design Decisions

### Why kernel mode only (no Ring 3)?
Inference is a single workload. Ring 0→3 transitions cost ~100ns each (syscall + sysret). For decode (one forward pass per token), there are hundreds of kernel calls per token in a traditional OS. Eliminating this boundary removes a measurable source of latency.

### Why not just patch Linux?
Linux's generality is the problem. Even with PREEMPT_RT, you can't eliminate:
- VFS overhead for GPU memory management
- Scheduler wake-up latency for GPU completion notifications
- Memory management overhead (page faults, TLB shootdowns on multi-core)
- The entire DRM/KFD subsystem sitting between you and the GPU

A purpose-built system eliminates these by construction.

### Why AMD GPU (not NVIDIA)?
- AMD's GPU driver stack (amdgpu, ROCm) is open source. NVIDIA's is closed.
- Open source means we can study and reimplement the minimal subset we need.
- AMD GPU documentation is more accessible than NVIDIA's.
- Strategic opportunity: most inference companies are locked into NVIDIA/CUDA. An optimized AMD path is differentiated.

### Why GGUF model format?
- Self-describing: model config + tokenizer + weights in one file.
- Quantization-native: Q4_0, Q8_0 etc. are first-class in the format.
- Ecosystem: every popular model has GGUF quantized versions on Hugging Face.
- Simple: can be parsed with ~500 lines of Rust. No protobuf, no pickle.

---

## References

### Inference Engine
- **llama2.c** by Andrej Karpathy — https://github.com/karpathy/llama2.c
  - Single-file C reference implementation. Best starting point for forward() logic.
- **llama.cpp** — https://github.com/ggerganov/llama.cpp
  - GGUF format authority. Reference for quantization kernels and AVX2 implementation.
- **Bare-Metal Tensor Virtualization** — https://arxiv.org/pdf/2601.03324
  - ARM64 bare-metal inference paper. Demand paging weights, cache optimization.

### GPU Driver
- **Linux amdgpu driver** — https://github.com/torvalds/linux/tree/master/drivers/gpu/drm/amd
  - The authoritative reference for AMD GPU initialization sequences.
- **AMD ROCm** — https://github.com/ROCm/ROCm
  - HIP runtime, compute queue management, .hsaco format.
- **LLVM AMDGPU backend** — https://llvm.org/docs/AMDGPUUsage.html
  - ISA documentation, register definitions, ABI.
- **Experimental Python AMD driver** — referenced in Tom's Hardware March 2026
  - Minimal compute driver via /dev/kfd interface. Architecture reference.

### Inference Serving
- **vLLM** — https://github.com/vllm-project/vllm
  - PagedAttention, continuous batching. Reference for scheduler design.
- **SGLang** — https://github.com/sgl-project/sglang
  - RadixAttention, efficient KV cache reuse. Reference for cache management.

### OS Development
- **Original MerlionOS** — https://github.com/MerlionOS/merlion-kernel
  - Source of migrated modules.
- **OSDev Wiki** — https://wiki.osdev.org
  - Hardware reference for x86_64, PCIe, NVMe, etc.
