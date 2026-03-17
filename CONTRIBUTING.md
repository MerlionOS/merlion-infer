# Contributing to MerlionOS Inference

Thanks for your interest in MerlionOS Inference! This is a bare-metal OS purpose-built for LLM inference on x86_64 + AMD GPU. Every line of code serves inference.

## Getting Started

### Prerequisites

- Rust nightly toolchain
- QEMU (for testing)
- x86_64 host (for KVM testing with real AVX2/AVX-512)

### Build & Run

```bash
# Install Rust nightly and required components
rustup toolchain install nightly
rustup component add rust-src llvm-tools --toolchain nightly

# Build
make build

# Run in QEMU (serial console)
make run

# Run in QEMU with framebuffer display
make run-gui

# Run with KVM on Linux (real SIMD)
make run-kvm
```

## How to Contribute

### Good First Issues

Look for issues labeled `good first issue` — these are scoped, well-defined tasks suitable for newcomers to the project.

### Areas Where Help is Needed

- **Drivers**: USB Ethernet, NVMe quirks for specific hardware
- **Inference kernels**: AVX-512 optimized kernels, ARM NEON port
- **GPU compute**: AMD RDNA3/CDNA driver work
- **Testing**: Running on different hardware, reporting boot issues
- **Documentation**: Architecture docs, hardware compatibility lists

### Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes — keep PRs focused on a single concern
3. Ensure `cargo build` succeeds with no new warnings
4. Test in QEMU: `make run` (serial) and `make run-gui` (framebuffer)
5. Submit a PR with a clear description of what and why

### Code Style

- All code is `no_std` — only `core` and `alloc` crates
- Minimize external dependencies. Every crate is a liability in a kernel
- Use `Result<T, E>` for fallible operations. Don't panic in production paths
- `panic!` is acceptable only in early boot (before inference engine init)
- All tensor buffers must be 64-byte aligned (x86 cache line)
- Provide scalar fallback for every SIMD kernel

### Commit Messages

- Use imperative mood: "Add framebuffer driver" not "Added framebuffer driver"
- Keep the first line under 72 characters
- Reference issue numbers where applicable

## Architecture

```
OpenAI-Compatible API
        |
  Inference Engine (AVX2/AVX-512 + GPU)
        |
  Memory Manager (physical frames, kernel heap)
        |
  Hardware: NVMe | PCIe | Network | AMD GPU
        |
  x86_64 HAL (GDT, IDT, APIC, SIMD state)
        |
  UEFI Boot (Limine)
```

This is NOT a general-purpose OS. It's firmware for LLM inference servers. Think ESXi for inference.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
