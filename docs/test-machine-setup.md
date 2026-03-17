# Test Machine Setup: MacBook Pro 2017 + Ubuntu 24

## Hardware
- CPU: Intel Core i7 (Kaby Lake) — AVX2 support
- GPU: AMD Radeon Pro 555/560 (Polaris 11, GCN 4)
- RAM: 16 GB
- Storage: SSD

## Prerequisites

```bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
rustup component add rust-src

# Install QEMU + KVM
sudo apt install qemu-system-x86 qemu-utils ovmf

# Verify KVM
sudo apt install cpu-checker
kvm-ok  # should say "KVM acceleration can be used"

# Install xorriso (for ISO creation)
sudo apt install xorriso

# Clone the repo
git clone https://github.com/MerlionOS/merlion-infer.git
cd merlion-infer
```

## Test 1: Build with AVX2

```bash
# Native x86_64 build with AVX2 kernels enabled
RUSTFLAGS="--cfg avx2_available" cargo build --release

# Verify
file target/x86_64-unknown-none/release/merlion-infer
# should show: ELF 64-bit LSB executable, x86-64
```

## Test 2: Run in QEMU with KVM

```bash
# Create test model + disk image
python3 tools/create_test_gguf.py

# Run with KVM (near-native speed!)
make run-kvm

# In the shell:
merlion> ai-load      # Load model from disk
merlion> ai-bench     # Benchmark (expect much faster than TCG)
merlion> ai Hello world from real hardware
```

## Test 3: Download and test SmolLM2-135M

```bash
# Download real model (~87 MiB)
./tools/download_model.sh

# Run with KVM + disk
make run-kvm

# In shell:
merlion> ai-load   # Will load SmolLM2-135M (takes a few seconds)
merlion> ai What is Singapore?
```

## Test 4: Boot from USB (bare metal)

```bash
# Build release + ISO
RUSTFLAGS="--cfg avx2_available" cargo build --release
./tools/mkimage.sh

# Write to USB (replace /dev/sdX with your USB device)
sudo ./tools/flash_usb.sh /dev/sdX

# Boot MacBook from USB:
# 1. Restart, hold Option key during boot
# 2. Select "EFI Boot" from the boot menu
# 3. MerlionOS should boot to serial console
#    (connect another machine via USB-serial or use internal display)
```

**Note:** Bare metal boot requires:
- Disable Secure Boot (should already be off with Ubuntu)
- The MacBook's display works as framebuffer (but our kernel uses serial only)
- For serial output, use a USB-to-serial adapter on another machine

## Test 5: GPU Detection

With KVM or bare metal boot:
```
merlion> gpu-info
gpu: AMD device_id=0x67ef BAR0=0x... BAR2=0x...
Chip: Polaris 11 (RX 460/560) [GCN4 (Polaris)]
Compute: GCN4/Polaris — gfx803 ISA, 4GB VRAM
```

## Test 6: Network + API

```bash
# Run with KVM + network
qemu-system-x86_64 \
  -machine q35 -enable-kvm -cpu host \
  -m 4G -serial stdio -nographic \
  -bios /usr/share/OVMF/OVMF_CODE.fd \
  -kernel target/x86_64-unknown-none/release/merlion-infer \
  -drive file=disk.img,format=raw,if=virtio \
  -netdev user,id=n0,hostfwd=tcp::8080-:8080 \
  -device virtio-net-pci,netdev=n0

# In shell:
merlion> ai-load
merlion> ai-serve 8080

# From another terminal:
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Expected Performance (KVM, test model)

| Metric | TCG (ARM Mac) | KVM (Intel Mac) | Improvement |
|--------|---------------|-----------------|-------------|
| Prefill | 16.8 tok/s | ~500+ tok/s | ~30x |
| Decode | 14.3 tok/s | ~200+ tok/s | ~15x |
| Boot time | ~3 seconds | ~1 second | ~3x |
| Model load | ~2 minutes | ~1 second | ~120x |
