#!/bin/bash
# Build a bootable disk image for MerlionOS Inference.
# Usage: ./tools/mkimage.sh [model.gguf]
#
# Prerequisites:
#   - Limine bootloader: git clone https://github.com/limine-bootloader/limine --branch=v8.x-binary --depth=1
#   - xorriso: brew install xorriso (macOS) / apt install xorriso (Linux)

set -euo pipefail

KERNEL="target/x86_64-unknown-none/release/merlion-infer"
ISO="merlionos-inference.iso"
LIMINE_DIR="${LIMINE_DIR:-limine}"

if [ ! -f "$KERNEL" ]; then
    echo "Building kernel (release)..."
    cargo build --release
fi

if [ ! -d "$LIMINE_DIR" ]; then
    echo "Error: Limine not found at $LIMINE_DIR"
    echo "Clone it: git clone https://github.com/limine-bootloader/limine --branch=v8.x-binary --depth=1"
    exit 1
fi

echo "Creating ISO..."
mkdir -p iso_root/boot iso_root/EFI/BOOT

cp "$KERNEL" iso_root/boot/kernel.elf
cp limine.conf iso_root/boot/
cp "$LIMINE_DIR/limine-bios.sys" iso_root/boot/ 2>/dev/null || true
cp "$LIMINE_DIR/limine-bios-cd.bin" iso_root/boot/ 2>/dev/null || true
cp "$LIMINE_DIR/limine-uefi-cd.bin" iso_root/boot/ 2>/dev/null || true
cp "$LIMINE_DIR/BOOTX64.EFI" iso_root/EFI/BOOT/ 2>/dev/null || true
cp "$LIMINE_DIR/BOOTIA32.EFI" iso_root/EFI/BOOT/ 2>/dev/null || true

# Optionally include a model file
if [ -n "${1:-}" ] && [ -f "$1" ]; then
    echo "Including model: $1"
    mkdir -p iso_root/models
    cp "$1" iso_root/models/
fi

xorriso -as mkisofs \
    -b boot/limine-bios-cd.bin \
    -no-emul-boot -boot-load-size 4 -boot-info-table \
    --efi-boot boot/limine-uefi-cd.bin \
    -efi-boot-part --efi-boot-image \
    --protective-msdos-label \
    iso_root -o "$ISO" 2>/dev/null

# Install Limine for BIOS boot
"$LIMINE_DIR/limine" bios-install "$ISO" 2>/dev/null || true

echo "Created: $ISO ($(du -h "$ISO" | cut -f1))"
echo ""
echo "To boot in QEMU (UEFI):"
echo "  qemu-system-x86_64 -bios /usr/share/OVMF/OVMF_CODE.fd -cdrom $ISO -serial stdio -m 1G"
echo ""
echo "To write to USB:"
echo "  sudo dd if=$ISO of=/dev/sdX bs=4M status=progress"

rm -rf iso_root
