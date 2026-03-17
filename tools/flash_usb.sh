#!/bin/bash
# Write MerlionOS Inference ISO to a USB drive.
# Usage: sudo ./tools/flash_usb.sh /dev/sdX

set -euo pipefail

ISO="merlionos-inference.iso"
DEVICE="${1:-}"

if [ -z "$DEVICE" ]; then
    echo "Usage: sudo $0 /dev/sdX"
    echo ""
    echo "Available devices:"
    lsblk -d -o NAME,SIZE,MODEL 2>/dev/null || diskutil list 2>/dev/null
    exit 1
fi

if [ ! -f "$ISO" ]; then
    echo "Error: $ISO not found. Run ./tools/mkimage.sh first."
    exit 1
fi

echo "WARNING: This will ERASE all data on $DEVICE"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -r

echo "Writing $ISO to $DEVICE..."
dd if="$ISO" of="$DEVICE" bs=4M status=progress conv=fsync
sync

echo "Done. Boot from USB with:"
echo "  - Disable Secure Boot in BIOS"
echo "  - Enable Above 4G Decoding"
echo "  - Enable Resizable BAR (for GPU)"
echo "  - Select USB in boot menu"
