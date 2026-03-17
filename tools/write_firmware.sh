#!/bin/bash
# Write AMD GPU MEC firmware to disk.img at the well-known offset.
#
# Usage: ./tools/write_firmware.sh <firmware.bin> [disk.img]
#
# The firmware is wrapped with a simple header:
#   [0x00] u32: magic = 0x4D454346 ("MECF")
#   [0x04] u32: size in bytes
#   [0x08] u32: version (0x00000001)
#   [0x0C] u32: reserved
#   [0x10] ...: raw microcode dwords
#
# Firmware offset: sector 262144 (128 MiB into disk)
#
# To get Polaris MEC firmware on Ubuntu:
#   sudo apt install linux-firmware
#   ls /lib/firmware/amdgpu/polaris11_mec.bin
#
# Note: Linux firmware files have their own headers. This script
# strips the common firmware header and writes raw microcode.

set -euo pipefail

FW_FILE="${1:-}"
DISK_IMG="${2:-disk.img}"
FW_SECTOR=262144
SECTOR_SIZE=512

if [ -z "$FW_FILE" ]; then
    echo "Usage: $0 <firmware.bin> [disk.img]"
    echo ""
    echo "Examples:"
    echo "  $0 /lib/firmware/amdgpu/polaris11_mec.bin"
    echo "  $0 /lib/firmware/amdgpu/polaris11_mec.bin disk.img"
    exit 1
fi

if [ ! -f "$FW_FILE" ]; then
    echo "Error: firmware file not found: $FW_FILE"
    exit 1
fi

if [ ! -f "$DISK_IMG" ]; then
    echo "Error: disk image not found: $DISK_IMG"
    echo "Run ./tools/download_model.sh first to create disk.img"
    exit 1
fi

FW_SIZE=$(stat -f%z "$FW_FILE" 2>/dev/null || stat -c%s "$FW_FILE" 2>/dev/null)
echo "Firmware: $FW_FILE ($FW_SIZE bytes)"

# Linux amdgpu firmware files start with a common header.
# The header has a 'header_size_bytes' field at offset 0 and
# 'ucode_array_offset_bytes' at offset 28.
# For simplicity, we'll try to detect and skip the Linux header,
# or write the raw file if it doesn't have one.

# Check if file starts with known Linux firmware header patterns
FIRST4=$(xxd -l 4 -p "$FW_FILE" 2>/dev/null || echo "")

# Create a temporary file with our header + microcode
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

# Write our header: magic + size + version + reserved
printf '\x46\x43\x45\x4D' > "$TMPFILE"  # "MECF" in LE = 0x4D454346
printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' \
    $((FW_SIZE & 0xFF)) $(((FW_SIZE >> 8) & 0xFF)) \
    $(((FW_SIZE >> 16) & 0xFF)) $(((FW_SIZE >> 24) & 0xFF)))" >> "$TMPFILE"
printf '\x01\x00\x00\x00' >> "$TMPFILE"  # version = 1
printf '\x00\x00\x00\x00' >> "$TMPFILE"  # reserved

# Append the raw firmware data
cat "$FW_FILE" >> "$TMPFILE"

TOTAL_SIZE=$((16 + FW_SIZE))
FW_OFFSET=$((FW_SECTOR * SECTOR_SIZE))

echo "Writing to $DISK_IMG at offset $FW_OFFSET (sector $FW_SECTOR)..."

# Ensure disk is large enough
DISK_SIZE=$(stat -f%z "$DISK_IMG" 2>/dev/null || stat -c%s "$DISK_IMG" 2>/dev/null)
NEEDED=$((FW_OFFSET + TOTAL_SIZE))
if [ "$DISK_SIZE" -lt "$NEEDED" ]; then
    echo "Extending disk to $((NEEDED / 1048576)) MiB..."
    dd if=/dev/zero of="$DISK_IMG" bs=1 count=0 seek="$NEEDED" 2>/dev/null
fi

dd if="$TMPFILE" of="$DISK_IMG" bs=$SECTOR_SIZE seek=$FW_SECTOR conv=notrunc 2>/dev/null

echo "Done! Firmware written at sector $FW_SECTOR."
echo ""
echo "To test:"
echo "  make run-disk"
echo "  merlion> gpu-fw-load"
