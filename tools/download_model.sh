#!/bin/bash
# Download a small GGUF model and create a disk image for QEMU.
# Usage: ./tools/download_model.sh
#
# Downloads SmolLM-135M Q4_0 (~80MB) from Hugging Face and writes it
# as raw data into disk.img for use with:
#   make run-disk   (virtio-blk)
#   make run-full   (virtio-blk + virtio-net)

set -euo pipefail

MODEL_DIR="models"
MODEL_NAME="smollm-135m-q4_0.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
DISK_IMG="disk.img"

# HuggingFace URL for SmolLM2-135M GGUF (Q4_0 quantization, ~87 MiB)
MODEL_URL="https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.Q4_0.gguf"

mkdir -p "$MODEL_DIR"

# Download model if not already present
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists: $MODEL_PATH"
else
    echo "Downloading SmolLM-135M Q4_0..."
    echo "URL: $MODEL_URL"

    if command -v curl &>/dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL"
    elif command -v wget &>/dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL"
    else
        echo "Error: curl or wget required"
        exit 1
    fi

    echo "Downloaded: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
fi

# Verify it's a GGUF file
MAGIC=$(xxd -l 4 -p "$MODEL_PATH" 2>/dev/null || echo "")
if [ "$MAGIC" != "47475546" ]; then
    echo "Warning: file does not start with GGUF magic (got: $MAGIC)"
    echo "The file may not be a valid GGUF model."
fi

# Create disk image with the model as raw data
MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
# Round up to next MB
DISK_SIZE=$(( (MODEL_SIZE / 1048576 + 2) * 1048576 ))

echo "Creating disk image: $DISK_IMG ($(( DISK_SIZE / 1048576 )) MiB)"
dd if=/dev/zero of="$DISK_IMG" bs=1M count=$(( DISK_SIZE / 1048576 )) 2>/dev/null
dd if="$MODEL_PATH" of="$DISK_IMG" bs=512 conv=notrunc 2>/dev/null

echo ""
echo "Done! Model written to $DISK_IMG at LBA 0."
echo ""
echo "To test:"
echo "  make run-disk    # boot with disk"
echo "  merlion> ai-load # parse GGUF from disk"
echo ""
echo "Model info:"
echo "  File: $MODEL_PATH"
echo "  Size: $(( MODEL_SIZE / 1048576 )) MiB ($MODEL_SIZE bytes)"
echo "  Disk: $DISK_IMG ($(( DISK_SIZE / 1048576 )) MiB)"
