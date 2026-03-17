.PHONY: build release run run-gui run-net run-disk run-full run-kvm image disk clean

KERNEL := target/x86_64-unknown-none/debug/merlion-infer
KERNEL_REL := target/x86_64-unknown-none/release/merlion-infer

# Auto-detect UEFI firmware path
OVMF_CODE := $(shell \
	for f in \
		/opt/homebrew/share/qemu/edk2-x86_64-code.fd \
		/opt/homebrew/share/qemu/edk2-x86_64-secure-code.fd \
		/usr/share/OVMF/OVMF_CODE.fd \
		/usr/share/edk2/x64/OVMF_CODE.fd \
		/usr/share/qemu/edk2-x86_64-code.fd \
	; do [ -f "$$f" ] && echo "$$f" && break; done)

# Build kernel (debug)
build:
	cargo build

# Build kernel (release)
release:
	cargo build --release

# Run in QEMU — serial only, headless, 1GB RAM
run: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-bios $(OVMF_CODE) \
		-kernel $(KERNEL)

# Run in QEMU with framebuffer display via Limine ISO.
# Must boot from ISO (not -kernel) so Limine provides the GOP framebuffer.
run-gui: image
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-display sdl \
		-bios $(OVMF_CODE) \
		-cdrom merlionos-inference.iso

# Run with network (virtio-net, port 8080 forwarded)
run-net: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-bios $(OVMF_CODE) \
		-kernel $(KERNEL) \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Run with disk (virtio-blk)
run-disk: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-bios $(OVMF_CODE) \
		-kernel $(KERNEL) \
		-drive file=disk.img,format=raw,if=virtio

# Run with disk + network
run-full: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-bios $(OVMF_CODE) \
		-kernel $(KERNEL) \
		-drive file=disk.img,format=raw,if=virtio \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Run with KVM (Linux, real AVX2/AVX-512)
run-kvm: build
	qemu-system-x86_64 \
		-machine q35 \
		-enable-kvm -cpu host \
		-m 4G \
		-nographic \
		-bios $(OVMF_CODE) \
		-kernel $(KERNEL) \
		-drive file=disk.img,format=raw,if=virtio \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Download SmolLM2-135M and create disk.img
disk:
	./tools/download_model.sh

# Build bootable ISO image
image: release
	./tools/mkimage.sh

# Clean build artifacts
clean:
	cargo clean
	rm -f merlionos-inference.iso
