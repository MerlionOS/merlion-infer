.PHONY: build release run run-gui run-net run-disk run-full run-kvm image clean

KERNEL := target/x86_64-unknown-none/debug/merlion-infer
KERNEL_REL := target/x86_64-unknown-none/release/merlion-infer

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
		-serial stdio \
		-nographic \
		-bios /opt/homebrew/share/qemu/edk2-x86_64-code.fd \
		-kernel $(KERNEL)

# Run in QEMU with framebuffer display (SDL window + serial)
run-gui: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-display sdl \
		-bios /opt/homebrew/share/qemu/edk2-x86_64-code.fd \
		-kernel $(KERNEL)

# Run with network (virtio-net, port 8080 forwarded)
run-net: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-nographic \
		-bios /opt/homebrew/share/qemu/edk2-x86_64-code.fd \
		-kernel $(KERNEL) \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Run with disk (virtio-blk)
run-disk: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-nographic \
		-bios /opt/homebrew/share/qemu/edk2-x86_64-code.fd \
		-kernel $(KERNEL) \
		-drive file=disk.img,format=raw,if=virtio

# Run with disk + network
run-full: build
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-nographic \
		-bios /opt/homebrew/share/qemu/edk2-x86_64-code.fd \
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
		-serial stdio \
		-nographic \
		-bios /usr/share/OVMF/OVMF_CODE.fd \
		-kernel $(KERNEL) \
		-drive file=disk.img,format=raw,if=virtio \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Build bootable ISO image
image: release
	./tools/mkimage.sh

# Clean build artifacts
clean:
	cargo clean
	rm -f merlionos-inference.iso
