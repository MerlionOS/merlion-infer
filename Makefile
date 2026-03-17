.PHONY: build release run run-gui run-fullscreen run-net run-disk run-full run-kvm image disk clean limine

KERNEL := target/x86_64-unknown-none/debug/merlion-infer
KERNEL_REL := target/x86_64-unknown-none/release/merlion-infer
ISO := merlionos-inference.iso

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

# Build debug ISO (for QEMU development)
debug-iso: build limine
	@mkdir -p iso_root/boot iso_root/EFI/BOOT
	@cp $(KERNEL) iso_root/boot/kernel.elf
	@cp limine.conf iso_root/boot/
	@cp limine/limine-bios.sys iso_root/boot/ 2>/dev/null || true
	@cp limine/limine-bios-cd.bin iso_root/boot/ 2>/dev/null || true
	@cp limine/limine-uefi-cd.bin iso_root/boot/ 2>/dev/null || true
	@cp limine/BOOTX64.EFI iso_root/EFI/BOOT/ 2>/dev/null || true
	@xorriso -as mkisofs \
		-b boot/limine-bios-cd.bin \
		-no-emul-boot -boot-load-size 4 -boot-info-table \
		--efi-boot boot/limine-uefi-cd.bin \
		-efi-boot-part --efi-boot-image \
		--protective-msdos-label \
		iso_root -o $(ISO) 2>/dev/null
	@limine/limine bios-install $(ISO) 2>/dev/null || true
	@rm -rf iso_root

# Ensure Limine is available
limine:
	@if [ ! -f limine/BOOTX64.EFI ]; then \
		echo "Cloning Limine bootloader..."; \
		git clone https://github.com/limine-bootloader/limine --branch=v8.x-binary --depth=1; \
	fi
	@if [ ! -f limine/limine ]; then \
		cc -o limine/limine limine/limine.c; \
	fi

# Run in QEMU — serial only, headless, 1GB RAM
run: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO)

# Run in QEMU with framebuffer display (SDL window + serial)
run-gui: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-display cocoa \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO)

# Run in QEMU fullscreen
run-fullscreen: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-serial stdio \
		-display cocoa \
		-full-screen \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO)

# Run with network (virtio-net, port 8080 forwarded)
run-net: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO) \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Run with disk (virtio-blk for model loading)
run-disk: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO) \
		-drive file=disk.img,format=raw,if=virtio

# Run with disk + network
run-full: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-cpu qemu64,+avx2,+sse4.1,+sse4.2,+ssse3 \
		-m 1G \
		-nographic \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO) \
		-drive file=disk.img,format=raw,if=virtio \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Run with KVM (Linux, real AVX2/AVX-512)
run-kvm: debug-iso
	qemu-system-x86_64 \
		-machine q35 \
		-enable-kvm -cpu host \
		-m 4G \
		-nographic \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-cdrom $(ISO) \
		-drive file=disk.img,format=raw,if=virtio \
		-netdev user,id=n0,hostfwd=tcp::8080-:8080 \
		-device virtio-net-pci,netdev=n0

# Download SmolLM2-135M and create disk.img
disk:
	./tools/download_model.sh

# Build release bootable ISO image
image: release limine
	./tools/mkimage.sh

# Clean build artifacts
clean:
	cargo clean
	rm -f $(ISO)
