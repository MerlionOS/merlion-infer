.PHONY: build run run-serial clean iso

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

# Run with KVM (Linux, for real AVX2/AVX-512)
run-kvm: build
	qemu-system-x86_64 \
		-machine q35 \
		-enable-kvm -cpu host \
		-m 1G \
		-serial stdio \
		-nographic \
		-bios /usr/share/OVMF/OVMF_CODE.fd \
		-kernel $(KERNEL)

# Clean build artifacts
clean:
	cargo clean
