/// AMD GPU firmware loader.
///
/// Loads MEC (Micro Engine Compute) microcode from disk.
/// Firmware is expected at a fixed disk offset so we can find it
/// without a filesystem.
///
/// Disk layout convention:
///   Sector 0:            GGUF model data
///   Sector FW_SECTOR:    Firmware header + MEC microcode
///
/// The firmware on disk starts with a simple header:
///   [0x00] u32: magic = 0x4D454346 ("MECF")
///   [0x04] u32: microcode size in bytes
///   [0x08] u32: ucode_version
///   [0x0C] u32: reserved
///   [0x10] ...: microcode dwords
///
/// To prepare firmware:
///   ./tools/write_firmware.sh polaris11_mec.bin disk.img

use alloc::vec;

/// Firmware starts at sector 262144 (= 128 MiB offset).
/// This gives plenty of room for models up to 128 MiB.
pub const FW_SECTOR: u64 = 262144;

/// Magic bytes at start of firmware region: "MECF" (MEC Firmware).
const FW_MAGIC: u32 = 0x4D454346;

/// Firmware header (16 bytes).
#[repr(C)]
struct FwHeader {
    magic: u32,
    size_bytes: u32,
    ucode_version: u32,
    reserved: u32,
}

/// Try to load MEC firmware from disk at the well-known sector.
/// Returns true if firmware was found and uploaded to the GPU.
pub fn load_from_disk() -> bool {
    let has_disk = crate::drivers::virtio_blk::is_detected()
        || crate::drivers::nvme::is_detected();
    if !has_disk {
        crate::serial_println!("[gpu-fw] No disk available");
        return false;
    }

    if !super::mmio::is_initialized() {
        crate::serial_println!("[gpu-fw] GPU MMIO not initialized");
        return false;
    }

    // Read the first sector of the firmware region to check for header
    let mut header_buf = [0u8; 512];
    let result = if crate::drivers::virtio_blk::is_detected() {
        crate::drivers::virtio_blk::read_sector(FW_SECTOR, &mut header_buf)
    } else {
        crate::drivers::nvme::read_sector(FW_SECTOR, &mut header_buf)
    };

    if result.is_err() {
        crate::serial_println!("[gpu-fw] Failed to read firmware sector {}", FW_SECTOR);
        return false;
    }

    // Check magic
    let magic = u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
    if magic != FW_MAGIC {
        crate::serial_println!("[gpu-fw] No firmware at sector {} (magic={:#010x}, expected {:#010x})",
            FW_SECTOR, magic, FW_MAGIC);
        return false;
    }

    let size_bytes = u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]) as usize;
    let ucode_version = u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]]);

    crate::serial_println!("[gpu-fw] Found MEC firmware: {} bytes, version {:#x}", size_bytes, ucode_version);

    if size_bytes == 0 || size_bytes > 512 * 1024 {
        crate::serial_println!("[gpu-fw] Invalid firmware size");
        return false;
    }

    // Read the full firmware (header is 16 bytes, data follows)
    let total_size = 16 + size_bytes;
    let total_sectors = (total_size + 511) / 512;
    let total_bytes = total_sectors * 512;
    let mut buf = vec![0u8; total_bytes];

    // Copy header sector we already read
    buf[..512].copy_from_slice(&header_buf);

    // Read remaining sectors if needed
    if total_sectors > 1 {
        let remaining = &mut buf[512..];
        let result = if crate::drivers::virtio_blk::is_detected() {
            crate::drivers::virtio_blk::read_sectors(FW_SECTOR + 1, remaining)
        } else {
            crate::drivers::nvme::read_sectors(FW_SECTOR + 1, remaining)
        };
        if result.is_err() {
            crate::serial_println!("[gpu-fw] Failed to read firmware data");
            return false;
        }
    }

    // Extract microcode dwords (skip 16-byte header)
    let ucode_data = &buf[16..16 + size_bytes];
    if size_bytes % 4 != 0 {
        crate::serial_println!("[gpu-fw] Firmware size not dword-aligned");
        return false;
    }

    let n_dwords = size_bytes / 4;
    let mut dwords = vec![0u32; n_dwords];
    for i in 0..n_dwords {
        let off = i * 4;
        dwords[i] = u32::from_le_bytes([
            ucode_data[off], ucode_data[off+1], ucode_data[off+2], ucode_data[off+3]
        ]);
    }

    crate::serial_println!("[gpu-fw] Uploading {} dwords to MEC...", n_dwords);

    // Upload to GPU
    super::compute::upload_mec_firmware(&dwords)
}
