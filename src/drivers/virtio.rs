/// Virtio device discovery.

use crate::drivers::pci;
use alloc::vec::Vec;
use alloc::string::String;

pub const VIRTIO_VENDOR: u16 = 0x1AF4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VirtioDeviceType {
    Network,
    Block,
    Console,
    Entropy,
    Unknown(u16),
}

impl VirtioDeviceType {
    pub fn from_device_id(id: u16) -> Self {
        match id {
            0x1000 | 0x1041 => Self::Network,
            0x1001 | 0x1042 => Self::Block,
            0x1003 | 0x1043 => Self::Console,
            0x1005 | 0x1044 => Self::Entropy,
            _ => Self::Unknown(id),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Network => "virtio-net",
            Self::Block => "virtio-blk",
            Self::Console => "virtio-console",
            Self::Entropy => "virtio-rng",
            Self::Unknown(_) => "virtio-unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VirtioDevice {
    pub pci: pci::PciDevice,
    pub device_type: VirtioDeviceType,
}

impl VirtioDevice {
    pub fn summary(&self) -> String {
        alloc::format!("{} at {:02x}:{:02x}.{}",
            self.device_type.name(),
            self.pci.bus, self.pci.device, self.pci.function)
    }
}

pub fn scan() -> Vec<VirtioDevice> {
    pci::scan()
        .into_iter()
        .filter(|d| d.vendor_id == VIRTIO_VENDOR)
        .map(|d| {
            let device_type = VirtioDeviceType::from_device_id(d.device_id);
            VirtioDevice { pci: d, device_type }
        })
        .collect()
}

pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;

pub const VIRTIO_STATUS_ACKNOWLEDGE: u8 = 1;
pub const VIRTIO_STATUS_DRIVER: u8 = 2;
pub const VIRTIO_STATUS_DRIVER_OK: u8 = 4;
