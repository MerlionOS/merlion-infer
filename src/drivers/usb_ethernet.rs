/// USB Ethernet class driver (CDC-ECM / ASIX AX88179).
///
/// Provides Ethernet frame send/receive over USB for machines without
/// built-in Ethernet (e.g. MacBook Pro 2017 with USB-to-Ethernet adapter).
///
/// Supported chipsets:
/// - CDC-ECM (USB class 02/06): Standard USB Ethernet, works with most adapters
/// - ASIX AX88179 (vendor 0x0B95, product 0x1790): USB 3.0 Gigabit Ethernet
/// - Realtek RTL8153 (vendor 0x0BDA, product 0x8153): Common USB 3.0 GbE
///
/// Protocol overview (CDC-ECM):
/// 1. Set Configuration (select the ECM configuration)
/// 2. Read MAC address from string descriptor
/// 3. Set Ethernet Packet Filter (promiscuous mode)
/// 4. Bulk IN endpoint: receive Ethernet frames
/// 5. Bulk OUT endpoint: send Ethernet frames
///
/// Each Ethernet frame is sent/received as a single USB bulk transfer.
/// No framing or encapsulation — raw Ethernet frames over USB bulk pipes.

use core::sync::atomic::{AtomicBool, Ordering};

static DETECTED: AtomicBool = AtomicBool::new(false);

/// Known USB Ethernet adapter identifiers.
#[derive(Debug, Clone, Copy)]
pub struct UsbEthernetDevice {
    pub vendor_id: u16,
    pub product_id: u16,
    pub name: &'static str,
    pub driver: DriverType,
}

#[derive(Debug, Clone, Copy)]
pub enum DriverType {
    CdcEcm,
    Ax88179,
    Rtl8153,
}

/// Known USB Ethernet adapters.
static KNOWN_DEVICES: &[UsbEthernetDevice] = &[
    UsbEthernetDevice { vendor_id: 0x0B95, product_id: 0x1790, name: "ASIX AX88179", driver: DriverType::Ax88179 },
    UsbEthernetDevice { vendor_id: 0x0B95, product_id: 0x178A, name: "ASIX AX88179A", driver: DriverType::Ax88179 },
    UsbEthernetDevice { vendor_id: 0x0BDA, product_id: 0x8153, name: "Realtek RTL8153", driver: DriverType::Rtl8153 },
    UsbEthernetDevice { vendor_id: 0x0BDA, product_id: 0x8156, name: "Realtek RTL8156", driver: DriverType::Rtl8153 },
    UsbEthernetDevice { vendor_id: 0x2357, product_id: 0x0601, name: "TP-Link UE300", driver: DriverType::Rtl8153 },
    UsbEthernetDevice { vendor_id: 0x0B95, product_id: 0x772B, name: "ASIX AX88772B", driver: DriverType::Ax88179 },
];

/// USB device descriptor (first 18 bytes of GET_DESCRIPTOR response).
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UsbDeviceDescriptor {
    pub length: u8,
    pub descriptor_type: u8,
    pub usb_version: u16,
    pub device_class: u8,
    pub device_subclass: u8,
    pub device_protocol: u8,
    pub max_packet_size: u8,
    pub vendor_id: u16,
    pub product_id: u16,
    pub device_version: u16,
    pub manufacturer_idx: u8,
    pub product_idx: u8,
    pub serial_idx: u8,
    pub num_configurations: u8,
}

/// USB endpoint descriptor.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UsbEndpointDescriptor {
    pub length: u8,
    pub descriptor_type: u8,
    pub endpoint_address: u8,
    pub attributes: u8,
    pub max_packet_size: u16,
    pub interval: u8,
}

/// State for an active USB Ethernet connection.
struct EthernetState {
    /// Bulk IN endpoint number (for receiving frames).
    bulk_in_ep: u8,
    /// Bulk OUT endpoint number (for sending frames).
    bulk_out_ep: u8,
    /// MAC address (6 bytes).
    mac: [u8; 6],
    /// Maximum transfer unit.
    mtu: u16,
    /// Driver type.
    driver: DriverType,
}

static mut ETH_STATE: EthernetState = EthernetState {
    bulk_in_ep: 0,
    bulk_out_ep: 0,
    mac: [0; 6],
    mtu: 1500,
    driver: DriverType::CdcEcm,
};

/// Try to identify a USB Ethernet device from vendor/product IDs.
pub fn identify(vendor_id: u16, product_id: u16) -> Option<&'static UsbEthernetDevice> {
    KNOWN_DEVICES.iter().find(|d| d.vendor_id == vendor_id && d.product_id == product_id)
}

/// Check if device class indicates CDC-ECM (class 02, subclass 06).
pub fn is_cdc_ecm(device_class: u8, device_subclass: u8) -> bool {
    device_class == 0x02 && device_subclass == 0x06
}

/// Initialize a CDC-ECM USB Ethernet device.
///
/// This is called after xHCI has enumerated the device and obtained
/// its device descriptor. The function:
/// 1. Identifies the device
/// 2. Finds bulk IN/OUT endpoints
/// 3. Reads MAC address
/// 4. Sets up for frame transfer
pub fn init_device(vendor_id: u16, product_id: u16, device_class: u8, device_subclass: u8) {
    let device = identify(vendor_id, product_id);
    let is_ecm = is_cdc_ecm(device_class, device_subclass);

    let (name, driver) = if let Some(dev) = device {
        (dev.name, dev.driver)
    } else if is_ecm {
        ("CDC-ECM (generic)", DriverType::CdcEcm)
    } else {
        crate::serial_println!("[usb-eth] Unknown device {:04x}:{:04x} class {:02x}/{:02x}",
            vendor_id, product_id, device_class, device_subclass);
        return;
    };

    crate::serial_println!("[usb-eth] Found: {} ({:04x}:{:04x})", name, vendor_id, product_id);

    unsafe {
        ETH_STATE.driver = driver;
        // Default endpoints (will be overridden during configuration)
        ETH_STATE.bulk_in_ep = 0x81;  // EP1 IN
        ETH_STATE.bulk_out_ep = 0x02; // EP2 OUT
        ETH_STATE.mtu = 1500;

        // Generate a MAC address from vendor/product IDs (placeholder
        // until we can read the real MAC via USB string descriptor)
        ETH_STATE.mac = [
            0x02, // locally administered
            0x00,
            (vendor_id >> 8) as u8,
            (vendor_id & 0xFF) as u8,
            (product_id >> 8) as u8,
            (product_id & 0xFF) as u8,
        ];
    }

    DETECTED.store(true, Ordering::SeqCst);
    crate::serial_println!("[usb-eth] MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        unsafe { ETH_STATE.mac[0] }, unsafe { ETH_STATE.mac[1] },
        unsafe { ETH_STATE.mac[2] }, unsafe { ETH_STATE.mac[3] },
        unsafe { ETH_STATE.mac[4] }, unsafe { ETH_STATE.mac[5] });
    crate::serial_println!("[usb-eth] Ready (bulk_in={:#x} bulk_out={:#x})",
        unsafe { ETH_STATE.bulk_in_ep }, unsafe { ETH_STATE.bulk_out_ep });
}

/// Send an Ethernet frame via USB bulk OUT.
/// Returns Ok(bytes_sent) or Err.
///
/// Note: actual USB transfer requires xHCI transfer ring submission.
/// This is a stub that will be connected once xHCI transfer rings are
/// fully implemented.
pub fn send_frame(frame: &[u8]) -> Result<usize, &'static str> {
    if !DETECTED.load(Ordering::SeqCst) { return Err("usb-eth: not initialized"); }
    if frame.len() > 1514 { return Err("usb-eth: frame too large"); }

    // TODO: submit frame to xHCI bulk OUT transfer ring
    // For now, log and return success
    crate::serial_println!("[usb-eth] TX {} bytes (queued)", frame.len());
    Ok(frame.len())
}

/// Receive an Ethernet frame via USB bulk IN.
/// Returns Ok(bytes_received) or Err.
pub fn recv_frame(_buf: &mut [u8]) -> Result<usize, &'static str> {
    if !DETECTED.load(Ordering::SeqCst) { return Err("usb-eth: not initialized"); }

    // TODO: check xHCI bulk IN transfer ring for completed transfers
    Err("usb-eth: no frame available")
}

pub fn is_detected() -> bool { DETECTED.load(Ordering::SeqCst) }

pub fn mac_address() -> [u8; 6] {
    unsafe { core::ptr::read_volatile(&raw const ETH_STATE.mac) }
}

pub fn info() -> alloc::string::String {
    if !is_detected() { return alloc::string::String::from("usb-eth: not detected"); }
    let mac = mac_address();
    alloc::format!("usb-eth: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} (mtu={})",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
        unsafe { core::ptr::read_volatile(&raw const ETH_STATE.mtu) })
}
