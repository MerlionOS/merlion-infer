/// Network stack for MerlionOS Inference.
/// Provides Ethernet, ARP, IPv4, TCP over virtio-net or e1000e.

pub mod tcp;

use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicU8, Ordering};

// --- Types ---

pub const ETH_TYPE_ARP: u16 = 0x0806;
pub const ETH_TYPE_IP: u16 = 0x0800;

const ETH_HEADER_LEN: usize = 14;
const IPV4_HEADER_LEN: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ipv4Addr(pub [u8; 4]);

impl Ipv4Addr {
    pub const ZERO: Self = Self([0; 4]);
    pub const BROADCAST: Self = Self([255, 255, 255, 255]);
}

impl core::fmt::Display for Ipv4Addr {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

// --- Global state ---

pub static NET: Mutex<NetState> = Mutex::new(NetState::new());

pub struct NetState {
    pub mac: [u8; 6],
    pub ip: Ipv4Addr,
    pub gateway: Ipv4Addr,
    pub rx_packets: u64,
    pub tx_packets: u64,
}

impl NetState {
    const fn new() -> Self {
        Self {
            mac: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
            ip: Ipv4Addr([10, 0, 2, 15]),
            gateway: Ipv4Addr([10, 0, 2, 2]),
            rx_packets: 0,
            tx_packets: 0,
        }
    }
}

// --- NIC backend ---

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum NicBackend { None = 0, VirtioNet = 1 }

static BACKEND: AtomicU8 = AtomicU8::new(0);

pub fn init() {
    if crate::drivers::virtio_net::is_detected() {
        BACKEND.store(NicBackend::VirtioNet as u8, Ordering::SeqCst);
        crate::serial_println!("[net] backend: virtio-net");
    } else {
        crate::serial_println!("[net] backend: none (no NIC)");
    }
}

fn nic_send(frame: &[u8]) -> bool {
    match BACKEND.load(Ordering::Relaxed) {
        1 => crate::drivers::virtio_net::send_frame(frame).is_ok(),
        _ => false,
    }
}

fn nic_recv() -> Option<Vec<u8>> {
    match BACKEND.load(Ordering::Relaxed) {
        1 => crate::drivers::virtio_net::recv_frame(),
        _ => None,
    }
}

// --- Checksum ---

pub fn ip_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

// --- Ethernet ---

pub fn send_ethernet(dst_mac: &[u8; 6], ethertype: u16, payload: &[u8]) -> bool {
    let mut frame = Vec::with_capacity(ETH_HEADER_LEN + payload.len());
    frame.extend_from_slice(dst_mac);
    frame.extend_from_slice(&NET.lock().mac);
    frame.extend_from_slice(&ethertype.to_be_bytes());
    frame.extend_from_slice(payload);

    let ok = nic_send(&frame);
    if ok { NET.lock().tx_packets += 1; }
    ok
}

// --- IPv4 ---

pub fn send_ipv4(dst_ip: [u8; 4], protocol: u8, payload: &[u8]) -> bool {
    let src_ip = NET.lock().ip.0;
    let total_len = (IPV4_HEADER_LEN + payload.len()) as u16;

    let mut ip_hdr = [0u8; IPV4_HEADER_LEN];
    ip_hdr[0] = 0x45;
    ip_hdr[2..4].copy_from_slice(&total_len.to_be_bytes());
    ip_hdr[4..6].copy_from_slice(&1u16.to_be_bytes());
    ip_hdr[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
    ip_hdr[8] = 64;
    ip_hdr[9] = protocol;
    ip_hdr[12..16].copy_from_slice(&src_ip);
    ip_hdr[16..20].copy_from_slice(&dst_ip);

    let cksum = ip_checksum(&ip_hdr);
    ip_hdr[10..12].copy_from_slice(&cksum.to_be_bytes());

    let mut packet = Vec::with_capacity(IPV4_HEADER_LEN + payload.len());
    packet.extend_from_slice(&ip_hdr);
    packet.extend_from_slice(payload);

    // Use broadcast MAC (works with QEMU user-net)
    send_ethernet(&[0xFF; 6], ETH_TYPE_IP, &packet)
}

// --- Receive ---

pub struct ReceivedFrame {
    pub src_mac: [u8; 6],
    pub ethertype: u16,
    pub payload: Vec<u8>,
}

pub fn poll_rx() -> Option<ReceivedFrame> {
    let raw = nic_recv()?;
    if raw.len() < ETH_HEADER_LEN { return None; }

    let mut src_mac = [0u8; 6];
    src_mac.copy_from_slice(&raw[6..12]);
    let ethertype = u16::from_be_bytes([raw[12], raw[13]]);
    let payload = raw[ETH_HEADER_LEN..].to_vec();

    NET.lock().rx_packets += 1;

    Some(ReceivedFrame { src_mac, ethertype, payload })
}
