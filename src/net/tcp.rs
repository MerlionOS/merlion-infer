/// TCP implementation for MerlionOS Inference.
/// Simplified from merlion-kernel's tcp_real.rs.
/// Supports connect, send, recv, close, and server-side accept.

use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;
use crate::net::{self, Ipv4Addr};

const IP_PROTO_TCP: u8 = 6;
const TCP_HEADER_LEN: usize = 20;
const MAX_SOCKETS: usize = 32;
const DEFAULT_WINDOW: u16 = 16384;
const RECV_BUF_CAP: usize = 65536;

pub const TCP_FIN: u8 = 0x01;
pub const TCP_SYN: u8 = 0x02;
pub const TCP_RST: u8 = 0x04;
pub const TCP_PSH: u8 = 0x08;
pub const TCP_ACK: u8 = 0x10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TcpState {
    Closed, Listen, SynSent, SynReceived, Established,
    FinWait1, FinWait2, CloseWait, Closing, LastAck, TimeWait,
}

pub struct TcpSocket {
    pub state: TcpState,
    pub seq_num: u32,
    pub ack_num: u32,
    pub local_ip: Ipv4Addr,
    pub local_port: u16,
    pub remote_ip: Ipv4Addr,
    pub remote_port: u16,
    pub recv_buf: Vec<u8>,
}

static SOCKETS: Mutex<Vec<TcpSocket>> = Mutex::new(Vec::new());

// --- Checksum ---

fn tcp_checksum(src_ip: [u8; 4], dst_ip: [u8; 4], tcp_data: &[u8]) -> u16 {
    let tcp_len = tcp_data.len() as u16;
    let mut buf = Vec::with_capacity(12 + tcp_data.len());
    buf.extend_from_slice(&src_ip);
    buf.extend_from_slice(&dst_ip);
    buf.push(0);
    buf.push(IP_PROTO_TCP);
    buf.extend_from_slice(&tcp_len.to_be_bytes());
    buf.extend_from_slice(tcp_data);
    net::ip_checksum(&buf)
}

// --- Build & parse ---

fn build_tcp_packet(
    src_ip: [u8; 4], dst_ip: [u8; 4],
    src_port: u16, dst_port: u16,
    seq: u32, ack: u32, flags: u8, payload: &[u8],
) -> Vec<u8> {
    let total = TCP_HEADER_LEN + payload.len();
    let mut seg = vec![0u8; total];
    seg[0..2].copy_from_slice(&src_port.to_be_bytes());
    seg[2..4].copy_from_slice(&dst_port.to_be_bytes());
    seg[4..8].copy_from_slice(&seq.to_be_bytes());
    seg[8..12].copy_from_slice(&ack.to_be_bytes());
    seg[12] = (TCP_HEADER_LEN as u8 / 4) << 4;
    seg[13] = flags;
    seg[14..16].copy_from_slice(&DEFAULT_WINDOW.to_be_bytes());
    if !payload.is_empty() {
        seg[TCP_HEADER_LEN..].copy_from_slice(payload);
    }
    let cksum = tcp_checksum(src_ip, dst_ip, &seg);
    seg[16..18].copy_from_slice(&cksum.to_be_bytes());
    seg
}

fn send_segment(
    src_ip: [u8; 4], dst_ip: [u8; 4],
    src_port: u16, dst_port: u16,
    seq: u32, ack: u32, flags: u8, payload: &[u8],
) -> bool {
    let seg = build_tcp_packet(src_ip, dst_ip, src_port, dst_port, seq, ack, flags, payload);
    net::send_ipv4(dst_ip, IP_PROTO_TCP, &seg)
}

struct ParsedTcp {
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    payload: Vec<u8>,
    src_ip: [u8; 4],
}

fn parse_from_ip(ip_payload: &[u8]) -> Option<ParsedTcp> {
    if ip_payload.len() < 20 || ip_payload[9] != IP_PROTO_TCP { return None; }
    let mut src_ip = [0u8; 4];
    src_ip.copy_from_slice(&ip_payload[12..16]);
    let ihl = ((ip_payload[0] & 0x0F) as usize) * 4;
    let tcp = &ip_payload[ihl..];
    if tcp.len() < TCP_HEADER_LEN { return None; }

    let src_port = u16::from_be_bytes([tcp[0], tcp[1]]);
    let dst_port = u16::from_be_bytes([tcp[2], tcp[3]]);
    let seq = u32::from_be_bytes([tcp[4], tcp[5], tcp[6], tcp[7]]);
    let ack = u32::from_be_bytes([tcp[8], tcp[9], tcp[10], tcp[11]]);
    let flags = tcp[13] & 0x3F;
    let data_off = ((tcp[12] >> 4) as usize) * 4;
    let payload = if data_off < tcp.len() { tcp[data_off..].to_vec() } else { Vec::new() };

    Some(ParsedTcp { src_port, dst_port, seq, ack, flags, payload, src_ip })
}

// --- ISN ---

fn generate_isn() -> u32 {
    (crate::arch::x86_64::timer::ticks().wrapping_mul(2654435761)) as u32
}

// --- Public API ---

/// Register an established connection (used by HTTP server after handshake).
pub fn register_established(
    local_ip: Ipv4Addr, local_port: u16,
    remote_ip: Ipv4Addr, remote_port: u16,
    seq_num: u32, ack_num: u32,
) -> usize {
    let mut sockets = SOCKETS.lock();
    let idx = sockets.len();
    sockets.push(TcpSocket {
        state: TcpState::Established,
        seq_num, ack_num,
        local_ip, local_port,
        remote_ip, remote_port,
        recv_buf: Vec::new(),
    });
    idx
}

/// Send data on an established connection.
pub fn send(sock_id: usize, data: &[u8]) -> Result<usize, &'static str> {
    let mut sockets = SOCKETS.lock();
    let sock = sockets.get_mut(sock_id).ok_or("bad socket")?;
    if sock.state != TcpState::Established { return Err("not established"); }

    send_segment(
        sock.local_ip.0, sock.remote_ip.0,
        sock.local_port, sock.remote_port,
        sock.seq_num, sock.ack_num,
        TCP_PSH | TCP_ACK, data,
    );
    sock.seq_num = sock.seq_num.wrapping_add(data.len() as u32);
    Ok(data.len())
}

/// Read received data.
pub fn recv(sock_id: usize) -> Result<Vec<u8>, &'static str> {
    let mut sockets = SOCKETS.lock();
    let sock = sockets.get_mut(sock_id).ok_or("bad socket")?;
    Ok(core::mem::take(&mut sock.recv_buf))
}

/// Close connection.
pub fn close(sock_id: usize) -> Result<(), &'static str> {
    let mut sockets = SOCKETS.lock();
    let sock = sockets.get_mut(sock_id).ok_or("bad socket")?;
    if sock.state == TcpState::Closed { return Ok(()); }

    send_segment(
        sock.local_ip.0, sock.remote_ip.0,
        sock.local_port, sock.remote_port,
        sock.seq_num, sock.ack_num,
        TCP_FIN | TCP_ACK, &[],
    );
    sock.seq_num = sock.seq_num.wrapping_add(1);
    sock.state = TcpState::Closed;
    Ok(())
}

/// Poll for incoming SYN on `port`. Complete handshake and return socket ID.
pub fn accept(port: u16) -> Option<usize> {
    let frame = net::poll_rx()?;
    if frame.ethertype != net::ETH_TYPE_IP { return None; }

    let parsed = parse_from_ip(&frame.payload)?;
    if parsed.dst_port != port { return None; }
    if parsed.flags & TCP_SYN == 0 || parsed.flags & TCP_ACK != 0 { return None; }

    let local_ip = net::NET.lock().ip;
    let isn = generate_isn();

    // Send SYN-ACK
    send_segment(
        local_ip.0, parsed.src_ip,
        port, parsed.src_port,
        isn, parsed.seq.wrapping_add(1),
        TCP_SYN | TCP_ACK, &[],
    );

    // Wait for ACK
    for _ in 0..500 {
        if let Some(frame2) = net::poll_rx() {
            if frame2.ethertype != net::ETH_TYPE_IP { continue; }
            if let Some(p2) = parse_from_ip(&frame2.payload) {
                if p2.dst_port == port && p2.flags & TCP_ACK != 0 {
                    let sock_id = register_established(
                        local_ip, port,
                        Ipv4Addr(parsed.src_ip), parsed.src_port,
                        isn.wrapping_add(1), parsed.seq.wrapping_add(1),
                    );
                    return Some(sock_id);
                }
            }
        }
        core::hint::spin_loop();
    }

    None
}

/// Poll and dispatch incoming segments to matching sockets.
pub fn poll_incoming() -> usize {
    let mut count = 0;
    while let Some(frame) = net::poll_rx() {
        if frame.ethertype != net::ETH_TYPE_IP { continue; }
        let parsed = match parse_from_ip(&frame.payload) {
            Some(p) => p,
            None => continue,
        };

        let mut sockets = SOCKETS.lock();
        if let Some(sock) = sockets.iter_mut().find(|s| {
            s.local_port == parsed.dst_port
            && s.remote_ip.0 == parsed.src_ip
            && s.state != TcpState::Closed
        }) {
            if parsed.flags & TCP_RST != 0 {
                sock.state = TcpState::Closed;
            } else if sock.state == TcpState::Established {
                if parsed.flags & TCP_FIN != 0 {
                    sock.ack_num = parsed.seq.wrapping_add(1);
                    send_segment(
                        sock.local_ip.0, sock.remote_ip.0,
                        sock.local_port, sock.remote_port,
                        sock.seq_num, sock.ack_num,
                        TCP_ACK, &[],
                    );
                    sock.state = TcpState::CloseWait;
                } else if !parsed.payload.is_empty() {
                    if sock.recv_buf.len() + parsed.payload.len() <= RECV_BUF_CAP {
                        sock.recv_buf.extend_from_slice(&parsed.payload);
                    }
                    sock.ack_num = sock.ack_num.wrapping_add(parsed.payload.len() as u32);
                    send_segment(
                        sock.local_ip.0, sock.remote_ip.0,
                        sock.local_port, sock.remote_port,
                        sock.seq_num, sock.ack_num,
                        TCP_ACK, &[],
                    );
                }
            }
            count += 1;
        }
    }
    count
}

/// List active sockets.
pub fn list_sockets() -> Vec<(usize, Ipv4Addr, u16, Ipv4Addr, u16, TcpState)> {
    SOCKETS.lock().iter().enumerate()
        .filter(|(_, s)| s.state != TcpState::Closed)
        .map(|(i, s)| (i, s.local_ip, s.local_port, s.remote_ip, s.remote_port, s.state))
        .collect()
}
