/// Minimal HTTP/1.1 server for inference API.
/// Single-threaded, polling-based, runs on the bare-metal TCP stack.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::net::tcp;

const MAX_REQUEST_SIZE: usize = 8192;
const RECV_POLL_LIMIT: usize = 500;

#[derive(Debug, PartialEq)]
pub enum Method { Get, Post, Other }

pub struct Request {
    pub method: Method,
    pub path: String,
    pub body: Vec<u8>,
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
    /// If true, body contains SSE events (text/event-stream).
    pub is_sse: bool,
}

impl Response {
    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.as_bytes().to_vec(),
            is_sse: false,
        }
    }

    pub fn text(status: u16, body: &str) -> Self {
        Self {
            status,
            content_type: "text/plain",
            body: body.as_bytes().to_vec(),
            is_sse: false,
        }
    }

    pub fn sse(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: body.as_bytes().to_vec(),
            is_sse: true,
        }
    }

    fn status_text(&self) -> &'static str {
        match self.status {
            200 => "OK",
            404 => "Not Found",
            405 => "Method Not Allowed",
            500 => "Internal Server Error",
            _ => "Unknown",
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        if self.is_sse {
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            );
            let mut out = header.into_bytes();
            out.extend_from_slice(&self.body);
            out
        } else {
            let header = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
                self.status, self.status_text(),
                self.content_type,
                self.body.len(),
            );
            let mut out = header.into_bytes();
            out.extend_from_slice(&self.body);
            out
        }
    }
}

fn parse_request(data: &[u8]) -> Option<Request> {
    let s = core::str::from_utf8(data).ok()?;
    let header_end = s.find("\r\n\r\n")?;
    let header = &s[..header_end];
    let body = &data[header_end + 4..];

    let first_line = header.lines().next()?;
    let mut parts = first_line.split_whitespace();
    let method_str = parts.next()?;
    let path = parts.next()?;

    let method = match method_str {
        "GET" => Method::Get,
        "POST" => Method::Post,
        _ => Method::Other,
    };

    Some(Request {
        method,
        path: String::from(path),
        body: body.to_vec(),
    })
}

/// Run the HTTP server on the given port.
/// This is a blocking loop — call from a shell command.
pub fn serve(port: u16, handler: fn(&Request) -> Response) {
    crate::serial_println!("[http] Listening on port {}", port);

    loop {
        // Try to accept a connection
        if let Some(sock_id) = tcp::accept(port) {
            // Read request
            let mut buf = Vec::new();
            for _ in 0..RECV_POLL_LIMIT {
                tcp::poll_incoming();
                if let Ok(data) = tcp::recv(sock_id) {
                    if !data.is_empty() {
                        buf.extend_from_slice(&data);
                        // Check if we have complete headers
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
                if buf.len() >= MAX_REQUEST_SIZE { break; }
                core::hint::spin_loop();
            }

            // Parse and handle
            if let Some(req) = parse_request(&buf) {
                crate::serial_println!("[http] {:?} {}", req.method, req.path);
                let resp = handler(&req);
                let data = resp.serialize();
                let _ = tcp::send(sock_id, &data);
            }

            // Close
            let _ = tcp::close(sock_id);
        }

        // Yield to avoid spinning
        x86_64::instructions::hlt();
    }
}
