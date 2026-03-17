/// Runtime configuration for the inference server.
/// Defaults that can be overridden at boot or via shell.

use core::sync::atomic::{AtomicU16, AtomicU32, Ordering};

/// HTTP API port.
static API_PORT: AtomicU16 = AtomicU16::new(8080);

/// Maximum batch size for continuous batching.
static MAX_BATCH_SIZE: AtomicU32 = AtomicU32::new(4);

/// Maximum sequence length.
static MAX_SEQ_LEN: AtomicU32 = AtomicU32::new(2048);

/// Temperature for sampling (fixed-point: value * 100).
static TEMPERATURE: AtomicU32 = AtomicU32::new(70); // 0.70

/// Top-p for nucleus sampling (fixed-point: value * 100).
static TOP_P: AtomicU32 = AtomicU32::new(90); // 0.90

pub fn api_port() -> u16 { API_PORT.load(Ordering::Relaxed) }
pub fn set_api_port(port: u16) { API_PORT.store(port, Ordering::SeqCst); }

pub fn max_batch_size() -> u32 { MAX_BATCH_SIZE.load(Ordering::Relaxed) }
pub fn set_max_batch_size(n: u32) { MAX_BATCH_SIZE.store(n, Ordering::SeqCst); }

pub fn max_seq_len() -> u32 { MAX_SEQ_LEN.load(Ordering::Relaxed) }

pub fn temperature() -> f32 { TEMPERATURE.load(Ordering::Relaxed) as f32 / 100.0 }
pub fn set_temperature(t: f32) { TEMPERATURE.store((t * 100.0) as u32, Ordering::SeqCst); }

pub fn top_p() -> f32 { TOP_P.load(Ordering::Relaxed) as f32 / 100.0 }
pub fn set_top_p(p: f32) { TOP_P.store((p * 100.0) as u32, Ordering::SeqCst); }

pub fn show() {
    crate::serial_println!("Configuration:");
    crate::serial_println!("  api_port:       {}", api_port());
    crate::serial_println!("  max_batch_size: {}", max_batch_size());
    crate::serial_println!("  max_seq_len:    {}", max_seq_len());
    crate::serial_println!("  temperature:    {:.2}", temperature());
    crate::serial_println!("  top_p:          {:.2}", top_p());
}
