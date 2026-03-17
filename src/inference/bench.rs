/// Inference benchmarking.
/// Measures prefill and decode throughput.

use crate::arch::x86_64::timer;

pub struct BenchResult {
    pub prefill_tokens: usize,
    pub prefill_ticks: u64,
    pub decode_tokens: usize,
    pub decode_ticks: u64,
    pub peak_memory_bytes: usize,
}

impl BenchResult {
    pub fn prefill_tok_per_sec(&self) -> f32 {
        if self.prefill_ticks == 0 { return 0.0; }
        let secs = self.prefill_ticks as f32 / timer::PIT_FREQUENCY_HZ as f32;
        self.prefill_tokens as f32 / secs
    }

    pub fn decode_tok_per_sec(&self) -> f32 {
        if self.decode_ticks == 0 { return 0.0; }
        let secs = self.decode_ticks as f32 / timer::PIT_FREQUENCY_HZ as f32;
        self.decode_tokens as f32 / secs
    }

    pub fn report(&self) {
        crate::serial_println!("[bench] Prefill: {} tokens in {} ticks ({:.1} tok/s)",
            self.prefill_tokens, self.prefill_ticks, self.prefill_tok_per_sec());
        crate::serial_println!("[bench] Decode:  {} tokens in {} ticks ({:.1} tok/s)",
            self.decode_tokens, self.decode_ticks, self.decode_tok_per_sec());
        crate::serial_println!("[bench] Peak memory: {} KiB",
            self.peak_memory_bytes / 1024);
    }
}

/// Measure elapsed ticks for a closure.
pub fn measure<F: FnOnce() -> R, R>(f: F) -> (R, u64) {
    let start = timer::ticks();
    let result = f();
    let elapsed = timer::ticks() - start;
    (result, elapsed)
}
