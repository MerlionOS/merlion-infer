/// Inference benchmarking.
/// Measures prefill and decode throughput with structured output.

use crate::arch::x86_64::timer;

pub struct BenchResult {
    pub prefill_tokens: usize,
    pub prefill_ticks: u64,
    pub decode_tokens: usize,
    pub decode_ticks: u64,
    pub ttft_ticks: u64,
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

    pub fn ttft_ms(&self) -> f32 {
        self.ttft_ticks as f32 / timer::PIT_FREQUENCY_HZ as f32 * 1000.0
    }

    pub fn report(&self) {
        let backend = crate::inference::kernels::dispatch::backend_name();

        crate::serial_println!("╔══════════════════════════════════════╗");
        crate::serial_println!("║     MerlionOS Inference Benchmark    ║");
        crate::serial_println!("╠══════════════════════════════════════╣");

        // Model info
        if let Some(info) = crate::inference::state::with_engine(|e| {
            (e.config.dim, e.config.n_layers, e.config.n_heads, e.config.vocab_size,
             e.weights.is_quantized, e.weights.memory_bytes(), e.state.memory_bytes())
        }) {
            let (dim, layers, heads, vocab, quantized, w_bytes, s_bytes) = info;
            let quant = if quantized { "Q4_0" } else { "F32" };
            crate::serial_println!("║ Model: dim={} L={} H={} V={}", dim, layers, heads, vocab);
            crate::serial_println!("║ Quant: {} | Weights: {} MiB | State: {} MiB",
                quant, w_bytes / (1024*1024), s_bytes / (1024*1024));
        }

        crate::serial_println!("║ Backend: {}", backend);
        crate::serial_println!("╠══════════════════════════════════════╣");
        crate::serial_println!("║ Prefill: {:>4} tokens | {:>7.1} tok/s",
            self.prefill_tokens, self.prefill_tok_per_sec());
        crate::serial_println!("║ Decode:  {:>4} tokens | {:>7.1} tok/s",
            self.decode_tokens, self.decode_tok_per_sec());
        crate::serial_println!("║ TTFT:    {:>7.1} ms", self.ttft_ms());
        crate::serial_println!("╠══════════════════════════════════════╣");

        let heap_used = crate::memory::heap::used();
        let heap_total = crate::memory::heap::heap_size();
        crate::serial_println!("║ Heap: {} / {} MiB ({:.0}%)",
            heap_used / (1024*1024), heap_total / (1024*1024),
            heap_used as f32 / heap_total as f32 * 100.0);
        crate::serial_println!("╚══════════════════════════════════════╝");
    }
}

/// Measure elapsed ticks for a closure.
pub fn measure<F: FnOnce() -> R, R>(f: F) -> (R, u64) {
    let start = timer::ticks();
    let result = f();
    let elapsed = timer::ticks() - start;
    (result, elapsed)
}
