/// Global inference engine state.
/// Holds the loaded model and provides thread-safe access.

use spin::Mutex;
use crate::inference::engine::LlamaEngine;

static ENGINE: Mutex<Option<LlamaEngine>> = Mutex::new(None);

/// Load an engine (replaces any previously loaded model).
pub fn load(engine: LlamaEngine) {
    *ENGINE.lock() = Some(engine);
}

/// Check if a model is loaded.
pub fn is_loaded() -> bool {
    ENGINE.lock().is_some()
}

/// Run a closure with mutable access to the engine.
/// Returns None if no model is loaded.
pub fn with_engine<R>(f: impl FnOnce(&mut LlamaEngine) -> R) -> Option<R> {
    let mut lock = ENGINE.lock();
    lock.as_mut().map(f)
}

/// Get model info string.
pub fn model_info() -> alloc::string::String {
    let lock = ENGINE.lock();
    match lock.as_ref() {
        Some(e) => alloc::format!(
            "Model loaded: dim={} layers={} heads={} vocab={}\nWeights: {} KiB | State: {} KiB",
            e.config.dim, e.config.n_layers, e.config.n_heads, e.config.vocab_size,
            e.weights.memory_bytes() / 1024, e.state.memory_bytes() / 1024,
        ),
        None => alloc::string::String::from("No model loaded"),
    }
}
