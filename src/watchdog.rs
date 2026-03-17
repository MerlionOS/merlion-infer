/// Software watchdog timer.
/// Detects hangs and triggers recovery (reboot or log).

use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Watchdog timeout in PIT ticks (default: 30 seconds at 100 Hz).
static TIMEOUT_TICKS: AtomicU64 = AtomicU64::new(3000);

/// Last feed timestamp.
static LAST_FEED: AtomicU64 = AtomicU64::new(0);

/// Whether the watchdog is enabled.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable the watchdog with the given timeout in seconds.
pub fn enable(timeout_secs: u64) {
    let ticks = timeout_secs * crate::arch::x86_64::timer::PIT_FREQUENCY_HZ;
    TIMEOUT_TICKS.store(ticks, Ordering::SeqCst);
    feed();
    ENABLED.store(true, Ordering::SeqCst);
    crate::serial_println!("[watchdog] enabled ({}s timeout)", timeout_secs);
}

/// Disable the watchdog.
pub fn disable() {
    ENABLED.store(false, Ordering::SeqCst);
}

/// Feed the watchdog (reset the timer).
/// Call this periodically from the main loop or inference path.
pub fn feed() {
    LAST_FEED.store(crate::arch::x86_64::timer::ticks(), Ordering::SeqCst);
}

/// Check if the watchdog has expired.
/// Call from timer interrupt or periodic check.
pub fn check() {
    if !ENABLED.load(Ordering::Relaxed) { return; }

    let now = crate::arch::x86_64::timer::ticks();
    let last = LAST_FEED.load(Ordering::Relaxed);
    let timeout = TIMEOUT_TICKS.load(Ordering::Relaxed);

    if now - last > timeout {
        crate::serial_println!("\n[watchdog] TIMEOUT — system appears hung!");
        crate::serial_println!("[watchdog] Last feed: {} ticks ago", now - last);
        crate::log::error("watchdog timeout");

        // Auto-reboot on hang
        crate::arch::x86_64::acpi::reboot();
    }
}

pub fn is_enabled() -> bool { ENABLED.load(Ordering::Relaxed) }
