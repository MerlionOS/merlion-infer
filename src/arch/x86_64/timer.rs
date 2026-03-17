/// PIT (Programmable Interval Timer) at 100 Hz.

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::instructions::port::Port;

static TICKS: AtomicU64 = AtomicU64::new(0);

pub const PIT_FREQUENCY_HZ: u64 = 100;
const PIT_BASE_FREQUENCY: u64 = 1_193_182;
const PIT_DIVISOR: u16 = (PIT_BASE_FREQUENCY / PIT_FREQUENCY_HZ) as u16;

pub fn init() {
    unsafe {
        let mut cmd = Port::<u8>::new(0x43);
        cmd.write(0x34); // Channel 0, lobyte/hibyte, rate generator

        let mut data = Port::<u8>::new(0x40);
        data.write((PIT_DIVISOR & 0xFF) as u8);
        data.write((PIT_DIVISOR >> 8) as u8);
    }
}

pub fn tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
}

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

pub fn uptime_secs() -> u64 {
    ticks() / PIT_FREQUENCY_HZ
}

pub fn uptime_hms() -> (u64, u64, u64) {
    let s = uptime_secs();
    (s / 3600, (s % 3600) / 60, s % 60)
}
