/// Kernel log ring buffer.
/// Fixed-size ring buffer for structured log messages.

use spin::Mutex;

const LOG_SIZE: usize = 8192;
const MAX_ENTRIES: usize = 128;

struct LogEntry {
    tick: u64,
    level: u8, // 0=debug, 1=info, 2=warn, 3=error
    msg: [u8; 60],
    len: u8,
}

impl LogEntry {
    const fn empty() -> Self {
        Self { tick: 0, level: 0, msg: [0; 60], len: 0 }
    }
}

struct KernelLog {
    entries: [LogEntry; MAX_ENTRIES],
    head: usize,
    count: usize,
}

static LOG: Mutex<KernelLog> = Mutex::new(KernelLog {
    entries: [const { LogEntry::empty() }; MAX_ENTRIES],
    head: 0,
    count: 0,
});

pub fn log(level: u8, msg: &str) {
    let mut log = LOG.lock();
    let idx = (log.head + log.count) % MAX_ENTRIES;
    if log.count == MAX_ENTRIES {
        log.head = (log.head + 1) % MAX_ENTRIES;
    } else {
        log.count += 1;
    }

    let bytes = msg.as_bytes();
    let len = core::cmp::min(bytes.len(), 60);
    log.entries[idx] = LogEntry {
        tick: crate::arch::x86_64::timer::ticks(),
        level,
        msg: {
            let mut buf = [0u8; 60];
            buf[..len].copy_from_slice(&bytes[..len]);
            buf
        },
        len: len as u8,
    };
}

pub fn info(msg: &str) { log(1, msg); }
pub fn warn(msg: &str) { log(2, msg); }
pub fn error(msg: &str) { log(3, msg); }

/// Dump recent log entries to serial.
pub fn dmesg() {
    let log = LOG.lock();
    let level_str = |l: u8| match l {
        0 => "DBG", 1 => "INF", 2 => "WRN", 3 => "ERR", _ => "???",
    };
    for i in 0..log.count {
        let idx = (log.head + i) % MAX_ENTRIES;
        let e = &log.entries[idx];
        let msg = core::str::from_utf8(&e.msg[..e.len as usize]).unwrap_or("?");
        crate::serial_println!("[{:>6}] {} {}", e.tick, level_str(e.level), msg);
    }
    if log.count == 0 {
        crate::serial_println!("(no log entries)");
    }
}
