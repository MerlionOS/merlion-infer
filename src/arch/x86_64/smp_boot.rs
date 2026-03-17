/// SMP (Symmetric Multi-Processing) AP core startup.
///
/// Wakes Application Processors (APs) via INIT-SIPI-SIPI sequence.
/// Each AP runs an idle loop waiting for work from the BSP.
///
/// For inference: BSP distributes attention heads across AP cores.
/// Communication is via lock-free SPSC (Single Producer, Single Consumer)
/// ring buffers — one per AP core.

use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Maximum number of CPU cores supported.
pub const MAX_CPUS: usize = 64;

/// Per-core state.
#[repr(C)]
pub struct CoreState {
    /// 0 = offline, 1 = idle, 2 = busy, 3 = done
    pub status: AtomicU8,
    /// Work item: pointer to a work descriptor (set by BSP, read by AP).
    pub work_ptr: AtomicU64,
    /// Result: set by AP when work is done.
    pub result: AtomicU64,
}

impl CoreState {
    const fn new() -> Self {
        Self {
            status: AtomicU8::new(0),
            work_ptr: AtomicU64::new(0),
            result: AtomicU64::new(0),
        }
    }
}

const STATUS_OFFLINE: u8 = 0;
const STATUS_IDLE: u8 = 1;
const STATUS_BUSY: u8 = 2;
const STATUS_DONE: u8 = 3;

/// Global per-core state array.
static CORES: [CoreState; MAX_CPUS] = {
    // const array init
    const INIT: CoreState = CoreState::new();
    [INIT; MAX_CPUS]
};

/// Number of online AP cores.
static AP_COUNT: AtomicU8 = AtomicU8::new(0);

/// Submit work to a specific AP core.
/// Returns true if the core accepted the work.
pub fn submit_work(core_id: usize, work_ptr: u64) -> bool {
    if core_id >= MAX_CPUS { return false; }
    let core = &CORES[core_id];

    // Only submit if core is idle
    if core.status.load(Ordering::Acquire) != STATUS_IDLE { return false; }

    core.work_ptr.store(work_ptr, Ordering::Release);
    core.status.store(STATUS_BUSY, Ordering::Release);
    true
}

/// Check if a core has finished its work.
pub fn is_done(core_id: usize) -> bool {
    if core_id >= MAX_CPUS { return false; }
    CORES[core_id].status.load(Ordering::Acquire) == STATUS_DONE
}

/// Collect result from a core and reset it to idle.
pub fn collect_result(core_id: usize) -> u64 {
    if core_id >= MAX_CPUS { return 0; }
    let core = &CORES[core_id];
    let result = core.result.load(Ordering::Acquire);
    core.status.store(STATUS_IDLE, Ordering::Release);
    result
}

/// Get the number of online AP cores.
pub fn online_ap_count() -> u8 {
    AP_COUNT.load(Ordering::Relaxed)
}

/// Mark a core as online (called by AP during startup).
pub fn mark_online(core_id: usize) {
    if core_id < MAX_CPUS {
        CORES[core_id].status.store(STATUS_IDLE, Ordering::Release);
        AP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

/// BSP: Send INIT-SIPI-SIPI to wake an AP core.
///
/// The trampoline code must be placed at a 4K-aligned physical address
/// below 1 MiB (SIPI vector is page number, so 0x8000 → vector 0x08).
///
/// This is a stub — actual implementation requires:
/// 1. Copy AP trampoline code to low memory (e.g. 0x8000)
/// 2. Set up AP's initial GDT, stack, and entry point
/// 3. Send INIT IPI via Local APIC
/// 4. Wait 10ms
/// 5. Send SIPI with vector = trampoline_page
/// 6. Wait for AP to signal online
pub fn wake_ap(apic_id: u8) {
    crate::serial_println!("[smp-boot] Wake AP {} (INIT-SIPI-SIPI stub)", apic_id);
    // TODO: implement when we have Local APIC MMIO access
    // For now, just log the intent
    crate::serial_println!("[smp-boot] AP startup requires Local APIC driver (future work)");
}

/// Distribute work across all online AP cores.
/// Calls `work_fn` with each core's assigned range.
/// BSP waits for all APs to finish, then collects results.
pub fn parallel_for(total_items: usize, work_fn: fn(start: usize, end: usize) -> u64) -> alloc::vec::Vec<u64> {
    let n_cores = online_ap_count() as usize;
    if n_cores == 0 {
        // No APs — run everything on BSP
        return alloc::vec![work_fn(0, total_items)];
    }

    let items_per_core = total_items / (n_cores + 1); // +1 for BSP
    let mut results = alloc::vec::Vec::with_capacity(n_cores + 1);

    // Submit to APs
    for i in 0..n_cores {
        let start = (i + 1) * items_per_core;
        let _end = if i + 1 == n_cores { total_items } else { (i + 2) * items_per_core };
        // In a real implementation, we'd pack start/end into work_ptr
        // For now, this is a framework showing the parallel dispatch pattern
        let _ = submit_work(i + 1, start as u64);
    }

    // BSP does its share
    let bsp_result = work_fn(0, items_per_core);
    results.push(bsp_result);

    // Wait for APs
    for i in 0..n_cores {
        let core_id = i + 1;
        while !is_done(core_id) {
            core::hint::spin_loop();
        }
        results.push(collect_result(core_id));
    }

    results
}
