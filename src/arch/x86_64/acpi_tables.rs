/// ACPI table parsing: RSDP → RSDT/XSDT → MADT.
/// Discovers CPU APIC IDs for SMP core startup.
///
/// ACPI spec: https://uefi.org/specs/ACPI/6.5/

use alloc::vec::Vec;
use crate::memory::phys;

const RSDP_SIGNATURE: &[u8; 8] = b"RSD PTR ";

/// Parsed MADT entry: one per CPU core.
#[derive(Debug, Clone)]
pub struct LocalApic {
    pub processor_id: u8,
    pub apic_id: u8,
    pub enabled: bool,
}

/// All APIC IDs found in the MADT.
static APIC_LIST: spin::Mutex<Vec<LocalApic>> = spin::Mutex::new(Vec::new());

/// Search for RSDP in standard BIOS memory locations.
/// Returns the physical address of the RSDP if found.
fn find_rsdp() -> Option<u64> {
    // Search EBDA (Extended BIOS Data Area) and main BIOS area
    let search_regions: &[(u64, u64)] = &[
        (0x000E0000, 0x000FFFFF), // Main BIOS area
    ];

    for &(start, end) in search_regions {
        let mut addr = start;
        while addr < end {
            let virt = phys::phys_to_virt(x86_64::PhysAddr::new(addr));
            let ptr = virt.as_ptr::<u8>();
            let sig = unsafe { core::slice::from_raw_parts(ptr, 8) };
            if sig == RSDP_SIGNATURE {
                // Verify checksum
                let rsdp_bytes = unsafe { core::slice::from_raw_parts(ptr, 20) };
                let sum: u8 = rsdp_bytes.iter().fold(0u8, |a, b| a.wrapping_add(*b));
                if sum == 0 {
                    return Some(addr);
                }
            }
            addr += 16; // RSDP is 16-byte aligned
        }
    }
    None
}

/// Parse the RSDP to get RSDT/XSDT address.
fn parse_rsdp(rsdp_phys: u64) -> Option<(u64, bool)> {
    let virt = phys::phys_to_virt(x86_64::PhysAddr::new(rsdp_phys));
    let ptr = virt.as_ptr::<u8>();

    unsafe {
        let revision = *ptr.add(15);

        if revision >= 2 {
            // ACPI 2.0+: use XSDT (64-bit addresses)
            let xsdt_addr = core::ptr::read_unaligned(ptr.add(24) as *const u64);
            if xsdt_addr != 0 {
                return Some((xsdt_addr, true));
            }
        }

        // ACPI 1.0: use RSDT (32-bit addresses)
        let rsdt_addr = core::ptr::read_unaligned(ptr.add(16) as *const u32) as u64;
        Some((rsdt_addr, false))
    }
}

/// Find the MADT (APIC table) in the RSDT/XSDT.
fn find_madt(sdt_phys: u64, is_xsdt: bool) -> Option<u64> {
    let virt = phys::phys_to_virt(x86_64::PhysAddr::new(sdt_phys));
    let ptr = virt.as_ptr::<u8>();

    unsafe {
        let length = core::ptr::read_unaligned(ptr.add(4) as *const u32) as usize;
        let entry_size = if is_xsdt { 8 } else { 4 };
        let header_size = 36;
        let n_entries = (length - header_size) / entry_size;

        for i in 0..n_entries {
            let entry_offset = header_size + i * entry_size;
            let table_phys = if is_xsdt {
                core::ptr::read_unaligned(ptr.add(entry_offset) as *const u64)
            } else {
                core::ptr::read_unaligned(ptr.add(entry_offset) as *const u32) as u64
            };

            // Read table signature
            let table_virt = phys::phys_to_virt(x86_64::PhysAddr::new(table_phys));
            let sig = core::slice::from_raw_parts(table_virt.as_ptr::<u8>(), 4);
            if sig == b"APIC" {
                return Some(table_phys);
            }
        }
    }
    None
}

/// Parse MADT to extract Local APIC entries.
fn parse_madt(madt_phys: u64) -> Vec<LocalApic> {
    let mut apics = Vec::new();
    let virt = phys::phys_to_virt(x86_64::PhysAddr::new(madt_phys));
    let ptr = virt.as_ptr::<u8>();

    unsafe {
        let length = core::ptr::read_unaligned(ptr.add(4) as *const u32) as usize;

        // MADT entries start at offset 44 (after header + Local APIC Address + Flags)
        let mut offset = 44;
        while offset + 2 <= length {
            let entry_type = *ptr.add(offset);
            let entry_len = *ptr.add(offset + 1) as usize;
            if entry_len < 2 { break; }

            match entry_type {
                0 => {
                    // Processor Local APIC (type 0, length 8)
                    if entry_len >= 8 {
                        let processor_id = *ptr.add(offset + 2);
                        let apic_id = *ptr.add(offset + 3);
                        let flags = core::ptr::read_unaligned(ptr.add(offset + 4) as *const u32);
                        let enabled = flags & 1 != 0 || flags & 2 != 0;
                        apics.push(LocalApic { processor_id, apic_id, enabled });
                    }
                }
                // Type 9: Processor Local x2APIC (length 16)
                9 => {
                    if entry_len >= 16 {
                        let x2apic_id = core::ptr::read_unaligned(ptr.add(offset + 4) as *const u32);
                        let flags = core::ptr::read_unaligned(ptr.add(offset + 8) as *const u32);
                        let processor_uid = core::ptr::read_unaligned(ptr.add(offset + 12) as *const u32);
                        let enabled = flags & 1 != 0 || flags & 2 != 0;
                        apics.push(LocalApic {
                            processor_id: processor_uid as u8,
                            apic_id: x2apic_id as u8,
                            enabled,
                        });
                    }
                }
                _ => {} // Skip IO APIC, interrupt overrides, etc.
            }

            offset += entry_len;
        }
    }
    apics
}

/// Discover all CPU cores via ACPI MADT.
/// Call after memory subsystem is initialized (needs HHDM for physical access).
pub fn discover_cpus() {
    let rsdp_phys = match find_rsdp() {
        Some(addr) => addr,
        None => {
            crate::serial_println!("[acpi] RSDP not found (QEMU may not expose BIOS tables with -kernel)");
            return;
        }
    };
    crate::serial_println!("[acpi] RSDP at {:#x}", rsdp_phys);

    let (sdt_phys, is_xsdt) = match parse_rsdp(rsdp_phys) {
        Some(v) => v,
        None => { crate::serial_println!("[acpi] Failed to parse RSDP"); return; }
    };
    crate::serial_println!("[acpi] {} at {:#x}", if is_xsdt { "XSDT" } else { "RSDT" }, sdt_phys);

    let madt_phys = match find_madt(sdt_phys, is_xsdt) {
        Some(addr) => addr,
        None => { crate::serial_println!("[acpi] MADT not found"); return; }
    };
    crate::serial_println!("[acpi] MADT at {:#x}", madt_phys);

    let apics = parse_madt(madt_phys);
    let enabled_count = apics.iter().filter(|a| a.enabled).count();

    for apic in &apics {
        crate::serial_println!("[acpi] CPU: processor={} apic_id={} enabled={}",
            apic.processor_id, apic.apic_id, apic.enabled);
    }
    crate::serial_println!("[acpi] {} CPUs found ({} enabled)", apics.len(), enabled_count);

    *APIC_LIST.lock() = apics;
}

/// Get the list of discovered APIC IDs.
pub fn cpu_list() -> Vec<LocalApic> {
    APIC_LIST.lock().clone()
}

/// Number of enabled CPUs.
pub fn cpu_count() -> usize {
    APIC_LIST.lock().iter().filter(|a| a.enabled).count()
}
