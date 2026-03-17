#![no_std]
#![feature(abi_x86_interrupt)]
#![allow(dead_code)]

extern crate alloc;

pub mod boot;
pub mod arch;
pub mod memory;
pub mod drivers;
pub mod inference;
pub mod shell;
