#![feature(box_patterns, slice_patterns)]
#![allow(clippy::identity_op, clippy::unreadable_literal)]

mod ast;
mod assembler;
mod bytecode;
mod executor;
mod interpreter;
mod jit;
mod llvm;
mod optimizer;
mod parser;

pub use crate::executor::{Executor, ExecutionModel};

extern crate libc;
extern crate itertools;
extern crate unreachable;
extern crate vec_map;
extern crate syscall;

#[cfg(feature="llvm")]
extern crate llvm as llvm_rs;
