#![feature(box_patterns, slice_patterns, convert)]

mod ast;
mod assembler;
mod bytecode;
mod executor;
mod interpreter;
mod jit;
mod llvm;
mod optimizer;
mod parser;

pub use executor::{Executor, ExecutionModel};

extern crate libc;
extern crate itertools;
extern crate unreachable;
extern crate vec_map;

#[cfg(feature="llvm")]
extern crate llvm as llvm_rs;
