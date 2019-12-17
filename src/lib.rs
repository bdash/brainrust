#![feature(box_patterns)]
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
