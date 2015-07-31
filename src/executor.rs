use super::ast::Node;
use super::bytecode::ByteCode;
use super::{interpreter, jit, optimizer};
use super::parser::Token;

#[cfg(feature="llvm")]
use super::llvm;

pub struct Executor {
  ast: Node,
}

impl Executor {
  pub fn new(source: Vec<u8>) -> Self {
    let tokens = source.into_iter().filter_map(Token::from_byte).collect();
    Executor { ast: Node::from_tokens(tokens) }
  }

  pub fn execute(&self, execution_model: ExecutionModel) {
    let bytecode = self.optimized_bytecode();
    match execution_model {
     ExecutionModel::Interpret => interpreter::execute_bytecode(&bytecode),
      ExecutionModel::JIT => jit::execute_bytecode(&bytecode),

      #[cfg(feature="llvm")]
      ExecutionModel::LLVM => llvm::execute_bytecode(&bytecode),
      #[cfg(not(feature="llvm"))]
      ExecutionModel::LLVM => panic!("LLVM is not available"),
    }
  }

  fn optimized_bytecode(&self) -> Vec<ByteCode> {
    ByteCode::from_ast(&optimizer::optimize(&self.ast))
  }
}

#[derive(Copy, Clone)]
pub enum ExecutionModel {
  Interpret,
  JIT,
  LLVM,
}
