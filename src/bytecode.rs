use super::ast::Node;
use std::fmt;

#[derive(PartialEq, Copy, Clone)]
pub enum ByteCode {
  Move(isize),
  Add{ amount: i8, offset: i32 },
  Set{ value: u8, offset: i32 },
  MultiplyAdd { multiplier: i8, source: i32, dest: i32 },
  Output { offset: i32 },
  Input,
  LoopStart { end: usize },
  LoopEnd { start: usize },
}

impl fmt::Debug for ByteCode {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    use self::ByteCode::*;

    match *self {
      Move(amount) => write!(f, "Move({})", amount),
      Add{ amount, offset } => write!(f, "Add{{ amount: {}, offset: {} }}", amount, offset),
      Set{ value, offset } => write!(f, "Set{{ value: {}, offset: {} }}", value, offset),
      MultiplyAdd{ multiplier, source, dest } => write!(f, "MultiplyAdd{{ multiplier: {}, source: {}, dest: {} }}", multiplier, source, dest),
      Output { offset } => write!(f, "Output {{ offset: {} }}", offset),
      Input => write!(f, "Input"),
      LoopStart { end } => write!(f, "LoopStart{{ end: {} }}", end),
      LoopEnd { start } => write!(f, "LoopEnd{{ start: {} }}", start),
    }
  }
}

impl ByteCode {
  fn from_ast_at_offset(node: &Node, offset: usize) -> Vec<ByteCode> {
    use super::ast::Node::*;

    let code = match *node {
      Move(amount) => Some(ByteCode::Move(amount)),
      Add{ amount, offset } => Some(ByteCode::Add{ amount, offset }),
      Set{ value, offset } => Some(ByteCode::Set{ value, offset }),
      MultiplyAdd { multiplier, source, dest } => Some(ByteCode::MultiplyAdd{ multiplier, source, dest }),
      Output { offset } => Some(ByteCode::Output { offset }),
      Input => Some(ByteCode::Input),
      Loop(..) | Node::Block(..) => None,
    };

    if let Some(code) = code {
      return vec![ code ]
    }

    match *node {
      Loop(box ref block) => {
        let block_bytecode = Self::from_ast_at_offset(block, offset + 1);
        let start = ByteCode::LoopStart { end: offset + block_bytecode.len() + 1};
        let end = ByteCode::LoopEnd { start: offset };

        let mut result = vec![ start ];
        result.extend(block_bytecode);
        result.push(end);
        result
      }
      Block(ref children) => {
        let mut bytecode = Vec::new();
        for node in children {
          let current_offset = bytecode.len() + offset;
          bytecode.extend(Self::from_ast_at_offset(node, current_offset));
        }
        bytecode
      }
      _ => unreachable!()
    }
  }

  pub fn from_ast(root: &Node) -> Vec<ByteCode> {
    Self::from_ast_at_offset(root, 0)
  }
}
