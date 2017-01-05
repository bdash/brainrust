use super::ast::Node;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ByteCode {
  MoveLeft(usize),
  MoveRight(usize),
  Add(u8, i32),
  Subtract(u8, i32),
  Set(u8, i32),
  Output,
  Input,
  LoopStart { end: usize },
  LoopEnd { start: usize },
}

impl ByteCode {
  fn from_ast_at_offset(node: &Node, offset: usize) -> Vec<ByteCode> {
    use super::ast::Node::*;

    let code = match *node {
      MoveLeft(amount) => Some(ByteCode::MoveLeft(amount)),
      MoveRight(amount) => Some(ByteCode::MoveRight(amount)),
      Add(amount, offset) => Some(ByteCode::Add(amount, offset)),
      Subtract(amount, offset) => Some(ByteCode::Subtract(amount, offset)),
      Set(value, offset) => Some(ByteCode::Set(value, offset)),
      Output => Some(ByteCode::Output),
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
