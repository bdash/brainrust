use super::parser::Token;
use std::fmt;

#[derive(PartialEq, Clone)]
pub enum Node {
  Block(Vec<Node>),
  Move(isize),
  Add{ amount: i8, offset: i32 },
  Set{ value: u8, offset: i32 },
  Output{ offset: i32 },
  Input,
  Loop(Box<Node>),
}

impl fmt::Debug for Node {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    use self::Node::*;

    match *self {
      Block(ref nodes) => write!(f, "{:#?}", nodes),
      Move(amount) => write!(f, "Move({})", amount),
      Add{ amount, offset } => write!(f, "Add{{ amount: {}, offset: {} }}", amount, offset),
      Set{ value, offset } => write!(f, "Set{{ value: {}, offset: {} }}", value, offset),
      Output{ offset } => write!(f, "Output{{ offset: {} }}", offset),
      Input => write!(f, "Input"),
      Loop(ref nodes) => write!(f, "Loop({:#?})", nodes),
    }
  }
}

impl Node {
  pub fn from_tokens(tokens: Vec<Token>) -> Node {
    use super::parser::Token::*;

    let mut nodes: Vec<Vec<Node>> = vec![ vec![] ];

    for token in tokens {
      let node = match token {
        MoveLeft => Some(Node::Move(-1)),
        MoveRight => Some(Node::Move(1)),
        Add => Some(Node::Add{ amount: 1, offset: 0 }),
        Subtract => Some(Node::Add{ amount: -1, offset: 0 }),
        Output => Some(Node::Output{ offset: 0 }),
        Input => Some(Node::Input),
        LoopStart => {
          nodes.push(Vec::new());
          None
        }
        LoopEnd => Some(Node::Loop(Box::new(Node::Block(nodes.pop().unwrap())))),
      };
      if let Some(node) = node {
        nodes.last_mut().unwrap().push(node);
      };
    }

    assert_eq!(nodes.len(), 1);
    Node::Block(nodes.pop().unwrap())
  }

  pub fn children(&self) -> Vec<Node> {
    match *self {
      Node::Block(ref children) => children.clone(),
      _ => vec![],
    }
  }

  pub fn is_mutation(&self) -> bool {
    use self::Node::*;

    match *self {
      Move(..) | Add{..} | Set{..} => true,
      _ => false,
    }
  }
}
