use super::parser::Token;

#[derive(Debug, PartialEq, Clone)]
pub enum Node {
  Block(Vec<Node>),
  MoveLeft(usize),
  MoveRight(usize),
  Add(u8, i32),
  Subtract(u8, i32),
  Set(u8, i32),
  Output,
  Input,
  Loop(Box<Node>),
}

impl Node {
  pub fn from_tokens(tokens: Vec<Token>) -> Node {
    use super::parser::Token::*;

    let mut nodes: Vec<Vec<Node>> = vec![ vec![] ];

    for token in tokens {
      let node = match token {
        MoveLeft => Some(Node::MoveLeft(1)),
        MoveRight => Some(Node::MoveRight(1)),
        Add => Some(Node::Add(1, 0)),
        Subtract => Some(Node::Subtract(1, 0)),
        Output => Some(Node::Output),
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
      MoveLeft(..) | MoveRight(..) | Add(..) | Subtract(..) | Set(..) => true,
      _ => false,
    }
  }
}
