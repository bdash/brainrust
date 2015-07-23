#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Token {
  MoveLeft,
  MoveRight,
  Add,
  Subtract,
  Output,
  Input,
  LoopStart,
  LoopEnd,
}

impl Token {
  pub fn from_byte(token: u8) -> Option<Token> {
    use self::Token::*;

    match token as char {
      '<' => Some(MoveLeft),
      '>' => Some(MoveRight),
      '+' => Some(Add),
      '-' => Some(Subtract),
      '.' => Some(Output),
      ',' => Some(Input),
      '[' => Some(LoopStart),
      ']' => Some(LoopEnd),
      _ => None
    }
  }
}
