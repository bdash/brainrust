#![feature(vecmap)]

use std::collections::VecMap;
use std::io::{Result, Read, Write, stdout};
use std::fs::File;
use std::path::Path;

#[inline(never)]
fn load_file(input_file_path: &str) -> Result<Vec<u8>> {
  let mut file = try!(File::open(&Path::new(input_file_path)));
  let mut contents: Vec<u8> = Vec::new();
  try!(file.read_to_end(&mut contents));
  Ok(contents)
}

#[derive(Debug)]
enum Direction {
  Left,
  Right
}

#[derive(Debug)]
enum Operation {
  Add,
  Subtract
}

#[derive(Debug)]
enum Instruction {
  Move(Direction, usize),
  Modify(Operation, u8),
  Output,
  Input,
  LoopStart { end: Option<usize> },
  LoopEnd { start: Option<usize> },
}

impl Instruction {
  fn from_token(token: u8) -> Option<Instruction> {
    match token as char {
      '<' => Some(Instruction::Move(Direction::Left, 1)),
      '>' => Some(Instruction::Move(Direction::Right, 1)),
      '+' => Some(Instruction::Modify(Operation::Add, 1)),
      '-' => Some(Instruction::Modify(Operation::Subtract, 1)),
      '.' => Some(Instruction::Output),
      ',' => Some(Instruction::Input),
      '[' => Some(Instruction::LoopStart { end: None }),
      ']' => Some(Instruction::LoopEnd { start: None }),
      _ => { /* println!("Unknown instruction {:?}", token as char); */ None }
    }
  }
}

#[inline(never)]
fn compile(source: Vec<u8>) -> Vec<Instruction> {
  source.into_iter().filter_map(|token| {
    if token as char == '\n' {
      None
    } else {
      Instruction::from_token(token)
    }
  }).collect()
}

#[inline(never)]
fn link(source: Vec<Instruction>) -> Vec<Instruction> {
  let mut loop_starts = Vec::new();
  let mut loop_ends = VecMap::new();
  let phase1: Vec<(usize, Instruction)>;
  {
    phase1 = source.into_iter().enumerate().map(|(i, instruction)| {
      match instruction {
        Instruction::LoopStart { end: None } => {
          loop_starts.push(i);
          (i, instruction)
        }
        Instruction::LoopEnd { start: None } => {
          let start = loop_starts.pop().unwrap();
          loop_ends.insert(start, i);
          (i, Instruction::LoopEnd { start: Some(start) })
        },
        _ => (i, instruction)
      }
    }).collect();
  }

  phase1.into_iter().map(|(i, instruction)| {
    match instruction {
      Instruction::LoopStart { end: None } => {
        let end = loop_ends.get(&i).unwrap();
        Instruction::LoopStart { end: Some(*end) }
      },
      _ => instruction,
      }
  }).collect()
}

#[inline(never)]
fn execute(instructions: Vec<Instruction>) {
  let mut tape = vec![0u8 ;30000];
  let mut tape_head = 0;
  let mut ip = 0;
  while ip < instructions.len() {
    let instruction = &instructions[ip];

    ip = match *instruction {
      Instruction::Modify(ref operation, amount) => {
        tape[tape_head] = match *operation {
          Operation::Add => tape[tape_head].wrapping_add(amount),
          Operation::Subtract => tape[tape_head].wrapping_sub(amount),
        };
        ip + 1
      },
      Instruction::Move(ref direction, amount) => {
        match *direction {
          Direction::Left => tape_head -= amount,
          Direction::Right => tape_head += amount,
        }
        ip + 1
      },
      Instruction::LoopStart { end: Some(end)} => {
        if tape[tape_head] == 0 {
          end + 1
        } else {
          ip + 1
        }
      },
      Instruction::LoopEnd { start: Some(start)} => {
        if tape[tape_head] != 0 {
          start + 1
        } else {
          ip + 1
        }
      },
      Instruction::Output => {
        stdout().write(&tape[tape_head..tape_head + 1]).unwrap();
        ip + 1
      }
      _ => panic!()
    }
  }
}

fn main() {
  let source = load_file("input.bf").unwrap();
  let instructions = link(compile(source));
  execute(instructions);
}
