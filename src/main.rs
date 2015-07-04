#![feature(vecmap, core_intrinsics)]

extern crate itertools;

use std::collections::VecMap;
use std::io::{Result, Read, Write, stdout};
use std::fs::File;
use std::path::Path;
use itertools::Itertools;


#[inline(never)]
fn load_file(input_file_path: &str) -> Result<Vec<u8>> {
  let mut file = try!(File::open(&Path::new(input_file_path)));
  let mut contents: Vec<u8> = Vec::new();
  try!(file.read_to_end(&mut contents));
  Ok(contents)
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Direction {
  Left,
  Right
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Operation {
  Add,
  Subtract
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Instruction {
  Move(Direction, usize),
  Modify(Operation, u8),
  Output,
  Input,
  LoopStart { end: Option<usize> },
  LoopEnd { start: Option<usize> },
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum LinkedInstruction {
  Move(Direction, usize),
  Modify(Operation, u8),
  Output,
  Input,
  LoopStart { end: usize },
  LoopEnd { start: usize },
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

impl LinkedInstruction {
  fn from_instruction(instruction: &Instruction) -> LinkedInstruction {
    match *instruction {
      Instruction::Move(direction, amount) => LinkedInstruction::Move(direction, amount),
      Instruction::Modify(operation, amount) => LinkedInstruction::Modify(operation, amount),
      Instruction::Output => LinkedInstruction::Output,
      Instruction::Input => LinkedInstruction::Input,
      Instruction::LoopStart { end: Some(end) } => LinkedInstruction::LoopStart { end: end },
      Instruction::LoopEnd { start: Some(start) } => LinkedInstruction::LoopEnd { start: start},
      _ => panic!(),
    }
  }
}

#[inline(never)]
fn compile(source: Vec<u8>) -> Vec<Instruction> {
  source.into_iter().filter_map(Instruction::from_token).collect()
}

#[inline(never)]
fn optimize(instructions: Vec<Instruction>) -> Vec<Instruction> {
  instructions.iter().group_by(|&instruction| instruction).flat_map(|(&key, group)| {
    match key {
      Instruction::Move(direction, _) => {
        vec![Instruction::Move(direction, group.len())]
      },
      Instruction::Modify(operation, _) => {
        group.chunks(u8::max_value() as usize).map(|g| {
          Instruction::Modify(operation, g.len() as u8)
        }).collect()
      },
      _ => group.iter().map(|&instruction| *instruction).collect()
    }
  }).collect()
}

#[inline(never)]
fn link(source: Vec<Instruction>) -> Vec<LinkedInstruction> {
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
        LinkedInstruction::LoopStart { end: *end }
      },
      _ => LinkedInstruction::from_instruction(&instruction)
      }
  }).collect()
}

#[inline(never)]
unsafe fn execute(instructions: Vec<LinkedInstruction>) {
//  println!("{:?}", instructions);
  let mut output = Vec::with_capacity(256);
  let mut tape = vec![0u8; 3000];
  let mut tape_head = 0;
  let mut ip = 0;
  while ip < instructions.len() {
    let instruction = instructions.get_unchecked(ip);

//    println!("{:?}", instruction);
    ip = match *instruction {
      LinkedInstruction::Modify(ref operation, amount) => {
        tape[tape_head] = match *operation {
          Operation::Add => tape.get_unchecked(tape_head).wrapping_add(amount),
          Operation::Subtract => tape.get_unchecked(tape_head).wrapping_sub(amount),
        };
        ip + 1
      },
      LinkedInstruction::Move(ref direction, amount) => {
        match *direction {
          Direction::Left => tape_head -= amount,
          Direction::Right => tape_head += amount,
        }
        ip + 1
      },
      LinkedInstruction::LoopStart { end } => {
        if *tape.get_unchecked(tape_head) == 0 {
          end + 1
        } else {
          ip + 1
        }
      },
      LinkedInstruction::LoopEnd { start } => {
        if *tape.get_unchecked(tape_head) != 0 {
          start + 1
        } else {
          ip + 1
        }
      },
      LinkedInstruction::Output => {
        let c = *tape.get_unchecked(tape_head);
        output.push(c);
        if c as char == '\n' {
          stdout().write(&output[..]).unwrap();
          output.clear();
        }
        ip + 1
      }
      _ => std::intrinsics::unreachable(),
    }
  }
}

fn main() {
  let source = load_file("input.bf").unwrap();
  let instructions = link(optimize(compile(source)));
  unsafe { execute(instructions) };
}
