#![feature(vecmap, core_intrinsics, libc)]

extern crate itertools;
extern crate libc;

use std::collections::VecMap;
use std::fs::{File, OpenOptions};
use std::io::{Result, Read, Write, stdout};
use std::mem;
use std::path::Path;
use std::ptr;

use itertools::Itertools;
use libc::{c_void, mmap, mprotect, PROT_EXEC, PROT_WRITE, MAP_ANON, MAP_PRIVATE};

#[inline(never)]
fn load_file(input_file_path: &str) -> Result<Vec<u8>> {
  let mut file = try!(File::open(&Path::new(input_file_path)));
  let mut contents: Vec<u8> = Vec::new();
  try!(file.read_to_end(&mut contents));
  Ok(contents)
}

fn write_to_file(file_path: &str, buffer: &Vec<u8>) -> Result<()> {
  let mut file = try!(OpenOptions::new().write(true).create(true).truncate(true).open(&Path::new(file_path)));
  try!(file.write_all(&buffer[..]));
  Ok(())
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Instruction {
  MoveLeft(usize),
  MoveRight(usize),
  Add(u8),
  Subtract(u8),
  Output,
  Input,
  LoopStart { end: Option<usize> },
  LoopEnd { start: Option<usize> },
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum LinkedInstruction {
  MoveLeft(usize),
  MoveRight(usize),
  Add(u8),
  Subtract(u8),
  Output,
  Input,
  LoopStart { end: usize },
  LoopEnd { start: usize },
}

impl Instruction {
  fn from_token(token: u8) -> Option<Instruction> {
    match token as char {
      '<' => Some(Instruction::MoveLeft(1)),
      '>' => Some(Instruction::MoveRight(1)),
      '+' => Some(Instruction::Add(1)),
      '-' => Some(Instruction::Subtract(1)),
      '.' => Some(Instruction::Output),
      ',' => Some(Instruction::Input),
      '[' => Some(Instruction::LoopStart { end: None }),
      ']' => Some(Instruction::LoopEnd { start: None }),
      _ => None
    }
  }
}

impl LinkedInstruction {
  fn from_instruction(instruction: &Instruction) -> LinkedInstruction {
    match *instruction {
      Instruction::MoveLeft(amount) => LinkedInstruction::MoveLeft(amount),
      Instruction::MoveRight(amount) => LinkedInstruction::MoveRight(amount),
      Instruction::Add(amount) => LinkedInstruction::Add(amount),
      Instruction::Subtract(amount) => LinkedInstruction::Subtract(amount),
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
      Instruction::MoveLeft(_) => vec![Instruction::MoveLeft(group.len())],
      Instruction::MoveRight(_) => vec![Instruction::MoveRight(group.len())],
      Instruction::Add(_) => {
        group.chunks(u8::max_value() as usize).map(|g| {
          Instruction::Add(g.len() as u8)
        }).collect()
      },
      Instruction::Subtract(_) => {
        group.chunks(u8::max_value() as usize).map(|g| {
          Instruction::Subtract(g.len() as u8)
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
  let mut output = Vec::with_capacity(256);
  let mut tape = vec![0u8; 1024];
  let mut tape_head = 0;
  let mut ip = 0;
  while ip < instructions.len() {
    let instruction = instructions.get_unchecked(ip);

    match *instruction {
      LinkedInstruction::MoveLeft(amount) => tape_head -= amount,
      LinkedInstruction::MoveRight(amount) => tape_head += amount,

      LinkedInstruction::Add(amount) => {
        let value = tape.get_unchecked_mut(tape_head);
        *value = value.wrapping_add(amount);
      }
      LinkedInstruction::Subtract(amount) => {
        let value = tape.get_unchecked_mut(tape_head);
        *value = value.wrapping_sub(amount);
      }

      LinkedInstruction::LoopStart { end } => {
        if *tape.get_unchecked(tape_head) == 0 {
          ip = end + 1;
          continue
        }
      }
      LinkedInstruction::LoopEnd { start } => {
        if *tape.get_unchecked(tape_head) != 0 {
          ip = start + 1;
          continue
        }
      }
      LinkedInstruction::Output => {
        let c = *tape.get_unchecked(tape_head);
        output.push(c);
        if c as char == '\n' {
          stdout().write(&output[..]).unwrap();
          output.clear();
        }
      }
      _ => std::intrinsics::unreachable(),
    }
    ip += 1
  }
  if output.len() > 0 {
    stdout().write(&output[..]).unwrap();
  }
}

#[inline(never)]
unsafe fn jit(instructions: &Vec<LinkedInstruction>) {
  let buffer_size = (instructions.len() * 32) as u64;
  let buffer = mmap(ptr::null::<u8>() as *mut libc::c_void, buffer_size, PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

  let prologue = vec![
    0x55, // push %rbp
    0x48, 0x89, 0xe5, // mov %rsp, %rbp
    0x48, 0x81, 0xec, 0xff, 0x0f, 0x00, 0x00, // subq  $0xfff, %rsp

    0x48, 0x8d, 0xbd, 0xf0, 0xfb, 0xff, 0xff, // leaq -1040(%rbp), %rdi
    0x48, 0xc7, 0xc1, 0x00, 0x04, 0x00, 0x00, // mov $1024, %rcx
    0xf3, 0xaa, // rep stosb
    0x48, 0x89, 0xf8, // mov %rdi, %rax
    0x48, 0x2d, 0x00, 0x04, 0x00, 0x00, // subl $1024, %rax
  ];

  let mut body = Vec::new();
  let mut loop_start_patch_points = VecMap::new();
  for (i, &instruction) in instructions.iter().enumerate() {

    match instruction {
      LinkedInstruction::MoveLeft(amount) => {
        assert!(amount < 255);
        if amount > 1 {
          body.extend(vec![
            0x48, 0x83, 0xe8, amount as u8, // subq $amount, %rax
          ]);
        } else {
          body.extend(vec![
            0x48, 0xff, 0xc8 // decq %rax
          ]);
        }
      }
      LinkedInstruction::MoveRight(amount) => {
        assert!(amount < 255);
        if amount > 1 {
          body.extend(vec![
            0x48, 0x83, 0xc0, amount as u8, // addq $amount, %rax
          ]);
        } else {
          body.extend(vec![
            0x48, 0xff, 0xc0, // inc %rax
          ]);
        }
      }
      LinkedInstruction::Add(amount) => {
        assert!(amount < 255);
        if amount > 1 {
          body.extend(vec![
            0x80, 0x00, amount as u8, // addq $amount, (%rax)
          ]);
        } else {
          body.extend(vec![
            0x48, 0xff, 0x00, // inc (%rax)
          ]);
        }
      }
      LinkedInstruction::Subtract(amount) => {
        assert!(amount < 255);
        if amount > 1 {
          body.extend(vec![
            0x80, 0x28, amount as u8, // addq $amount, (%rax)
          ]);
        } else {
          body.extend(vec![
            0x48, 0xff, 0x08, // sub (%rax)
          ]);
        }
      }
      LinkedInstruction::LoopStart { end: _ } => {
        body.extend(vec![
          0x80, 0x38, 0x00, // cmpb $0, (%rax)
          0x0f, 0x84, 0xff, 0xff, 0xff, 0xff, // jz placeholder
        ]);
        loop_start_patch_points.insert(i, body.len() - 4);
      }
      LinkedInstruction::LoopEnd { start } => {
        let loop_start_patch_point = loop_start_patch_points[start];
        let distance = body.len() - loop_start_patch_point + 5;
        let offset = -(distance as i64);

        body.extend(vec![
          0x80, 0x38, 0x00, // cmpb $0, (%rax)
          0x0f, 0x85, // jnz offset
          ((offset >>  0) & 0xff) as u8,
          ((offset >>  8) & 0xff) as u8,
          ((offset >> 16) & 0xff) as u8,
          ((offset >> 24) & 0xff) as u8,
        ]);

        body[loop_start_patch_point + 0] = ((distance >>  0) & 0xff) as u8;
        body[loop_start_patch_point + 1] = ((distance >>  8) & 0xff) as u8;
        body[loop_start_patch_point + 2] = ((distance >> 16) & 0xff) as u8;
        body[loop_start_patch_point + 3] = ((distance >> 24) & 0xff) as u8;
      }
      LinkedInstruction::Output => {
        body.extend(vec![
          0x50, // push %rax
          0x48, 0x89, 0xc6, // mov %rax, $rsi
          0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov $1, %rdi
          0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00, // mov $1, %rdx
          0x48, 0xc7, 0xc0, 0x04, 0x00, 0x00, 0x02, // mov $0x2000004, %rax
          0x0f, 0x05, // syscall
          0x58, // pop %rax
        ]);
      }
      _ => { println!("{:?}", instruction); panic!() }
    }
  }

  let epilogue = vec![
    0x48, 0x31, 0xc0, // xor %rax, %rax
    0x48, 0x81, 0xc4, 0xff, 0x0f, 0x00, 0x00, // addq $0xfff, %rsp
    0x5d, // pop %rbp
    0xc3, // ret
  ];

  let machine_code: Vec<u8> = prologue.into_iter().chain(body).chain(epilogue).collect();
  write_to_file("out.dat", &machine_code).unwrap();

  ptr::copy(machine_code.as_ptr(), buffer as *mut u8, machine_code.len());
  mprotect(buffer, buffer_size, PROT_EXEC);

  let function: extern "C" fn() -> libc::c_void = mem::transmute(buffer);
  function();
}

fn main() {
  let source = load_file("input.bf").unwrap();
  let instructions = link(optimize(compile(source)));
  unsafe { jit(&instructions) };
  unsafe { execute(instructions) };
}
