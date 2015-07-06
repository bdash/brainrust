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
use libc::{c_void, mmap, mprotect, munmap, PROT_EXEC, PROT_WRITE, MAP_ANON, MAP_PRIVATE};

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
fn compile_to_bytecode(source: Vec<u8>) -> Vec<Instruction> {
  source.into_iter().filter_map(Instruction::from_token).collect()
}

#[inline(never)]
fn optimize_bytecode(instructions: Vec<Instruction>) -> Vec<Instruction> {
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
fn link_bytecode(source: Vec<Instruction>) -> Vec<LinkedInstruction> {
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
unsafe fn execute_bytecode(instructions: &Vec<LinkedInstruction>) {
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

struct MemoryMap {
  size: u64,
  buffer: *mut libc::c_void,
}

impl MemoryMap {
  unsafe fn new(size: usize, protection: i32) -> MemoryMap {
    let buffer = mmap(ptr::null::<u8>() as *mut libc::c_void, size as u64, protection, MAP_ANON | MAP_PRIVATE, 0, 0);
    MemoryMap { size: size as u64, buffer: buffer }
  }

  unsafe fn reprotect(&self, protection: i32) {
    mprotect(self.buffer, self.size, protection);
  }
}

impl Drop for MemoryMap {
  fn drop(&mut self) {
    unsafe { munmap(self.buffer, self.size) };
  }
}

#[inline(never)]
fn compile_to_machinecode(instructions: &Vec<LinkedInstruction>) -> Vec<u8> {
  let prologue = vec![
    0x55, // push %rbp
    0x48, 0x89, 0xe5, // mov %rsp, %rbp

    // %rax is the tape head pointer.
    0x48, 0x89, 0xf8, // mov %rdi, %rax

    // %rbx is the output buffer insertion pointer.
    0x48, 0x89, 0xf3, // mov %rsi, %rbx

    // %r12 is the output buffer base.
    0x49, 0x89, 0xf4, // mov %rsi, %r12
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
          // Append byte to output buffer.
          0x44, 0x8a, 0x28, // movb (%rax), %r13b
          0x44, 0x88, 0x2b, // movb %r13b, (%rbx)
          0x48, 0xff, 0xc3, // inc %rbx

          // Don't call write until we see a newline character.
          0x41, 0x80, 0xfd, 0x0a, // cmp $10, %r13b
          0x75, 0x1e, // jneq +30

          0x50, // push %rax

          // Compute the number of bytes to write.
          0x48, 0x89, 0xda, // movq %rbx, %rdx
          0x4c, 0x29, 0xe2, // subq %r12, %rdx

          // Write bytes from %r12 to stdout.
          0x4c, 0x89, 0xe6, // mov %r12, %rsi
          0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov $1, %rdi
          0x48, 0xc7, 0xc0, 0x04, 0x00, 0x00, 0x02, // mov $0x2000004, %rax
          0x0f, 0x05, // syscall

          // Reset the output buffer tail to the start.
          0x4c, 0x89, 0xe3, // mov %r12, %rbx

          0x58, // pop %rax
        ]);
      }
      _ => { println!("{:?}", instruction); panic!() }
    }
  }

  let epilogue = vec![
    // Compute the number of bytes to write.
    0x48, 0x89, 0xda, // movq %rbx, %rdx
    0x4c, 0x29, 0xe2, // subq %r12, %rdx

    // Don't call write if we have nothing in the output buffer.
    0x74, 0x13, // jeq +19

    // Write bytes from %r12 to stdout.
    0x4c, 0x89, 0xe6, // mov %r12, %rsi
    0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov $1, %rdi
    0x48, 0xc7, 0xc0, 0x04, 0x00, 0x00, 0x02, // mov $0x2000004, %rax
    0x0f, 0x05, // syscall

    0x48, 0x31, 0xc0, // xor %rax, %rax (jump target)
    0x5d, // pop %rbp
    0xc3, // ret
  ];

  prologue.into_iter().chain(body).chain(epilogue).collect::<Vec<u8>>()
}

unsafe fn execute_machinecode(machine_code: &Vec<u8>) {
  write_to_file("out.dat", &machine_code).unwrap();

  let map = MemoryMap::new(machine_code.len(), PROT_WRITE);
  ptr::copy(machine_code.as_ptr(), map.buffer as *mut u8, machine_code.len());
  map.reprotect(PROT_EXEC);

  println!("Copied {:?} bytes of machine code to executable region at {:?}.", machine_code.len(), map.buffer);

  let function: extern "C" fn(*mut u8, *mut u8) -> u64 = mem::transmute(map.buffer);

  let tape = &mut [0u8;1024];
  let output_buffer = &mut [0u8;256];
  function(tape.as_mut_ptr(), output_buffer.as_mut_ptr());
}

fn main() {
  let source = load_file("input.bf").unwrap();
  let bytecode = link_bytecode(optimize_bytecode(compile_to_bytecode(source)));
  let machine_code = compile_to_machinecode(&bytecode);
  unsafe { execute_machinecode(&machine_code) };
  unsafe { execute_bytecode(&bytecode) };
}
