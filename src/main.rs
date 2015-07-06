extern crate argparse;
extern crate itertools;
extern crate libc;
extern crate unreachable;
extern crate vec_map;

use std::fs::{File, OpenOptions};
use std::io::{Result, Read, Write, stdout};
use std::mem;
use std::path::Path;
use std::ptr;

use argparse::{ArgumentParser, StoreConst, Store};
use itertools::Itertools;
use libc::{c_void, mmap, mprotect, munmap, PROT_EXEC, PROT_WRITE, MAP_ANON, MAP_PRIVATE};
use vec_map::VecMap;
use unreachable::unreachable;

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
      _ => unreachable(),
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

#[derive(Copy, Clone, Debug)]
enum RegisterNumber {
  RAX = 0,
  RCX = 1,
  RDX = 2,
  RBX = 3,
  RSP = 4,
  RBP = 5,
  RSI = 6,
  RDI = 7,
  R8  = 8,
  R9  = 9,
  R10 = 10,
  R11 = 11,
  R12 = 12,
  R13 = 13,
  R14 = 14,
  R15 = 15,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RegisterSize {
  Int8,
  Int16,
  Int32,
  Int64,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
enum Register {
  RAX,
  EAX,
  AX,
  AH,

  RSP,
  ESP,

  RBP,
  EBP,

  RCX,
  RDX,
  RBX,
  RSI,
  RDI,
  R8,
  R9,
  R10,
  R11,
  R12,
  R13,
  R14,
  R15,

  R8B,
  R9B,
  R10B,
  R11B,
  R12B,
  R13B,
  R14B,
  R15B,
}

impl Register {
  fn size(&self) -> RegisterSize {
    use Register::*;
    match *self {
      RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI => RegisterSize::Int64,
      R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 => RegisterSize::Int64,
      EAX | ESP | EBP => RegisterSize::Int32,
      AX => RegisterSize::Int16,
      AH => RegisterSize::Int8,
      R8B | R9B | R10B | R11B | R12B | R13B | R14B | R15B => RegisterSize::Int8,
    }
  }

  fn is_64_bit(&self) -> bool {
    self.size() == RegisterSize::Int64
  }

  fn number(&self) -> u8 {
    use Register::*;
    let number = match *self {
      RAX => RegisterNumber::RAX,
      RCX => RegisterNumber::RCX,
      RDX => RegisterNumber::RDX,
      RBX => RegisterNumber::RBX,
      RSP => RegisterNumber::RSP,
      RBP => RegisterNumber::RBP,
      RSI => RegisterNumber::RSI,
      RDI => RegisterNumber::RDI,
      R8  => RegisterNumber::R8,
      R9  => RegisterNumber::R9,
      R10 => RegisterNumber::R10,
      R11 => RegisterNumber::R11,
      R12 => RegisterNumber::R12,
      R13 => RegisterNumber::R13,
      R14 => RegisterNumber::R14,
      R15 => RegisterNumber::R15,

      EAX => RegisterNumber::RAX,
      AX  => RegisterNumber::RAX,
      AH  => RegisterNumber::RAX,

      ESP => RegisterNumber::RSP,

      EBP => RegisterNumber::RBP,

      R8B  => RegisterNumber::R8,
      R9B  => RegisterNumber::R9,
      R10B => RegisterNumber::R10,
      R11B => RegisterNumber::R11,
      R12B => RegisterNumber::R12,
      R13B => RegisterNumber::R13,
      R14B => RegisterNumber::R14,
      R15B => RegisterNumber::R15,
    };
    number as u8
  }

  fn is_extended_register(&self) -> bool {
    self.number() >= (RegisterNumber::R8 as u8)
  }
}

#[allow(dead_code)]
#[derive(Debug)]
enum MachineInstruction {
  Ret,

  Push(Register),
  Pop(Register),

  MovRR(Register, Register),
//  MovRM(Register, Register, u32),
//  MovMR(Register, u32, Register),

  IncR(Register),
  DecR(Register),
  IncM(RegisterSize, Register, u32),
  DecM(RegisterSize, Register, u32),

  AddIR(u64, Register),
  SubIR(u64, Register),

  AddIM(u64, Register, u32),
  SubIM(u64, Register, u32),

  AddRR(Register, Register),
  SubRR(Register, Register),
}

impl MachineInstruction {
  fn emit(&self, machine_code: &mut Vec<u8>) {
    use MachineInstruction::*;

    match *self {
      Ret => {
        machine_code.push(self.opcode());
      }
      Push(register) | Pop(register) => {
        assert!(register.size() == RegisterSize::Int16 || register.size() == RegisterSize::Int64);
        if register.size() == RegisterSize::Int16 {
          machine_code.push(0x66);
        }
        machine_code.push(self.opcode() | register.number());
      }
      MovRR(..) | AddRR(..) | SubRR(..) => {
        let modrm = self.modrm();
        modrm.emit_rex_if_needed(machine_code);
        machine_code.extend(&[
          self.opcode(), modrm.encode()
        ]);
        modrm.emit_offset_if_needed(machine_code);
        self.emit_constant_if_needed(machine_code);
      }
      AddIR(..) | SubIR(..) | AddIM(..) | SubIM(..) => {
        let modrm = self.modrm();
        modrm.emit_rex_if_needed(machine_code);
        machine_code.extend(&[
          self.opcode(), modrm.encode() | (self.group1_opcode() << 3)
        ]);
        modrm.emit_offset_if_needed(machine_code);
        self.emit_constant_if_needed(machine_code);
      }
      IncR(..) | DecR(..) | IncM(..) | DecM(..) => {
        let modrm = self.modrm();
        modrm.emit_rex_if_needed(machine_code);
        machine_code.extend(&[
          self.opcode(), modrm.encode() | (self.group3_opcode() << 3),
        ]);
        modrm.emit_offset_if_needed(machine_code);
        self.emit_constant_if_needed(machine_code);
      }
    }
  }

  fn opcode(&self) -> u8 {
    use MachineInstruction::*;

    match *self {
      Ret => 0xc3,
      Push(..) => 0x50,
      Pop(..) => 0x58,
      MovRR(..) => 0x89,
      AddIR(constant, _) | SubIR(constant, _) if constant < 256 => 0x83,
      AddIM(..) | SubIM(..) => 0x80,
      AddIR(..) | SubIR(..) => 0x81,
      IncM(RegisterSize::Int8, _, _) | DecM(RegisterSize::Int8, _, _) => 0xfe,
      IncR(..) | DecR(..) | IncM(..) | DecM(..) => 0xff,
      AddRR(..) => 0x1,
      SubRR(..) => 0x29,
    }
  }

  fn group1_opcode(&self) -> u8 {
    use MachineInstruction::*;

    match *self {
      AddIR(..) | AddIM(..) => 0x0,
      SubIR(..) | SubIM(..) => 0x5,
      _ => unreachable!()
    }
  }

  fn group3_opcode(&self) -> u8 {
    use MachineInstruction::*;

    match *self {
      IncR(..) | IncM(..) => 0x00,
      DecR(..) | DecM(..) => 0x01,
      _ => unreachable!()
    }
  }

  fn modrm(&self) -> ModRM {
    use MachineInstruction::*;

    match *self {
      MovRR(source, dest) | AddRR(source, dest) | SubRR(source, dest) => {
        ModRM::TwoRegisters(source, dest)
      }
      AddIR(_, register) | SubIR(_, register) | IncR(register) | DecR(register) => {
        ModRM::Register(register)
      }
      AddIM(_, register, offset) | SubIM(_, register, offset) if offset == 0 => {
        ModRM::Memory(register.size(), register)
      }
      AddIM(_, register, offset) | SubIM(_, register, offset) if offset < 255 => {
        ModRM::Memory8BitDisplacement(register.size(), register, offset as u8)
      }
      IncM(size, register, offset) | DecM(size, register, offset) if offset == 0 => {
        ModRM::Memory(size, register)
      }
      IncM(size, register, offset) | DecM(size, register, offset) if offset < 255 => {
        ModRM::Memory8BitDisplacement(size, register, offset as u8)
      }
      _ => { println!("{:?}", *self); panic!() },
    }
  }

  fn emit_constant_if_needed(&self, machine_code: &mut Vec<u8>) {
    use MachineInstruction::*;

    match *self {
      AddIR(constant, _) | SubIR(constant, _) | AddIM(constant, _, _) | SubIM(constant, _, _) => {
        if constant < 256 {
          machine_code.push(constant as u8);
        } else {
          machine_code.extend(&[
            ((constant >>  0) & 0xff) as u8,
            ((constant >>  8) & 0xff) as u8,
            ((constant >> 16) & 0xff) as u8,
            ((constant >> 24) & 0xff) as u8,
          ]);
        }
      }
      _ => {}
    }
  }
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
enum ModRM {
  Memory(RegisterSize, Register),
  Memory8BitDisplacement(RegisterSize, Register, u8),
  Memory32BitDisplacement(RegisterSize, Register, u32),
  Register(Register),
  TwoRegisters(Register, Register),
}

impl ModRM {
  fn encode(&self) -> u8 {
    match *self {
      ModRM::Register(register) => 0b11000000 | (register.number() & 0x7),
      ModRM::TwoRegisters(source, dest) => 0b11000000 | (source.number() & 0x7) << 3 | (dest.number() & 0x7),
      ModRM::Memory(_, dest) => 0x0 | (dest.number() & 0x7),
      ModRM::Memory8BitDisplacement(_, dest, _) => 0b01000000 | (dest.number() & 0x7),
      ModRM::Memory32BitDisplacement(_, dest, _) => 0b10000000 | (dest.number() & 0x7),
    }
  }

  fn emit_offset_if_needed(&self, machine_code: &mut Vec<u8>) {
    match *self {
      ModRM::Memory8BitDisplacement(_, _, offset) => machine_code.push(offset),
      _ => {}
    }
  }

  fn needs_rex(&self) -> bool {
    return self.is_64_bit() || self.has_extended_register()
  }

  fn emit_rex_if_needed(&self, machine_code: &mut Vec<u8>) {
    if !self.needs_rex() {
      return
    }

    let rex_marker = 0b01000000;
    match *self {
      ModRM::TwoRegisters(source, dest) => {
        assert!(source.is_64_bit() == dest.is_64_bit());
        let mut rex = rex_marker;
        rex |= (source.is_64_bit() as u8) << 3;
        rex |= (source.is_extended_register() as u8) << 2;
        rex |= dest.is_extended_register() as u8;
        machine_code.push(rex);
      }
      ModRM::Register(..) | ModRM::Memory(..) | ModRM::Memory8BitDisplacement(..) | ModRM::Memory32BitDisplacement(..) => {
        let mut rex = rex_marker;
        rex |= (self.is_64_bit() as u8) << 3;
        rex |= self.has_extended_register() as u8;
        machine_code.push(rex);
      }
    }
  }

  fn is_64_bit(&self) -> bool {
    match *self {
      ModRM::TwoRegisters(source, dest) => {
        assert!(source.is_64_bit() == dest.is_64_bit());
        source.is_64_bit()
      }
      ModRM::Register(register) => {
        register.is_64_bit()
      }
      ModRM::Memory(size, _) | ModRM::Memory8BitDisplacement(size, _, _) | ModRM::Memory32BitDisplacement(size, _, _) => {
        size == RegisterSize::Int64
      }
    }
  }

  fn has_extended_register(&self) -> bool {
    match *self {
      ModRM::TwoRegisters(source, dest) => {
        source.is_extended_register() || dest.is_extended_register()
      }
      ModRM::Register(register) | ModRM::Memory(_, register) | ModRM::Memory8BitDisplacement(_, register, _) | ModRM::Memory32BitDisplacement(_, register, _) => {
        register.is_extended_register()
      }
    }
  }

}

fn lower(instructions: &[MachineInstruction]) -> Vec<u8> {
  let mut machine_code = Vec::new();
  for instruction in instructions {
    instruction.emit(&mut machine_code);
  }
  machine_code
}

#[inline(never)]
fn compile_to_machinecode(instructions: &Vec<LinkedInstruction>) -> Vec<u8> {
  use MachineInstruction::*;

  let arguments = &[Register::RDI, Register::RSI];

  let tape_head = Register::RAX;
  let output_buffer_head = Register::R12;
  let output_buffer_tail = Register::RBX;

  let prologue = lower(&[
    Push(Register::RBP),
    MovRR(Register::RSP, Register::RBP),

    MovRR(arguments[0], tape_head),
    MovRR(arguments[1], output_buffer_head),
    MovRR(arguments[1], output_buffer_tail),
  ]);

  let mut body = Vec::new();
  let mut loop_start_patch_points = VecMap::new();
  for (i, &instruction) in instructions.iter().enumerate() {

    match instruction {
      LinkedInstruction::MoveLeft(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            DecR(tape_head)
          } else {
            SubIR(amount as u64, tape_head)
          }
        ]));
      }
      LinkedInstruction::MoveRight(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            IncR(tape_head)
          } else {
            AddIR(amount as u64, tape_head)
          }
        ]));
      }
      LinkedInstruction::Add(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            IncM(RegisterSize::Int8, tape_head, 0)
          } else {
            AddIM(amount as u64, tape_head, 0)
          }
        ]));
      }
      LinkedInstruction::Subtract(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            DecM(RegisterSize::Int8, tape_head, 0)
          } else {
            SubIM(amount as u64, tape_head, 0)
          }
        ]));
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

#[derive(Copy, Clone)]
enum ExecutionModel {
  Interpret,
  JIT,
}

fn main() {
  let mut execution_model = ExecutionModel::JIT;
  let mut input_file_path: String = "input.bf".to_string();
  {
    let mut ap = ArgumentParser::new();
    ap.set_description("A BrainFuck interpreter in Rust");
    ap.refer(&mut execution_model).add_option(&["-i", "--interpret"], StoreConst(ExecutionModel::Interpret), "Use the interpreter");
    ap.refer(&mut input_file_path).add_argument("script", Store, "BrainFuck script to execute.");
    ap.parse_args_or_exit();
  }

  let source = load_file(&input_file_path).unwrap();
  let bytecode = link_bytecode(optimize_bytecode(compile_to_bytecode(source)));

  match execution_model {
    ExecutionModel::Interpret => unsafe { execute_bytecode(&bytecode) },
    ExecutionModel::JIT => {
      let machine_code = compile_to_machinecode(&bytecode);
      unsafe { execute_machinecode(&machine_code) };
    }
  }
}
