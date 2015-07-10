#![feature(box_patterns, slice_patterns, convert)]

extern crate argparse;
extern crate itertools;
extern crate libc;
extern crate unreachable;
extern crate vec_map;

use std::collections::HashMap;
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
enum Token {
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
  fn from_byte(token: u8) -> Option<Token> {
    use Token::*;

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

#[derive(Debug, PartialEq, Clone)]
enum Node {
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
  fn from_tokens(tokens: Vec<Token>) -> Node {
    use Token::*;

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
    return Node::Block(nodes.pop().unwrap());
  }

  fn compile_to_bytecode(&self) -> Vec<ByteCode> {
    self.compile_to_bytecode_at_offset(0)
  }

  fn compile_to_bytecode_at_offset(&self, offset: usize) -> Vec<ByteCode> {
    use Node::*;

    let code = match self {
      &MoveLeft(amount) => Some(ByteCode::MoveLeft(amount)),
      &MoveRight(amount) => Some(ByteCode::MoveRight(amount)),
      &Add(amount, offset) => Some(ByteCode::Add(amount, offset)),
      &Subtract(amount, offset) => Some(ByteCode::Subtract(amount, offset)),
      &Set(value, offset) => Some(ByteCode::Set(value, offset)),
      &Output => Some(ByteCode::Output),
      &Input => Some(ByteCode::Input),
      &Loop(..) | &Node::Block(..) => None,
    };

    if let Some(code) = code {
      return vec![ code ]
    }

    match self {
      &Loop(box ref block) => {
        let block_bytecode = block.compile_to_bytecode_at_offset(offset + 1);
        let start = ByteCode::LoopStart { end: offset + block_bytecode.len() + 1};
        let end = ByteCode::LoopEnd { start: offset };

        let mut result = vec![ start ];
        result.extend(block_bytecode);
        result.push(end);
        result
      }
      &Block(ref children) => {
        let mut bytecode = Vec::new();
        for node in children {
          let current_offset = bytecode.len() + offset;
          bytecode.extend(node.compile_to_bytecode_at_offset(current_offset));
        }
        bytecode
      }
      _ => unreachable!()
    }
  }

  fn optimize(&self) -> Node {
    use Node::*;

    match *self {
      MoveLeft(..) | MoveRight(..) | Add(..) | Subtract(..) | Set(..) | Input | Output => self.clone(),
      Loop(box ref block) => {
        match block.children().as_slice() {
          [ Subtract(1, 0) ] => Set(0, 0),
          _ => Loop(Box::new(block.optimize())),
        }
      }
      Block(ref children) => {
        let simplified_children = Node::simplify_mutation_sequences(children);
        let optimized_children = simplified_children.iter().map(Node::optimize).collect();
        Block(optimized_children)
      }
    }
  }

  fn children(&self) -> Vec<Node> {
    match *self {
      Node::Block(ref children) => children.clone(),
      _ => vec![],
    }
  }

  fn is_mutation(&self) -> bool {
    use Node::*;

    match *self {
      MoveLeft(..) | MoveRight(..) | Add(..) | Subtract(..) | Set(..) => true,
      _ => false,
    }
  }

  fn simplify_mutation_sequences(children: &Vec<Node>) -> Vec<Node> {
    use Node::*;

    children.iter().group_by(|&node| node.is_mutation()).flat_map(|(key, group)| {
      match key {
        true => {
          let (index, mutations) = evaluate_mutations(&group);

          let mut modified_offsets = mutations.keys().collect::<Vec<&i32>>();
          modified_offsets.sort();

          modified_offsets.iter().flat_map(|offset| {
            let value = mutations[*offset];
            match value {
              Mutation::Subtract(value) => Some(Subtract(value as u8, **offset)),
              Mutation::Add(value) => Some(Add(value as u8, **offset)),
              Mutation::Set(value) => Some(Set(value as u8, **offset))
            }
          }).chain(
            match index.signum() {
               1 => Some(MoveRight(index as usize)),
              -1 => Some(MoveLeft(index.abs() as usize)),
               _ => None
            }
          ).collect::<Vec<_>>()
        }
        false => group.iter().map(|&node| node.clone()).collect()
      }
    }).collect()
  }
}

#[derive(Copy, Clone, Debug)]
enum Mutation {
  Add(i8),
  Subtract(i8),
  Set(i8),
}

impl Mutation {
  fn combine(&mut self, other: Mutation) {
    use Mutation::*;

    *self = match (*self, other) {
      (_,           Set(_))           => other,
      (Set(base),   Add(amount))      => Set(base + amount),
      (Set(base),   Subtract(amount)) => Set(base - amount),
      (Add(a),      Add(b))           => Add(a + b),
      (Subtract(a), Subtract(b))      => Subtract(a + b),
      (Add(a),      Subtract(b))      => {
        match (a - b).signum() {
          1 | 0 => Add(a - b),
          -1 => Subtract(b - a),
          _ => unreachable!(),
        }
      }
      (Subtract(a), Add(b)) => {
        match (b - a).signum() {
          1 | 0 => Add(a - b),
          -1 => Subtract(b - a),
          _ => unreachable!(),
        }
      }
    }
  }
}

fn evaluate_mutations(nodes: &Vec<&Node>) -> (i32, HashMap<i32, Mutation>) {
  use Node::*;

  let mut mutations = HashMap::new();
  let mut index = 0i32;

  for node in nodes {
    match **node {
      MoveLeft(amount) => {
        index -= amount as i32;
      }
      MoveRight(amount) => {
        index += amount as i32;
      }
      Add(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Add(0));
        value.combine(Mutation::Add(amount as i8));
      }
      Subtract(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Add(0));
        value.combine(Mutation::Subtract(amount as i8));
      }
      Set(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Set(0));
        value.combine(Mutation::Set(amount as i8));
      }
      _ => panic!()
    }
  }
  (index, mutations)
}


#[derive(Debug, PartialEq, Copy, Clone)]
enum ByteCode {
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

#[inline(never)]
fn parse_source(source: Vec<u8>) -> Node {
  let tokens = source.into_iter().filter_map(Token::from_byte).collect();
  Node::from_tokens(tokens)
}

#[inline(never)]
unsafe fn execute_bytecode(instructions: &Vec<ByteCode>) {
  let mut output = Vec::with_capacity(256);
  let mut tape = vec![0u8; 1024];
  let mut tape_head = 0;
  let mut ip = 0;
  while ip < instructions.len() {
    let instruction = instructions.get_unchecked(ip);

    match *instruction {
      ByteCode::MoveLeft(amount) => tape_head -= amount,
      ByteCode::MoveRight(amount) => tape_head += amount,

      ByteCode::Add(amount, offset) => {
        let value = tape.get_unchecked_mut(((tape_head as i32) + offset) as usize);
        *value = value.wrapping_add(amount);
      }
      ByteCode::Subtract(amount, offset) => {
        let value = tape.get_unchecked_mut(((tape_head as i32) + offset) as usize);
        *value = value.wrapping_sub(amount);
      }
      ByteCode::Set(constant, offset) => {
        let value = tape.get_unchecked_mut(((tape_head as i32) + offset) as usize);
        *value = constant;
      }

      ByteCode::LoopStart { end } => {
        if *tape.get_unchecked(tape_head) == 0 {
          ip = end + 1;
          continue
        }
      }
      ByteCode::LoopEnd { start } => {
        if *tape.get_unchecked(tape_head) != 0 {
          ip = start + 1;
          continue
        }
      }
      ByteCode::Output => {
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
  Int8 = 8,
  Int16 = 16,
  Int32 = 32,
  Int64 = 64,
}

impl RegisterSize {
  fn bits(&self) -> u8 {
    *self as u8
  }
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
enum Register {
  RAX,
  EAX,
  AX,
  AH,
  AL,

  RBX,
  EBX,
  BX,
  BH,
  BL,

  RSP,
  ESP,

  RBP,
  EBP,

  RCX,
  RDX,
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

  R8D,
  R9D,
  R10D,
  R11D,
  R12D,
  R13D,
  R14D,
  R15D,

  R8W,
  R9W,
  R10W,
  R11W,
  R12W,
  R13W,
  R14W,
  R15W,

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
      EAX | EBX | ESP | EBP => RegisterSize::Int32,
      AX | BX => RegisterSize::Int16,
      AH | AL | BH | BL => RegisterSize::Int8,
      R8D | R9D | R10D | R11D | R12D | R13D | R14D | R15D => RegisterSize::Int32,
      R8W | R9W | R10W | R11W | R12W | R13W | R14W | R15W => RegisterSize::Int16,
      R8B | R9B | R10B | R11B | R12B | R13B | R14B | R15B => RegisterSize::Int8,
    }
  }

  fn is_64_bit(&self) -> bool {
    self.size() == RegisterSize::Int64
  }

  fn is_16_bit(&self) -> bool {
    self.size() == RegisterSize::Int16
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
      AL  => RegisterNumber::RAX,
      AH  => RegisterNumber::RSP,

      EBX => RegisterNumber::RBX,
      BX  => RegisterNumber::RBX,
      BL  => RegisterNumber::RBX,
      BH  => RegisterNumber::RDI,

      ESP => RegisterNumber::RSP,

      EBP => RegisterNumber::RBP,

      R8D  => RegisterNumber::R8,
      R9D  => RegisterNumber::R9,
      R10D => RegisterNumber::R10,
      R11D => RegisterNumber::R11,
      R12D => RegisterNumber::R12,
      R13D => RegisterNumber::R13,
      R14D => RegisterNumber::R14,
      R15D => RegisterNumber::R15,

      R8W  => RegisterNumber::R8,
      R9W  => RegisterNumber::R9,
      R10W => RegisterNumber::R10,
      R11W => RegisterNumber::R11,
      R12W => RegisterNumber::R12,
      R13W => RegisterNumber::R13,
      R14W => RegisterNumber::R14,
      R15W => RegisterNumber::R15,

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
  Syscall,

  Push(Register),
  Pop(Register),

  MovIR(u64, Register),
  MovIM(RegisterSize, u32, Register, i32),
  MovRR(Register, Register),
  MovRM(Register, Register, i32),
  MovMR(Register, i32, Register),

  IncR(Register),
  DecR(Register),
  IncM(RegisterSize, Register, i32),
  DecM(RegisterSize, Register, i32),

  AddIR(u32, Register),
  SubIR(u32, Register),

  AddIM(RegisterSize, u32, Register, i32),
  SubIM(RegisterSize, u32, Register, i32),

  AddRR(Register, Register),
  SubRR(Register, Register),
  XorRR(Register, Register),

  CmpIR(u32, Register),
  CmpIM(RegisterSize, u32, Register, i32),

  Jmp(i32),
  Jz(i32),
  Jnz(i32),
  Call(i32),
}

impl MachineInstruction {
  fn emit(&self, machine_code: &mut MachineCode) {
    use MachineInstruction::*;

    self.validate();

    match *self {
      MovIR(_, _) if !self.mov_ir_needs_modrm_byte() => {
        let modrm = self.modrm();
        modrm.emit_prefixes_if_needed(machine_code);
        self.emit_opcode(machine_code);
        self.emit_constant_if_needed(machine_code);
      }
      MovIR(..) | MovIM(..) | MovRR(..) | MovRM(..) | MovMR(..) |
      AddRR(..) | SubRR(..) | XorRR(..) | AddIR(..) | SubIR(..) |
      AddIM(..) | SubIM(..) | CmpIM(..) | CmpIR(..) |
      IncR(..) | DecR(..) | IncM(..) | DecM(..)  => {
        let modrm = self.modrm();
        modrm.emit_prefixes_if_needed(machine_code);
        self.emit_opcode(machine_code);
        modrm.emit(machine_code, self.group_opcode());
        self.emit_constant_if_needed(machine_code);
      }
      Push(..) | Pop(..) | Ret | Syscall | Jmp(..) | Jz(..) | Jnz(..) | Call(..) => {
        let modrm = self.modrm();
        modrm.emit_prefixes_if_needed(machine_code);
        self.emit_opcode(machine_code);
        self.emit_constant_if_needed(machine_code);
      }
    }
  }

  fn validate(&self) {
    use MachineInstruction::*;

    match *self {
      MovRR(source, dest) | AddRR(source, dest) | SubRR(source, dest) | XorRR(source, dest) => {
        assert!(source.size() == dest.size());
      }
      MovIM(size, constant, _, _) => {
        match size {
          RegisterSize::Int8 => assert!(constant <= std::u8::MAX as u32),
          RegisterSize::Int16 => assert!(constant <= std::u16::MAX as u32),
          _ => {}
        }
      }
      Push(register) | Pop(register) => {
        assert!(register.size() == RegisterSize::Int16 || register.size() == RegisterSize::Int64);
      }
      _ => {}
    }
  }

  fn emit_opcode(&self, machine_code: &mut MachineCode) {
    use MachineInstruction::*;

    const TWO_BYTE_ESCAPE: u8 = 0x0f;

    let opcode = match *self {
      Ret => 0xc3,
      Syscall => TWO_BYTE_ESCAPE,
      Push(register) => 0x50 | (register.number() & 0x7),
      Pop(register) => 0x58 | (register.number() & 0x7),

      // addb / subb / cmpb
      AddIM(RegisterSize::Int8, _, _, _) | SubIM(RegisterSize::Int8, _, _, _) | CmpIM(RegisterSize::Int8, _, _, _) => 0x80,
      CmpIR(_, register) if register.size() == RegisterSize::Int8 => 0x80,

      // add / sub / cmp with 8-bit immediate.
      AddIR(constant, _) | SubIR(constant, _) | CmpIR(constant, _) |
      AddIM(_, constant, _, _) | SubIM(_, constant, _, _) | CmpIM(_, constant, _, _)
        if constant < 256 => 0x83,

      // Other add / sub / cmp
      AddIR(..) | SubIR(..) | CmpIR(..) | AddIM(..) | SubIM(..) | CmpIM(..) => 0x81,

      // movb      
      MovRR(source, _) | MovRM(source, _, _) if source.size() == RegisterSize::Int8 => 0x88,
      MovMR(_, _, dest) if dest.size() == RegisterSize::Int8 => 0x8a,

      // mov
      MovRR(..) | MovRM(..) => 0x89,
      MovMR(..) => 0x8b,

      MovIR(..) if !self.mov_ir_needs_modrm_byte() => 0xb8,
      MovIM(RegisterSize::Int8, _, _, _) => 0xc6,
      MovIR(..) | MovIM(..) => 0xc7,

      // incb / decb
      IncM(RegisterSize::Int8, _, _) | DecM(RegisterSize::Int8, _, _) => 0xfe,
      IncR(register) | DecR(register) if register.size() == RegisterSize::Int8 => 0xfe,

      // Other inc / dec
      IncR(..) | DecR(..) | IncM(..) | DecM(..) => 0xff,

      AddRR(source, _) if source.size() == RegisterSize::Int8 => 0x00,
      SubRR(source, _) if source.size() == RegisterSize::Int8 => 0x28,
      XorRR(source, _) if source.size() == RegisterSize::Int8 => 0x30,
      AddRR(..) => 0x01,
      SubRR(..) => 0x29,
      XorRR(..) => 0x31,

      Jmp(constant) if constant >= -128 && constant < 128 => 0xeb,
      Jz(constant) if constant >= -128 && constant < 128 => 0x74,
      Jnz(constant) if constant >= -128 && constant < 128 => 0x75,
      Jmp(_) => 0xe9,
      Jz(_) => TWO_BYTE_ESCAPE,
      Jnz(_) => TWO_BYTE_ESCAPE,

      Call(..) => 0xe8,
    };
    machine_code.push(opcode);

    if opcode == TWO_BYTE_ESCAPE {
      machine_code.push(match *self {
        Syscall => 0x05,
        Jz(..) => 0x84,
        Jnz(..) => 0x85,
        _ => unreachable!(),
      });
    }
  }

  fn group_opcode(&self) -> Option<u8> {
    use MachineInstruction::*;

    match *self {
      // Group 1
      AddIR(..) | AddIM(..) => Some(0b000),
      SubIR(..) | SubIM(..) => Some(0b101),
      CmpIR(..) | CmpIM(..) => Some(0b111),

      // Group 3
      IncR(..) | IncM(..) => Some(0b000),
      DecR(..) | DecM(..) => Some(0b001),

      // None
      _ => None,
    }
  }

  fn mov_ir_needs_modrm_byte(&self) -> bool {
    use MachineInstruction::*;

    match *self {
      MovIR(constant, register) => {
        register.size().bits() > 32 && constant <= std::u32::MAX as u64
      }
      _ => panic!()
    }
  }

  fn modrm(&self) -> ModRM {
    use MachineInstruction::*;

    match *self {
      MovRR(source, dest) | AddRR(source, dest) | SubRR(source, dest) | XorRR(source, dest) => {
        ModRM::TwoRegisters(source, dest)
      }
      MovRM(reg1, reg2, offset) | MovMR(reg2, offset, reg1) if offset == 0 => {
        // MovRM and MovMR encode the memory register second.
        ModRM::MemoryTwoRegisters(reg1, reg2)
      }
      MovRM(reg1, reg2, offset) | MovMR(reg2, offset, reg1) if constant_fits_in_i8(offset) => {
        // MovRM and MovMR encode the memory register second.
        ModRM::MemoryTwoRegisters8BitDisplacement(reg1, reg2, offset as i8)
      }
      MovRM(reg1, reg2, offset) | MovMR(reg2, offset, reg1) => {
        // MovRM and MovMR encode the memory register second.
        ModRM::MemoryTwoRegisters32BitDisplacement(reg1, reg2, offset)
      }
      MovIR(_, register) | AddIR(_, register) | SubIR(_, register) | IncR(register) | DecR(register) | CmpIR(_, register) => {
        ModRM::Register(register)
      }
      MovIM(size, _, register, offset) | AddIM(size, _, register, offset) | SubIM(size, _, register, offset) | CmpIM(size, _, register, offset) if offset == 0 => {
        ModRM::Memory(size, register)
      }
      MovIM(size, _, register, offset) | AddIM(size, _, register, offset) | SubIM(size, _, register, offset) | CmpIM(size, _, register, offset) if constant_fits_in_i8(offset) => {
        ModRM::Memory8BitDisplacement(size, register, offset as i8)
      }
      MovIM(size, _, register, offset) | AddIM(size, _, register, offset) | SubIM(size, _, register, offset) | CmpIM(size, _, register, offset) => {
        ModRM::Memory32BitDisplacement(size, register, offset)
      }
      IncM(size, register, offset) | DecM(size, register, offset) if offset == 0 => {
        ModRM::Memory(size, register)
      }
      IncM(size, register, offset) | DecM(size, register, offset) if constant_fits_in_i8(offset) => {
        ModRM::Memory8BitDisplacement(size, register, offset as i8)
      }
      IncM(size, register, offset) | DecM(size, register, offset) => {
        ModRM::Memory32BitDisplacement(size, register, offset)
      }
      Push(register) | Pop(register) => ModRM::Register64(register),
      Syscall | Ret | Jmp(..) | Jz(..) | Jnz(..) | Call(..) => ModRM::None,
    }
  }

  fn emit_constant_if_needed(&self, machine_code: &mut MachineCode) {
    use MachineInstruction::*;

    match *self {
      MovIM(RegisterSize::Int8, constant, _, _) | AddIM(RegisterSize::Int8, constant, _, _) | SubIM(RegisterSize::Int8, constant, _, _) | CmpIM(RegisterSize::Int8, constant, _, _) => {
        machine_code.emit_u8_constant(constant);
      }
      AddIR(constant, _) | SubIR(constant, _) | CmpIR(constant, _) | AddIM(_, constant, _, _) | SubIM(_, constant, _, _) | CmpIM(_, constant, _, _) => {
        if constant < 256 {
          machine_code.emit_u8_constant(constant);
        } else {
          machine_code.emit_u32_constant(constant);
        }
      }
      Jmp(constant) | Jz(constant) | Jnz(constant) => {
        if constant_fits_in_i8(constant) {
          machine_code.emit_i8_constant(constant);
        } else {
          machine_code.emit_i32_constant(constant);
        }
      },
      Call(constant) => {
        machine_code.emit_u32_constant(constant as u32);
      }
      MovIR(constant, _) if constant > std::u32::MAX as u64 => {
        machine_code.emit_u64_constant(constant);
      }
      MovIR(constant, register) if register.size() == RegisterSize::Int32 => {
        machine_code.emit_u32_constant(constant as u32);
      }
      MovIR(constant, register) if register.size() == RegisterSize::Int16 => {
        machine_code.emit_u16_constant(constant as u32);
      }
      MovIR(constant, register) if register.size() == RegisterSize::Int8 => {
        machine_code.emit_u8_constant(constant as u32);
      }
      MovIR(constant, _) => {
        machine_code.emit_u32_constant(constant as u32);
      }
      MovIM(RegisterSize::Int16, constant, _, _) => {
        machine_code.emit_u16_constant(constant);
      }
      MovIM(_, constant, _, _) => {
        machine_code.emit_u32_constant(constant);
      }
      _ => {}
    }
  }
}

fn constant_fits_in_i8(offset: i32) -> bool {
  offset >= std::i8::MIN as i32 && offset <= std::i8::MAX as i32
}

struct MachineCode {
  buffer: Vec<u8>
}

impl MachineCode {
  fn new() -> MachineCode {
    MachineCode { buffer: Vec::new() }
  }

  fn push(&mut self, byte: u8) {
    self.buffer.push(byte);
  }

  fn emit_u8_constant(&mut self, constant: u32) {
    assert!(constant <= std::u8::MAX as u32);
    self.buffer.push(constant as u8);
  }

  fn emit_u16_constant(&mut self, constant: u32) {
    assert!(constant <= std::u16::MAX as u32);
    self.buffer.extend(&[
      ((constant >>  0) & 0xff) as u8,
      ((constant >>  8) & 0xff) as u8,
    ]);
  }

  fn emit_u32_constant(&mut self, constant: u32) {
    self.buffer.extend(&[
      ((constant >>  0) & 0xff) as u8,
      ((constant >>  8) & 0xff) as u8,
      ((constant >> 16) & 0xff) as u8,
      ((constant >> 24) & 0xff) as u8,
    ]);
  }

  fn emit_u64_constant(&mut self, constant: u64) {
    self.buffer.extend(&[
      ((constant >>  0) & 0xff) as u8,
      ((constant >>  8) & 0xff) as u8,
      ((constant >> 16) & 0xff) as u8,
      ((constant >> 24) & 0xff) as u8,
      ((constant >> 32) & 0xff) as u8,
      ((constant >> 40) & 0xff) as u8,
      ((constant >> 48) & 0xff) as u8,
      ((constant >> 56) & 0xff) as u8,
    ]);
  }

  fn emit_i8_constant(&mut self, constant: i32) {
    assert!(constant_fits_in_i8(constant));
    self.emit_u8_constant(constant as u8 as u32);
  }

  fn emit_i32_constant(&mut self, constant: i32) {
    self.emit_u32_constant(constant as u32);
  }
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
enum ModRM {
  None,
  Memory(RegisterSize, Register),
  MemoryTwoRegisters(Register, Register),
  MemoryTwoRegisters8BitDisplacement(Register, Register, i8),
  MemoryTwoRegisters32BitDisplacement(Register, Register, i32),
  Memory8BitDisplacement(RegisterSize, Register, i8),
  Memory32BitDisplacement(RegisterSize, Register, i32),
  Register(Register),
  Register64(Register),
  TwoRegisters(Register, Register),
}

impl ModRM {
  fn encode(&self) -> u8 {
    use ModRM::*;

    match *self {
      Register(register) => 0b11000000 | (register.number() & 0x7),
      TwoRegisters(source, dest) => 0b11000000 | (source.number() & 0x7) << 3 | (dest.number() & 0x7),
      Memory(_, dest) => 0x0 | (dest.number() & 0x7),
      MemoryTwoRegisters(source, dest) => 0x0 | (source.number() & 0x7) << 3 | (dest.number() & 0x7),
      MemoryTwoRegisters8BitDisplacement(source, dest, _) => 0b01000000 | (source.number() & 0x7) << 3 | (dest.number() & 0x7),
      MemoryTwoRegisters32BitDisplacement(source, dest, _) => 0b10000000 | (source.number() & 0x7) << 3 | (dest.number() & 0x7),
      Memory8BitDisplacement(_, dest, _) => 0b01000000 | (dest.number() & 0x7),
      Memory32BitDisplacement(_, dest, _) => 0b10000000 | (dest.number() & 0x7),

      Register64(..) => {
        // Register64 is currently only used to emit prefixes for push / pop, therefore it
        // is never asked to emit a ModRM byte.
        unreachable!()
      }

      None => unreachable!()
    }
  }

  fn emit(&self, machine_code: &mut MachineCode, group_opcode: Option<u8>) {
    machine_code.push(self.encode() | (group_opcode.unwrap_or(0) << 3));
    self.emit_offset_if_needed(machine_code);
  }

  fn emit_offset_if_needed(&self, machine_code: &mut MachineCode) {
    use ModRM::*;

    match *self {
      Memory8BitDisplacement(_, _, offset) | MemoryTwoRegisters8BitDisplacement(_, _, offset) => {
        machine_code.emit_i8_constant(offset as i32)
      }
      Memory32BitDisplacement(_, _, offset) | MemoryTwoRegisters32BitDisplacement(_, _, offset) => {
        machine_code.emit_i32_constant(offset)
      }
      _ => {}
    }
  }

  fn needs_rex(&self) -> bool {
    use ModRM::*;

    match *self {
      MemoryTwoRegisters(source, _) | MemoryTwoRegisters8BitDisplacement(source, _, _) |
      MemoryTwoRegisters32BitDisplacement(source, _, _) => {
        source.is_64_bit() || source.is_extended_register()
      }
      Register64(..) => self.has_extended_register(),
      None => false,
      _ => self.is_64_bit() || self.has_extended_register()
    }
  }

  fn needs_operand_size_override(&self) -> bool {
    use ModRM::*;

    match *self {
      TwoRegisters(source, dest) => {
        assert!(source.is_16_bit() == dest.is_16_bit());
        source.is_16_bit()
      }
      MemoryTwoRegisters(source, dest) | MemoryTwoRegisters8BitDisplacement(source, dest, _) | MemoryTwoRegisters32BitDisplacement(source, dest, _) => {
        // FIXME: Check only source?
        source.is_16_bit() || dest.is_16_bit()
      }
      Register(register) | Register64(register) => {
        register.is_16_bit()
      }
      Memory(size, _) | Memory8BitDisplacement(size, _, _) | Memory32BitDisplacement(size, _, _) => {
        size == RegisterSize::Int16
      }
      None => false,
    }
  }

  fn needs_address_size_override(&self) -> bool {
    use ModRM::*;

    match *self {
      Memory(_, register) | Memory8BitDisplacement(_, register, _) | Memory32BitDisplacement(_, register, _) => {
        register.size() != RegisterSize::Int64
      }
      _ => false,
    }
  }

  fn emit_prefixes_if_needed(&self, machine_code: &mut MachineCode) {
    self.emit_operand_size_override_if_needed(machine_code);
    self.emit_address_size_override_if_needed(machine_code);
    self.emit_rex_if_needed(machine_code);
  }

  fn emit_rex_if_needed(&self, machine_code: &mut MachineCode) {
    use ModRM::*;

    if !self.needs_rex() {
      return
    }

    let rex_marker = 0b01000000;
    match *self {
      TwoRegisters(source, dest) | MemoryTwoRegisters(source, dest) | MemoryTwoRegisters8BitDisplacement(source, dest, _) |
      MemoryTwoRegisters32BitDisplacement(source, dest, _) => {
        let mut rex = rex_marker;
        rex |= (source.is_64_bit() as u8) << 3;
        rex |= (source.is_extended_register() as u8) << 2;
        rex |= dest.is_extended_register() as u8;
        machine_code.push(rex);
      }
      Register(..) | Register64(..) | Memory(..) | Memory8BitDisplacement(..) | Memory32BitDisplacement(..) => {
        let mut rex = rex_marker;
        rex |= (self.is_64_bit() as u8) << 3;
        rex |= self.has_extended_register() as u8;
        machine_code.push(rex);
      }
      None => unreachable!()
    }
  }

  fn emit_operand_size_override_if_needed(&self, machine_code: &mut MachineCode) {
    if !self.needs_operand_size_override() {
      return
    }

    machine_code.push(0x66);
  }

  fn emit_address_size_override_if_needed(&self, machine_code: &mut MachineCode) {
    if !self.needs_address_size_override() {
      return
    }

    machine_code.push(0x67);
  }

  fn is_64_bit(&self) -> bool {
    use ModRM::*;

    match *self {
      TwoRegisters(source, dest) => {
        assert!(source.is_64_bit() == dest.is_64_bit());
        source.is_64_bit()
      }
      MemoryTwoRegisters(source, dest) | MemoryTwoRegisters8BitDisplacement(source, dest, _) |
      MemoryTwoRegisters32BitDisplacement(source, dest, _) => {
        source.is_64_bit() || dest.is_64_bit()
      }

      Register(register) => register.is_64_bit(),

      Memory(size, _) | Memory8BitDisplacement(size, _, _) | Memory32BitDisplacement(size, _, _) => {
        size == RegisterSize::Int64
      }
      Register64(..) | None => false,
    }
  }

  fn has_extended_register(&self) -> bool {
    use ModRM::*;

    match *self {
      TwoRegisters(source, dest) | MemoryTwoRegisters(source, dest) | MemoryTwoRegisters8BitDisplacement(source, dest, _) |
      MemoryTwoRegisters32BitDisplacement(source, dest, _) => {
        source.is_extended_register() || dest.is_extended_register()
      }
      Register(register) | Register64(register) | Memory(_, register) |
      Memory8BitDisplacement(_, register, _) | Memory32BitDisplacement(_, register, _) => {
        register.is_extended_register()
      }
      None => false,
    }
  }
}

fn lower(instructions: &[MachineInstruction]) -> Vec<u8> {
  let mut machine_code = MachineCode::new();
  for instruction in instructions {
    instruction.emit(&mut machine_code);
  }
  machine_code.buffer
}

#[allow(dead_code)]
fn show_representation(instructions: &[MachineInstruction]) {
  println!("{:?}", instructions);
  println!("  {:}", lower(instructions).iter().map(|b| format!("{:02x}", b)).collect::<Vec<String>>().connect(" "));
}

#[cfg(test)]
 mod test {
  use super::{lower, RegisterSize};
  use super::MachineInstruction::*;
  use super::Register::*;

  #[test]
  fn test_ret() {
    assert_eq!(lower(&[ Ret ]), vec![ 0xc3 ]);
  }

  #[test]
  fn test_syscall() {
    assert_eq!(lower(&[ Syscall ]), vec![ 0x0f, 0x05 ]);
  }

    #[test]
  fn test_push() {
    assert_eq!(lower(&[ Push(RAX ) ]), vec![ 0x50 ]);
    assert_eq!(lower(&[ Push(RCX ) ]), vec![ 0x51 ]);
    assert_eq!(lower(&[ Push(RDX ) ]), vec![ 0x52 ]);
    assert_eq!(lower(&[ Push(RBX ) ]), vec![ 0x53 ]);
    assert_eq!(lower(&[ Push(RSP ) ]), vec![ 0x54 ]);
    assert_eq!(lower(&[ Push(RBP ) ]), vec![ 0x55 ]);
    assert_eq!(lower(&[ Push(RSI ) ]), vec![ 0x56 ]);
    assert_eq!(lower(&[ Push(RDI ) ]), vec![ 0x57 ]);
    assert_eq!(lower(&[ Push(R8  ) ]), vec![ 0b01000001, 0x50 ]);
    assert_eq!(lower(&[ Push(R9  ) ]), vec![ 0b01000001, 0x51 ]);
    assert_eq!(lower(&[ Push(R10 ) ]), vec![ 0b01000001, 0x52 ]);
    assert_eq!(lower(&[ Push(R11 ) ]), vec![ 0b01000001, 0x53 ]);
    assert_eq!(lower(&[ Push(R12 ) ]), vec![ 0b01000001, 0x54 ]);
    assert_eq!(lower(&[ Push(R13 ) ]), vec![ 0b01000001, 0x55 ]);
    assert_eq!(lower(&[ Push(R14 ) ]), vec![ 0b01000001, 0x56 ]);
    assert_eq!(lower(&[ Push(R15 ) ]), vec![ 0b01000001, 0x57 ]);

    assert_eq!(lower(&[ Push(AX ) ]), vec![              0x66, 0x50 ]);
    assert_eq!(lower(&[ Push(R8W ) ]), vec![ 0x66, 0b01000001, 0x50 ]);
  }

  #[test]
  fn test_pop() {
    assert_eq!(lower(&[ Pop(RAX ) ]), vec![ 0x58 ]);
    assert_eq!(lower(&[ Pop(RCX ) ]), vec![ 0x59 ]);
    assert_eq!(lower(&[ Pop(RDX ) ]), vec![ 0x5a ]);
    assert_eq!(lower(&[ Pop(RBX ) ]), vec![ 0x5b ]);
    assert_eq!(lower(&[ Pop(RSP ) ]), vec![ 0x5c ]);
    assert_eq!(lower(&[ Pop(RBP ) ]), vec![ 0x5d ]);
    assert_eq!(lower(&[ Pop(RSI ) ]), vec![ 0x5e ]);
    assert_eq!(lower(&[ Pop(RDI ) ]), vec![ 0x5f ]);
    assert_eq!(lower(&[ Pop(R8  ) ]), vec![ 0b01000001, 0x58 ]);
    assert_eq!(lower(&[ Pop(R9  ) ]), vec![ 0b01000001, 0x59 ]);
    assert_eq!(lower(&[ Pop(R10 ) ]), vec![ 0b01000001, 0x5a ]);
    assert_eq!(lower(&[ Pop(R11 ) ]), vec![ 0b01000001, 0x5b ]);
    assert_eq!(lower(&[ Pop(R12 ) ]), vec![ 0b01000001, 0x5c ]);
    assert_eq!(lower(&[ Pop(R13 ) ]), vec![ 0b01000001, 0x5d ]);
    assert_eq!(lower(&[ Pop(R14 ) ]), vec![ 0b01000001, 0x5e ]);
    assert_eq!(lower(&[ Pop(R15 ) ]), vec![ 0b01000001, 0x5f ]);

    assert_eq!(lower(&[ Pop(AX ) ]), vec![              0x66, 0x58 ]);
    assert_eq!(lower(&[ Pop(R8W ) ]), vec![ 0x66, 0b01000001, 0x58 ]);
  }

  #[test]
  fn test_mov_ir() {
    assert_eq!(lower(&[ MovIR(1, RAX) ]), vec![ 0b1001000, 0xc7, 0b11000000, 0x01, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovIR(1, RBX) ]), vec![ 0b1001000, 0xc7, 0b11000011, 0x01, 0x00, 0x00, 0x00 ]);

    assert_eq!(lower(&[ MovIR(0x123456789abcde, RBX) ]), vec![ 0b1001000, 0xb8, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x00 ]);

    assert_eq!(lower(& [ MovIR(1, EAX) ]), vec![       0xb8, 0x01, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(& [ MovIR(1, AX ) ]), vec![ 0x66, 0xb8, 0x01, 0x00 ]);
  }

  #[test]
  #[should_panic]
  fn test_invalid_mov_ir() {
    // FIXME: Is this only testing that at least one of the statements panics?
    lower(& [ MovIR(0x123456789abcde, EAX) ]);
    lower(& [ MovIR(0x1234, AX) ]);
  }

  #[test]
  fn test_mov_im() {
    assert_eq!(lower(&[ MovIM(RegisterSize::Int64, 1, RAX, 0) ]), vec![ 0b1001000, 0xc7, 0b00000000, 0x01, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovIM(RegisterSize::Int32, 1, RAX, 0) ]), vec![ 0xc7, 0b00000000, 0x01, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovIM(RegisterSize::Int16, 1, RAX, 0) ]), vec![ 0x66, 0xc7, 0b00000000, 0x01, 0x00 ]);
    assert_eq!(lower(&[ MovIM(RegisterSize::Int8,  1, RAX, 0) ]), vec![ 0xc6, 0b00000000, 0x01 ]);

    assert_eq!(lower(&[ MovIM(RegisterSize::Int32, 1, RAX, 1) ]), vec![ 0xc7, 0b01000000, 0x01, 0x01, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovIM(RegisterSize::Int32, 1, RAX, 1024) ]), vec![ 0xc7, 0b10000000, 0x00, 0x04, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 ]);
  }

  #[test]
  #[should_panic]
  fn test_invalid_mov_im() {
    // FIXME: Is this only testing that at least one of the statements panics?
    lower(&[ MovIM(RegisterSize::Int16, 0x12345678, RAX, 0) ]);
    lower(&[ MovIM(RegisterSize::Int8,  0x1234, RAX, 0) ]);
  }

  #[test]
  fn test_mov_rr() {
    assert_eq!(lower(&[ MovRR(RAX, RBX) ]), vec![ 0b1001000, 0x89, 0b11000011 ]);
    assert_eq!(lower(&[ MovRR(RBX, RAX) ]), vec![ 0b1001000, 0x89, 0b11011000 ]);
    assert_eq!(lower(&[ MovRR(RAX, R8 ) ]), vec![ 0b1001001, 0x89, 0b11000000 ]);

    assert_eq!(lower(&[ MovRR(EAX, EBX) ]), vec![            0x89, 0b11000011 ]);
    assert_eq!(lower(&[ MovRR(EAX, R8D) ]), vec![ 0b1000001, 0x89, 0b11000000 ]);

    assert_eq!(lower(&[ MovRR(AX, BX ) ]),  vec![            0x66, 0x89, 0b11000011 ]);
    assert_eq!(lower(&[ MovRR(AX, R8W) ]),  vec![ 0x66, 0b1000001, 0x89, 0b11000000 ]);

    assert_eq!(lower(&[ MovRR(AH, BH ) ]),  vec![            0x88, 0b11100111 ]);
    assert_eq!(lower(&[ MovRR(AH, R8B) ]),  vec![ 0b1000001, 0x88, 0b11100000 ]);
  }

  #[test]
  fn test_mov_rm() {
    assert_eq!(lower(&[ MovRM(RAX, RBX, 0) ]), vec![ 0b1001000, 0x89, 0b00000011 ]);
    assert_eq!(lower(&[ MovRM(R8,  RBX, 0) ]), vec![ 0b1001100, 0x89, 0b00000011 ]);
    assert_eq!(lower(&[ MovRM(RBX, R8,  0) ]), vec![ 0b1001001, 0x89, 0b00011000 ]);

    assert_eq!(lower(&[ MovRM(EAX, RBX, 0) ]), vec![            0x89, 0b00000011 ]);
    assert_eq!(lower(&[ MovRM(R8D, RBX, 0) ]), vec![ 0b1000100, 0x89, 0b00000011 ]);

    assert_eq!(lower(&[ MovRM(AX , RBX, 0) ]), vec![            0x66, 0x89, 0b00000011 ]);
    assert_eq!(lower(&[ MovRM(R8W, RBX, 0) ]), vec![ 0x66, 0b1000100, 0x89, 0b00000011 ]);

    assert_eq!(lower(&[ MovRM(AH , RBX, 0) ]), vec![            0x88, 0b00100011 ]);
    assert_eq!(lower(&[ MovRM(AL , RBX, 0) ]), vec![            0x88, 0b00000011 ]);
    assert_eq!(lower(&[ MovRM(R8B, RBX, 0) ]), vec![ 0b1000100, 0x88, 0b00000011 ]);


    assert_eq!(lower(&[ MovRM(RAX, RBX, 16) ]), vec![ 0b1001000, 0x89, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovRM(R8,  RBX, 16) ]), vec![ 0b1001100, 0x89, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovRM(EAX, RBX, 16) ]), vec![            0x89, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovRM(R8D, RBX, 16) ]), vec![ 0b1000100, 0x89, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovRM(AX , RBX, 16) ]), vec![            0x66, 0x89, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovRM(R8W, RBX, 16) ]), vec![ 0x66, 0b1000100, 0x89, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovRM(AH , RBX, 16) ]), vec![            0x88, 0b01100011, 0x10 ]);
    assert_eq!(lower(&[ MovRM(AL , RBX, 16) ]), vec![            0x88, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovRM(R8B, RBX, 16) ]), vec![ 0b1000100, 0x88, 0b01000011, 0x10 ]);


    assert_eq!(lower(&[ MovRM(RAX, RBX, 1024) ]), vec![ 0b1001000, 0x89, 0b10000011, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovRM(R8,  RBX, 1024) ]), vec![ 0b1001100, 0x89, 0b10000011, 0x00, 0x04, 0x00, 0x00 ]);
  }

  #[test]
  fn test_mov_mr() {
    assert_eq!(lower(&[ MovMR(RBX, 0, RAX) ]), vec![ 0b1001000, 0x8b, 0b00000011 ]);
    assert_eq!(lower(&[ MovMR(RBX, 0, R8 ) ]), vec![ 0b1001100, 0x8b, 0b00000011 ]);
    assert_eq!(lower(&[ MovMR(R8 , 0, RBX) ]), vec![ 0b1001001, 0x8b, 0b00011000 ]);

    assert_eq!(lower(&[ MovMR(RBX, 0, EAX) ]), vec![            0x8b, 0b00000011 ]);
    assert_eq!(lower(&[ MovMR(RBX, 0, R8D) ]), vec![ 0b1000100, 0x8b, 0b00000011 ]);

    assert_eq!(lower(&[ MovMR(RBX, 0, AX ) ]), vec![            0x66, 0x8b, 0b00000011 ]);
    assert_eq!(lower(&[ MovMR(RBX, 0, R8W) ]), vec![ 0x66, 0b1000100, 0x8b, 0b00000011 ]);

    assert_eq!(lower(&[ MovMR(RBX, 0, AH ) ]), vec![            0x8a, 0b00100011 ]);
    assert_eq!(lower(&[ MovMR(RBX, 0, AL ) ]), vec![            0x8a, 0b00000011 ]);
    assert_eq!(lower(&[ MovMR(RBX, 0, R8B) ]), vec![ 0b1000100, 0x8a, 0b00000011 ]);


    assert_eq!(lower(&[ MovMR(RBX, 16, RAX) ]), vec![ 0b1001000, 0x8b, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovMR(RBX, 16, R8 ) ]), vec![ 0b1001100, 0x8b, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovMR(RBX, 16, EAX) ]), vec![            0x8b, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovMR(RBX, 16, R8D) ]), vec![ 0b1000100, 0x8b, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovMR(RBX, 16, AX ) ]), vec![            0x66, 0x8b, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovMR(RBX, 16, R8W) ]), vec![ 0x66, 0b1000100, 0x8b, 0b01000011, 0x10 ]);

    assert_eq!(lower(&[ MovMR(RBX, 16, AH ) ]), vec![            0x8a, 0b01100011, 0x10 ]);
    assert_eq!(lower(&[ MovMR(RBX, 16, AL ) ]), vec![            0x8a, 0b01000011, 0x10 ]);
    assert_eq!(lower(&[ MovMR(RBX, 16, R8B) ]), vec![ 0b1000100, 0x8a, 0b01000011, 0x10 ]);


    assert_eq!(lower(&[ MovMR(RBX, 1024, RAX) ]), vec![ 0b1001000, 0x8b, 0b10000011, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ MovMR(RBX, 1024, R8 ) ]), vec![ 0b1001100, 0x8b, 0b10000011, 0x00, 0x04, 0x00, 0x00 ]);
  }

  #[test]
  fn test_inc_r() {
    assert_eq!(lower(&[ IncR(RAX) ]), vec![ 0b1001000, 0xff, 0b11000000 ]);
    assert_eq!(lower(&[ IncR(R8 ) ]), vec![ 0b1001001, 0xff, 0b11000000 ]);

    assert_eq!(lower(&[ IncR(EAX) ]), vec![            0xff, 0b11000000 ]);
    assert_eq!(lower(&[ IncR(R8D) ]), vec![ 0b1000001, 0xff, 0b11000000 ]);

    assert_eq!(lower(&[ IncR(AX ) ]), vec![            0x66, 0xff, 0b11000000 ]);
    assert_eq!(lower(&[ IncR(R8W) ]), vec![ 0x66, 0b1000001, 0xff, 0b11000000 ]);

    assert_eq!(lower(&[ IncR(AL ) ]), vec![            0xfe, 0b11000000 ]);
    assert_eq!(lower(&[ IncR(AH ) ]), vec![            0xfe, 0b11000100 ]);
    assert_eq!(lower(&[ IncR(R8B) ]), vec![ 0b1000001, 0xfe, 0b11000000 ]);
  }

  #[test]
  fn test_dec_r() {
    assert_eq!(lower(&[ DecR(RAX) ]), vec![ 0b1001000, 0xff, 0b11001000 ]);
    assert_eq!(lower(&[ DecR(R8 ) ]), vec![ 0b1001001, 0xff, 0b11001000 ]);

    assert_eq!(lower(&[ DecR(EAX) ]), vec![            0xff, 0b11001000 ]);
    assert_eq!(lower(&[ DecR(R8D) ]), vec![ 0b1000001, 0xff, 0b11001000 ]);

    assert_eq!(lower(&[ DecR(AX ) ]), vec![            0x66, 0xff, 0b11001000 ]);
    assert_eq!(lower(&[ DecR(R8W) ]), vec![ 0x66, 0b1000001, 0xff, 0b11001000 ]);

    assert_eq!(lower(&[ DecR(AL ) ]), vec![            0xfe, 0b11001000 ]);
    assert_eq!(lower(&[ DecR(AH ) ]), vec![            0xfe, 0b11001100 ]);
    assert_eq!(lower(&[ DecR(R8B) ]), vec![ 0b1000001, 0xfe, 0b11001000 ]);
  }

  #[test]
  fn test_inc_m() {
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, RAX,  0) ]), vec![       0b1001000, 0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, R8,   0) ]), vec![       0b1001001, 0xff, 0b00000000 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int32, RAX,  0) ]), vec![                  0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int16, RAX,  0) ]), vec![            0x66, 0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int8,  RAX,  0) ]), vec![                  0xfe, 0b00000000 ]);
           
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, EAX,  0) ]), vec![ 0x67, 0b1001000, 0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, R8D,  0) ]), vec![ 0x67, 0b1001001, 0xff, 0b00000000 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int32, EAX,  0) ]), vec![            0x67, 0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int16, EAX,  0) ]), vec![      0x66, 0x67, 0xff, 0b00000000 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int8,  EAX,  0) ]), vec![            0x67, 0xfe, 0b00000000 ]);


    assert_eq!(lower(&[ IncM(RegisterSize::Int64, RAX, 16) ]), vec![       0b1001000, 0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, R8,  16) ]), vec![       0b1001001, 0xff, 0b01000000, 0x10 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int32, RAX, 16) ]), vec![                  0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int16, RAX, 16) ]), vec![            0x66, 0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int8,  RAX, 16) ]), vec![                  0xfe, 0b01000000, 0x10 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int64, EAX, 16) ]), vec![ 0x67, 0b1001000, 0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, R8D, 16) ]), vec![ 0x67, 0b1001001, 0xff, 0b01000000, 0x10 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int32, EAX, 16) ]), vec![            0x67, 0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int16, EAX, 16) ]), vec![      0x66, 0x67, 0xff, 0b01000000, 0x10 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int8,  EAX, 16) ]), vec![            0x67, 0xfe, 0b01000000, 0x10 ]);

    assert_eq!(lower(&[ IncM(RegisterSize::Int64, EAX, 1024) ]), vec![ 0x67, 0b1001000, 0xff, 0b10000000, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ IncM(RegisterSize::Int64, R8D, 1024) ]), vec![ 0x67, 0b1001001, 0xff, 0b10000000, 0x00, 0x04, 0x00, 0x00 ]);
  }

  #[test]
  fn test_dec_m() {
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, RAX, 0) ]), vec![       0b1001000, 0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, R8,  0) ]), vec![       0b1001001, 0xff, 0b00001000 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int32, RAX, 0) ]), vec![                  0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int16, RAX, 0) ]), vec![            0x66, 0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int8,  RAX, 0) ]), vec![                  0xfe, 0b00001000 ]);
           
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, EAX, 0) ]), vec![ 0x67, 0b1001000, 0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, R8D, 0) ]), vec![ 0x67, 0b1001001, 0xff, 0b00001000 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int32, EAX, 0) ]), vec![            0x67, 0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int16, EAX, 0) ]), vec![      0x66, 0x67, 0xff, 0b00001000 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int8,  EAX, 0) ]), vec![            0x67, 0xfe, 0b00001000 ]);


    assert_eq!(lower(&[ DecM(RegisterSize::Int64, RAX, 16) ]), vec![       0b1001000, 0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, R8,  16) ]), vec![       0b1001001, 0xff, 0b01001000, 0x10 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int32, RAX, 16) ]), vec![                  0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int16, RAX, 16) ]), vec![            0x66, 0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int8,  RAX, 16) ]), vec![                  0xfe, 0b01001000, 0x10 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int64, EAX, 16) ]), vec![ 0x67, 0b1001000, 0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, R8D, 16) ]), vec![ 0x67, 0b1001001, 0xff, 0b01001000, 0x10 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int32, EAX, 16) ]), vec![            0x67, 0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int16, EAX, 16) ]), vec![      0x66, 0x67, 0xff, 0b01001000, 0x10 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int8,  EAX, 16) ]), vec![            0x67, 0xfe, 0b01001000, 0x10 ]);

    assert_eq!(lower(&[ DecM(RegisterSize::Int64, EAX, 1024) ]), vec![ 0x67, 0b1001000, 0xff, 0b10001000, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ DecM(RegisterSize::Int64, R8D, 1024) ]), vec![ 0x67, 0b1001001, 0xff, 0b10001000, 0x00, 0x04, 0x00, 0x00 ]);
  }

  #[test]
  fn test_add_ir() {
    assert_eq!(lower(&[ AddIR(1, RAX) ]), vec![ 0b1001000, 0x83, 0b11000000, 0x01 ]);
    assert_eq!(lower(&[ AddIR(1, RBX) ]), vec![ 0b1001000, 0x83, 0b11000011, 0x01 ]);

    // FIXME: AddIR(0x1234, RAX) could use opcode 0x05 to avoid the ModR/M byte.
    assert_eq!(lower(&[ AddIR(0x1234, RAX) ]), vec![ 0b1001000, 0x81, 0b11000000, 0x34, 0x12, 0x00, 0x00 ]);
    assert_eq!(lower(&[ AddIR(0x1234, RBX) ]), vec![ 0b1001000, 0x81, 0b11000011, 0x34, 0x12, 0x00, 0x00 ]);

    assert_eq!(lower(&[ AddIR(0x12345678, RAX) ]), vec![ 0b1001000, 0x81, 0b11000000, 0x78, 0x56, 0x34, 0x12 ]);
    assert_eq!(lower(&[ AddIR(0x12345678, RBX) ]), vec![ 0b1001000, 0x81, 0b11000011, 0x78, 0x56, 0x34, 0x12 ]);

    assert_eq!(lower(&[ AddIR(1,      EAX) ]), vec![ 0x83, 0b11000000, 0x01 ]);
    // FIXME: AddIR(0x1234, EAX) could use opcode 0x05 to avoid the ModR/M byte.
    assert_eq!(lower(&[ AddIR(0x1234, EAX) ]), vec![ 0x81, 0b11000000, 0x34, 0x12, 0x00, 0x00 ]);
  }

  #[test]
  fn test_sub_ir() {
    assert_eq!(lower(&[ SubIR(1, RAX) ]), vec![ 0b1001000, 0x83, 0b11101000, 0x01 ]);
    assert_eq!(lower(&[ SubIR(1, RBX) ]), vec![ 0b1001000, 0x83, 0b11101011, 0x01 ]);

    // FIXME: SubIR(0x1234, RAX) could use opcode 0x2d to avoid the ModR/M byte.
    assert_eq!(lower(&[ SubIR(0x1234, RAX) ]), vec![ 0b1001000, 0x81, 0b11101000, 0x34, 0x12, 0x00, 0x00 ]);
    assert_eq!(lower(&[ SubIR(0x1234, RBX) ]), vec![ 0b1001000, 0x81, 0b11101011, 0x34, 0x12, 0x00, 0x00 ]);

    assert_eq!(lower(&[ SubIR(0x12345678, RAX) ]), vec![ 0b1001000, 0x81, 0b11101000, 0x78, 0x56, 0x34, 0x12 ]);
    assert_eq!(lower(&[ SubIR(0x12345678, RBX) ]), vec![ 0b1001000, 0x81, 0b11101011, 0x78, 0x56, 0x34, 0x12 ]);

    assert_eq!(lower(&[ SubIR(1,      EAX) ]), vec![ 0x83, 0b11101000, 0x01 ]);
    // FIXME: SubIR(0x1234, EAX) could use opcode 0x2d to avoid the ModR/M byte.
    assert_eq!(lower(&[ SubIR(0x1234, EAX) ]), vec![ 0x81, 0b11101000, 0x34, 0x12, 0x00, 0x00 ]);
  }

  #[test]
  fn test_add_im() {
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, RAX, 0) ]), vec![ 0b1001000, 0x83, 0b0000000, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, R8,  0) ]), vec![ 0b1001001, 0x83, 0b0000000, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, RAX, 0) ]), vec![ 0b1001000, 0x81, 0b0000000, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, R8,  0) ]), vec![ 0b1001001, 0x81, 0b0000000, 0x00, 0x04, 0x00, 0x00 ]);


    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, RAX, 16) ]), vec![ 0b1001000, 0x83, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, R8,  16) ]), vec![ 0b1001001, 0x83, 0b01000000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, RAX, 16) ]), vec![ 0b1001000, 0x81, 0b01000000, 0x10, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, R8,  16) ]), vec![ 0b1001001, 0x81, 0b01000000, 0x10, 0x00, 0x04, 0x00, 0x00 ]);


    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, RAX, 1024) ]), vec![ 0b1001000, 0x83, 0b10000000, 0x00, 0x04, 0x00, 0x00, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, R8,  1024) ]), vec![ 0b1001001, 0x83, 0b10000000, 0x00, 0x04, 0x00, 0x00, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, RAX, 1024) ]), vec![ 0b1001000, 0x81, 0b10000000, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1024, R8,  1024) ]), vec![ 0b1001001, 0x81, 0b10000000, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00 ]);



    assert_eq!(lower(&[ AddIM(RegisterSize::Int32, 1, RAX, 16) ]), vec![            0x83, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int32, 1, R8,  16) ]), vec![ 0b1000001, 0x83, 0b01000000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int16, 1, RAX, 16) ]), vec![            0x66, 0x83, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int16, 1, R8,  16) ]), vec![ 0x66, 0b1000001, 0x83, 0b01000000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int8, 1, RAX, 16) ]), vec![            0x80, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int8, 1, R8,  16) ]), vec![ 0b1000001, 0x80, 0b01000000, 0x10, 0x01 ]);


    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, EAX, 16) ]), vec![ 0x67, 0b1001000, 0x83, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int64, 1, R8D, 16) ]), vec![ 0x67, 0b1001001, 0x83, 0b01000000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ AddIM(RegisterSize::Int16, 1, EAX, 16) ]), vec![            0x66, 0x67, 0x83, 0b01000000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ AddIM(RegisterSize::Int16, 1, R8D, 16) ]), vec![ 0x66, 0x67, 0b1000001, 0x83, 0b01000000, 0x10, 0x01 ]);

    // FIXME: Is AddIM(RegisterSize::Int64, 1, AX, 16) something we should be able to assemble?
    // If not, we should test that we don't allow it.
  }

  #[test]
  fn test_sub_im() {
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, RAX, 0) ]), vec![ 0b1001000, 0x83, 0b0101000, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, R8,  0) ]), vec![ 0b1001001, 0x83, 0b0101000, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, RAX, 0) ]), vec![ 0b1001000, 0x81, 0b0101000, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, R8,  0) ]), vec![ 0b1001001, 0x81, 0b0101000, 0x00, 0x04, 0x00, 0x00 ]);


    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, RAX, 16) ]), vec![ 0b1001000, 0x83, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, R8,  16) ]), vec![ 0b1001001, 0x83, 0b01101000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, RAX, 16) ]), vec![ 0b1001000, 0x81, 0b01101000, 0x10, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, R8,  16) ]), vec![ 0b1001001, 0x81, 0b01101000, 0x10, 0x00, 0x04, 0x00, 0x00 ]);


    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, RAX, 1024) ]), vec![ 0b1001000, 0x83, 0b10101000, 0x00, 0x04, 0x00, 0x00, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, R8,  1024) ]), vec![ 0b1001001, 0x83, 0b10101000, 0x00, 0x04, 0x00, 0x00, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, RAX, 1024) ]), vec![ 0b1001000, 0x81, 0b10101000, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1024, R8,  1024) ]), vec![ 0b1001001, 0x81, 0b10101000, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00 ]);



    assert_eq!(lower(&[ SubIM(RegisterSize::Int32, 1, RAX, 16) ]), vec![            0x83, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int32, 1, R8,  16) ]), vec![ 0b1000001, 0x83, 0b01101000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int16, 1, RAX, 16) ]), vec![            0x66, 0x83, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int16, 1, R8,  16) ]), vec![ 0x66, 0b1000001, 0x83, 0b01101000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int8, 1, RAX, 16) ]), vec![            0x80, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int8, 1, R8,  16) ]), vec![ 0b1000001, 0x80, 0b01101000, 0x10, 0x01 ]);


    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, EAX, 16) ]), vec![ 0x67, 0b1001000, 0x83, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int64, 1, R8D, 16) ]), vec![ 0x67, 0b1001001, 0x83, 0b01101000, 0x10, 0x01 ]);

    assert_eq!(lower(&[ SubIM(RegisterSize::Int16, 1, EAX, 16) ]), vec![            0x66, 0x67, 0x83, 0b01101000, 0x10, 0x01 ]);
    assert_eq!(lower(&[ SubIM(RegisterSize::Int16, 1, R8D, 16) ]), vec![ 0x66, 0x67, 0b1000001, 0x83, 0b01101000, 0x10, 0x01 ]);

    // FIXME: Is SubIM(RegisterSize::Int64, 1, AX, 16) something we should be able to assemble?
    // If not, we should test that we don't allow it.
  }

  #[test]
  fn test_add_rr() {
    assert_eq!(lower(&[ AddRR(RAX, RBX) ]), vec![ 0b1001000, 0x01, 0b11000011 ]);
    assert_eq!(lower(&[ AddRR(R8,  RBX) ]), vec![ 0b1001100, 0x01, 0b11000011 ]);

    assert_eq!(lower(&[ AddRR(EAX, EBX) ]), vec![            0x01, 0b11000011 ]);
    assert_eq!(lower(&[ AddRR(R8D, EBX) ]), vec![ 0b1000100, 0x01, 0b11000011 ]);

    assert_eq!(lower(&[ AddRR( AX,  BX) ]), vec![            0x66, 0x01, 0b11000011 ]);
    assert_eq!(lower(&[ AddRR(R8W,  BX) ]), vec![ 0x66, 0b1000100, 0x01, 0b11000011 ]);

    assert_eq!(lower(&[ AddRR( AH,  BH) ]), vec![            0x00, 0b11100111 ]);
    assert_eq!(lower(&[ AddRR( AL,  BH) ]), vec![            0x00, 0b11000111 ]);
    assert_eq!(lower(&[ AddRR(R8B,  BH) ]), vec![ 0b1000100, 0x00, 0b11000111 ]);
  }

  #[test]
  #[should_panic]
  fn test_invalid_add_rr() {
    // FIXME: Is this only testing that at least one of the statements panics?
    lower(&[ AddRR(RAX, EAX) ]);
    lower(&[ AddRR(EAX, AH) ]);
  }

  #[test]
  fn test_sub_rr() {
    assert_eq!(lower(&[ SubRR(RAX, RBX) ]), vec![ 0b1001000, 0x29, 0b11000011 ]);
    assert_eq!(lower(&[ SubRR(R8,  RBX) ]), vec![ 0b1001100, 0x29, 0b11000011 ]);

    assert_eq!(lower(&[ SubRR(EAX, EBX) ]), vec![            0x29, 0b11000011 ]);
    assert_eq!(lower(&[ SubRR(R8D, EBX) ]), vec![ 0b1000100, 0x29, 0b11000011 ]);

    assert_eq!(lower(&[ SubRR( AX,  BX) ]), vec![            0x66, 0x29, 0b11000011 ]);
    assert_eq!(lower(&[ SubRR(R8W,  BX) ]), vec![ 0x66, 0b1000100, 0x29, 0b11000011 ]);

    assert_eq!(lower(&[ SubRR( AH,  BH) ]), vec![            0x28, 0b11100111 ]);
    assert_eq!(lower(&[ SubRR( AL,  BH) ]), vec![            0x28, 0b11000111 ]);
    assert_eq!(lower(&[ SubRR(R8B,  BH) ]), vec![ 0b1000100, 0x28, 0b11000111 ]);
  }

  #[test]
  #[should_panic]
  fn test_invalid_sub_rr() {
    // FIXME: Is this only testing that at least one of the statements panics?
    lower(&[ SubRR(RAX, EAX) ]);
    lower(&[ SubRR(EAX, AH) ]);
  }

  #[test]
  fn test_xor_rr() {
    assert_eq!(lower(&[ XorRR(RAX, RBX) ]), vec![ 0b1001000, 0x31, 0b11000011 ]);
    assert_eq!(lower(&[ XorRR(R8,  RBX) ]), vec![ 0b1001100, 0x31, 0b11000011 ]);

    assert_eq!(lower(&[ XorRR(EAX, EBX) ]), vec![            0x31, 0b11000011 ]);
    assert_eq!(lower(&[ XorRR(R8D, EBX) ]), vec![ 0b1000100, 0x31, 0b11000011 ]);

    assert_eq!(lower(&[ XorRR( AX,  BX) ]), vec![            0x66, 0x31, 0b11000011 ]);
    assert_eq!(lower(&[ XorRR(R8W,  BX) ]), vec![ 0x66, 0b1000100, 0x31, 0b11000011 ]);

    assert_eq!(lower(&[ XorRR( AH,  BH) ]), vec![            0x30, 0b11100111 ]);
    assert_eq!(lower(&[ XorRR( AL,  BH) ]), vec![            0x30, 0b11000111 ]);
    assert_eq!(lower(&[ XorRR(R8B,  BH) ]), vec![ 0b1000100, 0x30, 0b11000111 ]);
  }

  #[test]
  #[should_panic]
  fn test_invalid_xor_rr() {
    // FIXME: Is this only testing that at least one of the statements panics?
    lower(&[ XorRR(RAX, EAX) ]);
    lower(&[ XorRR(EAX, AH) ]);
  }

  #[test]
  fn test_cmp_ir() {
    assert_eq!(lower(&[ CmpIR(1, RBX) ]), vec![ 0b1001000, 0x83, 0b11111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIR(1024, RBX) ]), vec![ 0b1001000, 0x81, 0b11111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIR(1, EBX) ]), vec![ 0x83, 0b11111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIR(1024, EBX) ]), vec![ 0x81, 0b11111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIR(1, BX) ]), vec![ 0x66, 0x83, 0b11111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIR(1024, BX) ]), vec![ 0x66, 0x81, 0b11111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIR(1, BH) ]), vec![ 0x80, 0b11111111, 0x01 ]);
    assert_eq!(lower(&[ CmpIR(1, BL) ]), vec![ 0x80, 0b11111011, 0x01 ]);
  }

  #[test]
  fn test_cmp_im() {
    assert_eq!(lower(&[ CmpIM(RegisterSize::Int64, 1,    RBX, 0) ]), vec![ 0b1001000, 0x83, 0b00111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIM(RegisterSize::Int64, 1024, RBX, 0) ]), vec![ 0b1001000, 0x81, 0b00111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIM(RegisterSize::Int64, 1,    EBX, 0) ]), vec![ 0x67, 0b1001000, 0x83, 0b00111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIM(RegisterSize::Int64, 1024, EBX, 0) ]), vec![ 0x67, 0b1001000, 0x81, 0b00111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIM(RegisterSize::Int32, 1,    RBX, 0) ]), vec![ 0x83, 0b00111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIM(RegisterSize::Int32, 1024, RBX, 0) ]), vec![ 0x81, 0b00111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIM(RegisterSize::Int16, 1,    RBX, 0) ]), vec![ 0x66, 0x83, 0b00111011, 0x01 ]);
    assert_eq!(lower(&[ CmpIM(RegisterSize::Int16, 1024, RBX, 0) ]), vec![ 0x66, 0x81, 0b00111011, 0x00, 0x04, 0x00, 0x00 ]);

    assert_eq!(lower(&[ CmpIM(RegisterSize::Int8, 1,    RBX, 0) ]), vec![ 0x80, 0b00111011, 0x01 ]);

    // FIXME: Is CmpIM(RegisterSize::Int64, 1,    BX, 0) something we should be able to assemble?
    // If not, we should test that we don't allow it.
  }

  #[test]
  fn test_jmp() {
    assert_eq!(lower(&[ Jmp( 16) ]), vec![ 0xeb, 0x10 ]);
    assert_eq!(lower(&[ Jmp(-16) ]), vec![ 0xeb, 0xf0 ]);
    assert_eq!(lower(&[ Jmp( 1024) ]), vec![ 0xe9, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ Jmp(-1024) ]), vec![ 0xe9, 0x00, 0xfc, 0xff, 0xff ]);
  }

  #[test]
  fn test_jz() {
    assert_eq!(lower(&[ Jz( 16) ]), vec![ 0x74, 0x10 ]);
    assert_eq!(lower(&[ Jz(-16) ]), vec![ 0x74, 0xf0 ]);
    assert_eq!(lower(&[ Jz( 1024) ]), vec![ 0x0f, 0x84, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ Jz(-1024) ]), vec![ 0x0f, 0x84, 0x00, 0xfc, 0xff, 0xff ]);
  }

  #[test]
  fn test_jnz() {
    assert_eq!(lower(&[ Jnz( 16) ]), vec![ 0x75, 0x10 ]);
    assert_eq!(lower(&[ Jnz(-16) ]), vec![ 0x75, 0xf0 ]);
    assert_eq!(lower(&[ Jnz( 1024) ]), vec![ 0x0f, 0x85, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ Jnz(-1024) ]), vec![ 0x0f, 0x85, 0x00, 0xfc, 0xff, 0xff ]);
  }

  #[test]
  fn test_call() {
    assert_eq!(lower(&[ Call( 16) ]), vec![ 0xe8, 0x10, 0x00, 0x00, 0x00 ]);
    assert_eq!(lower(&[ Call(-16) ]), vec![ 0xe8, 0xf0, 0xff, 0xff, 0xff ]);
    assert_eq!(lower(&[ Call( 1024) ]), vec![ 0xe8, 0x00, 0x04, 0x00, 0x00 ]);
    assert_eq!(lower(&[ Call(-1024) ]), vec![ 0xe8, 0x00, 0xfc, 0xff, 0xff ]);
  }
}

#[inline(never)]
fn compile_to_machinecode(instructions: &Vec<ByteCode>) -> Vec<u8> {
  use MachineInstruction::*;

  let arguments = &[Register::RDI, Register::RSI, Register::RDX];

  let tape_head = Register::RAX;
  let output_buffer_head = Register::R12;
  let output_buffer_tail = Register::RBX;
  let system_call_number = Register::RAX;
  let write_scratch_byte = Register::R13B;

  let write_function_length = 0x2e;

  let prologue = lower(&[
    Push(Register::RBP),
    MovRR(Register::RSP, Register::RBP),

    MovRR(arguments[0], tape_head),
    MovRR(arguments[1], output_buffer_head),
    MovRR(arguments[1], output_buffer_tail),

    // Jump over our `write` function.
    // FIXME: Don't hard-code the jump displacement.
    Jmp(write_function_length),

    // write
    // FIXME: Put this at the end of the generated code to avoid having to jump over it.

      // Append byte to output buffer.
      MovMR(tape_head, 0, write_scratch_byte),
      MovRM(write_scratch_byte, output_buffer_tail, 0),
      IncR(output_buffer_tail),

      // Don't call write until we see a newline character.
      CmpIR(10, write_scratch_byte),
      // FIXME: Don't hard-code the jump displacement.
      Jnz(0x1e),

      Push(tape_head),

      // Compute the number of bytes written
      MovRR(output_buffer_tail, arguments[2]),
      SubRR(output_buffer_head, arguments[2]),

      // Write output buffer to stdout
      MovRR(output_buffer_head, arguments[1]),
      MovIR(1, arguments[0]),
      MovIR(0x2000004, system_call_number),
      Syscall,

      // Reset the output buffer tail to the start.
      MovRR(output_buffer_head, output_buffer_tail),

      Pop(tape_head),

      Ret,

    // Jump target
  ]);

  let mut body = Vec::new();
  let mut loop_start_patch_points = VecMap::new();
  for (i, &instruction) in instructions.iter().enumerate() {

    match instruction {
      ByteCode::MoveLeft(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            DecR(tape_head)
          } else {
            SubIR(amount as u32, tape_head)
          }
        ]));
      }
      ByteCode::MoveRight(amount) => {
        body.extend(lower(&[
          if amount == 1 {
            IncR(tape_head)
          } else {
            AddIR(amount as u32, tape_head)
          }
        ]));
      }
      ByteCode::Add(amount, offset) => {
        body.extend(lower(&[
          if amount == 1 {
            IncM(RegisterSize::Int8, tape_head, offset)
          } else {
            AddIM(RegisterSize::Int8, amount as u32, tape_head, offset)
          }
        ]));
      }
      ByteCode::Subtract(amount, offset) => {
        body.extend(lower(&[
          if amount == 1 {
            DecM(RegisterSize::Int8, tape_head, offset)
          } else {
            SubIM(RegisterSize::Int8, amount as u32, tape_head, offset)
          }
        ]));
      }
      ByteCode::Set(value, offset) => {
        body.extend(lower(&[
          MovIM(RegisterSize::Int8, value as u32, tape_head, offset)
        ]));
      }
      ByteCode::LoopStart { end: _ } => {
        body.extend(lower(&[
          CmpIM(RegisterSize::Int8, 0, tape_head, 0),
          // Large placeholder address to force 32-bit displacement on the jump.
          Jz(std::i32::MAX),
        ]));
        loop_start_patch_points.insert(i, body.len() - 4);
      }
      ByteCode::LoopEnd { start } => {
        let loop_start_patch_point = loop_start_patch_points[start];
        let mut distance = body.len() - loop_start_patch_point + 1;

        if distance > 128 {
        // Adjust the jump distance to account for the larger instruction required to represent distances > 128.
          distance += 4;
        }

        body.extend(lower(&[
          CmpIM(RegisterSize::Int8, 0, tape_head, 0),
          Jnz(-(distance as i32)),
        ]));

        body[loop_start_patch_point + 0] = ((distance >>  0) & 0xff) as u8;
        body[loop_start_patch_point + 1] = ((distance >>  8) & 0xff) as u8;
        body[loop_start_patch_point + 2] = ((distance >> 16) & 0xff) as u8;
        body[loop_start_patch_point + 3] = ((distance >> 24) & 0xff) as u8;
      }
      ByteCode::Output => {
        let call_instruction_size = 5;
        let write_function_offset = -(body.len() as i32) - write_function_length - call_instruction_size;
        body.extend(lower(&[
          Call(write_function_offset),
        ]));
      }
      _ => { println!("{:?}", instruction); panic!() }
    }
  }

  let epilogue = lower(&[
    // Compute the number of bytes written
    MovRR(output_buffer_tail, arguments[2]),
    SubRR(output_buffer_head, arguments[2]),

    // Don't call write if we have nothing in the output buffer.
    // FIXME: Don't hard-code the jump displacement.
    Jz(0x13),

    // Write output buffer to stdout
    MovRR(output_buffer_head, arguments[1]),
    MovIR(1, arguments[0]),
    MovIR(0x2000004, system_call_number),
    Syscall,

    XorRR(Register::RAX, Register::RAX),
    Pop(Register::RBP),
    Ret,
  ]);

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
  let root = parse_source(source);
  let optimized = root.optimize();
  let bytecode = optimized.compile_to_bytecode();

  match execution_model {
    ExecutionModel::Interpret => unsafe { execute_bytecode(&bytecode) },
    ExecutionModel::JIT => {
      let machine_code = compile_to_machinecode(&bytecode);
      unsafe { execute_machinecode(&machine_code) };
    }
  }
}
