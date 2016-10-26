use std;

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
pub enum RegisterSize {
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
pub enum Register {
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
    use self::Register::*;
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
    use self::Register::*;
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
pub enum MachineInstruction {
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
    use self::MachineInstruction::*;

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
    use self::MachineInstruction::*;

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
      MovIR(constant, register) => {
        match register.size() {
          RegisterSize::Int8 => assert!(constant <= std::u8::MAX as u64),
          RegisterSize::Int16 => assert!(constant <= std::u16::MAX as u64),
          RegisterSize::Int32 => assert!(constant <= std::u32::MAX as u64),
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
    use self::MachineInstruction::*;

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
    use self::MachineInstruction::*;

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
    use self::MachineInstruction::*;

    match *self {
      MovIR(constant, register) => {
        register.size().bits() > 32 && constant <= std::u32::MAX as u64
      }
      _ => panic!()
    }
  }

  fn modrm(&self) -> ModRM {
    use self::MachineInstruction::*;

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
    use self::MachineInstruction::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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
    use self::ModRM::*;

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

pub fn lower(instructions: &[MachineInstruction]) -> Vec<u8> {
  let mut machine_code = MachineCode::new();
  for instruction in instructions {
    instruction.emit(&mut machine_code);
  }
  machine_code.buffer
}

#[allow(dead_code)]
fn show_representation(instructions: &[MachineInstruction]) {
  println!("{:?}", instructions);
  println!("  {:}", lower(instructions).iter().map(|b| format!("{:02x}", b)).collect::<Vec<String>>().join(" "));
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
