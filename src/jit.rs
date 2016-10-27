use super::assembler::x86_64::*;
use super::bytecode::*;

use std;
use std::mem;
use std::ptr;

use std::fs::OpenOptions;
use std::io::{Result, Write};
use std::path::Path;

use libc::{c_void, mmap, mprotect, munmap, PROT_EXEC, PROT_WRITE, MAP_ANON, MAP_PRIVATE};
use vec_map::VecMap;
use syscall;

pub fn execute_bytecode(instructions: &Vec<ByteCode>) {
  let machine_code = compile_to_machinecode(instructions);
  unsafe { execute_machinecode(&machine_code) };
}

#[inline(never)]
fn compile_to_machinecode(instructions: &Vec<ByteCode>) -> Vec<u8> {
  use super::assembler::x86_64::MachineInstruction::*;

  let arguments = &[Register::RDI, Register::RSI, Register::RDX];

  let tape_head = Register::RAX;
  let output_buffer_head = Register::R12;
  let output_buffer_tail = Register::RBX;
  let system_call_number = Register::RAX;
  let write_scratch_byte = Register::R13B;
  let write_scratch_byte_register = Register::R13;

  let write_function_length = 0x2e;

  let prologue = lower(&[
    Push(Register::RBP),
    Push(output_buffer_head),
    Push(output_buffer_tail),
    Push(write_scratch_byte_register),
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
      MovIR(syscall::nr::WRITE as u64, system_call_number),
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
      ByteCode::Input => {
        body.extend(lower(&[
          // Zero byte at tape head so that EOF will map to zero.
          MovIM(RegisterSize::Int8, 0, tape_head, 0),
          Push(tape_head),

          // Read one byte into tape head.
          MovIR(1, arguments[2]),
          MovRR(tape_head, arguments[1]),
          MovIR(0, arguments[0]),
          MovIR(syscall::nr::READ as u64, system_call_number),

          Syscall,
          Pop(tape_head),
        ]));
      }
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
    MovIR(syscall::nr::WRITE as u64, system_call_number),
    Syscall,

    XorRR(Register::RAX, Register::RAX),
    Pop(write_scratch_byte_register),
    Pop(output_buffer_tail),
    Pop(output_buffer_head),
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

fn write_to_file(file_path: &str, buffer: &Vec<u8>) -> Result<()> {
  let mut file = try!(OpenOptions::new().write(true).create(true).truncate(true).open(&Path::new(file_path)));
  try!(file.write_all(&buffer[..]));
  Ok(())
}

struct MemoryMap {
  size: u64,
  buffer: *mut c_void,
}

impl MemoryMap {
  unsafe fn new(size: usize, protection: i32) -> MemoryMap {
    let buffer = mmap(ptr::null::<u8>() as *mut c_void, size as u64, protection, MAP_ANON | MAP_PRIVATE, 0, 0);
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
