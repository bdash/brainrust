use super::bytecode::*;

use std::io::{Read, Write, stdin, stdout};

#[inline(never)]
pub fn execute_bytecode(instructions: &[ByteCode]) {
  unsafe {
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
            stdout().write_all(&output[..]).unwrap();
            output.clear();
          }
        }
        ByteCode::Input => {
          let mut input = [0u8; 1];
          stdin().read_exact(&mut input[..]).unwrap();
          tape[tape_head] = input[0];
        }
      }
      ip += 1
    }
    if !output.is_empty() {
      stdout().write_all(&output[..]).unwrap();
    }
  }
}
