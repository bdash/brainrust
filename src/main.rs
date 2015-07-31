extern crate argparse;
extern crate brainrust;

use std::fs::File;
use std::io::{Result, Read};
use std::path::Path;

use argparse::{ArgumentParser, StoreConst, Store};
use brainrust::{Executor, ExecutionModel};

#[cfg_attr(test, allow(dead_code))]
fn load_file(input_file_path: &str) -> Result<Vec<u8>> {
  let mut file = try!(File::open(&Path::new(input_file_path)));
  let mut contents: Vec<u8> = Vec::new();
  try!(file.read_to_end(&mut contents));
  Ok(contents)
}

#[cfg_attr(test, allow(dead_code))]
fn main() {
  let mut execution_model = ExecutionModel::JIT;
  let mut input_file_path: String = "input.bf".to_string();
  {
    let mut ap = ArgumentParser::new();
    ap.set_description("A BrainFuck interpreter in Rust");
    ap.refer(&mut execution_model)
      .add_option(&["-i", "--interpret"], StoreConst(ExecutionModel::Interpret), "Use the interpreter")
      .add_option(&["-j", "--jit"], StoreConst(ExecutionModel::JIT), "Use the JIT")
      .add_option(&["-l", "--llvm"], StoreConst(ExecutionModel::LLVM), "Use LLVM");
    ap.refer(&mut input_file_path).add_argument("script", Store, "BrainFuck script to execute.");
    ap.parse_args_or_exit();
  }

  let source = load_file(&input_file_path).unwrap();
  let executor = Executor::new(source);
  executor.execute(execution_model);
}
