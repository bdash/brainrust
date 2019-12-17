#![cfg(feature="llvm")]

use crate::bytecode::*;

use llvm_rs::*;

fn label(instruction: &ByteCode, index: usize) -> String {
  format!("i-{}-{}", index, match *instruction {
      ByteCode::MoveLeft(amount) => format!("move-left-{}", amount),
      ByteCode::MoveRight(amount) => format!("move-right-{}", amount),
      ByteCode::Add(amount, offset) => format!("add-{}-{}", amount, offset),
      ByteCode::Subtract(amount, offset) => format!("sub-{}-{}", amount, offset),
      ByteCode::Set(value, offset) => format!("set-{}-{}", value, offset),
      ByteCode::LoopStart { end } => format!("loop-start-{}", end),
      ByteCode::LoopEnd { start } => format!("loop-end-{}", start),
      ByteCode::Output => "output".to_string(),
      ByteCode::Input => "input".to_string(),
  })
}

const TAPE_SIZE: usize = 1024;
const OUTPUT_BUFFER_SIZE: usize = 1024;

struct ModuleHelper<'a> {
  context: &'a Context,
  module: &'a Module,
  builder: &'a Builder,
}

impl<'a> ModuleHelper<'a> {
  fn new(context: &'a Context, module: &'a Module, builder: &'a Builder) -> Self {
    ModuleHelper { context, module, builder }
  }

  fn emit_address_of_first_element(&'a self, array: &'a Value) -> &'a Value {
    let zero = 0.compile(self.context);
    self.builder.build_gep(array, &[ zero, zero ])
  }

  fn emit_memset(&'a self, address: &'a Value, value: u8, count: usize) {
    let context = self.context;
    let memset = self.module.get_function("llvm.memset.p0i8.i32").unwrap_or_else(|| {
      self.module.add_function("llvm.memset.p0i8.i32", Type::get::<fn(*const i8, i8, i32, i32, bool)>(context))
    });

   self.builder.build_call(memset, &[ address, value.compile(context), (count as i32).compile(context),
                                      0i32.compile(context), false.compile(context)
                                    ]);
  }
}

struct BufferedWriter<'a> {
  module_helper: &'a ModuleHelper<'a>,
}

impl<'a> BufferedWriter<'a> {
  fn new(module_helper: &'a ModuleHelper<'a>) -> Self {
    let s = BufferedWriter { module_helper };
    s.emit_buffered_write_function();
    s.emit_flush_buffered_writes_function();
    s
  }

  fn context(&self) -> &Context {
    self.module_helper.context
  }

  fn module(&self) -> &Module {
    self.module_helper.module
  }

  fn builder(&self) -> &Builder {
    self.module_helper.builder
  }

  fn write(&self) -> &Function {
    self.module().get_function("write").unwrap_or_else(|| {
      self.module().add_function("write", Type::get::<fn(i32, *const i8, usize) -> i64>(self.context()))
    })
  }

  fn buffered_write(&self) -> &Function {
    self.module().get_function("buffered_write").unwrap()
  }

  fn flush_buffered_writes(&self) -> &Function {
    self.module().get_function("flush_buffered_writes").unwrap()
  }

  fn emit_buffered_write_function(&'a self) -> &Function {
    let context = self.context();
    let module = self.module();
    let builder = self.builder();

    let expect = module.add_function("llvm.expect.i1", Type::get::<fn(bool, bool) -> bool>(context));

    let buffered_write_type = FunctionType::new(Type::get::<usize>(context),
      &[
        PointerType::new(ArrayType::new(Type::get::<i8>(context), OUTPUT_BUFFER_SIZE)),
        Type::get::<usize>(context),
        Type::get::<i8>(context),
      ]);
    let function = module.add_function("buffered_write", buffered_write_type);

    {
      let output_buffer = &function[0];
      let output_buffer_index = &function[1];
      let character = &function[2];
      let zero = 0.compile(context);

      let entry = function.append("entry");
      let do_write = function.append("do-write");
      let no_write = function.append("no-write");

      builder.position_at_end(entry);

      let output_buffer_tail = builder.build_gep(output_buffer, &[ zero, output_buffer_index ]);
      builder.build_store(character, output_buffer_tail);
      let count = builder.build_add(output_buffer_index, 1usize.compile(context));

      let cond = builder.build_cmp(character, 10u8.compile(context), Predicate::Equal);
      let cond = builder.build_call(expect, &[ cond, false.compile(context) ]);
      builder.build_cond_br(cond, do_write, Some(no_write));


      builder.position_at_end(do_write);

      let output_buffer_start = self.module_helper.emit_address_of_first_element(output_buffer);
      builder.build_call(self.write(), &[ 1i32.compile(context), output_buffer_start, count ]);
      builder.build_ret(0usize.compile(context));


      builder.position_at_end(no_write);
      builder.build_ret(count);
    }

    function
  }

  fn emit_flush_buffered_writes_function(&'a self) -> &'a mut Function {
    let context = self.context();
    let module = self.module();
    let builder = self.builder();

    let function_type = FunctionType::new(Type::get::<()>(context),
      &[
        PointerType::new(ArrayType::new(Type::get::<i8>(context), OUTPUT_BUFFER_SIZE)),
        Type::get::<usize>(context),
      ]);
    let function = module.add_function("flush_buffered_writes", function_type);

    {
      let output_buffer = &function[0];
      let output_buffer_size = &function[1];

      let entry = function.append("entry");
      let do_write = function.append("do-write");
      let no_write = function.append("no-write");

      builder.position_at_end(entry);

      let cond = builder.build_cmp(output_buffer_size, 0usize.compile(context), Predicate::GreaterThan);
      builder.build_cond_br(cond, do_write, Some(no_write));

      builder.position_at_end(do_write);
      let output_buffer_start = self.module_helper.emit_address_of_first_element(output_buffer);
      builder.build_call(self.write(), &[ 1i32.compile(context), output_buffer_start, output_buffer_size ]);
      builder.build_br(no_write);

      builder.position_at_end(no_write);
      builder.build_ret_void();
    }

    function
  }
}

struct StackFrame<'a> {
  tape: &'a Value,
  tape_head: &'a Value,
  output_buffer: &'a Value,
  output_buffer_size: &'a Value,
}

impl<'a> StackFrame<'a> {
    fn new(module_helper: &'a ModuleHelper<'a>) -> Self {
    let context = module_helper.context;
    let builder = module_helper.builder;

    let tape = builder.build_alloca(ArrayType::new(Type::get::<u8>(context), TAPE_SIZE));
    let output_buffer = builder.build_alloca(ArrayType::new(Type::get::<u8>(context), OUTPUT_BUFFER_SIZE));
    let tape_head = builder.build_alloca(Type::get::<usize>(context));
    let output_buffer_size = builder.build_alloca(Type::get::<usize>(context));

    module_helper.emit_memset(module_helper.emit_address_of_first_element(tape), 0, TAPE_SIZE);
    module_helper.emit_memset(module_helper.emit_address_of_first_element(output_buffer), 0, OUTPUT_BUFFER_SIZE);

    builder.build_store(0i64.compile(context), tape_head);
    builder.build_store(0i64.compile(context), output_buffer_size);

    StackFrame { tape, tape_head, output_buffer, output_buffer_size }
  }
}

struct InstructionHelper<'a> {
  module_helper: &'a ModuleHelper<'a>,
  stack_frame: &'a StackFrame<'a>,
}

impl<'a> InstructionHelper<'a> {
  fn new(module_helper: &'a ModuleHelper<'a>, stack_frame: &'a StackFrame<'a>) -> Self {
    InstructionHelper { module_helper, stack_frame }
  }

  fn context(&self) -> &Context {
    self.module_helper.context
  }

  fn builder(&self) -> &Builder {
    self.module_helper.builder
  }

  fn emit_move_tape_head<F>(&'a self, f: F, amount: usize) where F: Fn(&'a Builder, &'a Value, &'a Value) -> &'a Value {
    let value = self.builder().build_load(self.stack_frame.tape_head);
    let updated_value = f(self.builder(), value, amount.compile(self.context()));
    self.builder().build_store(updated_value, self.stack_frame.tape_head);
  }

  fn emit_address_of_value_at_tape_head(&'a self, offset: i32) -> &'a Value {
    let index = self.builder().build_load(self.stack_frame.tape_head);
    let index = self.builder().build_add(index, (offset as i64).compile(self.context()));
    self.builder().build_gep(self.stack_frame.tape, &[ 0.compile(self.context()), index ])
  }

  fn emit_load_value_at_tape_head(&'a self, offset: i32) -> &'a Value {
    let index = self.builder().build_load(self.stack_frame.tape_head);
    let index = self.builder().build_add(index, (offset as i64).compile(self.context()));
    let address = self.builder().build_gep(self.stack_frame.tape, &[ 0.compile(self.context()), index ]);
    self.builder().build_load(address)
  }

  fn emit_mutate_value_at_tape_head<F>(&'a self, offset: i32, f: F, amount: u8) where F: Fn(&'a Builder, &'a Value, &'a Value) -> &'a Value {
    let addr = self.emit_address_of_value_at_tape_head(offset);
    let value = self.builder().build_load(addr);
    let updated_value = f(self.builder(), value, amount.compile(self.context()));
    self.builder().build_store(updated_value, addr);
  }

  fn emit_set_value_at_tape_head(&'a self, offset: i32, value: u8) {
    let addr = self.emit_address_of_value_at_tape_head(offset);
    self.builder().build_store(value.compile(self.context()), addr);
  }
}

pub fn execute_bytecode(instructions: &Vec<ByteCode>) {
  let context = Context::new();
  let module = Module::new("module", &context);

  let builder = Builder::new(&context);

  let module_helper = ModuleHelper::new(&context, &module, &builder);

  let writer = BufferedWriter::new(&module_helper);

  let function = module.add_function("main", Type::get::<fn(i32) -> i32>(&context));
  let entry = function.append("entry");
  builder.position_at_end(entry);

  let getchar = module.add_function("getchar", Type::get::<fn() -> i32>(&context));

  let stack_frame = StackFrame::new(&module_helper);
  let helper = InstructionHelper::new(&module_helper, &stack_frame);

  let mut blocks = instructions.iter().enumerate().map(|(i, instruction)| {
    function.append(label(instruction, i).as_str())
  }).collect::<Vec<_>>();
  blocks.push(function.append("after-last-instruction"));

  builder.build_br(blocks.first().unwrap());

  for (i, &instruction) in instructions.iter().enumerate() {
    let block = blocks[i];
    let next_block = blocks[i + 1];
    builder.position_at_end(block);
    match instruction {
      ByteCode::MoveLeft(amount) => helper.emit_move_tape_head(Builder::build_sub, amount),
      ByteCode::MoveRight(amount) => helper.emit_move_tape_head(Builder::build_add, amount),

      ByteCode::Add(amount, offset) => helper.emit_mutate_value_at_tape_head(offset, Builder::build_add, amount),
      ByteCode::Subtract(amount, offset) => helper.emit_mutate_value_at_tape_head(offset, Builder::build_sub, amount),
      ByteCode::Set(value, offset) => helper.emit_set_value_at_tape_head(offset, value),

      ByteCode::LoopStart { end } => {
        let value = helper.emit_load_value_at_tape_head(0);
        let cond = builder.build_cmp(value, 0u8.compile(&context), Predicate::Equal);
        let true_block = blocks[end + 1];
        builder.build_cond_br(cond, true_block, Some(next_block));
      }
      ByteCode::LoopEnd { start } => { builder.build_br(blocks[start]); }

      ByteCode::Output => {
        let value = helper.emit_load_value_at_tape_head(0);
        let size = builder.build_load(stack_frame.output_buffer_size);
        let result = builder.build_call(writer.buffered_write(), &[ stack_frame.output_buffer, size, value ]);
        builder.build_store(result, stack_frame.output_buffer_size);
      }
      ByteCode::Input => {
        let result = builder.build_call(getchar, &[]);
        let value = builder.build_trunc(result, Type::get::<u8>(&context));
        let addr = helper.emit_address_of_value_at_tape_head(0);
        builder.build_store(value, addr);
      }
    }
    match instruction {
      ByteCode::LoopStart { .. } | ByteCode::LoopEnd { .. } => {},

      _ => { builder.build_br(next_block); }
    }
  }

  builder.position_at_end(blocks.last().unwrap());

  let size = builder.build_load(stack_frame.output_buffer_size);
  builder.build_call(writer.flush_buffered_writes(), &[ stack_frame.output_buffer, size ]);
  builder.build_ret(0i32.compile(&context));

  module.write_bitcode("/tmp/out.bc").unwrap();
  module.verify().unwrap();

  // FIXME: JITEngine doesn't appear to link the calls to external functions correctly.
  // They jump into unmapped memory.
  let ee = JitEngine::new(&module, JitOptions {opt_level: 3}).unwrap();
  ee.run_function(function, &[ &0.to_generic(&context) ]);
}
