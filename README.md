# BrainRust: an x86_64 BrainFuck JIT in Rust

BrainRust is an interpreter and JIT for the [BrainFuck][bf] programming language. The interpreter is
available on all platforms supported by Rust, while the JIT is for x86_64 only. It performs high-level
optimizations that provide a 3x increase in performance for both JIT and interpreter, while the JIT
provides a > 5x increase over the interpreter.

[bf]: https://en.wikipedia.org/wiki/Brainfuck

## Supported optimizations
* Straight-line sequences of mutation operations are simplified. Adjustments of the tape head are merged
  into a single adjustment at the end of the block. Modifications to the tape are merged into a single
  add / subtract / set operation per tape offset that performs the aggregate of all modifications to that
  location within the block. This allows sequences such as `>>+<<` to be converted to `Add(1, 1)`, and
  `->>+>>` to be represented as `Add(-1, 0), Add(1, 2), Move(4)`.
* Loops that adjust the location under the tape head by a fixed amount, such as `[>++<-]`, can be recognized
  as multiplications by the value under the tape head. `[-]` zeroing the location under the tape head is 
  covered by this optimization.

## Future optimizations
* Loops that iterate a fixed number of times can be detected.
* Constant propagation can be performed based on the initially-zeroed contents of the tape.
