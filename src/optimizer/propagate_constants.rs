use crate::{ast::Node, optimizer};
use itertools::Itertools;
use std::collections::HashMap;

pub struct PropagateConstants;

#[derive(Copy, Clone, Debug)]
enum Value<T> {
  Unknown,
  Constant(T),
}

impl<T> Default for Value<T> {
  fn default() -> Self { Self::Unknown }
}

#[derive(Debug)]
struct State {
  tape: HashMap<i32, Value<u8>>,
  tape_head: i32,
  default: Value<u8>,
}

impl State {
  fn new(default: Value<u8>) -> Self {
    State { tape: HashMap::default(), tape_head: 0 , default }
  }

  fn at(&mut self, offset: i32) -> &mut Value<u8> {
    self.tape.entry(self.tape_head + offset).or_insert(self.default)
  }
}

fn loop_node_allows_constant_iteration_count(node: &Node) -> bool {
  use Node::*;

  match node {
    Loop(..) | Move(..) | Input => false,
    Add{ offset: 0, amount } if *amount != -1 => false,
    Add{..} | Set{..} | MultiplyAdd{..} | Output{..} => true,
    Block(..) => unreachable!(),
  }
}


impl PropagateConstants {
  fn transform_subrange(&self, nodes: Vec<Node>, is_initial_block: bool) -> Vec<Node> {
    let default_tape_value = if is_initial_block { Value::Constant(0) } else { Value::Unknown };
    let mut state = State::new(default_tape_value);
    self.transform_subrange_with_state(nodes, &mut state)
  }

  fn transform_subrange_with_state(&self, nodes: Vec<Node>, mut state: &mut State) -> Vec<Node> {
    use Node::*;

    let nodes = nodes.into_iter().flat_map(|node| {
      match node {
        Set{ value, offset } => {
          *state.at(offset) = Value::Constant(value);
          vec!(Set{ value, offset })
        }
        Add{ amount, offset } => {
          let value = state.at(offset);
          *value = match value {
            Value::Unknown => Value::Unknown,
            Value::Constant(old_constant) => {
              let new_constant = if amount > 0 {
                old_constant.wrapping_add(amount as u8)
              } else {
                old_constant.wrapping_sub(amount.abs() as u8)
              };
              Value::Constant(new_constant)
            }
          };
          vec!(match value {
            Value::Unknown => Add { amount, offset },
            Value::Constant(value) => Set{ value: *value, offset },
          })
        }
        MultiplyAdd{ multiplier, source, dest } => {
          let source_value = state.at(source).clone();
          let dest_value = state.at(dest);
          *dest_value = match (&source_value, &dest_value) {
            (Value::Unknown, _) | (_, Value::Unknown) => Value::Unknown,
            (Value::Constant(source_constant), Value::Constant(dest_constant)) => {
              let result = if multiplier > 0 {
                dest_constant.wrapping_add(source_constant.wrapping_mul(multiplier as u8))
              } else {
                dest_constant.wrapping_sub(source_constant.wrapping_mul(multiplier.abs() as u8))
              };
              Value::Constant(result)
            },
          };
          vec!(match dest_value {
            Value::Unknown => MultiplyAdd { multiplier, source, dest },
            Value::Constant(value) => Set{ value: *value, offset: dest },
          })
        },
        Move(amount) => {
          state.tape_head += amount as i32;
          vec!(Move(amount))
        }
        Input => {
          *state.at(0) = Value::Unknown;
          vec!(Input)
        }
        Output{..} => vec!(node),
        Loop(box Block(ref children)) => {
          match state.at(0) {
            // Loops where the condition variable is known to be zero can be stripped as their body will never
            // be entered.
            Value::Constant(0) => vec!(),

            // Loops with a constant iteration count (no nested loops, the tape head doesn't move,
            // and the condition variable is decremented by 1 each time) can be unrolled.
            Value::Constant(constant) if children.iter().all(loop_node_allows_constant_iteration_count) => {
                let mut result = None;
                for i in (1..*constant + 1).rev() {
                  *state.at(0) = Value::Constant(i);
                  result = Some(self.transform_subrange_with_state(children.to_vec(), &mut state));
                }
                result.unwrap()
            }
            _ => vec!(node),
          }
        }
        Loop(..) => unreachable!(),
        Block(..) => unreachable!(),
      }
    }).collect();
    nodes
  }
}

impl optimizer::Transformation for PropagateConstants {
  fn transform_block(&self, children: &[Node], is_top_level: bool) -> Vec<Node> {
    let mut loop_count = 0;
    children.iter().group_by(|&node| {
      let group = loop_count;
      if node.is_loop() {
        loop_count += 1;
      }
      group
    }).into_iter().flat_map(|(i, group)| {
      self.transform_subrange(group.cloned().collect(), is_top_level && i == 0)
    }).collect()
  }
}
