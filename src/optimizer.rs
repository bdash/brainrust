use super::ast::Node;

use std::collections::HashMap;

use itertools::Itertools;

pub fn optimize_all(node: &Node) -> Vec<Node> {
  let mut optimized = vec!(node.clone());
  let optimizers = [
    Box::new(SimplifyMoves) as Box<dyn Transformation>,
    Box::new(SimplifyMutationSequences),
    Box::new(SimplifyLoops),
  ];
  loop {
    let prior_length = optimized.len();
    for optimizer in &optimizers {
      let last = optimized.last().unwrap();
      let transformed = optimizer.transform(last);
      if transformed != *last {
        optimized.push(transformed);
      }
    }
    if optimized.len() == prior_length {
      // If the optimization pass didn't produce a different AST than it started with, we're done.
      break;
    }
  }
  optimized
}

pub fn optimize(node: &Node) -> Node {
  optimize_all(node).pop().unwrap()
}

trait Transformation {
  fn transform_block(&self, &[Node]) -> Vec<Node>;

  fn transform(&self, node: &Node) -> Node {
    use super::ast::Node::*;

    match node {
      Move(..) | Add{..} | Set{..} | MultiplyAdd{..} | Input | Output{..} => node.clone(),
      Loop(box block) => Loop(Box::new(self.transform(block))),
      Block(children) => {
        Block(
          self.transform_block(children).iter().map(|n| self.transform(n)).collect()
        )
      }
    }
  }
}

struct SimplifyMoves;

impl SimplifyMoves {
  fn propagate_offsets(&self, nodes: &[&Node]) -> Vec<Node> {
    use super::ast::Node::*;

    let mut index = 0i32;

    let mut result: Vec<_> = nodes.iter().flat_map(|node| {
      match **node {
        Move(amount) => {
          index += amount as i32;
          None
        }
        Add { amount, offset } => Some(Add { amount, offset: offset + index }),
        Set { value, offset } => Some(Set { value, offset: offset + index }),
        MultiplyAdd { multiplier, source, dest } => Some(MultiplyAdd { multiplier, source: source + index, dest: dest + index }),
        Output { offset } => Some(Output { offset: offset + index }),
        _ => panic!()
      }
    }).collect();
    if index != 0 {
      result.push(Move(index as isize));
    }
    result
  }
}

impl Transformation for SimplifyMoves {
  fn transform_block(&self, children: &[Node]) -> Vec<Node> {
    children.iter().group_by(|&node| node.supports_offset()).into_iter().flat_map(|(key, group)| {
      if !key {
        group.cloned().collect()
      } else {
        let group_nodes: Vec<_> = group.collect();
        self.propagate_offsets(&group_nodes)
      }
    }).collect()
  }
}

#[derive(Copy, Clone, Debug)]
enum Mutation {
  Add(i8),
  Set(i8),
}

impl Mutation {
  fn combine(&mut self, other: Mutation) {
    use self::Mutation::*;

    *self = match (*self, other) {
      (_,           Set(_))           => other,
      (Set(base),   Add(amount))      => Set(base + amount),
      (Add(a),      Add(b))           => Add(a + b),
    }
  }
}

struct SimplifyMutationSequences;

impl SimplifyMutationSequences {
  fn evaluate_mutations(&self, nodes: &[&Node]) -> HashMap<i32, Mutation> {
    use super::ast::Node::*;

    let mut mutations = HashMap::new();

    for node in nodes {
      match **node {
        Add{ amount, offset } => {
          let value = mutations.entry(offset).or_insert(Mutation::Add(0));
          value.combine(Mutation::Add(amount));
        }
        Set{ value: amount, offset } => {
          let value = mutations.entry(offset).or_insert(Mutation::Set(0));
          value.combine(Mutation::Set(amount as i8));
        }
        _ => panic!()
      }
    }
    mutations
  }
}

impl Transformation for SimplifyMutationSequences {
  fn transform_block(&self, children: &[Node]) -> Vec<Node> {
    use super::ast::Node::*;

    children.iter().group_by(|&node| node.is_add_or_set()).into_iter().flat_map(|(key, group)| {
      if !key {
        group.cloned().collect()
      } else {
        let group_nodes: Vec<_> = group.collect();
        let mutations = self.evaluate_mutations(&group_nodes);

        let mut modified_offsets = mutations.keys().cloned().collect::<Vec<_>>();
        modified_offsets.sort();

        modified_offsets.iter().flat_map(|offset| {
          let value = mutations[offset];
          match value {
            Mutation::Add(value) => Some(Add{ amount: value, offset: *offset }),
            Mutation::Set(value) => Some(Set{ value: value as u8, offset: *offset })
          }
        }).collect::<Vec<_>>()
      }
    }).collect()
  }
}

struct SimplifyLoops;

impl SimplifyLoops {
  fn is_substract_at_offset_0(node: &&Node) -> bool {
    match **node {
      // TODO: It should be possible to generalize this optimization to handle loops with a stride larger than one.
      Node::Add{ amount: -1, offset: 0 } => true,
      _ => false,
    }
  }

  fn simplify_loop(&self, children: &[Node]) -> Vec<Node> {
    use super::ast::Node::*;

    if !children.iter().all(Node::is_add) {
      return vec!(Loop(Box::new(Block(children.to_vec()))))
    }

    if let Some(Node::Add{ offset: 0, .. }) = children.iter().find(Self::is_substract_at_offset_0) {
      let mut result = Vec::new();
      for child in children {
        if Self::is_substract_at_offset_0(&&child) {
          continue;
        }

        result.push(match child {
          Add { offset: 0, .. } => panic!("Unexpected secondary addition to offset 0"),
          Add { amount, offset } => MultiplyAdd { multiplier: *amount, source: 0, dest: *offset },
          Set { .. } => child.clone(),
          _ => unreachable!(),
        });
      }
      result.push(Set { value: 0, offset: 0 });
      return result;
    }
    return vec!(Loop(Box::new(Block(children.to_vec()))))
  }
}

impl Transformation for SimplifyLoops {
  fn transform_block(&self, children: &[Node]) -> Vec<Node> {
    use super::ast::Node::*;

    children.iter().flat_map(|node| {
      match node {
        Loop(box Block(children)) => self.simplify_loop(&children).into_iter(),
        _ => vec!(node.clone()).into_iter(),
      }
    }).collect()
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use super::Node::*;

  #[test]
  fn test_simplify_loop() {
    assert_eq!(
      SimplifyLoops.simplify_loop(&[]),
      vec!(Loop(Box::new(Block(vec!())))),
    );

    assert_eq!(
      SimplifyLoops.simplify_loop(
        &[Add{ amount: -1, offset: 0 }],
      ),
      vec!(Set { value: 0, offset: 0 }),
    );


    // TODO: It should be possible to generalize this optimization to handle loops with a stride larger than one.
    // assert_eq!(
    //   SimplifyLoops.transform_block(
    //     &vec!(Add{ amount: -5, offset: 0 }),
    //   ),
    //   vec!(Set { value: 0, offset: 0 }),
    // );

    assert_eq!(
      SimplifyLoops.simplify_loop(
        &[Add{ amount: -1, offset: 0 },
          Add{ amount: 1, offset: 1}],
      ),
      vec!(
        MultiplyAdd{ multiplier: 1, source: 0, dest: 1 },
        Set{ value: 0, offset: 0 },
      ),
    );
  }

  #[test]
  fn test_optimize() {
    // Two adds of different offsets that cannot be optimized.
    assert_eq!(
      optimize(
        &Node::from_bytes(b"->+"),
      ),
      Block(vec!(
        Add{ amount: -1, offset: 0 },
        Add{ amount: 1, offset: 1},
        Move(1),
      )),
    );

    // Simplifying of adds, subtracts, and moves.
    assert_eq!(
      optimize(
        &Node::from_bytes(b"-->++>-"),
      ),
      Block(vec!(
        Add{ amount: -2, offset: 0 },
        Add{ amount: 2, offset: 1},
        Add{ amount: -1, offset: 2},
        Move(2),
      )),
    );

    // Simple multiply / add loops.
    assert_eq!(
      optimize(
        &Node::from_bytes(b"[->+>---<<]")
      ),
      Block(vec!(
        MultiplyAdd{ multiplier: 1, source: 0, dest: 1 },
        MultiplyAdd{ multiplier: -3, source: 0, dest: 2 },
        Set{ value: 0, offset: 0 },
      )),
    );

    // This is a loop being used as an if statement. If offset 0 is non-zero, offset 1 will be cleared.
    // The outer loop never loops.
    assert_eq!(
      optimize(
        &Node::from_bytes(b"[>[-]<[-]]")
      ),
      Block(vec!(
        Loop(Box::new(Block(vec!(
            Set{ value: 0, offset: 0 },
            Set{ value: 0, offset: 1 },
        )))),
      )),
    );

    // Adds the values 2 + 5, converts to ASCII, and prints the result.
    assert_eq!(
      optimize(
        &Node::from_bytes(b"++>+++++[<+>-]++++++++[<++++++>-]<.")
      ),
      Block(vec!(
        Add{ amount: 2, offset: 0 },
        Add{ amount: 5, offset: 1 },
        MultiplyAdd{ multiplier: 1, source: 1, dest: 0 },
        Set{ value: 8, offset: 1 },
        MultiplyAdd{ multiplier: 6, source: 1, dest: 0 },
        Set{ value: 0, offset: 1 },
        Output{ offset: 0 },
      )),
    );
  }
}
