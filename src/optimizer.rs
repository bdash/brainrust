use crate::ast::Node;

mod loops;
mod moves;
mod mutations;
mod propagate_constants;

pub fn optimize_all(node: &Node) -> Vec<Node> {
  let mut optimized = vec!(node.clone());
  let optimizers = [
    Box::new(moves::SimplifyMoves) as Box<dyn Transformation>,
    Box::new(mutations::SimplifyMutationSequences),
    Box::new(loops::SimplifyLoops),
    Box::new(propagate_constants::PropagateConstants)
  ];
  loop {
    let prior_length = optimized.len();
    for optimizer in &optimizers {
      let last = optimized.last().unwrap();
      let transformed = optimizer.transform(last, true);
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
  fn transform_block(&self, children: &[Node], is_top_level: bool) -> Vec<Node>;

  fn transform(&self, node: &Node, is_top_level: bool) -> Node {
    use super::ast::Node::*;

    match node {
      Move(..) | Add{..} | Set{..} | MultiplyAdd{..} | Input | Output{..} => node.clone(),
      Loop(block) => Loop(Box::new(self.transform(block, false))),
      Block(children) => {
        Block(
          self.transform_block(children, is_top_level).iter().map(|n| self.transform(n, false)).collect()
        )
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use Node::*;

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
