use crate::{ast::Node, optimizer};

pub struct SimplifyLoops;

impl SimplifyLoops {
  fn is_substract_at_offset_0(node: &&Node) -> bool {
    match **node {
      // TODO: It should be possible to generalize this optimization to handle loops with a stride larger than one.
      Node::Add{ amount: -1, offset: 0 } => true,
      _ => false,
    }
  }

  fn simplify_loop(&self, children: &[Node]) -> Vec<Node> {
    use Node::*;

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

impl optimizer::Transformation for SimplifyLoops {
  fn transform_block(&self, children: &[Node], _is_top_level: bool) -> Vec<Node> {
    use Node::*;

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
  use Node::*;

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
}
