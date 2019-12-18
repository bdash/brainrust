use crate::{ast::Node, optimizer};

use itertools::Itertools;

pub struct SimplifyMoves;

impl SimplifyMoves {
  fn propagate_offsets(&self, nodes: &[&Node]) -> Vec<Node> {
    use crate::ast::Node::*;

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

impl optimizer::Transformation for SimplifyMoves {
  fn transform_block(&self, children: &[Node], _is_top_level: bool) -> Vec<Node> {
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
