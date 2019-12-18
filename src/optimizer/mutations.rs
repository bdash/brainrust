use crate::{ast::Node, optimizer};
use std::collections::HashMap;
use itertools::Itertools;

pub struct SimplifyMutationSequences;

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


impl SimplifyMutationSequences {
  fn evaluate_mutations(&self, nodes: &[&Node]) -> HashMap<i32, Mutation> {
    use Node::*;

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

impl optimizer::Transformation for SimplifyMutationSequences {
  fn transform_block(&self, children: &[Node]) -> Vec<Node> {
    use Node::*;

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
