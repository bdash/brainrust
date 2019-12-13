use super::ast::Node;

use std::collections::HashMap;

use itertools::Itertools;

pub fn optimize(node: &Node) -> Node {
  let mut node = node.clone();
  loop {
    let optimized = optimize_once(&node);
    if optimized == node {
      break;
    }
    node = optimized;
  }
  node
}

fn optimize_once(node: &Node) -> Node {
  use super::ast::Node::*;

  match *node {
    Move(..) | Add{..} | Set{..} | Input | Output => node.clone(),
    Loop(box ref block) => {
      match *block.children().as_slice() {
        [ Add{ amount: -1, offset: 0 } ] => Set{ value: 0, offset: 0 },
        _ => Loop(Box::new(optimize_once(block))),
      }
    }
    Block(ref children) => {
      let simplified_children = simplify_mutation_sequences(children);
      let optimized_children = simplified_children.iter().map(optimize_once).collect();
      Block(optimized_children)
    }
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

fn simplify_mutation_sequences(children: &[Node]) -> Vec<Node> {
  use super::ast::Node::*;

  children.iter().group_by(|&node| node.is_mutation()).into_iter().flat_map(|(key, group)| {
    if !key {
      group.cloned().collect()
    } else {
      let group_nodes: Vec<_> = group.collect();
      let (index, mutations) = evaluate_mutations(&group_nodes);

      let mut modified_offsets = mutations.keys().collect::<Vec<&i32>>();
      modified_offsets.sort();

      modified_offsets.iter().flat_map(|offset| {
        let value = mutations[*offset];
        match value {
          Mutation::Add(value) => Some(Add{ amount: value, offset: **offset }),
          Mutation::Set(value) => Some(Set{ value: value as u8, offset: **offset })
        }
      }).chain(
        if index == 0 { None } else { Some(Move(index as isize)) }
      ).collect::<Vec<_>>()
    }
  }).collect()
}

fn evaluate_mutations(nodes: &[&Node]) -> (i32, HashMap<i32, Mutation>) {
  use super::ast::Node::*;

  let mut mutations = HashMap::new();
  let mut index = 0i32;

  for node in nodes {
    match **node {
      Move(amount) => index += amount as i32,
      Add{ amount, offset } => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Add(0));
        value.combine(Mutation::Add(amount));
      }
      Set{ value: amount, offset } => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Set(0));
        value.combine(Mutation::Set(amount as i8));
      }
      _ => panic!()
    }
  }
  (index, mutations)
}
