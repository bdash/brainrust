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
    MoveLeft(..) | MoveRight(..) | Add(..) | Subtract(..) | Set(..) | Input | Output => node.clone(),
    Loop(box ref block) => {
      match block.children().as_slice() {
        [ Subtract(1, 0) ] => Set(0, 0),
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
  Subtract(i8),
  Set(i8),
}

impl Mutation {
  fn combine(&mut self, other: Mutation) {
    use self::Mutation::*;

    *self = match (*self, other) {
      (_,           Set(_))           => other,
      (Set(base),   Add(amount))      => Set(base + amount),
      (Set(base),   Subtract(amount)) => Set(base - amount),
      (Add(a),      Add(b))           => Add(a + b),
      (Subtract(a), Subtract(b))      => Subtract(a + b),
      (Add(a),      Subtract(b))      => {
        match (a - b).signum() {
          1 | 0 => Add(a - b),
          -1 => Subtract(b - a),
          _ => unreachable!(),
        }
      }
      (Subtract(a), Add(b)) => {
        match (b - a).signum() {
          1 | 0 => Add(a - b),
          -1 => Subtract(b - a),
          _ => unreachable!(),
        }
      }
    }
  }
}

fn simplify_mutation_sequences(children: &Vec<Node>) -> Vec<Node> {
  use super::ast::Node::*;

  children.iter().group_by(|&node| node.is_mutation()).flat_map(|(key, group)| {
    match key {
      true => {
        let (index, mutations) = evaluate_mutations(&group);

        let mut modified_offsets = mutations.keys().collect::<Vec<&i32>>();
        modified_offsets.sort();

        modified_offsets.iter().flat_map(|offset| {
          let value = mutations[*offset];
          match value {
            Mutation::Subtract(value) => Some(Subtract(value as u8, **offset)),
            Mutation::Add(value) => Some(Add(value as u8, **offset)),
            Mutation::Set(value) => Some(Set(value as u8, **offset))
          }
        }).chain(
          match index.signum() {
             1 => Some(MoveRight(index as usize)),
            -1 => Some(MoveLeft(index.abs() as usize)),
             _ => None
          }
        ).collect::<Vec<_>>()
      }
      false => group.iter().map(|&node| node.clone()).collect()
    }
  }).collect()
}

fn evaluate_mutations(nodes: &Vec<&Node>) -> (i32, HashMap<i32, Mutation>) {
  use super::ast::Node::*;

  let mut mutations = HashMap::new();
  let mut index = 0i32;

  for node in nodes {
    match **node {
      MoveLeft(amount) => {
        index -= amount as i32;
      }
      MoveRight(amount) => {
        index += amount as i32;
      }
      Add(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Add(0));
        value.combine(Mutation::Add(amount as i8));
      }
      Subtract(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Add(0));
        value.combine(Mutation::Subtract(amount as i8));
      }
      Set(amount, offset) => {
        let value = mutations.entry(index + offset).or_insert(Mutation::Set(0));
        value.combine(Mutation::Set(amount as i8));
      }
      _ => panic!()
    }
  }
  (index, mutations)
}
