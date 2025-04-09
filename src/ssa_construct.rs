use crate::cfg::{BlockLabel, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use std::collections::HashMap;
use std::collections::HashSet;

fn add_block_to_var_def(
    def: &mut HashMap<VarLabel, HashSet<BlockLabel>>,
    var: &VarLabel,
    block_id: BlockLabel,
) {
    match def.get_mut(&var) {
        Some(t) => {
            t.insert(block_id);
        }
        None => {
            let mut set = HashSet::new();
            set.insert(block_id);
            def.insert(var.clone(), set);
        }
    }
}

fn var_to_def_locs(m: CfgMethod) -> HashMap<VarLabel, HashSet<BlockLabel>> {
    let mut def: HashMap<VarLabel, HashSet<BlockLabel>> = HashMap::new();

    for (label, block) in m.blocks {
        for instr in block.body {
            match instr {
                Instruction::ThreeOp {
                    source1,
                    source2,
                    dest,
                    op,
                } => add_block_to_var_def(&mut def, &dest, label),
                Instruction::Constant { dest, constant } => {
                    add_block_to_var_def(&mut def, &dest, label)
                }
                Instruction::MoveOp { source, dest } => {
                    add_block_to_var_def(&mut def, &dest, label)
                }
                Instruction::TwoOp { source1, dest, op } => {
                    add_block_to_var_def(&mut def, &dest, label)
                }
                _ => (),
            }
        }
    }

    def
}

fn intersect_all<'a>(
    vals: &mut impl Iterator<Item = &'a HashSet<BlockLabel>>,
) -> HashSet<BlockLabel> {
    let mut int: HashSet<BlockLabel> = vals.next().unwrap().clone();
    for val in vals {
        int = int.intersection(&val).map(|x| *x).collect();
    }
    int
}

// from https://en.wikipedia.org/wiki/Dominator_(graph_theory)
fn dominator_sets(
    start_node: BlockLabel,
    g: HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_sets: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();
    let all_keys: HashSet<BlockLabel> = g.keys().map(|x| *x).collect::<HashSet<_>>();
    let mut pred = HashMap::new();
    // dominator of the start node is the start itself
    // for all other nodes, set all nodes as the dominators
    for x in g.keys() {
        if *x == start_node {
            dom_sets.insert(*x, vec![*x].into_iter().collect::<HashSet<BlockLabel>>());
        } else {
            dom_sets.insert(*x, all_keys.clone());
        }
        pred.insert(*x, HashSet::new());
    }

    // compute predecessors
    for (x, ys) in g.iter() {
        for y in ys {
            pred.get_mut(&y).unwrap().insert(x);
        }
    }

    let mut changes = true;
    while changes {
        changes = false;
        for (n, preds) in pred.iter() {
            if *n != start_node {
                // Dom(n) = {n} union with intersection over Dom(p) for all p in pred(n)
                let mut new_dom_n: HashSet<BlockLabel> =
                    intersect_all(&mut preds.iter().map(|p| dom_sets.get(&p).unwrap()));
                new_dom_n.insert(*n);

                let dom_n = dom_sets.get(n).unwrap();
                if *dom_n != new_dom_n {
                    changes = true;
                    dom_sets.insert(*n, new_dom_n);
                }
            }
        }
    }

    dom_sets
}

fn dominator_tree(
    m: CfgMethod,
    dom_sets: HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_tree: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();
    let blocks = m.blocks.iter();
    for (label, block) in blocks {
        // compute the immediate dominator of the block
        let mut agenda = block.parents.clone();
        let mut seen: HashSet<BlockLabel> = HashSet::new();
        seen.insert(*label);
        while agenda.len() > 0 {
            let curr = agenda.pop().unwrap();
            if seen.contains(&curr) {
                continue;
            }; // cycle

            let mut dom_set = dom_sets.get(&curr).unwrap();
            if dom_set.contains(&label) {
                // found immediate dominator
                match dom_tree.get_mut(&curr) {
                    None => {
                        let mut new_set = HashSet::new();
                        new_set.insert(*label);

                        dom_tree.insert(curr, new_set);
                    }
                    Some(s) => {
                        s.insert(*label);
                    }
                }
                break;
            }

            seen.insert(curr);

            // add the new parents
            agenda.extend(m.blocks.get(&curr).unwrap().parents.clone())
        }
    }

    dom_tree
}

fn dominance_frontiers(
    g: HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_sets: HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_tree: HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut df = HashMap::new();
    let mut idom = HashMap::new();
    for (p, children) in dominance_tree.iter() {
        for c in children {
            idom.insert(c, *p);
            df.insert(*c, HashSet::new());
        }
    }
    // Algorithm 3.2 in SSA book
    for (a, children) in g {
        for b in children {
            let mut x = a;
            let x_dom_b = dominance_sets.get(&x).unwrap().contains(&b);
            // while x does not strictly dominate b
            while !(x_dom_b && x != b) {
                df.get_mut(&x).unwrap().insert(b);
                x = *idom.get(&x).unwrap();
            }
        }
    }
    df
}

fn insert_phis(m: &mut CfgMethod) {}
