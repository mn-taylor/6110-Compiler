use crate::cfg::{BlockLabel, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use crate::ir::Block;
use std::arch::aarch64::int16x8_t;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env::var;

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

fn var_to_def_locs(m: &CfgMethod) -> HashMap<VarLabel, HashSet<BlockLabel>> {
    let mut def: HashMap<VarLabel, HashSet<BlockLabel>> = HashMap::new();

    for (label, block) in m.blocks.iter() {
        for instr in block.body.clone() {
            match instr {
                Instruction::ThreeOp {
                    source1,
                    source2,
                    dest,
                    op,
                } => add_block_to_var_def(&mut def, &dest, *label),
                Instruction::Constant { dest, constant } => {
                    add_block_to_var_def(&mut def, &dest, *label)
                }
                Instruction::MoveOp { source, dest } => {
                    add_block_to_var_def(&mut def, &dest, *label)
                }
                Instruction::TwoOp { source1, dest, op } => {
                    add_block_to_var_def(&mut def, &dest, *label)
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
    let mut int: HashSet<BlockLabel>;
    if let Some(set) = vals.next() {
        int = set.clone();
    } else {
        int = HashSet::new();
    }

    for val in vals {
        int = int.intersection(&val).map(|x| *x).collect();
    }
    int
}
// n -> set of blocks that dominate n
// from https://en.wikipedia.org/wiki/Dominator_(graph_theory)
pub fn dominator_sets(
    start_node: BlockLabel,
    g: &HashMap<BlockLabel, HashSet<BlockLabel>>,
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

pub fn dominator_tree(
    m: &CfgMethod,
    dom_sets: &HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_tree: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();
    let blocks = m.blocks.iter();

    for (label, block) in blocks {
        // compute the immediate dominator of the block

        let mut agenda = vec![];
        for parent in block.parents.iter() {
            if dom_sets.contains_key(&parent) {
                agenda.push(*parent);
            }
        }

        if agenda.len() == 0 {
            continue;
        };

        let target_dom_set = dom_sets.get(label).unwrap();
        let mut seen: HashSet<BlockLabel> = HashSet::new();
        seen.insert(*label);
        while agenda.len() > 0 {
            let curr = agenda.pop().unwrap();
            if seen.contains(&curr) {
                continue;
            }; // cycle

            if target_dom_set.contains(&curr) {
                // found immediate dominator
                let mut new_set: HashSet<BlockLabel> = match dom_tree.get(&curr) {
                    Some(set) => set.clone(),
                    None => HashSet::new(),
                };

                new_set.insert(*label);

                dom_tree.insert(curr, new_set.clone());
                break;
            }

            seen.insert(curr);

            // add the new parents
            for parent in m.blocks.get(&curr).unwrap().parents.iter() {
                if dom_sets.contains_key(&parent) {
                    agenda.push(*parent);
                }
            }
        }
    }

    dom_tree
}

pub fn dominance_frontiers(
    g: HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_sets: HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_tree: HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut df = HashMap::new();
    let mut idom = HashMap::new();
    for (p, children) in dominance_tree.iter() {
        df.insert(*p, HashSet::new());
        for c in children {
            idom.insert(c, *p);
            df.insert(*c, HashSet::new());
        }
    }

    println!("Initial dominance frontier{:?}", df);
    println!("g : {:?}", g);
    // Algorithm 3.2 in SSA book
    for (a, children) in g {
        for b in children {
            let mut x = a;
            let mut x_dom_b = dominance_sets.get(&b).unwrap().contains(&x);

            // while x does not strictly dominate b
            while !(x_dom_b && x != b) {
                println!("{} dominates {}", x, b);
                df.get_mut(&x).unwrap().insert(b);
                if idom.get(&x).is_some() {
                    x = *idom.get(&x).unwrap();
                    x_dom_b = dominance_sets.get(&b).unwrap().contains(&x);
                } else {
                    break;
                }
            }
        }
    }
    df
}

pub fn get_graph(m: &mut CfgMethod) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut g: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();

    for (label, block) in m.blocks.iter() {
        if (block.parents.len() == 0 && *label != 0) {
            // these nodes are never accessed by the program and confuse the computation of dominance sets
            continue;
        }

        let mut edges: HashSet<BlockLabel> = HashSet::new();
        match block.jump_loc {
            Jump::Uncond(b) => {
                edges.insert(b);
            }
            Jump::Cond {
                source: _,
                true_block: b1,
                false_block: b2,
            } => {
                edges.insert(b1);
                edges.insert(b2);
            }
            Jump::Nowhere => {}
        }

        g.insert(*label, edges);
    }

    g
}

pub fn insert_phis(m: &mut CfgMethod) -> &CfgMethod {
    let var_defs = var_to_def_locs(m);
    let g = get_graph(m);
    let dom_sets: HashMap<usize, HashSet<usize>> = dominator_sets(0, &g);
    let dom_tree = dominator_tree(m, &dom_sets);
    // let dom_frontier = dominance_frontiers(g, dom_sets, dom_tree);

    // Algorithm 3.1 in SSA-Based Compiler Design
    for (var, defs) in var_defs.iter() {
        let mut F: HashSet<BlockLabel> = HashSet::new();
        let mut W: Vec<BlockLabel> = vec![];

        for def in defs {
            W.push(*def);
        }

        while W.len() != 0 {
            let X = W.pop().unwrap();
            for Y in dom_sets.get(&X).unwrap() {
                // change
                if !F.contains(Y) {
                    let block = m.blocks.get_mut(Y).unwrap();
                    block.body.insert(
                        0,
                        Instruction::PhiExpr {
                            dest: *var,
                            sources: vec![],
                        },
                    );

                    F.insert(*Y);

                    if !defs.contains(Y) {
                        W.push(*Y);
                    }
                }
            }
        }
    }

    return m;
}
