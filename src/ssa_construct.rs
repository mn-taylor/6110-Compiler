use crate::cfg::{BlockLabel, Instruction};
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

fn dominator_sets(start_node: BlockLabel, g: HashMap<BlockLabel, HashSet<BlockLabel>>) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_sets = HashMap::new();
    for (x, 
    dom_sets
}

fn dominator_tree(m: CfgMethod) -> HashMap<BlockLabel, HashSet<BlockLabel>> {}

fn dominance_frontiers(
    g: HashMap<BlockLabel, HashSet<BlockLabel>>,
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
            // while x does not strictly dominate child
            while !dominance_tree.get(&x).unwrap().contains(&b) {
                df.get_mut(&x).unwrap().insert(b);
                x = *idom.get(&x).unwrap();
            }
        }
    }
    df
}

fn insert_phis(m: &mut CfgMethod) {}
