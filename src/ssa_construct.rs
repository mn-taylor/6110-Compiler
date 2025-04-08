use crate::cfg::BlockLabel;
use crate::cfg_build::{CfgMethod, VarLabel};
use std::collections::HashMap;
use std::collections::HashSet;

fn var_to_def_locs(m: CfgMethod) -> HashMap<VarLabel, HashSet<BlockLabel>> {}

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
