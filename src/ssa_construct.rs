use crate::cfg::BlockLabel;
use crate::cfg_build::{CfgMethod, VarLabel};
use std::collections::HashMap;
use std::collections::HashSet;

fn var_to_def_locs(m: CfgMethod) -> HashMap<VarLabel, HashSet<BlockLabel>> {}

fn dominator_tree(m: CfgMethod) -> HashMap<BlockLabel, HashSet<BlockLabel>> {}

fn dominance_frontier(
    dominance_tree: HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashSet<BlockLabel> {
}

fn insert_phis(m: &mut CfgMethod) {}
