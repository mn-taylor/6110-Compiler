use crate::cfg;
use crate::cfg::{Arg, BasicBlock, BlockLabel, ImmVar, Instruction, IsImmediate, Jump};
use crate::cfg_build::get_parents;
use crate::ir::{Bop, UnOp};
use crate::scan::{AddOp, EqOp, MulOp, RelOp, Sum};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, prune_method, SSAVarLabel};
use maplit::hashset;
use std::collections::{HashMap, HashSet};

fn find_loops_at(
    bid: BlockLabel,
    dominance_tree: HashMap<BlockLabel, HashSet<BlockLabel>>,
    g: HashMap<BlockLabel, HashSet<BasicBlock<SSAVarLabel>>>,
) -> Option<HashSet<BlockLabel>> {
    // iterate through all blocks that are dominated by bid. If there is a backedge, iterate up the dominance tree and add all blocks to the loop
    todo!();
}

fn find_invariant_code(
    blocks: Vec<BlockLabel>,
    m: cfg::CfgMethod<SSAVarLabel>,
) -> cfg::BasicBlock<SSAVarLabel> {
    // find all variables defined inside of the loop,
    //  - if the variable is defined in terms of a constant it can be moved
    //  - if the variable is defined in terms of varibles defined outside of the loop, then it can be moved.
    // returns a block of all of the invariant code
    todo!();
}

pub fn loop_invariant_code_motion(
    m: &mut cfg::CfgMethod<SSAVarLabel>,
) -> cfg::CfgMethod<SSAVarLabel> {
    // iterate through blocks (potentially in a dfs of the dominance tree but it does not matter)
    // for a given block, call find_loops_at
    // if find loops_at returns a loop call find_invariant_code
    // if the outputed block has at least one instruction make a detached header block to contain the code and stitch

    // might have to reset the dominance and dom
    todo!();
}
