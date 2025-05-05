// following https://courses.cs.washington.edu/courses/cse501/06wi/reading/click-pldi95.pdf
use crate::{cfg, regalloc};
use cfg::{BlockLabel, CfgMethod, Instruction};
use regalloc::InsnLoc;
use std::collections::{HashMap, HashSet};

use Instruction::*;
fn pinned<T>(insn: Instruction<T>) -> bool {
    match insn {
        Spill { .. } | Reload { .. } => panic!(),
        _ => todo!(),
    }
}

// bottom of pg 249
fn schedule_all_early<T>(m: &CfgMethod<T>) -> HashMap<InsnLoc, BlockLabel> {
    let done: HashSet<InsnLoc> = HashSet::new();
    for i in regalloc::all_insn_locs(m) {}
    todo!()
}
