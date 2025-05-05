// following https://courses.cs.washington.edu/courses/cse501/06wi/reading/click-pldi95.pdf
use crate::{cfg, regalloc, scan};
use cfg::{BlockLabel, CfgMethod, Instruction};
use regalloc::InsnLoc;
use scan::Sum;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use Instruction::*;
fn insn_pinned<T>(insn: &Instruction<T>) -> bool {
    match insn {
        Spill { .. } | Reload { .. } | ParMov(_) | StoreParam(_, _) | NoArgsCall(_, _) => panic!(),
        PhiExpr { .. }
        | ArrayAccess { .. }
        | ArrayStore { .. }
        | LoadParam { .. }
        | Call(_, _, _)
        | Ret(_) => true,
        // maybe want to leave divisions alone
        ThreeOp { .. } => false,
        MoveOp { .. } | TwoOp { .. } | Constant { .. } => false,
    }
}

fn pinned<T>(insn_loc: InsnLoc, m: &CfgMethod<T>) -> bool {
    match regalloc::get_insn(m, insn_loc) {
        Sum::Inl(insn) => insn_pinned(insn),
        Sum::Inr(_) => true,
    }
}

// bottom of pg 249
pub fn schedule_all_early<T: Copy + Hash + Eq>(m: &CfgMethod<T>) -> HashMap<InsnLoc, BlockLabel> {
    let mut done: HashSet<InsnLoc> = HashSet::new();
    let all_locs = regalloc::all_insn_locs(m);
    let _loc_of_lbl: HashMap<T, InsnLoc> = all_locs
        .iter()
        .filter_map(|iloc| regalloc::get_dest(&regalloc::get_insn(m, *iloc)).map(|v| (v, *iloc)))
        .collect();
    for i in all_locs.iter() {
        if pinned(*i, m) {
            done.insert(*i);
        }
    }
    todo!()
}

fn schedule_early<T: Hash>(m: &CfgMethod<T>, done: HashSet<InsnLoc>) -> BlockLabel {
    todo!()
}
