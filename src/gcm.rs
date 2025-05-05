use crate::regalloc::get_insn;
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

// see definition of "pinned" on page 249
fn pinned<T>(insn_loc: InsnLoc, m: &CfgMethod<T>) -> bool {
    match regalloc::get_insn(m, insn_loc) {
        Sum::Inl(insn) => insn_pinned(insn),
        Sum::Inr(_) => true,
    }
}

// bottom of pg 249
pub fn schedule_all_early<T: Copy + Hash + Eq>(m: &CfgMethod<T>) -> HashMap<InsnLoc, BlockLabel> {
    let mut schedule: HashMap<InsnLoc, BlockLabel> = HashMap::new();
    let mut done: HashSet<InsnLoc> = HashSet::new();
    let all_locs = regalloc::all_insn_locs(m);
    let loc_of_lbl: HashMap<T, InsnLoc> = all_locs
        .iter()
        .filter_map(|iloc| regalloc::get_dest(&regalloc::get_insn(m, *iloc)).map(|v| (v, *iloc)))
        .collect();
    for i in all_locs.iter() {
        if pinned(*i, m) {
            done.insert(*i);
            for x in regalloc::get_sources(&get_insn(m, *i)) {
                let x = *loc_of_lbl.get(&x).unwrap();
                schedule_early(x, m, &mut done, &mut schedule, &loc_of_lbl);
            }
        }
    }
    schedule
}

fn schedule_early<T: Hash + Copy + Eq>(
    i: InsnLoc,
    m: &CfgMethod<T>,
    done: &mut HashSet<InsnLoc>,
    schedule: &mut HashMap<InsnLoc, BlockLabel>,
    loc_of_lbl: &HashMap<T, InsnLoc>,
) {
    if pinned(i, m) || done.contains(&i) {
        return;
    }
    done.insert(i);
    // 0 is the root
    schedule.insert(i, 0);
    for x in regalloc::get_sources(&get_insn(m, i)) {
        let x = *loc_of_lbl.get(&x).unwrap();
        schedule_early(x, m, done, schedule, loc_of_lbl);
        if false
        /*TODO define dominator depths; cnoditiion is very very wrong*/
        {
            schedule.insert(i, *schedule.get(&x).unwrap());
        }
    }
}
