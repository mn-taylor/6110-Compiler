// following https://courses.cs.washington.edu/courses/cse501/06wi/reading/click-pldi95.pdf
use crate::{cfg, regalloc, scan, ssa_construct};
use cfg::{BlockLabel, CfgMethod, Instruction};
use regalloc::InsnLoc;
use scan::Sum;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use Instruction::*;
fn insn_pinned<T>(insn: &Instruction<T>) -> bool {
    match insn {
        Spill { .. } | Reload { .. } | ParMov(_) | StoreParam(_, _) | Push(_) | Pop(_) => panic!(),
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

fn dominator_tree<T>(m: &CfgMethod<T>) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let g = ssa_construct::get_graph(m);
    let dom_sets = ssa_construct::dominator_sets(0, &g);
    ssa_construct::dominator_tree(m, &dom_sets)
}

fn build_idoms<T>(m: &CfgMethod<T>) -> HashMap<BlockLabel, BlockLabel> {
    dominator_tree(m)
        .into_iter()
        .flat_map(|(u, vs)| vs.into_iter().map(|v| (v, u)).collect::<Vec<_>>())
        .collect()
}

fn dom_depths<T>(m: &CfgMethod<T>) -> HashMap<BlockLabel, u32> {
    let dom_tree = dominator_tree(m);
    let mut agenda = vec![0];
    let mut depths = HashMap::from([(0, 0)]);
    while agenda.len() > 0 {
        let mut next_level = vec![];
        for v in agenda {
            for child in dom_tree.get(&v).unwrap() {
                depths.insert(*child, *depths.get(&v).unwrap());
                next_level.push(*child);
            }
        }
        agenda = next_level;
    }
    depths
}

// see definition of "pinned" on page 249
fn pinned<T>(insn_loc: InsnLoc, m: &CfgMethod<T>) -> bool {
    match regalloc::get_insn(m, insn_loc) {
        Sum::Inl(insn) => insn_pinned(insn),
        Sum::Inr(_) => true,
    }
}

fn build_loc_of_lbl<T: Copy + Hash + Eq>(
    m: &CfgMethod<T>,
    all_locs: &HashSet<InsnLoc>,
) -> HashMap<T, InsnLoc> {
    all_locs
        .iter()
        .filter_map(|iloc| regalloc::get_dest(&regalloc::get_insn(m, *iloc)).map(|v| (v, *iloc)))
        .collect()
}

pub fn schedule_all<T: Copy + Hash + Eq>(
    m: &CfgMethod<T>,
    early: Option<&HashMap<InsnLoc, BlockLabel>>,
) -> HashMap<InsnLoc, BlockLabel> {
    let mut schedule: HashMap<InsnLoc, BlockLabel> = HashMap::new();
    let mut done: HashSet<InsnLoc> = HashSet::new();
    let all_locs = regalloc::all_insn_locs(m);
    let loc_of_lbl: HashMap<T, InsnLoc> = build_loc_of_lbl(m, &all_locs);
    let depths = dom_depths(m);
    let uses = build_uses(m, &loc_of_lbl);
    let idoms = build_idoms(m);
    for i in all_locs.iter() {
        if pinned(*i, m) {
            done.insert(*i);
            for x in regalloc::get_sources(&regalloc::get_insn(m, *i)) {
                let x = *loc_of_lbl.get(&x).unwrap();
                match early {
                    None => schedule_early(x, m, &mut done, &mut schedule, &loc_of_lbl, &depths),
                    Some(early) => schedule_best(
                        x,
                        m,
                        &mut done,
                        early,
                        &mut schedule,
                        &loc_of_lbl,
                        &depths,
                        &uses,
                        &idoms,
                    ),
                }
            }
        }
    }
    schedule
}

// bottom of pg 249
fn schedule_early<T: Hash + Copy + Eq>(
    i: InsnLoc,
    m: &CfgMethod<T>,
    done: &mut HashSet<InsnLoc>,
    schedule: &mut HashMap<InsnLoc, BlockLabel>,
    loc_of_lbl: &HashMap<T, InsnLoc>,
    depths: &HashMap<BlockLabel, u32>,
) {
    if pinned(i, m) || done.contains(&i) {
        return;
    }
    done.insert(i);
    // 0 is the root
    schedule.insert(i, 0);
    for x in regalloc::get_sources(&regalloc::get_insn(m, i)) {
        let x = *loc_of_lbl.get(&x).unwrap();
        schedule_early(x, m, done, schedule, loc_of_lbl, depths);
        if depths.get(schedule.get(&i).unwrap()).unwrap()
            < depths.get(schedule.get(&x).unwrap()).unwrap()
        {
            schedule.insert(i, *schedule.get(&x).unwrap());
        }
    }
}

fn build_uses<T: Copy + Eq + Hash>(
    m: &CfgMethod<T>,
    loc_of_lbl: &HashMap<T, InsnLoc>,
) -> HashMap<InsnLoc, Vec<InsnLoc>> {
    let mut uses: HashMap<_, _> = regalloc::all_insn_locs(m)
        .into_iter()
        .map(|i| (i, vec![]))
        .collect();
    for i in regalloc::all_insn_locs(m) {
        for j in regalloc::get_sources(&regalloc::get_insn(m, i)) {
            uses.get_mut(&loc_of_lbl.get(&j).unwrap()).unwrap().push(i);
        }
    }
    uses
}

// top of pg 251
// unlike schedule_early, schedule_best might make schedule a partial map.
// if schedule.get(&x) = None, then x is never used and thus does not need to be scheduled at all.
fn schedule_best<T: Copy + Hash + PartialEq + Eq>(
    i: InsnLoc,
    m: &CfgMethod<T>,
    done: &mut HashSet<InsnLoc>,
    early: &HashMap<InsnLoc, BlockLabel>,
    sched: &mut HashMap<InsnLoc, BlockLabel>,
    loc_of_lbl: &HashMap<T, InsnLoc>,
    depths: &HashMap<BlockLabel, u32>,
    uses: &HashMap<InsnLoc, Vec<InsnLoc>>,
    idoms: &HashMap<BlockLabel, BlockLabel>,
) {
    if pinned(i, m) || done.contains(&i) {
        return;
    }
    done.insert(i);
    let mut lca = None;
    for y in uses.get(&i).unwrap() {
        schedule_best(*y, m, done, early, sched, loc_of_lbl, depths, uses, idoms);
        let use_of_y =
            if let Sum::Inl(Instruction::PhiExpr { sources, .. }) = regalloc::get_insn(m, *y) {
                let (block_with_i, _) = sources
                    .into_iter()
                    .find(|(_, ii)| loc_of_lbl.get(ii).unwrap() == &i)
                    .unwrap();
                block_with_i
            } else {
                sched.get(&y).unwrap()
            };
        lca = Some(find_lca(sched.get(y).map(|x| *x), *use_of_y, depths, idoms));
    }
    if let Some(lca) = lca {
        sched.insert(i, best_block(*early.get(&i).unwrap(), lca, idoms));
    } // else don't need to shcedule at all, is dead code
}

fn find_lca(
    a: Option<BlockLabel>,
    mut b: BlockLabel,
    depths: &HashMap<BlockLabel, u32>,
    idoms: &HashMap<BlockLabel, BlockLabel>,
) -> BlockLabel {
    let mut a = match a {
        None => return b,
        Some(a) => a,
    };
    while depths.get(&a).unwrap() < depths.get(&b).unwrap() {
        a = *idoms.get(&a).unwrap();
    }
    while depths.get(&b).unwrap() < depths.get(&a).unwrap() {
        b = *idoms.get(&b).unwrap();
    }
    while a != b {
        a = *idoms.get(&a).unwrap();
        b = *idoms.get(&b).unwrap();
    }
    a
}

// alg at end of section 2 (pg 251).
fn best_block(
    earliest: BlockLabel,
    mut latest: BlockLabel,
    idoms: &HashMap<BlockLabel, BlockLabel>,
) -> BlockLabel {
    let best = latest;
    while latest != earliest {
        // if loop_nest(latest) < loop_nest(best) {
        //     best = latest;
        // }
        latest = *idoms.get(&latest).unwrap();
        todo!();
    }
    best
}
