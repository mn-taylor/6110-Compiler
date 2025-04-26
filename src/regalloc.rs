use crate::cfg_build::VarLabel;
use crate::{cfg, scan};
use cfg::BlockLabel;
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::deadcode::get_dest;

type CfgMethod = cfg::CfgMethod<VarLabel>;
type Instruction = cfg::Instruction<VarLabel>;
type Jump = cfg::Jump<VarLabel>;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct InsnLoc {
    blk: BlockLabel,
    idx: usize,
}

fn get_defs(m: CfgMethod) -> HashMap<VarLabel, HashSet<(BlockLabel, usize)>> {
    let mut defs: HashMap<VarLabel, HashSet<(BlockLabel, usize)>> = HashMap::new();

    for (bid, block) in m.blocks {
        for (iid, instruction) in block.body.into_iter().enumerate() {
            if let Some(dest) = get_dest(instruction) {
                defs.entry(dest)
                    .or_insert_with(HashSet::new)
                    .insert((bid, iid));
            }
        }
    }
    defs
}

fn get_insn(m: &CfgMethod, i: InsnLoc) -> Sum<&Instruction, &Jump> {
    let blk = m.blocks.get(&i.blk).unwrap();
    if blk.body.len() == i.idx {
        Sum::Inr(&blk.jump_loc)
    } else {
        Sum::Inl(blk.body.get(i.idx).unwrap())
    }
}

fn get_children(m: &CfgMethod, i: InsnLoc) -> Vec<InsnLoc> {
    let blk = m.blocks.get(&i.blk).unwrap();
    if blk.body.len() == i.idx {
        match blk.jump_loc {
            Jump::Nowhere => vec![],
            Jump::Uncond(next_blk) => vec![InsnLoc {
                blk: next_blk,
                idx: 0,
            }],
            Jump::Cond {
                true_block,
                false_block,
                ..
            } => vec![
                InsnLoc {
                    blk: true_block,
                    idx: 0,
                },
                InsnLoc {
                    blk: false_block,
                    idx: 0,
                },
            ],
        }
    } else {
        vec![InsnLoc {
            blk: i.blk,
            idx: i.idx + 1,
        }]
    }
}

fn get_uses(m: &CfgMethod, x: VarLabel, i: InsnLoc) -> HashSet<InsnLoc> {
    let mut uses = HashSet::new();
    let mut seen = HashSet::new();
    let mut next = vec![i];
    while let Some(v) = next.pop() {
        seen.insert(v);
        if
        /*TODO: v has x as a source*/
        true {
            uses.insert(v);
        }
        if
        /*TODO: v does not have x as dest*/
        true {
            next.append(
                &mut get_children(m, v)
                    .into_iter()
                    .filter(|i| !seen.contains(i))
                    .collect(),
            );
        }
    }
    uses
}

fn get_webs(m: CfgMethod) -> HashSet<(VarLabel, Vec<HashSet<(BlockLabel, usize)>>)> {
    let defs = get_defs(m);
    let mut webs = HashSet::new();

    for (varname, def_set) in defs {
        let mut sets: Vec<HashSet<InsnLoc>> = def_set
            .iter()
            .map(|(bid, iid)| {
                get_uses(
                    &m,
                    varname,
                    InsnLoc {
                        blk: *bid,
                        idx: *iid,
                    },
                )
            })
            .collect::<Vec<_>>();
        if sets.is_empty() {
            continue;
        }

        let mut new_sets = vec![];
        while !sets.is_empty() {
            let curr = sets.pop().unwrap();

            let mut found_overlap = false;
            let mut to_merge = None;

            for set in &sets {
                if !curr.is_disjoint(set) {
                    to_merge = Some(set.clone());
                    found_overlap = true;
                    break;
                }
            }

            if let Some(merge_set) = to_merge {
                sets.remove(&merge_set);
                let merged = curr.union(&merge_set).cloned().collect();
                sets.insert(merged);
            } else {
                new_sets.insert(curr);
            }
        }

        webs.insert((varname, new_sets));
    }

    webs
}

fn find_inter_instructions(_m: CfgMethod, _s: HashSet<InsnLoc>) -> HashSet<InsnLoc> {
    todo!()
}

fn interference_graph(_m: CfgMethod) {
    todo!()
}
