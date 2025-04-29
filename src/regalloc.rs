use crate::cfg_build::{BasicBlock, VarLabel};
use crate::ssa_construct::get_graph;
use crate::{cfg, deadcode, scan};
use cfg::{BlockLabel, ImmVar};
use maplit::{hashmap, hashset};
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;
type CfgMethod = cfg::CfgMethod<VarLabel>;
type Instruction = cfg::Instruction<VarLabel>;
type Jump = cfg::Jump<VarLabel>;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct InsnLoc {
    blk: BlockLabel,
    idx: usize,
}

fn get_defs(m: &CfgMethod) -> HashMap<VarLabel, HashSet<InsnLoc>> {
    let mut defs: HashMap<VarLabel, HashSet<InsnLoc>> = HashMap::new();

    for (bid, block) in m.blocks.iter() {
        for (iid, instruction) in block.body.iter().enumerate() {
            if let Some(dest) = get_dest(&Sum::Inl(&instruction.clone())) {
                defs.entry(dest)
                    .or_insert_with(HashSet::new)
                    .insert(InsnLoc {
                        blk: *bid,
                        idx: iid,
                    });
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

fn get_sources(insn: &Sum<&Instruction, &Jump>) -> HashSet<VarLabel> {
    match insn {
        Sum::Inl(i) => deadcode::get_sources(i),
        Sum::Inr(j) => deadcode::get_jump_sources(j),
    }
}

fn get_dest(insn: &Sum<&Instruction, &Jump>) -> Option<VarLabel> {
    match insn {
        Sum::Inl(i) => get_insn_dest(i),
        Sum::Inr(_) => None,
    }
}

fn get_insn_dest(insn: &Instruction) -> Option<VarLabel> {
    match insn {
        Instruction::PhiExpr { .. } => panic!(),
        Instruction::MemPhiExpr { .. } => panic!(),
        Instruction::ParMov(_) => panic!(),
        Instruction::ArrayAccess { dest, .. } => Some(*dest),
        Instruction::Call(_, _, dest) => *dest,
        Instruction::Constant { dest, .. } => Some(*dest),
        Instruction::MoveOp { dest, .. } => Some(*dest),
        Instruction::ThreeOp { dest, .. } => Some(*dest),
        Instruction::TwoOp { dest, .. } => Some(*dest),
        Instruction::ArrayStore { .. } => None,
        Instruction::Spill { .. } => None,
        Instruction::Reload { ord_var, .. } => Some(*ord_var),
        Instruction::Ret(r) => match r {
            Some(ImmVar::Var(x)) => Some(*x),
            Some(ImmVar::Imm(_)) => None,
            None => None,
        },
    }
}

fn get_uses(m: &CfgMethod, x: VarLabel, i: InsnLoc) -> HashSet<InsnLoc> {
    let mut uses = HashSet::new();
    let mut seen = HashSet::new();
    let mut next = vec![i];
    while let Some(v) = next.pop() {
        seen.insert(v);
        let insn = get_insn(m, v);
        if get_sources(&insn).contains(&x) {
            uses.insert(v);
        }
        if get_dest(&insn) != Some(x) {
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

fn get_webs(m: &CfgMethod) -> HashMap<VarLabel, Vec<(Vec<InsnLoc>, HashSet<InsnLoc>)>> {
    let defs = get_defs(m);
    let mut webs = HashMap::new();

    for (varname, def_set) in defs {
        // sets contains vec of def and set of uses
        let mut sets: Vec<(Vec<InsnLoc>, HashSet<InsnLoc>)> = def_set
            .iter()
            .map(|(InsnLoc { blk: bid, idx: iid })| {
                (
                    vec![InsnLoc {
                        blk: *bid,
                        idx: *iid,
                    }],
                    get_uses(
                        &m,
                        varname,
                        InsnLoc {
                            blk: *bid,
                            idx: *iid,
                        },
                    ),
                )
            })
            .collect::<Vec<_>>();
        if sets.is_empty() {
            continue;
        }

        let mut new_sets = vec![];
        while let Some((mut curr_defs, curr_uses)) = sets.pop() {
            let mut to_merge = None;

            for (i, (defs, uses)) in sets.iter_mut().enumerate() {
                if !curr_uses.is_disjoint(uses) {
                    to_merge = Some((i, (defs.clone(), uses.clone())));
                    break;
                }
            }

            if let Some((i, (defs, uses))) = to_merge {
                sets.remove(i);
                let merged: HashSet<InsnLoc> = curr_uses.union(&uses).cloned().collect();
                curr_defs.extend(defs.clone());
                sets.push((curr_defs, merged));
            } else {
                new_sets.push((curr_defs, curr_uses));
            }
        }

        webs.insert(varname, new_sets);
    }
    webs
}

fn contains_def(block: &BasicBlock, var: VarLabel, after: usize) -> bool {
    for ((i, instr)) in block.body.iter().enumerate() {
        if i <= after {
            continue;
        }
        match get_insn_dest(instr) {
            Some(v) => {
                if var == v {
                    return true;
                }
            }
            None => (),
        }
    }
    return false;
}

fn find_inter_instructions(
    m: &mut CfgMethod,
    var: VarLabel,
    defs: &Vec<InsnLoc>,
    uses: HashSet<InsnLoc>,
) -> HashSet<InsnLoc> {
    let g = get_graph(m);

    let mut all_instructions = hashset! {};
    let target_blocks: HashSet<usize> = uses.iter().map(|InsnLoc { blk, idx }| *blk).collect();

    // find all instructions/blocks that lie on some path from a def to a use.
    // find all blocks that are reachable from the def, then find all blocks that can reach a use.
    let mut reachable_from_defs = hashset! {}; // blocks to consider

    for def in defs {
        let InsnLoc { blk: bid, idx: iid } = def;

        let mut agenda: Vec<(usize, usize)> = vec![(*bid, *iid)];
        let mut seen: HashSet<usize> = hashset! {};
        while agenda.len() != 0 {
            let (curr_block, curr_instr) = agenda.pop().unwrap();
            if seen.contains(&curr_block) {
                continue;
            };
            reachable_from_defs.insert(curr_block);

            // check if curr contains a redifinition of
            if !contains_def(m.blocks.get(&curr_block).unwrap(), var, curr_instr) {
                let children: Vec<(usize, usize)> = g
                    .get(&curr_block)
                    .unwrap()
                    .iter()
                    .map(|c| (*c, 0))
                    .collect(); // start at the top of child blocks
                agenda.extend(children)
            }

            seen.insert(curr_block);
        }
    }

    let mut can_reach_use: HashSet<BlockLabel> = hashset! {};
    for block in reachable_from_defs.iter() {
        if can_reach_use.contains(&block) {
            continue;
        };

        let mut agenda = vec![block];
        let mut seen = hashset! {};
        while agenda.len() != 0 {
            let curr_block = agenda.pop().unwrap();
            if seen.contains(&curr_block) {
                continue;
            } else {
                seen.insert(curr_block);
            }

            if target_blocks.contains(&curr_block) {
                can_reach_use.insert(*block);
                break;
            } else {
                let children = g.get(&curr_block).unwrap().iter().collect::<Vec<_>>();
                agenda.extend(children)
            }
        }
    }

    let blocks_of_interest: HashSet<usize> = reachable_from_defs
        .intersection(&can_reach_use)
        .cloned()
        .collect();

    for bid in blocks_of_interest.iter() {
        // check if it contains a def
        let potential_start = defs.iter().find(|InsnLoc { blk, idx }| blk == bid).cloned();

        let starting_idx = match potential_start {
            Some(InsnLoc { blk: _, idx: iid }) => iid,
            None => 0,
        };

        let mut start = starting_idx;
        for (i, instr) in m.blocks.get(bid).unwrap().body.iter().enumerate() {
            if i < starting_idx {
                continue;
            }

            if uses.contains(&InsnLoc { blk: *bid, idx: i }) {
                all_instructions.extend(
                    (start..i + 1)
                        .collect::<Vec<_>>()
                        .iter()
                        .map(|c| InsnLoc { blk: *bid, idx: *c }),
                );
                start = i + 1;
            }

            if i == m.blocks.get(bid).unwrap().body.len() - 1 {
                if uses.contains(&InsnLoc {
                    blk: *bid,
                    idx: i + 1,
                }) {
                    // jump instruction
                    all_instructions.extend(
                        (start..i + 2)
                            .collect::<Vec<_>>()
                            .iter()
                            .map(|c| InsnLoc { blk: *bid, idx: *c }),
                    );
                } else {
                    // check if the remaining instruction are on a path to a use.
                    let children = g.get(bid).unwrap();
                    if !children.is_disjoint(&can_reach_use) {
                        // add all of the remaining instructions
                        all_instructions.extend(
                            (start..i + 2)
                                .collect::<Vec<_>>()
                                .iter()
                                .map(|c| InsnLoc { blk: *bid, idx: *c }),
                        );
                    }
                }
            }
        }
    }

    all_instructions
}

// returns a tuple: a map taking a label k to the corresponding convex closure of a web, and the interference graph, where nodes are labels
fn interference_graph(
    m: &mut CfgMethod,
) -> (HashMap<u32, HashSet<InsnLoc>>, HashMap<u32, HashSet<u32>>) {
    let webs = get_webs(m);
    let mut convex_closures_of_webs = Vec::new();
    for (var, def_use_s) in webs {
        for (defs, uses) in def_use_s {
            convex_closures_of_webs.push(find_inter_instructions(m, var, &defs, uses));
        }
    }
    let ret1: HashMap<u32, HashSet<InsnLoc>> = convex_closures_of_webs
        .into_iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();

    let mut ret2 = HashMap::new();
    for (i, _) in ret1.iter() {
        ret2.insert(*i, HashSet::new());
    }
    // TODO: fill in ret2 with intersections
    (ret1, ret2)
}

fn remove_node(g: &mut HashMap<u32, HashSet<u32>>, v: u32) -> HashSet<u32> {
    let result = g.remove(&v).unwrap();
    for (_, es) in g {
        es.remove(&v);
    }
    result
}

fn color_not_in_set(num_colors: u32, s: HashSet<u32>) -> u32 {
    let all_colors: HashSet<u32> = (0..num_colors).into_iter().collect();
    *all_colors.difference(&s).collect::<Vec<_>>().pop().unwrap()
}

fn color(
    mut g: HashMap<u32, HashSet<u32>>,
    num_colors: u32,
) -> Result<HashMap<u32, u32>, HashSet<u32>> {
    let mut color_later = Vec::new();

    loop {
        let mut to_remove = None;
        for (v, es) in g.iter() {
            if (es.len() as u32) < num_colors {
                to_remove = Some(*v);
                break;
            };
        }
        match to_remove {
            Some(v) => color_later.push((v, remove_node(&mut g, v))),
            None => break,
        }
    }
    if g.is_empty() {
        let mut colors = HashMap::new();
        while let Some((v, es)) = color_later.pop() {
            colors.insert(
                v,
                color_not_in_set(
                    num_colors,
                    es.into_iter().map(|u| *colors.get(&u).unwrap()).collect(),
                ),
            );
        }
        // return an assignment of colors
        Ok(colors)
    } else {
        // return nodes that could not be colored
        Err(g.into_keys().collect())
    }
}
