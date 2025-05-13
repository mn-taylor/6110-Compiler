use crate::cfg::IsImmediate;
use crate::cfg_build::VarLabel;
use crate::regalloc::{get_dest, get_insn, get_insn_dest, get_sources, InsnLoc};
use crate::{cfg, deadcode, parse, reg_asm, scan};
use cfg::{Arg, BlockLabel, CfgType, ImmVar, MemVarLabel};
use core::fmt;
use maplit::{hashmap, hashset};
use parse::Primitive;
use reg_asm::Reg;
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
type CfgMethod = cfg::CfgMethod<VarLabel>;
type Jump = cfg::Jump<VarLabel>;

struct GenKill<T> {
    gen: HashMap<T, HashSet<VarLabel>>,
    kill: HashMap<T, HashSet<VarLabel>>,
}

fn get_block_gen_kills(b: &cfg::BasicBlock<VarLabel>) -> (HashSet<VarLabel>, HashSet<VarLabel>) {
    let mut gen: HashSet<VarLabel> = hashset! {};
    let mut kill: HashSet<VarLabel> = hashset! {};

    match b.jump_loc {
        Jump::Cond {
            source: ImmVar::Var(s),
            ..
        } => {
            gen.insert(s);
        }
        _ => (),
    }

    for i in b.body.iter().rev() {
        get_sources(&Sum::Inl(i)).iter().for_each(|var| {
            if !kill.contains(var) {
                gen.insert(*var);
            }
        });

        get_dest(&Sum::Inl(i)).iter().for_each(|var| {
            kill.insert(*var);
        });
    }

    (gen, kill)
}

fn successors(b: &cfg::BasicBlock<VarLabel>) -> Vec<BlockLabel> {
    match &b.jump_loc {
        Jump::Uncond(label) => vec![*label],
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => vec![*true_block, *false_block],
        Jump::Nowhere => vec![],
    }
}

fn in_out_for_blocks(
    m: &cfg::CfgMethod<VarLabel>,
) -> (
    HashMap<BlockLabel, HashSet<VarLabel>>,
    HashMap<BlockLabel, HashSet<VarLabel>>,
) {
    let mut in_map: HashMap<BlockLabel, HashSet<VarLabel>> = HashMap::new();
    let mut out_map: HashMap<BlockLabel, HashSet<VarLabel>> = HashMap::new();
    let mut gen_map: HashMap<BlockLabel, HashSet<VarLabel>> = HashMap::new();
    let mut kill_map: HashMap<BlockLabel, HashSet<VarLabel>> = HashMap::new();

    // initalize the maps
    for (bid, block) in m.blocks.iter() {
        let (gen, kill) = get_block_gen_kills(block);

        gen_map.insert(*bid, gen);
        kill_map.insert(*bid, kill);
        in_map.insert(*bid, HashSet::new());
        out_map.insert(*bid, HashSet::new());
    }

    // WorkList Algorithm

    let mut changed = true;
    while changed {
        changed = false;

        for (bid, block) in m.blocks.iter() {
            let old_in = &in_map[&bid];
            let old_out = &out_map[&bid];

            let mut out_set: HashSet<VarLabel> = hashset! {};
            for succ in successors(&block) {
                if let Some(in_succ) = in_map.get(&succ) {
                    out_set.extend(in_succ.iter().cloned());
                }
            }

            // Compute in[B] = gen[B] ∪ (out[B] - kill[B])
            let gen = &gen_map[&bid];
            let kill = &kill_map[&bid];
            let mut in_set = gen.clone();
            in_set.extend(out_set.difference(kill).cloned());

            if in_set != *old_in || out_set != *old_out {
                changed = true;
                in_map.insert(*bid, in_set);
                out_map.insert(*bid, out_set);
            }
        }
    }
    (in_map, out_map)
}

fn live_in_blocks(
    b_label: BlockLabel,
    block: &cfg::BasicBlock<VarLabel>,
    b_out: HashSet<VarLabel>,
) -> (
    HashMap<InsnLoc, HashSet<VarLabel>>,
    HashMap<InsnLoc, HashSet<VarLabel>>,
) {
    let mut live_in: HashMap<InsnLoc, HashSet<VarLabel>> = hashmap! {};
    let mut live_out: HashMap<InsnLoc, HashSet<VarLabel>> = hashmap! {};

    let mut live = b_out.clone();

    // handle jump
    match block.jump_loc {
        Jump::Cond {
            source: ImmVar::Var(s),
            true_block: _,
            false_block: _,
        } => {
            let loc = InsnLoc {
                blk: b_label,
                idx: block.body.len(),
            };

            live_out.insert(loc, live.clone());
            live.insert(s);
            live_in.insert(loc, live.clone());
        }
        _ => (),
    }

    for (i, instr) in block.body.iter().enumerate().rev() {
        let loc = InsnLoc {
            blk: b_label,
            idx: i,
        };

        live_out.insert(loc, live.clone());

        let def_i = get_dest(&Sum::Inl(instr)); // def[i]
        let use_i = get_sources(&Sum::Inl(instr)); // use[i]

        let mut live_in_i: HashSet<_> = use_i.iter().copied().collect();
        for v in &live {
            if !def_i.contains(v) {
                live_in_i.insert(*v);
            }
        }

        live_in.insert(loc, live_in_i.clone());
        live = live_in_i;
    }

    (live_in, live_out)
}

fn get_interference_graph(m: &cfg::CfgMethod<VarLabel>) -> HashMap<VarLabel, HashSet<VarLabel>> {
    // get in and outs of blocks
    let (in_blocks, out_blocks) = in_out_for_blocks(m);

    // get in and out of instructions
    let mut live_out: HashMap<InsnLoc, HashSet<VarLabel>> = HashMap::new();
    for (b_label, block) in m.blocks.iter() {
        let (_, live_out_in_block) =
            live_in_blocks(*b_label, block, out_blocks.get(b_label).unwrap().clone());

        live_out.extend(live_out_in_block)
    }

    // build graph by iterating over instructions
    let mut graph: HashMap<VarLabel, HashSet<VarLabel>> = HashMap::new();
    for (i_loc, variables) in live_out.iter() {
        let defns = get_dest(&get_insn(m, *i_loc));

        for def in defns {
            for var in variables {
                if *var != def {
                    graph.entry(def).or_insert_with(HashSet::new).insert(*var);
                    graph.entry(*var).or_insert_with(HashSet::new).insert(def);
                }
            }
        }
    }

    return graph;
}
