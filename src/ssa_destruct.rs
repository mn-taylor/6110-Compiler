use crate::cfg::{self, ImmVar};
use crate::cfg::{Arg, BasicBlock, CfgType, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use crate::ssa_construct::SSAVarLabel;
use std::collections::HashMap;
use std::collections::HashSet;

fn union(
    a: SSAVarLabel,
    b: SSAVarLabel,
    sets: Vec<HashSet<SSAVarLabel>>,
) -> Vec<HashSet<SSAVarLabel>> {
    let mut a_set = None;
    let mut b_set = None;
    let mut new_sets = Vec::new();
    for set in sets {
        if set.contains(&a) {
            a_set = Some(set);
        } else if set.contains(&b) {
            b_set = Some(set);
        } else {
            new_sets.push(set);
        }
    }
    let a_set = match a_set {
        Some(a_set) => a_set,
        None => HashSet::from_iter(vec![a].into_iter()),
    };
    let b_set = match b_set {
        Some(b_set) => b_set,
        None => HashSet::from_iter(vec![b].into_iter()),
    };

    let new_set: HashSet<_> = a_set.union(&b_set).map(|x| x.clone()).collect();
    new_sets.push(new_set);
    new_sets
}

fn get_phi_webs(ssa_method: &cfg::CfgMethod<SSAVarLabel>) -> Vec<HashSet<SSAVarLabel>> {
    let mut phi_web: Vec<HashSet<SSAVarLabel>> = vec![];
    for block in ssa_method.blocks.values() {
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { dest, sources } => {
                    for (_, var) in sources {
                        phi_web = union(*dest, *var, phi_web);
                    }
                }
                _ => break,
            }
        }
    }
    phi_web
}

fn convert_imm_var_name(
    imm_var: &ImmVar<SSAVarLabel>,
    coallesced_name: &HashMap<SSAVarLabel, SSAVarLabel>,
    lookup: &mut HashMap<SSAVarLabel, VarLabel>,
    all_fields: &mut HashMap<u32, (CfgType, String)>,
) -> ImmVar<VarLabel> {
    match imm_var {
        ImmVar::Var(var) => ImmVar::Var(convert_name(var, coallesced_name, lookup, all_fields)),
        ImmVar::Imm(i) => ImmVar::Imm(*i),
    }
}

fn convert_name(
    var: &SSAVarLabel,
    coallesced_name: &HashMap<SSAVarLabel, SSAVarLabel>,
    lookup: &mut HashMap<SSAVarLabel, VarLabel>,
    all_fields: &mut HashMap<u32, (CfgType, String)>,
) -> VarLabel {
    let name = match coallesced_name.get(&var) {
        Some(new_name) => new_name.clone(),
        None => var.clone(),
    };

    // check if this is a global variable
    if !all_fields.contains_key(&var.name) {
        return var.name;
    }

    match lookup.get(&name) {
        Some(flat_name) => *flat_name,
        None => {
            let new_name = *all_fields.keys().max().unwrap() as usize + 1;
            lookup.insert(name.clone(), new_name as u32);

            // update all_fields
            let original_var = lookup
                .get(&SSAVarLabel {
                    name: name.name.clone(),
                    version: 1,
                })
                .unwrap();
            all_fields.insert(
                new_name as u32,
                all_fields.get(original_var).unwrap().clone(),
            );

            new_name as u32
        }
    }
}

fn destruct_instruction(
    instr: Instruction<SSAVarLabel>,
    coallesced_name: &HashMap<SSAVarLabel, SSAVarLabel>,
    lookup: &mut HashMap<SSAVarLabel, VarLabel>,
    all_fields: &mut HashMap<u32, (CfgType, String)>,
) -> Vec<Instruction<VarLabel>> {
    match instr {
        Instruction::StoreParam(_, _) | Instruction::Pop(_) | Instruction::Push(_) => panic!(),
        Instruction::ParMov(copies) => vec![Instruction::ParMov(
            copies
                .into_iter()
                .map(|om| cfg::OneMove {
                    src: convert_name(&om.src, coallesced_name, lookup, all_fields),
                    dest: convert_name(&om.dest, coallesced_name, lookup, all_fields),
                })
                .collect(),
        )],
        Instruction::ArrayAccess { dest, name, idx } => vec![Instruction::ArrayAccess {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            name: name,
            idx: convert_imm_var_name(&idx, coallesced_name, lookup, all_fields),
        }],
        Instruction::ArrayStore { source, arr, idx } => vec![Instruction::ArrayStore {
            source: convert_imm_var_name(&source, coallesced_name, lookup, all_fields),
            arr: arr,
            idx: convert_imm_var_name(&idx, coallesced_name, lookup, all_fields),
        }],
        Instruction::Call(string, args, opt_ret_val) => {
            let new_args = args
                .iter()
                .map(|arg| match arg {
                    Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                    Arg::VarArg(var) => Arg::VarArg(convert_imm_var_name(
                        &var.clone(),
                        coallesced_name,
                        lookup,
                        all_fields,
                    )),
                })
                .collect::<Vec<_>>();
            let new_ret_val = match opt_ret_val {
                Some(ret_var) => Some(convert_name(&ret_var, coallesced_name, lookup, all_fields)),
                None => None,
            };
            vec![Instruction::Call(string, new_args, new_ret_val)]
        }
        Instruction::Constant { dest, constant } => vec![Instruction::Constant {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            constant: constant,
        }],
        Instruction::LoadParam { dest, param } => vec![Instruction::LoadParam {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            param,
        }],
        Instruction::MoveOp { source, dest } => vec![Instruction::MoveOp {
            source: convert_imm_var_name(&source, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
        }],
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => vec![Instruction::ThreeOp {
            source1: convert_imm_var_name(&source1, coallesced_name, lookup, all_fields),
            source2: convert_imm_var_name(&source2, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            op: op.clone(),
        }],
        Instruction::TwoOp { source1, dest, op } => vec![Instruction::TwoOp {
            source1: convert_imm_var_name(&source1, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            op: op.clone(),
        }],
        Instruction::PhiExpr { dest, .. } => vec![Instruction::PhiExpr {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            sources: vec![], // don't care about these anymore
        }],
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(var) => vec![Instruction::Ret(Some(convert_imm_var_name(
                &var,
                coallesced_name,
                lookup,
                all_fields,
            )))],
            None => vec![Instruction::Ret(None)],
        },
        Instruction::LeftShift {
            dest,
            source,
            shift,
        } => vec![Instruction::LeftShift {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            source: convert_name(&source, coallesced_name, lookup, all_fields),
            shift: shift,
        }],
        Instruction::RightShift {
            dest,
            source,
            shift,
        } => vec![Instruction::RightShift {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            source: convert_name(&source, coallesced_name, lookup, all_fields),
            shift: shift,
        }],
        Instruction::Spill { .. } => panic!(),
        Instruction::Reload { .. } => panic!(),
    }
}

pub fn seq_method(m: &mut CfgMethod) {
    for (_, blk) in m.blocks.iter_mut() {
        blk.body = blk
            .body
            .clone() // gross
            .into_iter()
            .map(|insn| sequentialize(insn, &mut m.fields))
            .collect::<Vec<_>>()
            .concat();
    }
}

fn sequentialize(
    insn: Instruction<VarLabel>,
    all_fields: &mut HashMap<u32, (CfgType, String)>,
) -> Vec<Instruction<VarLabel>> {
    if let Instruction::ParMov(mut copies) = insn {
        // algorithm 3.6 in SSA-Based Compiler Design
        let mut instructions: Vec<Instruction<VarLabel>> = vec![];

        copies.retain(|om| om.src != om.dest);
        while copies.len() > 0 {
            let all_srcs: HashSet<VarLabel> = copies.iter().map(|om| om.src).collect();
            let all_dests: HashSet<VarLabel> = copies.iter().map(|om| om.dest).collect();
            match (&all_dests - &all_srcs).iter().next() {
                Some(b) => {
                    // retain with an impure function?  gross.
                    let mut found_it = false;
                    copies.retain(|om| {
                        if !found_it && om.dest == *b {
                            found_it = true;
                            instructions.push(Instruction::MoveOp {
                                source: ImmVar::Var(om.src),
                                dest: om.dest,
                            });
                            return false;
                        }
                        true
                    });
                }
                None => {
                    let om = copies.pop().unwrap();
                    // create new variable and add it to lookup and translations
                    let inter_name = (*all_fields.keys().max().unwrap() as usize + 1) as u32;
                    all_fields.insert(inter_name, (*all_fields.get(&om.src).unwrap()).clone());
                    instructions.push(Instruction::MoveOp {
                        source: ImmVar::Var(om.src),
                        dest: inter_name,
                    });
                    copies.push(cfg::OneMove {
                        src: inter_name,
                        dest: om.dest,
                    });
                }
            }
            copies.retain(|om| om.src != om.dest);
        }
        instructions
    } else {
        vec![insn]
    }
}

pub fn destruct_jump(
    jump: cfg::Jump<SSAVarLabel>,
    coallesced_name: &HashMap<SSAVarLabel, SSAVarLabel>,
    lookup: &mut HashMap<SSAVarLabel, VarLabel>,
    all_fields: &mut HashMap<u32, (CfgType, String)>,
) -> cfg::Jump<VarLabel> {
    match jump {
        Jump::Uncond(block) => Jump::Uncond(block),
        Jump::Nowhere => Jump::Nowhere,
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => Jump::Cond {
            source: match source {
                ImmVar::Var(v) => {
                    ImmVar::Var(convert_name(&v, coallesced_name, lookup, all_fields))
                }
                ImmVar::Imm(i) => ImmVar::Imm(i),
            },
            true_block: true_block,
            false_block: false_block,
        },
    }
}

// function to destruct conventional SSA
pub fn destruct(ssa_method: &mut cfg::CfgMethod<SSAVarLabel>) -> CfgMethod {
    let webs = get_phi_webs(&ssa_method);
    let mut coallesced_name: HashMap<SSAVarLabel, SSAVarLabel> = HashMap::new();
    let mut flat_name_lookup: HashMap<SSAVarLabel, VarLabel> = HashMap::new();

    // collapse all var labels in the web into the same name
    for web in webs {
        let new_name = web.iter().min_by_key(|c| c.version).unwrap();
        for var in web.iter() {
            coallesced_name.insert(var.clone(), new_name.clone());
        }
    }

    let mut de_ssa_method: cfg::CfgMethod<VarLabel> = cfg::CfgMethod {
        name: ssa_method.name.clone(),
        num_params: ssa_method.num_params,
        fields: ssa_method.fields.clone(),
        blocks: HashMap::new(),
        return_type: ssa_method.return_type.clone(),
    };

    // need to initalize lookup with 0 and 1. 0 represent variables that are never assigned (global vars or method params). 1 represent the first time that variables are defined.
    for i in ssa_method.fields.keys() {
        flat_name_lookup.insert(
            SSAVarLabel {
                name: *i,
                version: 1,
            },
            *i,
        );
        flat_name_lookup.insert(
            SSAVarLabel {
                name: *i,
                version: 0,
            },
            *i,
        );
    }

    for (id, block) in ssa_method.blocks.iter_mut() {
        // skip phi instructions, and rewrite phi webs with same name
        let mut new_instructions: Vec<Instruction<VarLabel>> = vec![];
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { .. } => (),
                _ => new_instructions.extend(destruct_instruction(
                    instruction.clone(),
                    &coallesced_name,
                    &mut flat_name_lookup,
                    &mut ssa_method.fields,
                )),
            }
        }

        let mut new_block: BasicBlock<VarLabel> = cfg::BasicBlock {
            body: vec![],
            jump_loc: destruct_jump(
                block.jump_loc.clone(),
                &coallesced_name,
                &mut flat_name_lookup,
                &mut ssa_method.fields,
            ),
        };

        new_block.body = new_instructions;
        de_ssa_method.blocks.insert(*id, new_block);
    }

    de_ssa_method.fields = ssa_method.fields.clone();
    de_ssa_method
}

use crate::cfg_build;
use cfg::BlockLabel;

fn change_parent<T>(old: BlockLabel, new: BlockLabel, blk_insns: &mut Vec<Instruction<T>>) {
    for insn in blk_insns.iter_mut() {
        if let Instruction::PhiExpr { sources, .. } = insn {
            for (par, _) in sources.iter_mut() {
                if *par == old {
                    *par = new;
                }
            }
        } else {
            break;
        }
    }
}

// algorithm 3.5 of SSA book
pub fn split_crit_edges(method: &mut cfg::CfgMethod<SSAVarLabel>) {
    let cfg = &mut method.blocks;
    let parents = cfg_build::get_parents(cfg);
    let all_lbls: HashSet<BlockLabel> = cfg.keys().map(|x| *x).collect();
    for lbl in all_lbls {
        let mut blk = cfg.get(&lbl).unwrap().clone();
        for parent in parents.get(&lbl).unwrap().iter() {
            let new_par_name = cfg.keys().max().unwrap_or(&0) + 1;
            match &mut cfg.get_mut(parent).unwrap().jump_loc {
                Jump::Nowhere => panic!("parent jumping nowhere?"),
                Jump::Uncond(_) => (),
                Jump::Cond {
                    true_block: t,
                    false_block: f,
                    ..
                } => {
                    let new_par = BasicBlock::<SSAVarLabel> {
                        body: vec![],
                        jump_loc: Jump::Uncond(lbl),
                    };
                    change_parent(*parent, new_par_name, &mut blk.body);
                    if *t == lbl {
                        *t = new_par_name;
                    } else if *f == lbl {
                        *f = new_par_name;
                    } else {
                        panic!("oops");
                    }
                    cfg.insert(new_par_name, new_par);
                }
            }
        }

        // given a parent of blk, what are the corresponding copies
        let mut copies: HashMap<BlockLabel, Vec<cfg::OneMove<SSAVarLabel>>> = HashMap::new();
        for insn in blk.body.iter_mut() {
            if let Instruction::PhiExpr { sources, .. } = insn {
                for (par, var) in sources {
                    let fresh_var = method.fields.keys().max().unwrap_or(&0) + 1;
                    method.fields.insert(
                        fresh_var as u32,
                        method.fields.get(&var.name).unwrap().clone(),
                    );
                    let fresh_var = SSAVarLabel {
                        name: fresh_var,
                        version: 1,
                    };

                    let par_copies = copies.entry(*par).or_insert(vec![]);
                    par_copies.push(cfg::OneMove {
                        src: *var,
                        dest: fresh_var,
                    });
                    *var = fresh_var;
                }
            } else {
                break;
            }
        }
        for (par, parcops) in copies {
            if !cfg.contains_key(&par) {
                continue; // block was removed by the optimizations
            };
            cfg.get_mut(&par)
                .unwrap()
                .body
                .push(Instruction::ParMov(parcops));
        }
        // another annoying artifact of my inability to convince a hashmap that two values are different...
        cfg.insert(lbl, blk);
    }
}
