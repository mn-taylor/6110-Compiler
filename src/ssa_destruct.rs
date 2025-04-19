use crate::cfg::{self, ImmVar, OneMove};
use crate::cfg::{Arg, BasicBlock, CfgType, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use crate::ssa_construct::SSAVarLabel;
use maplit::{hashmap, hashset};
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

            // println!(
            //     "BEFORE field len: {}, coallesced name: {}",
            //     all_fields.len(),
            //     new_name
            // );

            // println!("coalesced: {:?}", coallesced_name.keys());

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

            // println!(
            //     "AFTER field len: {}, coallesced name: {}",
            //     all_fields.len(),
            //     new_name
            // );

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
        Instruction::ParMov(copies) => {
            // algorithm 3.6 in SSA-Based Compiler Design
            let mut all_srcs: HashSet<SSAVarLabel> = hashset! {};
            let mut all_dests: HashSet<SSAVarLabel> = hashset! {};
            let mut instructions: Vec<Instruction<VarLabel>> = vec![];
            let mut inter_instructions: Vec<Instruction<VarLabel>> = vec![];
            copies.iter().for_each(|om| {
                all_srcs.insert(om.src.clone());
                all_dests.insert(om.dest.clone());
            });

            copies.iter().for_each(|om| {
                if true {
                    // should modify conditional so that we do not have to always do t1 < temp < t2 when doing parallel moves.
                    let source = convert_name(&om.src, coallesced_name, lookup, all_fields);

                    // create new variable and add it to lookup and translations
                    let inter_name = (*all_fields.keys().max().unwrap() as usize + 1) as u32;
                    let inter_ssa_name = SSAVarLabel {
                        name: inter_name,
                        version: 1,
                    };
                    lookup.insert(inter_ssa_name, inter_name);
                    all_fields.insert(inter_name, (*all_fields.get(&source).unwrap()).clone());

                    let dest = convert_name(&om.dest, coallesced_name, lookup, all_fields);

                    inter_instructions.push(Instruction::MoveOp {
                        source: ImmVar::Var(source),
                        dest: inter_name,
                    });
                    instructions.push(Instruction::MoveOp {
                        source: ImmVar::Var(inter_name),
                        dest: dest,
                    });
                } else {
                    instructions.push(Instruction::MoveOp {
                        source: ImmVar::Var(convert_name(
                            &om.src,
                            coallesced_name,
                            lookup,
                            all_fields,
                        )),
                        dest: convert_name(&om.dest, coallesced_name, lookup, all_fields),
                    })
                }
            });

            inter_instructions.extend(instructions);
            return inter_instructions;
        }
        Instruction::ArrayAccess { dest, name, idx } => vec![Instruction::ArrayAccess {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            name: convert_name(&name, coallesced_name, lookup, all_fields),
            idx: convert_imm_var_name(&idx, coallesced_name, lookup, all_fields),
        }],
        Instruction::ArrayStore { source, arr, idx } => vec![Instruction::ArrayStore {
            source: convert_imm_var_name(&source, coallesced_name, lookup, all_fields),
            arr: convert_name(&arr, coallesced_name, lookup, all_fields),
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

    println!("phi webs: {:?}\n\n", webs);

    // collapse all var labels in the web into the same name
    for web in webs {
        let new_name = web.iter().min_by_key(|c| c.version).unwrap();
        for var in web.iter() {
            coallesced_name.insert(var.clone(), new_name.clone());
        }
    }

    println!("new_names {:?}\n\n", coallesced_name);
    let mut de_ssa_method: cfg::CfgMethod<VarLabel> = cfg::CfgMethod {
        name: ssa_method.name.clone(),
        params: ssa_method.params.clone(),
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
            parents: block.parents.clone(),
            block_id: block.block_id.clone(),
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
    cfg_build::get_parents(cfg);
    let all_lbls: HashSet<BlockLabel> = cfg.keys().map(|x| *x).collect();
    for lbl in all_lbls {
        let mut blk = cfg.get(&lbl).unwrap().clone();
        for parent in blk.parents.iter() {
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
                        parents: vec![],
                        block_id: new_par_name,
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
            println!("instruction before: {}", insn);
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
            println!("instruction after: {}", insn);
        }
        for (par, parcops) in copies {
            cfg.get_mut(&par)
                .unwrap()
                .body
                .push(Instruction::ParMov(parcops));
        }
        // another annoying artifact of my inability to convince a hashmap that two values are different...
        cfg.insert(lbl, blk);
    }
}
