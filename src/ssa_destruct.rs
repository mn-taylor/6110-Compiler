use crate::cfg;
use crate::cfg::{Arg, BasicBlock, CfgType, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use crate::ssa_construct::SSAVarLabel;
use std::collections::HashMap;
use std::collections::HashSet;

// fn get_phi_webs
fn get_phi_webs(ssa_method: &cfg::CfgMethod<SSAVarLabel>) -> Vec<HashSet<SSAVarLabel>> {
    let mut phi_web: HashMap<SSAVarLabel, HashSet<SSAVarLabel>> = HashMap::new();

    // initalize each phi web
    for block in ssa_method.blocks.values() {
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { dest, sources } => {
                    phi_web.insert(
                        dest.clone(),
                        [dest].iter().map(|c| (*c).clone()).collect::<HashSet<_>>(),
                    );
                    let _ = sources
                        .iter()
                        .map(|(_, var)| {
                            phi_web
                                .insert(var.clone(), [var].iter().map(|c| (*c).clone()).collect())
                        })
                        .collect::<Vec<_>>();
                }
                _ => break,
            }
        }
    }

    for block in ssa_method.blocks.values() {
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { dest, sources } => {
                    for (_, var) in sources {
                        let new_web: HashSet<SSAVarLabel> = phi_web
                            .get(dest)
                            .unwrap()
                            .union(phi_web.get(var).unwrap())
                            .cloned()
                            .collect();
                        phi_web.insert(dest.clone(), new_web.clone());
                        phi_web.insert(var.clone(), new_web.clone());
                    }
                }
                _ => break,
            }
        }
    }

    let mut webs: Vec<HashSet<SSAVarLabel>> = vec![];
    // get rid of redundant webs
    while phi_web.len() != 0 {
        let (_, set) = phi_web.iter_mut().next().unwrap();
        webs.push(set.clone());

        for ssa_var in set.clone().iter() {
            phi_web.remove(ssa_var);
        }
    }

    return webs;
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

            println!("original name {}, lookup: {:?}", name.name, lookup.keys());

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
) -> Instruction<VarLabel> {
    match instr {
        Instruction::ParMov(_) => panic!("these should not be here yet"),
        Instruction::ArrayAccess { dest, name, idx } => Instruction::ArrayAccess {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            name: convert_name(&name, coallesced_name, lookup, all_fields),
            idx: convert_name(&idx, coallesced_name, lookup, all_fields),
        },
        Instruction::ArrayStore { source, arr, idx } => Instruction::ArrayStore {
            source: convert_name(&source, coallesced_name, lookup, all_fields),
            arr: convert_name(&arr, coallesced_name, lookup, all_fields),
            idx: convert_name(&idx, coallesced_name, lookup, all_fields),
        },
        Instruction::Call(string, args, opt_ret_val) => {
            let new_args = args
                .iter()
                .map(|arg| match arg {
                    Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                    Arg::VarArg(var) => Arg::VarArg(convert_name(
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
            Instruction::Call(string, new_args, new_ret_val)
        }
        Instruction::Constant { dest, constant } => Instruction::Constant {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            constant: constant,
        },
        Instruction::MoveOp { source, dest } => Instruction::MoveOp {
            source: convert_name(&source, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
        },
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => Instruction::ThreeOp {
            source1: convert_name(&source1, coallesced_name, lookup, all_fields),
            source2: convert_name(&source2, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            op: op.clone(),
        },
        Instruction::TwoOp { source1, dest, op } => Instruction::TwoOp {
            source1: convert_name(&source1, coallesced_name, lookup, all_fields),
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            op: op.clone(),
        },
        Instruction::PhiExpr { dest, .. } => Instruction::PhiExpr {
            dest: convert_name(&dest, coallesced_name, lookup, all_fields),
            sources: vec![], // don't care about these anymore
        },
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(var) => Instruction::Ret(Some(convert_name(
                &var,
                coallesced_name,
                lookup,
                all_fields,
            ))),
            None => Instruction::Ret(None),
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
            source: convert_name(&source, coallesced_name, lookup, all_fields),
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
        let _ = web
            .iter()
            .map(|var| coallesced_name.insert(var.clone(), new_name.clone()))
            .collect::<Vec<_>>();
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
                _ => new_instructions.push(destruct_instruction(
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

// function to convert Non-conventional SSA into convential SSA
