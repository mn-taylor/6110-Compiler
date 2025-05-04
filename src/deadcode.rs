use crate::cfg::{self, Jump};
use crate::cfg::{Arg, ImmVar, Instruction};
use crate::scan::Sum;
use crate::ssa_construct::SSAVarLabel;
use maplit::{hashmap, hashset};
use std::collections::HashSet;
use std::hash::Hash;

pub fn get_sources<VarLabel: Eq + Hash + Copy>(
    instruction: &Instruction<VarLabel>,
) -> HashSet<VarLabel> {
    match instruction {
        Instruction::PhiExpr { sources, .. } => sources
            .iter()
            .map(|(_, var)| match var {
                Sum::Inl(v) => *v,
                _ => panic!("Should not do register allocation before optimizations"),
            })
            .collect(),
        Instruction::ArrayAccess { idx, .. } => {
            let mut set = hashset! {};
            match idx {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }

            set
        }
        Instruction::ArrayStore { source, idx, .. } => {
            let mut set = hashset! {};
            match idx {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }
            match source {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }
            set
        }
        Instruction::Call(_, args, _) => {
            let mut sources = hashset! {};
            args.iter().for_each(|arg| match arg {
                Arg::VarArg(var) => match var {
                    ImmVar::Var(v) => {
                        sources.insert(v.clone());
                    }
                    _ => {}
                },
                _ => {}
            });
            sources
        }
        Instruction::MoveOp { source, .. } => {
            let mut set = hashset! {};
            match source {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }

            set
        }
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(var) => match var {
                ImmVar::Var(v) => hashset! {*v},
                _ => hashset! {},
            },
            None => hashset! {},
        },
        Instruction::ThreeOp {
            source1, source2, ..
        } => {
            let mut set = hashset! {};

            match source1 {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }
            match source2 {
                ImmVar::Var(v) => {
                    set.insert(*v);
                }
                _ => {}
            }

            set
        }
        Instruction::TwoOp { source1, .. } => match source1 {
            ImmVar::Var(v) => hashset! {*v},
            _ => hashset! {},
        },
        _ => hashset! {}, // Excludes phis and constant loads
    }
}

pub fn get_jump_sources<VarLabel: Eq + Hash + Copy>(j: &Jump<VarLabel>) -> HashSet<VarLabel> {
    match j {
        Jump::Cond { source, .. } => match source {
            ImmVar::Var(v) => hashset! {*v},
            ImmVar::Imm(_) => hashset! {},
        },
        _ => hashset! {},
    }
}

pub fn get_dest<T: Copy>(instruction: &Instruction<T>) -> Option<T> {
    match *instruction {
        Instruction::PhiExpr { dest, .. } => Some(dest),
        Instruction::ArrayAccess { dest, .. } => Some(dest),
        Instruction::Call(_, _, _) => None, // We always want to call functions whether or not their return values are used, because they may modify global variables.
        Instruction::Constant { dest, .. } => Some(dest),
        Instruction::MoveOp { dest, .. } => Some(dest),
        Instruction::ThreeOp { dest, .. } => Some(dest),
        Instruction::TwoOp { dest, .. } => Some(dest),
        _ => None, // Excludes Array Stores, Returns, and Parallel Moves
    }
}

pub fn dead_code_elimination(m: &mut cfg::CfgMethod<SSAVarLabel>) -> cfg::CfgMethod<SSAVarLabel> {
    let mut new_method: cfg::CfgMethod<SSAVarLabel> = cfg::CfgMethod {
        name: m.name.clone(),
        num_params: m.num_params,
        blocks: hashmap! {},
        fields: m.fields.clone(),
        return_type: m.return_type.clone(),
    };

    // compute the set of all variables that are used
    let mut all_used_vars: HashSet<SSAVarLabel> = hashset! {};
    for (_, block) in m.blocks.iter() {
        for instruction in block.body.iter() {
            all_used_vars = all_used_vars
                .union(&get_sources(instruction))
                .cloned()
                .collect()
        }

        all_used_vars = all_used_vars
            .union(&get_jump_sources(&block.jump_loc))
            .cloned()
            .collect()
    }

    let mut removed_var = false;
    // remove definitions of non global variables that are never sources
    for (id, block) in m.blocks.iter() {
        let mut new_block = block.clone();

        let mut new_instructions = vec![];
        for instruction in block.body.iter() {
            match get_dest(instruction) {
                Some(dest_var) => {
                    // add instruction if dest is used later or dest is global
                    if !m.fields.contains_key(&dest_var.name) || all_used_vars.contains(&dest_var) {
                        new_instructions.push(instruction.clone());
                    } else {
                        removed_var = true;
                    }
                }
                None => {
                    new_instructions.push(instruction.clone());
                }
            }
        }
        new_block.body = new_instructions;

        new_method.blocks.insert(*id, new_block);
    }

    // should probably do some stack based removal instead of recursive calls so that we dont have to iterate through all the blocks multiple times
    if removed_var {
        return dead_code_elimination(&mut new_method);
    } else {
        return new_method;
    }
}
