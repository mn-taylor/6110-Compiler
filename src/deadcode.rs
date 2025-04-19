use crate::cfg::{self, Jump};
use crate::cfg::{Arg, BasicBlock, ImmVar, Instruction};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, SSAVarLabel};
use maplit::{hashmap, hashset};
use std::collections::HashSet;

fn get_sources(instruction: Instruction<SSAVarLabel>) -> HashSet<SSAVarLabel> {
    match instruction {
        Instruction::PhiExpr { dest, sources } => sources.iter().map(|(_, var)| *var).collect(),
        Instruction::ArrayAccess { dest, name, idx } => {
            let mut set = hashset! {};
            match idx {
                ImmVar::Var(v) => {
                    set.insert(v);
                }
                _ => {}
            }

            set
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let mut set = hashset! {};
            match idx {
                ImmVar::Var(v) => {
                    set.insert(v);
                }
                _ => {}
            }
            match source {
                ImmVar::Var(v) => {
                    set.insert(v);
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
        Instruction::MoveOp { source, dest } => {
            let mut set = hashset! {};
            match source {
                ImmVar::Var(v) => {
                    set.insert(v);
                }
                _ => {}
            }

            set
        }
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(var) => match var {
                ImmVar::Var(v) => hashset! {v},
                _ => hashset! {},
            },
            None => hashset! {},
        },
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => {
            let mut set = hashset! {};

            match source1 {
                ImmVar::Var(v) => {
                    set.insert(v);
                }
                _ => {}
            }
            match source2 {
                ImmVar::Var(v) => {
                    set.insert(v);
                }
                _ => {}
            }

            set
        }
        Instruction::TwoOp { source1, dest, op } => match source1 {
            ImmVar::Var(v) => hashset! {v},
            _ => hashset! {},
        },
        _ => hashset! {}, // Excludes phis and constant loads
    }
}

fn get_jump_sources(j: Jump<SSAVarLabel>) -> HashSet<SSAVarLabel> {
    match j {
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => match source {
            ImmVar::Var(v) => hashset! {v},
            ImmVar::Imm(i) => hashset! {},
        },
        _ => hashset! {},
    }
}

fn get_dest(instruction: Instruction<SSAVarLabel>) -> Option<SSAVarLabel> {
    match instruction {
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
        params: m.params.clone(),
        blocks: hashmap! {},
        fields: m.fields.clone(),
        return_type: m.return_type.clone(),
    };

    // compute the set of all variables that are used
    let mut all_used_vars: HashSet<SSAVarLabel> = hashset! {};
    for (_, block) in m.blocks.iter() {
        for instruction in block.body.iter() {
            all_used_vars = all_used_vars
                .union(&get_sources(instruction.clone()))
                .cloned()
                .collect()
        }

        all_used_vars = all_used_vars
            .union(&get_jump_sources(block.jump_loc.clone()))
            .cloned()
            .collect()
    }

    // remove definitions of non global variables
    for (id, block) in m.blocks.iter() {
        let mut new_block = block.clone();

        let mut new_instructions = vec![];
        for instruction in block.body.iter() {
            match get_dest(instruction.clone()) {
                Some(dest_var) => {
                    // add instruction if dest is used later or dest is global
                    if !m.fields.contains_key(&dest_var.name) || all_used_vars.contains(&dest_var) {
                        new_instructions.push(instruction.clone());
                    } else {
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

    return new_method;
}
