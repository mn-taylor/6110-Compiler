use crate::cfg::{self, Jump};
use crate::cfg::{Arg, Instruction};
use crate::ssa_construct::SSAVarLabel;
use maplit::{hashmap, hashset};
use std::collections::HashSet;

fn get_sources(instruction: Instruction<SSAVarLabel>) -> HashSet<SSAVarLabel> {
    match instruction {
        Instruction::PhiExpr { sources, .. } => sources.iter().map(|(_, var)| *var).collect(),
        Instruction::ArrayAccess { idx, .. } => hashset! {idx},
        Instruction::ArrayStore { source, idx, .. } => hashset! {source, idx},
        Instruction::Call(_, args, _) => {
            let mut sources = hashset! {};
            args.iter().for_each(|arg| match arg {
                Arg::VarArg(var) => {
                    sources.insert(var.clone());
                }
                _ => {}
            });
            sources
        }
        Instruction::MoveOp { source, .. } => hashset! {source},
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(var) => hashset! {var},
            None => hashset! {},
        },
        Instruction::ThreeOp {
            source1, source2, ..
        } => hashset! {source1, source2},
        Instruction::TwoOp { source1, .. } => hashset! {source1},
        _ => hashset! {}, // Excludes phis and constant loads
    }
}

fn get_jump_sources(j: Jump<SSAVarLabel>) -> HashSet<SSAVarLabel> {
    match j {
        Jump::Cond { source, .. } => hashset! {source},
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
