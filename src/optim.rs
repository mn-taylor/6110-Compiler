use crate::cfg;
use crate::cfg::{Arg, BasicBlock, Instruction};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, SSAVarLabel};
use std::collections::HashMap;

// find first variable not in coppy lookup
fn check_copy(var: SSAVarLabel, copy_lookup: &HashMap<SSAVarLabel, SSAVarLabel>) -> SSAVarLabel {
    match copy_lookup.get(&var) {
        Some(original) => check_copy(original.clone(), copy_lookup),
        None => var,
    }
}

fn prop_copies(
    instr: Instruction<SSAVarLabel>,
    copy_lookup: &HashMap<SSAVarLabel, SSAVarLabel>,
) -> Instruction<SSAVarLabel> {
    match instr {
        // can be smarter and not call check copy when we know there cant be a copy
        Instruction::ArrayAccess { dest, name, idx } => Instruction::ArrayAccess {
            dest: dest,
            name: check_copy(name, copy_lookup),
            idx: check_copy(idx, copy_lookup),
        },
        Instruction::ArrayStore { source, arr, idx } => Instruction::ArrayStore {
            source: check_copy(source, copy_lookup),
            arr: arr,
            idx: check_copy(idx, copy_lookup),
        },
        Instruction::Call(string, args, opt_ret_val) => {
            let new_args = args
                .iter()
                .map(|arg| match arg {
                    Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                    Arg::VarArg(var) => Arg::VarArg(check_copy(var.clone(), copy_lookup)),
                })
                .collect::<Vec<_>>();
            let new_ret_val = match opt_ret_val {
                Some(ret_var) => Some(check_copy(ret_var, copy_lookup)),
                None => None,
            };
            Instruction::Call(string, new_args, new_ret_val)
        }
        Instruction::Constant { dest, constant } => Instruction::Constant {
            dest: dest.clone(),
            constant: constant,
        },
        Instruction::MoveOp { source, dest } => Instruction::MoveOp {
            source: check_copy(source, copy_lookup),
            dest: dest,
        },
        Instruction::PhiExpr { dest, sources } => Instruction::PhiExpr {
            dest: dest,
            sources: sources
                .iter()
                .map(|(block_id, var)| (*block_id, check_copy(var.clone(), copy_lookup)))
                .collect::<Vec<_>>(),
        },
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(val) => Instruction::Ret(Some(check_copy(val, copy_lookup))),
            None => Instruction::Ret(None),
        },
        Instruction::TwoOp { source1, dest, op } => Instruction::TwoOp {
            source1: check_copy(source1, copy_lookup),
            dest: dest,
            op: op.clone(),
        },
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => Instruction::ThreeOp {
            source1: check_copy(source1, copy_lookup),
            source2: check_copy(source2, copy_lookup),
            dest: dest,
            op: op.clone(),
        },
    }
}

pub fn copy_propagation(method: &mut cfg::CfgMethod<SSAVarLabel>) -> cfg::CfgMethod<SSAVarLabel> {
    let mut new_method: cfg::CfgMethod<SSAVarLabel> = cfg::CfgMethod {
        name: method.name.clone(),
        params: method.params.clone(),
        blocks: HashMap::new(),
        fields: method.fields.clone(),
        return_type: method.return_type.clone(),
    };

    let g = get_graph(method);
    let dom_sets = dominator_sets(0, &g);
    let dom_tree = dominator_tree(method, &dom_sets);

    // Iterate through cfg in pre-order traversal. Should maximize ability to place copies, i think!!
    let mut copy_lookup: &mut HashMap<SSAVarLabel, SSAVarLabel> = &mut HashMap::new();
    let mut agenda: Vec<usize> = vec![0];

    while agenda.len() != 0 {
        let curr = agenda.pop().unwrap();
        let curr_block = method.blocks.get(&curr).unwrap();

        let mut new_instructions: Vec<Instruction<SSAVarLabel>> = vec![];
        for instruction in curr_block.body.iter() {
            // first propogate copies
            let new_instr = prop_copies(instruction.clone(), copy_lookup);

            // check for new copies and update table
            match new_instr {
                Instruction::MoveOp { source, dest } => {
                    copy_lookup.insert(dest, source);
                }
                _ => {
                    new_instructions.push(new_instr); // We can get rid of all copy instructions!! I think
                }
            }
        }

        // make new block and add it to new_method
        let mut new_block: BasicBlock<SSAVarLabel> = cfg::BasicBlock {
            parents: curr_block.parents.clone(),
            block_id: curr_block.block_id.clone(),
            body: new_instructions,
            jump_loc: curr_block.jump_loc.clone(),
        };

        new_method.blocks.insert(curr, new_block);

        match dom_tree.get(&curr) {
            Some(children) => agenda.extend(children.iter().collect::<Vec<_>>()),
            None => {}
        }
    }

    // should take one final pass through the instructions make sure all copies are propagated.
    // Somewhat concerned about copies being propagated across the dominance frontier, one more forward pass through the method with no move instructions would fix this.

    new_method
}
