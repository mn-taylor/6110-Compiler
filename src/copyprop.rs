use crate::cfg;
use crate::cfg::{Arg, BasicBlock, ImmVar, Instruction};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, SSAVarLabel};
use std::collections::HashMap;

// find first variable not in copy lookup
fn check_copy(var: SSAVarLabel, copy_lookup: &HashMap<SSAVarLabel, SSAVarLabel>) -> SSAVarLabel {
    match copy_lookup.get(&var) {
        Some(original) => check_copy(original.clone(), copy_lookup),
        None => var,
    }
}

fn check_imm_var_copy(
    imm_var: ImmVar<SSAVarLabel>,
    copy_lookup: &HashMap<SSAVarLabel, SSAVarLabel>,
) -> ImmVar<SSAVarLabel> {
    match imm_var {
        ImmVar::Var(var) => ImmVar::Var(check_copy(var, copy_lookup)),
        ImmVar::Imm(i) => ImmVar::Imm(i),
    }
}

fn prop_copies(
    instr: Instruction<SSAVarLabel>,
    copy_lookup: &HashMap<SSAVarLabel, SSAVarLabel>,
) -> Instruction<SSAVarLabel> {
    match instr {
        Instruction::NoArgsCall(_, _) => instr,
        Instruction::StoreParam(dest, arg) => Instruction::StoreParam(
            dest,
            match arg {
                Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                Arg::VarArg(var) => Arg::VarArg(check_imm_var_copy(var.clone(), copy_lookup)),
            },
        ),
        Instruction::ParMov(_) => todo!(),
        // can be smarter and not call check copy when we know there cant be a copy
        Instruction::ArrayAccess { dest, name, idx } => Instruction::ArrayAccess {
            dest,
            name: name,
            idx: check_imm_var_copy(idx, copy_lookup),
        },
        Instruction::ArrayStore { source, arr, idx } => Instruction::ArrayStore {
            source: check_imm_var_copy(source, copy_lookup),
            arr,
            idx: check_imm_var_copy(idx, copy_lookup),
        },
        Instruction::Call(string, args, opt_ret_val) => {
            let new_args = args
                .iter()
                .map(|arg| match arg {
                    Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                    Arg::VarArg(var) => Arg::VarArg(check_imm_var_copy(var.clone(), copy_lookup)),
                })
                .collect::<Vec<_>>();
            let new_ret_val = match opt_ret_val {
                Some(ret_var) => Some(check_copy(ret_var, copy_lookup)),
                None => None,
            };
            Instruction::Call(string, new_args, new_ret_val)
        }
        Instruction::Constant { dest, constant } => Instruction::Constant { dest, constant },
        Instruction::LoadParam { dest, param } => Instruction::LoadParam { dest, param },
        Instruction::MoveOp { source, dest } => Instruction::MoveOp {
            source: check_imm_var_copy(source, copy_lookup),
            dest,
        },
        Instruction::PhiExpr { dest, sources } => Instruction::PhiExpr {
            dest,
            sources: sources
                .iter()
                .map(|(block_id, var)| (*block_id, check_copy(*var, copy_lookup)))
                .collect::<Vec<_>>(),
        },
        Instruction::Ret(opt_ret_val) => match opt_ret_val {
            Some(val) => Instruction::Ret(Some(check_imm_var_copy(val, copy_lookup))),
            None => Instruction::Ret(None),
        },
        Instruction::TwoOp { source1, dest, op } => Instruction::TwoOp {
            source1: check_imm_var_copy(source1, copy_lookup),
            dest,
            op: op.clone(),
        },
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => Instruction::ThreeOp {
            source1: check_imm_var_copy(source1, copy_lookup),
            source2: check_imm_var_copy(source2, copy_lookup),
            dest,
            op: op.clone(),
        },
        Instruction::Spill { .. } => panic!(),
        Instruction::Reload { .. } => panic!(),
    }
}

pub fn copy_propagation(method: &mut cfg::CfgMethod<SSAVarLabel>) -> cfg::CfgMethod<SSAVarLabel> {
    let mut new_method: cfg::CfgMethod<SSAVarLabel> = cfg::CfgMethod {
        name: method.name.clone(),
        num_params: method.num_params,
        blocks: HashMap::new(),
        fields: method.fields.clone(),
        return_type: method.return_type.clone(),
    };

    let g = get_graph(method);
    let dom_sets = dominator_sets(0, &g);
    let dom_tree = dominator_tree(method, &dom_sets);

    // Iterate through cfg in pre-order traversal. Should maximize ability to place copies, i think!!
    let copy_lookup: &mut HashMap<SSAVarLabel, SSAVarLabel> = &mut HashMap::new();
    let mut agenda: Vec<usize> = vec![0];

    let mut num_iters = 0;

    while agenda.len() != 0 && num_iters < 2 {
        let curr = agenda.pop().unwrap();
        let curr_block = method.blocks.get(&curr).unwrap();

        let mut new_instructions: Vec<Instruction<SSAVarLabel>> = vec![];
        for instruction in curr_block.body.iter() {
            // first propogate copies
            let new_instr = prop_copies(instruction.clone(), copy_lookup);

            // check for new copies and update table
            match &new_instr {
                Instruction::MoveOp { source, dest } => {
                    // don't propagate global variables
                    if method.fields.get(&dest.name.clone()).is_none() {
                        new_instructions.push(Instruction::MoveOp {
                            source: source.clone(),
                            dest: dest.clone(),
                        });
                    } else {
                        // idea: leave old move instructions to simplify code. Then clean up with dead code elimination.
                        match source.clone() {
                            ImmVar::Var(s) => {
                                copy_lookup.insert(dest.clone(), s.clone());
                            }
                            _ => {}
                        };
                        //  new_instructions.push(new_instr.clone());
                    }
                }
                _ => {
                    new_instructions.push(new_instr); // We can get rid of all copy instructions!! I think
                }
            }
        }

        let new_jmp = match &curr_block.jump_loc {
            cfg::Jump::Cond {
                source,
                true_block,
                false_block,
            } => cfg::Jump::Cond {
                true_block: *true_block,
                false_block: *false_block,
                source: match source {
                    ImmVar::Var(v) => ImmVar::Var(check_copy(*v, copy_lookup)),
                    ImmVar::Imm(i64) => ImmVar::Imm(*i64),
                },
            },
            _ => curr_block.jump_loc.clone(),
        };

        // make new block and add it to new_method
        let new_block: BasicBlock<SSAVarLabel> = cfg::BasicBlock {
            body: new_instructions,
            jump_loc: new_jmp,
        };

        new_method.blocks.insert(curr, new_block);

        match dom_tree.get(&curr) {
            Some(children) => agenda.extend(children.iter().collect::<Vec<_>>()),
            None => {}
        }

        if agenda.len() == 0 {
            num_iters += 1;
            agenda.push(0);
        }
    }

    // should take one final pass through the instructions make sure all copies are propagated.
    // Somewhat concerned about copies being propagated across the dominance frontier, one more forward pass through the method with no move instructions would fix this.

    new_method
}
