use crate::cfg;
use crate::cfg::{Arg, BasicBlock, ImmVar, Instruction, IsImmediate, Jump};
use crate::cfg_build::get_parents;
use crate::ir::{Bop, UnOp};
use crate::scan::{AddOp, EqOp, MulOp, RelOp, Sum};
use crate::ssa_construct::{prune_method, SSAVarLabel};
use maplit::hashset;
use std::collections::{HashMap, HashSet};

fn binary_operation(imm1: i64, imm2: i64, bop: &Bop) -> i64 {
    match bop {
        Bop::AddBop(aop) => match aop {
            AddOp::Add => imm1 + imm2,
            AddOp::Sub => imm1 - imm2,
        },
        Bop::MulBop(mop) => match mop {
            MulOp::Mul => imm1 * imm2,
            MulOp::Div => imm1 / imm2,
            MulOp::Mod => imm1 % imm2,
        },
        Bop::RelBop(rop) => match rop {
            RelOp::Lt => (imm1 < imm2) as i64,
            RelOp::Le => (imm1 <= imm2) as i64,
            RelOp::Ge => (imm1 >= imm2) as i64,
            RelOp::Gt => (imm1 > imm2) as i64,
        },
        Bop::EqBop(eop) => match eop {
            EqOp::Eq => (imm1 == imm2) as i64,
            EqOp::Neq => (imm1 != imm2) as i64,
        },
        _ => panic!("And and Or should not be allowed in the cfg"),
    }
}

fn simplify(i: Instruction<SSAVarLabel>) -> Instruction<SSAVarLabel> {
    match &i {
        Instruction::MoveOp { source, dest } => {
            if source.is_immediate() {
                let imm = match source {
                    ImmVar::Imm(i) => i,
                    ImmVar::Var(_) => panic!(),
                };

                Instruction::Constant {
                    dest: *dest,
                    constant: *imm,
                }
            } else {
                i
            }
        }
        Instruction::TwoOp { source1, dest, op } => {
            if source1.is_immediate() {
                let imm = match source1 {
                    ImmVar::Imm(i) => i,
                    ImmVar::Var(_) => panic!(),
                };

                let new_imm = match op {
                    UnOp::Neg => -imm,
                    UnOp::Not => match imm {
                        0 => 1,
                        1 => 0,
                        _ => panic!("not acting on non-boolean value"),
                    },
                    UnOp::IntCast => (*imm as i32) as i64,
                    UnOp::LongCast => *imm,
                };

                Instruction::Constant {
                    dest: *dest,
                    constant: new_imm,
                }
            } else {
                i
            }
        }
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => {
            if source1.is_immediate() && source2.is_immediate() {
                let imm1 = match source1 {
                    ImmVar::Imm(i) => i,
                    ImmVar::Var(_) => panic!(),
                };

                let imm2 = match source2 {
                    ImmVar::Imm(i) => i,
                    ImmVar::Var(_) => panic!(),
                };

                Instruction::Constant {
                    dest: *dest,
                    constant: binary_operation(*imm1, *imm2, &op),
                }
            } else {
                i
            }
        }
        _ => i,
    }
}

fn check_const(
    src: ImmVar<SSAVarLabel>,
    const_lookup: &HashMap<ImmVar<SSAVarLabel>, ImmVar<SSAVarLabel>>,
) -> ImmVar<SSAVarLabel> {
    match const_lookup.get(&src) {
        Some(var) => var.clone(),
        None => src,
    }
}

fn prop_const(
    i: Instruction<SSAVarLabel>,
    const_lookup: &HashMap<ImmVar<SSAVarLabel>, ImmVar<SSAVarLabel>>,
) -> Instruction<SSAVarLabel> {
    let new_instruction = match i {
        Instruction::ArrayAccess { dest, name, idx } => Instruction::ArrayAccess {
            dest: dest,
            name: name,
            idx: check_const(idx, const_lookup),
        },
        Instruction::ArrayStore { source, arr, idx } => Instruction::ArrayStore {
            source: check_const(source, const_lookup),
            arr: arr,
            idx: check_const(idx, const_lookup),
        },
        Instruction::Call(func_name, args, opt_return_val) => {
            let new_args: Vec<Arg<SSAVarLabel>> = args
                .iter()
                .map(|arg| match arg {
                    Arg::VarArg(imm_var) => Arg::VarArg(check_const(imm_var.clone(), const_lookup)),
                    Arg::StrArg(s) => Arg::StrArg(s.to_string()),
                })
                .collect::<Vec<_>>();

            Instruction::Call(func_name, new_args, opt_return_val)
        }
        Instruction::MoveOp { source, dest } => Instruction::MoveOp {
            source: check_const(source, const_lookup),
            dest: dest,
        },
        Instruction::Ret(opt_ret_val) => Instruction::Ret(match opt_ret_val {
            Some(ret_val) => Some(check_const(ret_val, const_lookup)),
            None => None,
        }),
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => Instruction::ThreeOp {
            source1: check_const(source1, const_lookup),
            source2: check_const(source2, const_lookup),
            dest: dest,
            op: op,
        },
        Instruction::TwoOp { source1, dest, op } => Instruction::TwoOp {
            source1: check_const(source1, const_lookup),
            dest: dest,
            op: op,
        },
        _ => i, // covers phi expressions, parallel moves, and constant loads
    };

    simplify(new_instruction)
}

pub fn constant_propagation(
    method: &mut cfg::CfgMethod<SSAVarLabel>,
) -> cfg::CfgMethod<SSAVarLabel> {
    let mut new_method: cfg::CfgMethod<SSAVarLabel> = cfg::CfgMethod {
        name: method.name.clone(),
        params: method.params.clone(),
        blocks: HashMap::new(),
        fields: method.fields.clone(),
        return_type: method.return_type.clone(),
    };

    // should take imm_var::vars to imm_var::imm
    let mut const_lookup: HashMap<ImmVar<SSAVarLabel>, ImmVar<SSAVarLabel>> = HashMap::new();

    // variables that appear inside of phi nodes should not be replaced by constants. If we replaced them, it is unclear what the behavior of the phi node is
    let mut forbidden: HashSet<SSAVarLabel> = hashset! {};
    for (_, block) in method.blocks.iter() {
        for instr in block.body.iter() {
            match instr {
                Instruction::PhiExpr { sources, .. } => {
                    let new_forbidden: HashSet<SSAVarLabel> = sources
                        .iter()
                        .map(|(_, var)| match var {
                            Sum::Inl(v) => *v,
                            Sum::Inr(_) => panic!(),
                        })
                        .collect();

                    forbidden = forbidden
                        .union(&new_forbidden)
                        .cloned()
                        .collect::<HashSet<_>>();
                }
                _ => (),
            }
        }
    }

    let mut found_constant_load = false;

    // find all constants
    for (_, block) in method.blocks.iter() {
        for instr in block.body.iter() {
            match instr {
                Instruction::Constant { dest, constant } => {
                    if !forbidden.contains(dest) && method.fields.contains_key(&dest.name) {
                        const_lookup.insert(ImmVar::Var(*dest), ImmVar::Imm(*constant));
                        found_constant_load = true;
                    }
                }
                _ => (),
            }
        }
    }

    // update new_instructions
    for (i, block) in method.blocks.iter() {
        let mut new_instructions = vec![];
        for instr in block.body.iter() {
            match instr {
                Instruction::Constant { dest, .. } => {
                    if !forbidden.contains(dest) && method.fields.contains_key(&dest.name) {
                        continue;
                    }
                }
                _ => (), // remove constant loads
            }
            new_instructions.push(prop_const(instr.clone(), &const_lookup))
        }

        let new_block: BasicBlock<SSAVarLabel> = BasicBlock {
            parents: block.parents.clone(),
            block_id: *i,
            body: new_instructions,
            jump_loc: prop_const_jump(block.jump_loc.clone(), &const_lookup),
        };

        new_method.blocks.insert(*i, new_block);
    }

    if found_constant_load {
        // get_parents(&mut method.blocks); // parents might change after prop const jump
        constant_propagation(&mut new_method)
    } else {
        prune_method(&mut new_method);
        get_parents(&mut new_method.blocks);
        new_method
    }
}

fn prop_const_jump(
    j: Jump<SSAVarLabel>,
    const_lookup: &HashMap<ImmVar<SSAVarLabel>, ImmVar<SSAVarLabel>>,
) -> Jump<SSAVarLabel> {
    // should modify conditional jumps if possible. Should turn conditional jumps with immediate soures into unconditional jumps.
    match &j {
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => match check_const(source.clone(), const_lookup) {
            ImmVar::Imm(imm) => {
                if imm == 0 {
                    Jump::Uncond(*false_block)
                } else {
                    Jump::Uncond(*true_block)
                }
            }
            _ => j,
        },
        _ => j,
    }
}

// Might be time consuming to all of these identies for each instruction.
fn use_identities(i: Instruction<SSAVarLabel>) -> Instruction<SSAVarLabel> {
    /*
     * Identities to check for
     * x + 0 = x
     * 0 + x = x
     * x - x = 0
     *
     * x * 1 = x
     * 1 * x = x
     * x * 0 = 0
     * 0 * x = 0
     *
     * x / 1 = x
     * 0 / x = 0 // Is it?
     *
     * x % 1 = 0
     * x % x = 0
     *
     * x > x = 0
     * x >= x = 1
     * x <= x = 1
     * x < x = 0
     *
     * x == x = 1
     * x != x = 0
     *
     *
     */
    i
}
