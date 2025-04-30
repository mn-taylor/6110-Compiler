use crate::cfg::{self};
use crate::cfg::{ImmVar, Instruction, IsImmediate};
use crate::deadcode::get_dest;
use crate::ir::{Bop, UnOp};
use crate::scan::AddOp;
use crate::ssa_construct::SSAVarLabel;
use maplit::hashmap;
use std::collections::HashMap;

#[derive(Clone)]
enum PosNeg {
    Pos(ImmVar<SSAVarLabel>),
    Neg(ImmVar<SSAVarLabel>),
}

fn negate(v: Vec<PosNeg>) -> Vec<PosNeg> {
    return v
        .iter()
        .map(|c| match c {
            PosNeg::Pos(p) => PosNeg::Neg(p.clone()),
            PosNeg::Neg(p) => PosNeg::Pos(p.clone()),
        })
        .collect();
}

fn get_def(m: &mut cfg::CfgMethod<SSAVarLabel>) -> HashMap<SSAVarLabel, Instruction<SSAVarLabel>> {
    let mut defs: HashMap<SSAVarLabel, Instruction<SSAVarLabel>> = hashmap! {};

    for (_, block) in m.blocks.iter() {
        for (_, instruction) in block.body.iter().enumerate() {
            if let Some(dest) = get_dest(instruction) {
                if m.fields.contains_key(&dest.name) {
                    // no global variables
                    defs.insert(dest, instruction.clone());
                }
            }
        }
    }
    defs
}

fn sum_distill_variable(
    i: Instruction<SSAVarLabel>,
    defs: &HashMap<SSAVarLabel, Instruction<SSAVarLabel>>,
    seen: &mut HashMap<SSAVarLabel, Option<Vec<PosNeg>>>, // modify seen so it does not take an option
) -> Option<Vec<PosNeg>> {
    // check if we are at a sum atomic variable
    let can_recurse = match &i {
        Instruction::ThreeOp { op, .. } => match op {
            Bop::AddBop(_) => true,
            _ => false,
        },
        Instruction::TwoOp { op, .. } => match op {
            UnOp::Neg => true,
            _ => false,
        },
        _ => false,
    };
    // recurse on the sources and apply positive negative switches
    let dest = get_dest(&i).unwrap();
    if !can_recurse {
        // update seen
        seen.insert(dest, Some(vec![PosNeg::Pos(ImmVar::Var(dest))]));
        return Some(vec![PosNeg::Pos(ImmVar::Var(dest))]);
    }

    //recurse on sources
    match &i {
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => match op {
            Bop::AddBop(aop) => match aop {
                AddOp::Add => {
                    // check if the sources are global
                    let mut pns = vec![];
                    if source1.is_immediate() {
                        pns.push(PosNeg::Pos(source1.clone()));
                    } else {
                        match source1 {
                            ImmVar::Var(v) => {
                                if !defs.contains_key(v) {
                                    seen.insert(*dest, None);
                                    return None;
                                } else {
                                    // bypassing
                                    match seen.get(v) {
                                        Some(Some(vecpns)) => pns.extend((*vecpns).clone()),
                                        _ => {
                                            let reduced = sum_distill_variable(
                                                defs.get(v).unwrap().clone(),
                                                defs,
                                                seen,
                                            );
                                            if reduced.is_some() {
                                                pns.extend(reduced.unwrap());
                                            } else {
                                                return None;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => panic!(),
                        }
                    }

                    if source2.is_immediate() {
                        pns.push(PosNeg::Pos(source2.clone()));
                    } else {
                        match source2 {
                            ImmVar::Var(v) => {
                                if !defs.contains_key(v) {
                                    seen.insert(*dest, None);
                                    return None;
                                } else {
                                    // bypassing
                                    match seen.get(v) {
                                        Some(Some(vecpns)) => pns.extend((*vecpns).clone()),
                                        _ => {
                                            let reduced = sum_distill_variable(
                                                defs.get(v).unwrap().clone(),
                                                defs,
                                                seen,
                                            );
                                            if reduced.is_some() {
                                                pns.extend(reduced.unwrap());
                                            } else {
                                                return None;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => panic!(),
                        }
                    }

                    seen.insert(*dest, Some(pns.clone()));
                    return Some(pns);
                }
                AddOp::Sub => {
                    // check if the sources are global
                    let mut pns = vec![];
                    if source1.is_immediate() {
                        pns.push(PosNeg::Pos(source1.clone()));
                    } else {
                        match source1 {
                            ImmVar::Var(v) => {
                                if !defs.contains_key(v) {
                                    seen.insert(*dest, None);
                                    return None;
                                } else {
                                    // bypassing
                                    match seen.get(v) {
                                        Some(Some(vecpns)) => pns.extend((*vecpns).clone()),
                                        _ => {
                                            let reduced = sum_distill_variable(
                                                defs.get(v).unwrap().clone(),
                                                defs,
                                                seen,
                                            );
                                            if reduced.is_some() {
                                                pns.extend(reduced.unwrap());
                                            } else {
                                                return None;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => panic!(),
                        }
                    }

                    if source2.is_immediate() {
                        pns.push(PosNeg::Pos(source2.clone()));
                    } else {
                        match source2 {
                            ImmVar::Var(v) => {
                                if !defs.contains_key(v) {
                                    // seen.insert(*dest, None);
                                    return None;
                                } else {
                                    // bypassing
                                    match seen.get(v) {
                                        Some(Some(vecpns)) => pns.extend((*vecpns).clone()),
                                        _ => {
                                            let reduced = sum_distill_variable(
                                                defs.get(v).unwrap().clone(),
                                                defs,
                                                seen,
                                            );
                                            if reduced.is_some() {
                                                pns.extend(negate(reduced.unwrap()));
                                            } else {
                                                return None;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => panic!(),
                        }
                    }

                    seen.insert(*dest, Some(pns.clone()));
                    return Some(pns);
                }
            },
            _ => (),
        },
        Instruction::TwoOp { source1, dest, op } => match op {
            UnOp::Neg => {
                // check if the sources are global
                let mut pns = vec![];
                if source1.is_immediate() {
                    pns.push(PosNeg::Pos(source1.clone()));
                } else {
                    match source1 {
                        ImmVar::Var(v) => {
                            if !defs.contains_key(v) {
                                seen.insert(*dest, None);
                                return None;
                            } else {
                                // bypassing
                                match seen.get(v) {
                                    Some(Some(vecpns)) => pns.extend((*vecpns).clone()),
                                    _ => {
                                        let reduced = sum_distill_variable(
                                            defs.get(v).unwrap().clone(),
                                            defs,
                                            seen,
                                        );
                                        if reduced.is_some() {
                                            pns.extend(negate(reduced.unwrap()));
                                        } else {
                                            return None;
                                        }
                                    }
                                }
                            }
                        }
                        _ => panic!(),
                    }
                }

                seen.insert(*dest, Some(pns.clone()));
                return Some(pns);
            }
            _ => (),
        },
        _ => (),
    }

    panic!()
}

fn get_sum_reducible_defintions(
    m: &mut cfg::CfgMethod<SSAVarLabel>,
) -> HashMap<SSAVarLabel, Option<Vec<PosNeg>>> {
    let defs = get_def(m);
    let mut seen = hashmap! {};

    for (_, block) in m.blocks.iter() {
        for (_, instruction) in block.body.iter().enumerate() {
            sum_distill_variable(instruction.clone(), &defs, &mut seen);
        }
    }

    return seen;
}

fn sum_merge(
    m: &mut cfg::CfgMethod<SSAVarLabel>,
    all_sums: &mut HashMap<SSAVarLabel, Vec<PosNeg>>,
) {
}

// reorganize additions/subtractions and multiplactions/divisions to reduce the number of times we perform a given computation.
fn restructure_ops(m: &mut cfg::CfgMethod<SSAVarLabel>) -> cfg::CfgMethod<SSAVarLabel> {
    todo!()
}
