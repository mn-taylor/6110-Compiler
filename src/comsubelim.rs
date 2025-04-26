use crate::cfg::{self, GetNameVer};
use crate::cfg::{Arg, BasicBlock, BlockLabel, CfgType, ImmVar, Instruction, IsImmediate, Jump};
use crate::cfg_build::get_parents;
use crate::deadcode::get_dest;
use crate::ir::{Bop, UnOp};
use crate::scan::{AddOp, EqOp, MulOp, RelOp, Sum};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, SSAVarLabel};
use maplit::{hashmap, hashset};
use std::cmp::min;
use std::collections::{HashMap, HashSet};

// convention, in commuatitive expressions containing immediates, immediate will always go on right.
// for expressions containing two variables that commute, the variable with the smaller label will be on the left.
// Ties are broken by version number, with smaller version number on the left.
#[derive(Eq, Hash, PartialEq, Clone)]
pub enum CSEHash<T> {
    Add(ImmVar<T>, ImmVar<T>),
    Sub(ImmVar<T>, ImmVar<T>),
    Mul(ImmVar<T>, ImmVar<T>),
    Div(ImmVar<T>, ImmVar<T>),
    Mod(ImmVar<T>, ImmVar<T>),
    Lt(ImmVar<T>, ImmVar<T>),
    Lte(ImmVar<T>, ImmVar<T>),
    Eq(ImmVar<T>, ImmVar<T>),
    Neq(ImmVar<T>, ImmVar<T>),
    Neg(ImmVar<T>),
    Not(ImmVar<T>),
}

fn get_commutative_ordering(
    source1: ImmVar<SSAVarLabel>,
    source2: ImmVar<SSAVarLabel>,
) -> (ImmVar<SSAVarLabel>, ImmVar<SSAVarLabel>) {
    if source1.is_immediate() {
        (source2, source1)
    } else if source2.is_immediate() {
        (source1, source2)
    } else {
        let (var1, ver1) = source1.get_name_ver().unwrap();
        let (var2, ver2) = source2.get_name_ver().unwrap();

        if (var1 < var2) {
            (source1, source2)
        } else if (var1 > var2) {
            (source2, source1)
        } else if ver1 < ver2 {
            (source1, source2)
        } else {
            (source2, source1)
        }
    }
}

fn generate_hash(i: Instruction<SSAVarLabel>) -> Option<CSEHash<SSAVarLabel>> {
    // Generate a hash for binops and two ops
    match i {
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => match op {
            Bop::AddBop(aop) => match aop {
                AddOp::Add => {
                    let (var1, var2) = get_commutative_ordering(source1, source2);
                    Some(CSEHash::Add(var1, var2))
                }
                AddOp::Sub => Some(CSEHash::Sub(source1, source2)),
            },
            Bop::MulBop(mop) => match mop {
                MulOp::Mul => {
                    let (var1, var2) = get_commutative_ordering(source1, source2);
                    Some(CSEHash::Add(var1, var2))
                }
                MulOp::Div => Some(CSEHash::Div(source1, source2)),
                MulOp::Mod => Some(CSEHash::Mod(source1, source2)),
            },
            Bop::RelBop(rop) => match rop {
                RelOp::Lt => Some(CSEHash::Lt(source1, source2)),
                RelOp::Le => Some(CSEHash::Lte(source1, source2)),
                RelOp::Gt => Some(CSEHash::Lt(source2, source1)),
                RelOp::Ge => Some(CSEHash::Lte(source2, source1)),
            },
            Bop::EqBop(eop) => match eop {
                EqOp::Eq => {
                    let (var1, var2) = get_commutative_ordering(source1, source2);
                    Some(CSEHash::Eq(var1, var2))
                }
                EqOp::Neq => {
                    let (var1, var2) = get_commutative_ordering(source1, source2);
                    Some(CSEHash::Neq(var1, var2))
                }
            },
            _ => None,
        },
        Instruction::TwoOp { source1, dest, op } => match op {
            UnOp::Neg => Some(CSEHash::Neg(source1)),
            UnOp::Not => Some(CSEHash::Not(source1)),
            _ => None,
        },
        _ => None,
    }
}

fn convert_hash_to_instr(
    hash: &CSEHash<SSAVarLabel>,
    dest: SSAVarLabel,
) -> Instruction<SSAVarLabel> {
    match hash.clone() {
        CSEHash::Add(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::AddBop(AddOp::Add),
        },
        CSEHash::Sub(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::AddBop(AddOp::Sub),
        },
        CSEHash::Mul(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::MulBop(MulOp::Mul),
        },
        CSEHash::Div(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::MulBop(MulOp::Div),
        },
        CSEHash::Mod(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::MulBop(MulOp::Mod),
        },
        CSEHash::Eq(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::EqBop(EqOp::Eq),
        },
        CSEHash::Neq(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::EqBop(EqOp::Neq),
        },
        CSEHash::Lt(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::RelBop(RelOp::Lt),
        },
        CSEHash::Lte(s1, s2) => Instruction::ThreeOp {
            source1: s1,
            source2: s2,
            dest: dest,
            op: Bop::RelBop(RelOp::Le),
        },
        CSEHash::Neg(s1) => Instruction::TwoOp {
            source1: s1,
            dest: dest,
            op: UnOp::Not,
        },
        CSEHash::Not(s1) => Instruction::TwoOp {
            source1: s1,
            dest: dest,
            op: UnOp::Not,
        },
    }
}

fn find_common_subexpressions(
    m: &cfg::CfgMethod<SSAVarLabel>,
) -> HashMap<CSEHash<SSAVarLabel>, Vec<(BlockLabel, usize, (CfgType, String))>> {
    // easier to keep the type information of the sub expression that to compute it later.
    let mut subexprs: HashMap<CSEHash<SSAVarLabel>, Vec<(BlockLabel, usize, (CfgType, String))>> =
        hashmap! {};

    todo!("Fix to handle assignments to global variables");
    for (id, block) in m.blocks.iter() {
        for (idx, instruction) in block.body.iter().enumerate() {
            match generate_hash(instruction.clone()) {
                Some(hash) => {
                    if !m
                        .fields
                        .contains_key(&get_dest(instruction.clone()).unwrap().name)
                    {
                        println!("{}", instruction);
                    }

                    if !subexprs.contains_key(&hash) {
                        subexprs.insert(
                            hash,
                            vec![(
                                *id,
                                idx,
                                m.fields
                                    .get(&get_dest(instruction.clone()).unwrap().name)
                                    .unwrap()
                                    .clone(),
                            )],
                        );
                    } else {
                        subexprs.get_mut(&hash).unwrap().push((
                            *id,
                            idx,
                            m.fields
                                .get(&get_dest(instruction.clone()).unwrap().name)
                                .unwrap()
                                .clone(),
                        ))
                    }
                }
                None => (),
            }
        }
    }
    subexprs
}

fn invert_dom_tree(dom_tree: HashMap<usize, HashSet<usize>>) -> HashMap<usize, usize> {
    let mut new_tree: HashMap<usize, usize> = hashmap! {};

    for (parent, children) in dom_tree.iter() {
        for child in children {
            new_tree.insert(*child, *parent);
        }
    }

    return new_tree;
}

pub fn eliminate_common_subexpressions(
    m: &mut cfg::CfgMethod<SSAVarLabel>,
) -> cfg::CfgMethod<SSAVarLabel> {
    let subexprs = find_common_subexpressions(m);

    let g = &get_graph(m);
    println!("{:?}\n\n", g);
    let dominator_sets = dominator_sets(0, g);
    println!("{:?}", dominator_sets);
    let dominator_tree = dominator_tree(m, &dominator_sets);
    let inverted_tree = invert_dom_tree(dominator_tree);

    let mut hash_conversions: HashMap<CSEHash<SSAVarLabel>, SSAVarLabel> = hashmap! {};

    for (hash, block_labels) in subexprs.iter() {
        // check if there are multiple occurrences of the hash
        if block_labels.len() < 2 {
            continue;
        }

        // if there are multiple occurences of the hash and the blocks are different, go up dominator tree to find first block that dominates all of them
        let (block_id, instr_id, typ) = block_labels.get(0).unwrap();
        let mut all_the_same = true;

        let _ = block_labels
            .iter()
            .map(|(bid, _, _)| {
                if bid != block_id {
                    all_the_same = false;
                }
            })
            .collect::<Vec<_>>();

        let mut curr = block_id;
        while *curr != 0 {
            // if current dominates all of the children then we should place the instruction there
            let dom_set = dominator_sets.get(&curr).unwrap();
            let dominates_all: bool = block_labels
                .iter()
                .map(|(bid, _, _)| dom_set.contains(bid))
                .all(|c| c);

            if dominates_all {
                break;
            } else {
                curr = inverted_tree.get(curr).unwrap()
            }
        }

        // compute where to put the instruction
        let mut min_idx = m.blocks.get(curr).unwrap().body.len();
        for (bid, iid, _) in block_labels {
            if bid == curr {
                min_idx = min(min_idx, *iid);
            }
        }

        // make new variable
        let new_var = SSAVarLabel {
            name: m.fields.keys().max().unwrap() + 1,
            version: 1,
        };
        m.fields.insert(new_var.name, typ.clone());
        hash_conversions.insert(hash.clone(), new_var);

        let new_instruction = convert_hash_to_instr(hash, new_var);

        m.blocks
            .get_mut(&curr)
            .unwrap()
            .body
            .insert(min_idx, new_instruction)
    }

    let mut new_method = m.clone();
    // modify instructions using hash conversions
    for (id, block) in m.blocks.iter() {
        let mut new_instructions = vec![];
        for instruction in block.body.iter() {
            let dest = match get_dest(instruction.clone()) {
                Some(v) => v,
                None => {
                    new_instructions.push(instruction.clone());
                    continue;
                }
            };

            match generate_hash(instruction.clone()) {
                Some(hash) => match hash_conversions.get(&hash) {
                    Some(var) => {
                        if *var != dest {
                            // want to skip over the defining subexpression and replace its successors
                            new_instructions.push(Instruction::MoveOp {
                                source: ImmVar::Var(var.clone()),
                                dest: dest,
                            })
                        } else {
                            new_instructions.push(instruction.clone())
                        }
                    }
                    None => new_instructions.push(instruction.clone()),
                },
                None => new_instructions.push(instruction.clone()),
            }
        }

        let mut new_block = block.clone();
        new_block.body = new_instructions;
        new_method.blocks.insert(new_block.block_id, new_block);
    }

    new_method
}
