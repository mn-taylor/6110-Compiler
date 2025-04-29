use crate::cfg::{self, GetNameVer};
use crate::cfg::{BlockLabel, CfgType, ImmVar, Instruction, IsImmediate};
use crate::deadcode::{get_dest, get_sources};
use crate::ir::{Bop, UnOp};
use crate::parse::Primitive;
use crate::scan::{AddOp, EqOp, MulOp, RelOp};
use crate::ssa_construct::{dominator_sets, dominator_tree, get_graph, SSAVarLabel};
use maplit::{hashmap, hashset};
use std::cmp::min;
use std::collections::{HashMap, HashSet};

// convention, in commuatitive expressions containing immediates, immediate will always go on right.
// for expressions containing two variables that commute, the variable with the smaller label will be on the left.
// Ties are broken by version number, with smaller version number on the left.
#[derive(Eq, Hash, PartialEq, Clone, Debug)]
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

        if var1 < var2 {
            (source1, source2)
        } else if var1 > var2 {
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
                    Some(CSEHash::Mul(var1, var2))
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
) -> (
    HashMap<CSEHash<SSAVarLabel>, Vec<(BlockLabel, usize)>>,
    HashMap<CSEHash<SSAVarLabel>, CfgType>,
) {
    let mut all_sub_expr_types: HashMap<CSEHash<SSAVarLabel>, CfgType> = hashmap! {};
    let mut sub_exprs: HashMap<CSEHash<SSAVarLabel>, Vec<(BlockLabel, usize)>> = hashmap! {};

    for (bid, block) in m.blocks.iter() {
        for (iid, instruction) in block.body.iter().enumerate() {
            // check if any of the source operands are global variables
            let sources = get_sources(instruction);
            let mut is_global = false;
            let _ = sources
                .iter()
                .map(|c| {
                    if !m.fields.contains_key(&c.name) {
                        is_global = true
                    }
                })
                .collect::<Vec<_>>();

            // compute the type of the expression
            let expr_type = match get_dest(instruction) {
                Some(var) => {
                    // If this var is global then we cannot look up its type in the global scope so we have no easy way of concluding the type. Assume if var is global, type is long
                    match m.fields.get(&var.name) {
                        Some((var_type, _)) => var_type.clone(),
                        None => CfgType::Scalar(Primitive::LongType),
                    }
                }
                None => continue,
            };

            // if the instruction is hashable, add to the hashmaps
            match generate_hash(instruction.clone()) {
                Some(hash) => {
                    let hash = &hash.clone();
                    all_sub_expr_types.insert(hash.clone(), expr_type);

                    if !sub_exprs.contains_key(&hash.clone()) {
                        sub_exprs.insert(hash.clone(), vec![]);
                    }
                    sub_exprs.get_mut(&hash.clone()).unwrap().push((*bid, iid));
                }
                None => {}
            };
        }
    }

    // keep only the subexpressions that occur more than once
    let mut common_sub_exprs: HashMap<CSEHash<SSAVarLabel>, Vec<(usize, usize)>> = hashmap! {};
    for (key, vec) in sub_exprs {
        if vec.len() > 1 {
            common_sub_exprs.insert(key, vec);
        }
    }

    (common_sub_exprs, all_sub_expr_types)
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

fn invert_dom_sets(dom_sets: HashMap<usize, HashSet<usize>>) -> HashMap<usize, HashSet<usize>> {
    let mut new_sets: HashMap<usize, HashSet<usize>> = hashmap! {};

    for (bid, dom_set) in dom_sets.iter() {
        for dom in dom_set.iter() {
            if !new_sets.contains_key(dom) {
                new_sets.insert(*dom, hashset! {});
            }
            new_sets.get_mut(dom).unwrap().insert(*bid);
        }
    }

    return new_sets;
}

pub fn eliminate_common_subexpressions(
    m: &mut cfg::CfgMethod<SSAVarLabel>,
) -> cfg::CfgMethod<SSAVarLabel> {
    let (common_sub_exprs, expr_types) = find_common_subexpressions(m);
    println!("{:?}", common_sub_exprs);

    let g = &get_graph(m);
    println!("{:?}\n\n", g);
    let dominator_sets = dominator_sets(0, g);
    println!("{:?}", dominator_sets);
    let dominator_tree = dominator_tree(m, &dominator_sets);
    let inverted_dom_sets = invert_dom_sets(dominator_sets.clone());
    let inverted_tree = invert_dom_tree(dominator_tree);

    println!("{:?}", inverted_tree);

    let mut hash_conversions: HashMap<CSEHash<SSAVarLabel>, SSAVarLabel> = hashmap! {};

    // iterate over all common sub expressions {
    for (hash, bid_iid) in common_sub_exprs.iter() {
        //      We want to find the first block that dominates all of the uses of the subexpressions
        //      this means just following the parent points in the inverted dominator tree

        let (mut curr_bid, _) = &bid_iid.iter().next().unwrap();
        loop {
            let mut dominates_all = true;
            let curr_dominates = inverted_dom_sets.get(&curr_bid).unwrap();

            let _ = bid_iid
                .iter()
                .map(|(bid, iid)| {
                    if !curr_dominates.contains(bid) {
                        dominates_all = false
                    }
                })
                .collect::<Vec<_>>();

            if dominates_all {
                break;
            } else {
                curr_bid = *inverted_tree.get(&curr_bid).unwrap_or(&0);
            }
        }

        //      Once we've found the block that dominates all of the subexpression uses, we need to find out where to place it
        //      set the instruction location to be the length of the block, then iterate over all of the uses of the subexpression, and update the min accordingly

        let mut curr_block = m.blocks.get(&curr_bid).unwrap().clone();
        let mut instr_loc = curr_block.body.len();

        let _ = bid_iid
            .iter()
            .map(|(bid, iid)| {
                if *bid == curr_bid {
                    instr_loc = min(instr_loc, *iid);
                    println!("{} {}", instr_loc, *iid);
                }
            })
            .collect::<Vec<_>>();

        // make new variable add it method fields
        let var_name = m.fields.keys().max().unwrap() + 1;
        let ssa_var_name = SSAVarLabel {
            name: var_name,
            version: 1,
        };
        let var_type = expr_types.get(hash).unwrap().clone();

        m.fields.insert(var_name, (var_type, "temp".to_string()));

        // insert the instruction in the correct place
        curr_block
            .body
            .insert(instr_loc, convert_hash_to_instr(hash, ssa_var_name));

        m.blocks.insert(curr_bid, curr_block);

        let blocks_of_interest = bid_iid
            .iter()
            .map(|(bid, _)| *bid)
            .collect::<HashSet<usize>>();

        let mut new_method = m.clone();
        for (bid, block) in m.blocks.iter_mut() {
            let mut new_block = block.clone();
            let mut new_instructions = vec![];

            if !blocks_of_interest.contains(bid) {
                new_method.blocks.insert(*bid, new_block);
                continue;
            }

            for (iid, instruction) in block.body.iter_mut().enumerate() {
                if *bid == curr_bid && iid == instr_loc {
                    new_instructions.push(instruction.clone());
                    continue;
                }

                // check the hash
                if generate_hash(instruction.clone()) == Some(hash.clone()) {
                    println!("found a match");
                    match get_dest(instruction) {
                        Some(var) => new_instructions.push(Instruction::MoveOp {
                            source: ImmVar::Var(ssa_var_name),
                            dest: var,
                        }),
                        None => panic!("instructions without destinations are not hashable"),
                    }
                } else {
                    new_instructions.push(instruction.clone());
                }
            }
            new_block.body = new_instructions;
            new_method.blocks.insert(*bid, new_block);
        }

        m.blocks = new_method.blocks.clone();
    }

    m.clone()
}
