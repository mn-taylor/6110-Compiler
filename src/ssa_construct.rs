use crate::cfg;
use crate::cfg::{Arg, BasicBlock, BlockLabel, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use crate::ir::Block;
use std::arch::aarch64::int16x8_t;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env::var;
use std::fmt;

fn add_block_to_var_def(
    def: &mut HashMap<VarLabel, HashSet<BlockLabel>>,
    var: &VarLabel,
    block_id: BlockLabel,
) {
    match def.get_mut(&var) {
        Some(t) => {
            t.insert(block_id);
        }
        None => {
            let mut set = HashSet::new();
            set.insert(block_id);
            def.insert(var.clone(), set);
        }
    }
}

fn var_to_def_locs(m: &CfgMethod) -> HashMap<VarLabel, HashSet<BlockLabel>> {
    let mut def: HashMap<VarLabel, HashSet<BlockLabel>> = HashMap::new();

    for (label, block) in m.blocks.iter() {
        for instr in block.body.clone() {
            match instr {
                Instruction::ThreeOp {
                    source1,
                    source2,
                    dest,
                    op,
                } => add_block_to_var_def(&mut def, &dest, *label),
                Instruction::Constant { dest, constant } => {
                    add_block_to_var_def(&mut def, &dest, *label)
                }
                Instruction::MoveOp { source, dest } => {
                    add_block_to_var_def(&mut def, &dest, *label)
                }
                Instruction::TwoOp { source1, dest, op } => {
                    add_block_to_var_def(&mut def, &dest, *label)
                }
                _ => (),
            }
        }
    }

    def
}

fn intersect_all<'a>(
    vals: &mut impl Iterator<Item = &'a HashSet<BlockLabel>>,
) -> HashSet<BlockLabel> {
    let mut int: HashSet<BlockLabel>;
    if let Some(set) = vals.next() {
        int = set.clone();
    } else {
        int = HashSet::new();
    }

    for val in vals {
        int = int.intersection(&val).map(|x| *x).collect();
    }
    int
}
// n -> set of blocks that dominate n
// from https://en.wikipedia.org/wiki/Dominator_(graph_theory)
pub fn dominator_sets(
    start_node: BlockLabel,
    g: &HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_sets: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();
    let all_keys: HashSet<BlockLabel> = g.keys().map(|x| *x).collect::<HashSet<_>>();
    let mut pred = HashMap::new();
    // dominator of the start node is the start itself
    // for all other nodes, set all nodes as the dominators
    for x in g.keys() {
        if *x == start_node {
            dom_sets.insert(*x, vec![*x].into_iter().collect::<HashSet<BlockLabel>>());
        } else {
            dom_sets.insert(*x, all_keys.clone());
        }
        pred.insert(*x, HashSet::new());
    }

    // compute predecessors
    for (x, ys) in g.iter() {
        for y in ys {
            pred.get_mut(&y).unwrap().insert(x);
        }
    }

    let mut changes = true;
    while changes {
        changes = false;
        for (n, preds) in pred.iter() {
            if *n != start_node {
                // Dom(n) = {n} union with intersection over Dom(p) for all p in pred(n)
                let mut new_dom_n: HashSet<BlockLabel> =
                    intersect_all(&mut preds.iter().map(|p| dom_sets.get(&p).unwrap()));
                new_dom_n.insert(*n);

                let dom_n = dom_sets.get(n).unwrap();
                if *dom_n != new_dom_n {
                    changes = true;
                    dom_sets.insert(*n, new_dom_n);
                }
            }
        }
    }

    dom_sets
}

pub fn dominator_tree(
    m: &CfgMethod,
    dom_sets: &HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut dom_tree: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();
    let blocks = m.blocks.iter();

    for (label, block) in blocks {
        // compute the immediate dominator of the block

        let mut agenda = vec![];
        for parent in block.parents.iter() {
            if dom_sets.contains_key(&parent) {
                agenda.push(*parent);
            }
        }

        if agenda.len() == 0 {
            continue;
        };

        let target_dom_set = dom_sets.get(label).unwrap();
        let mut seen: HashSet<BlockLabel> = HashSet::new();
        seen.insert(*label);
        while agenda.len() > 0 {
            let curr = agenda.pop().unwrap();
            if seen.contains(&curr) {
                continue;
            }; // cycle

            if target_dom_set.contains(&curr) {
                // found immediate dominator
                let mut new_set: HashSet<BlockLabel> = match dom_tree.get(&curr) {
                    Some(set) => set.clone(),
                    None => HashSet::new(),
                };

                new_set.insert(*label);

                dom_tree.insert(curr, new_set.clone());
                break;
            }

            seen.insert(curr);

            // add the new parents
            for parent in m.blocks.get(&curr).unwrap().parents.iter() {
                if dom_sets.contains_key(&parent) {
                    agenda.push(*parent);
                }
            }
        }
    }

    dom_tree
}

pub fn dominance_frontiers(
    g: HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_sets: &HashMap<BlockLabel, HashSet<BlockLabel>>,
    dominance_tree: &HashMap<BlockLabel, HashSet<BlockLabel>>,
) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut df = HashMap::new();
    let mut idom = HashMap::new();
    for (p, children) in dominance_tree.iter() {
        df.insert(*p, HashSet::new());
        for c in children {
            idom.insert(c, *p);
            df.insert(*c, HashSet::new());
        }
    }

    // Algorithm 3.2 in SSA book
    for (a, children) in g {
        for b in children {
            let mut x = a;
            let mut x_dom_b = dominance_sets.get(&b).unwrap().contains(&x);

            // while x does not strictly dominate b
            while !(x_dom_b && x != b) {
                df.get_mut(&x).unwrap().insert(b);
                if idom.get(&x).is_some() {
                    x = *idom.get(&x).unwrap();
                    x_dom_b = dominance_sets.get(&b).unwrap().contains(&x);
                } else {
                    break;
                }
            }
        }
    }
    df
}

pub fn get_graph(m: &mut CfgMethod) -> HashMap<BlockLabel, HashSet<BlockLabel>> {
    let mut g: HashMap<BlockLabel, HashSet<BlockLabel>> = HashMap::new();

    for (label, block) in m.blocks.iter() {
        if (block.parents.len() == 0 && *label != 0) {
            // these nodes are never accessed by the program and confuse the computation of dominance sets
            continue;
        }

        let mut edges: HashSet<BlockLabel> = HashSet::new();
        match block.jump_loc {
            Jump::Uncond(b) => {
                edges.insert(b);
            }
            Jump::Cond {
                source: _,
                true_block: b1,
                false_block: b2,
            } => {
                edges.insert(b1);
                edges.insert(b2);
            }
            Jump::Nowhere => {}
        }

        g.insert(*label, edges);
    }

    g
}

pub fn insert_phis(m: &mut CfgMethod) -> &CfgMethod {
    let var_defs = var_to_def_locs(m);
    let g = get_graph(m);
    let dom_sets: HashMap<usize, HashSet<usize>> = dominator_sets(0, &g);
    let dom_tree = dominator_tree(m, &dom_sets);
    let dom_frontier = dominance_frontiers(g, &dom_sets, &dom_tree);

    // Algorithm 3.1 in SSA-Based Compiler Design
    for (var, defs) in var_defs.iter() {
        let mut F: HashSet<BlockLabel> = HashSet::new();
        let mut W: Vec<BlockLabel> = vec![];

        for def in defs {
            W.push(*def);
        }

        while W.len() != 0 {
            let X = W.pop().unwrap();
            for Y in dom_frontier.get(&X).unwrap() {
                if !F.contains(Y) {
                    let block = m.blocks.get_mut(Y).unwrap();
                    block.body.insert(
                        0,
                        Instruction::PhiExpr {
                            dest: *var,
                            sources: vec![],
                        },
                    );

                    F.insert(*Y);

                    if !defs.contains(Y) {
                        W.push(*Y);
                    }
                }
            }
        }
    }

    return m;
}

#[derive(Debug, Clone)]
pub struct SSAVarLabel {
    pub name: u32,
    pub version: u32,
}

impl fmt::Display for SSAVarLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the struct as: t{name}_v{version}
        write!(f, "{}_v{}", self.name, self.version)
    }
}

fn rewrite_source(
    source: VarLabel,
    reaching_defs: &HashMap<BlockLabel, HashMap<VarLabel, SSAVarLabel>>,
    all_fields: &HashSet<VarLabel>,
    block_id: BlockLabel,
) -> SSAVarLabel {
    let new_source: SSAVarLabel;
    if all_fields.contains(&source) {
        new_source = reaching_defs
            .get(&block_id)
            .unwrap()
            .get(&source)
            .unwrap()
            .clone();
    } else {
        // global variable
        new_source = SSAVarLabel {
            name: source,
            version: 0,
        }
    }

    new_source
}

fn rewrite_dest(
    dest: VarLabel,
    reaching_defs: &mut HashMap<BlockLabel, HashMap<VarLabel, SSAVarLabel>>,
    latest_defs: &mut HashMap<u32, u32>,
    all_fields: &HashSet<VarLabel>,
    block_id: BlockLabel,
) -> SSAVarLabel {
    let new_dest: SSAVarLabel;
    if all_fields.contains(&dest) {
        let curr_version = latest_defs.get(&dest).unwrap();
        new_dest = SSAVarLabel {
            name: dest,
            version: *curr_version + 1,
        };

        latest_defs.insert(dest, curr_version + 1);
        let block_vars = reaching_defs.get_mut(&block_id).unwrap();
        block_vars.insert(dest, new_dest.clone());
    } else {
        // global variable
        new_dest = SSAVarLabel {
            name: dest,
            version: 0,
        };
    }

    new_dest
}

fn rewrite_jump(
    jump: Jump<VarLabel>,
    reaching_defs: &mut HashMap<BlockLabel, HashMap<VarLabel, SSAVarLabel>>,
    all_fields: &HashSet<VarLabel>,
    block_id: BlockLabel,
) -> Jump<SSAVarLabel> {
    match jump {
        Jump::Uncond(new_block) => Jump::Uncond(new_block),
        Jump::Nowhere => Jump::Nowhere,
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => {
            let new_source = rewrite_source(source, reaching_defs, all_fields, block_id);
            Jump::Cond {
                source: new_source,
                true_block: true_block,
                false_block: false_block,
            }
        }
    }
}

fn rewrite_instr(
    instr: &Instruction<VarLabel>,
    reaching_defs: &mut HashMap<BlockLabel, HashMap<VarLabel, SSAVarLabel>>,
    latest_defs: &mut HashMap<u32, u32>,
    all_fields: &HashSet<VarLabel>,
    block_id: BlockLabel,
) -> Instruction<SSAVarLabel> {
    let new_instr: Instruction<SSAVarLabel>;

    match instr {
        Instruction::MoveOp { source, dest } => {
            // replace source by its reaching_def
            let new_source = rewrite_source(*source, reaching_defs, all_fields, block_id);

            // update dest
            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);
            Instruction::MoveOp {
                source: new_source,
                dest: new_dest,
            }
        }
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => {
            let new_source1 = rewrite_source(*source1, reaching_defs, all_fields, block_id);
            let new_source2 = rewrite_source(*source2, reaching_defs, all_fields, block_id);

            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);

            Instruction::ThreeOp {
                source1: new_source1,
                source2: new_source2,
                dest: new_dest,
                op: op.clone(),
            }
        }
        Instruction::TwoOp { source1, dest, op } => {
            let new_source1 = rewrite_source(*source1, reaching_defs, all_fields, block_id);

            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);

            Instruction::TwoOp {
                source1: new_source1,
                dest: new_dest,
                op: op.clone(),
            }
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let new_idx = rewrite_source(*idx, reaching_defs, all_fields, block_id);
            let new_name = SSAVarLabel {
                name: *name,
                version: 0,
            }; // arrays all get the same version
            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);

            Instruction::ArrayAccess {
                dest: new_dest,
                name: new_name,
                idx: new_idx,
            }
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let new_idx = rewrite_source(*idx, reaching_defs, all_fields, block_id);
            let new_source = rewrite_source(*source, reaching_defs, all_fields, block_id);

            let new_arr = SSAVarLabel {
                name: *arr,
                version: 0,
            }; // arrays all get the same version

            Instruction::ArrayStore {
                source: new_source,
                arr: new_arr,
                idx: new_idx,
            }
        }
        Instruction::Constant { dest, constant } => {
            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);

            Instruction::Constant {
                dest: new_dest,
                constant: *constant,
            }
        }
        Instruction::Call(func_name, args, opt_dest) => {
            let new_args = args
                .iter()
                .map(|c| match c {
                    Arg::VarArg(name) => {
                        Arg::VarArg(rewrite_source(*name, reaching_defs, all_fields, block_id))
                    }
                    Arg::StrArg(string) => Arg::StrArg(string.to_string()),
                })
                .collect::<Vec<_>>();

            let new_opt_dest = match opt_dest {
                Some(dest) => Some(rewrite_dest(
                    *dest,
                    reaching_defs,
                    latest_defs,
                    all_fields,
                    block_id,
                )),
                None => None,
            };

            Instruction::Call(func_name.to_string(), new_args, new_opt_dest)
        }
        Instruction::Ret(opt_source) => {
            let new_opt_source = match opt_source {
                Some(var) => Some(rewrite_source(*var, reaching_defs, all_fields, block_id)),
                None => None,
            };

            Instruction::Ret(new_opt_source)
        }
        Instruction::PhiExpr { dest, sources } => {
            // only modify destination, sources are modified by loop in rename_variables

            let new_dest = rewrite_dest(*dest, reaching_defs, latest_defs, all_fields, block_id);

            let new_sources = vec![]; // will handle after loop

            Instruction::PhiExpr {
                dest: new_dest,
                sources: new_sources,
            }
        }
    }
}

// reaching def is maps varlabel to its reaching definition
pub fn rename_variables(
    method: &CfgMethod,
    dominance_tree: &HashMap<BlockLabel, HashSet<BlockLabel>>,
    all_fields: HashSet<VarLabel>,
) -> cfg::CfgMethod<SSAVarLabel> {
    // initalize latest names
    let mut latest_defs: HashMap<u32, u32> = HashMap::new();
    for name in all_fields.iter() {
        latest_defs.insert(*name, 0);
    }

    // intialize reaching defs
    // maps a given var_label to the

    let mut ssa_method: cfg::CfgMethod<SSAVarLabel> = cfg::CfgMethod {
        name: method.name.clone(),
        params: method.params.clone(),
        fields: method.fields.clone(),
        blocks: HashMap::new(),
        return_type: method.return_type.clone(),
    };

    // block label -> fn mapping var to most recent SSAVarLabel(version)
    let mut reaching_defs: HashMap<usize, HashMap<u32, SSAVarLabel>> = HashMap::new();
    reaching_defs.insert(0, HashMap::new());

    // initalize all fields to have version 0, until they are defined
    for var in all_fields.iter() {
        reaching_defs.get_mut(&0).unwrap().insert(
            *var,
            SSAVarLabel {
                name: *var,
                version: 0,
            },
        );
    }

    println!("{:?}", reaching_defs.get(&0));

    let mut agenda: Vec<usize> = vec![0];
    while agenda.len() != 0 {
        let curr = agenda.pop().unwrap();
        let curr_block = method.blocks.get(&curr).unwrap();

        let new_instructions = curr_block
            .body
            .iter()
            .map(|c| rewrite_instr(c, &mut reaching_defs, &mut latest_defs, &all_fields, curr))
            .collect::<Vec<_>>();

        let ssa_block = BasicBlock {
            parents: curr_block.parents.clone(),
            block_id: curr,
            body: new_instructions,
            jump_loc: rewrite_jump(
                curr_block.jump_loc.clone(),
                &mut reaching_defs,
                &all_fields,
                curr,
            ),
        };

        ssa_method.blocks.insert(curr, ssa_block);

        match dominance_tree.get(&curr) {
            Some(children) => {
                // propagate reaching defs
                for child in children.iter() {
                    reaching_defs.insert(*child, reaching_defs.get(&curr).unwrap().clone());
                }

                // process the children
                agenda.extend(children.iter().collect::<Vec<_>>())
            }
            None => {}
        }
    }

    println!("{:?}", reaching_defs.get(&0));

    // after the loop, we must still indicate the sources for each of the phi nodes.
    for block in ssa_method.blocks.values_mut() {
        for instruction in block.body.iter_mut() {
            match instruction {
                Instruction::PhiExpr { dest, sources } => {
                    let new_sources = block
                        .parents
                        .iter()
                        .map(|parent| {
                            // println!("{}, {}, {:?}", parent, dest, reaching_defs.get(parent));
                            (
                                *parent,
                                reaching_defs
                                    .get(parent)
                                    .unwrap()
                                    .get(&dest.name)
                                    .unwrap()
                                    .clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    sources.extend(new_sources);
                }
                _ => break,
            }
        }
    }

    ssa_method
}
