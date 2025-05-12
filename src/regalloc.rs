use crate::cfg_build::VarLabel;
use crate::{cfg, deadcode, parse, reg_asm, scan};
use cfg::{Arg, BlockLabel, CfgType, ImmVar, MemVarLabel};
use maplit::{hashmap, hashset};
use parse::Primitive;
use reg_asm::Reg;
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
type CfgMethod = cfg::CfgMethod<VarLabel>;
type Jump = cfg::Jump<VarLabel>;
use crate::spilling_heuristics::rank_webs;
use std::cmp::max;

#[derive(PartialEq, Debug, Eq, Hash, Clone, Copy)]
pub struct InsnLoc {
    pub blk: BlockLabel,
    pub idx: usize,
}

#[derive(Clone, Debug)]
pub struct Web {
    pub var: VarLabel,
    pub defs: Vec<InsnLoc>,
    pub uses: Vec<InsnLoc>,
}

fn get_defs(m: &CfgMethod) -> HashMap<VarLabel, HashSet<InsnLoc>> {
    let mut defs: HashMap<VarLabel, HashSet<InsnLoc>> = HashMap::new();

    for (bid, block) in m.blocks.iter() {
        for (iid, instruction) in block.body.iter().enumerate() {
            for dest in get_insn_dest(&instruction) {
                if m.fields.keys().any(|x| *x == dest) {
                    defs.entry(dest)
                        .or_insert_with(HashSet::new)
                        .insert(InsnLoc {
                            blk: *bid,
                            idx: iid,
                        });
                }
            }
        }
    }
    defs
}

pub fn get_insn<T>(m: &cfg::CfgMethod<T>, i: InsnLoc) -> Sum<&cfg::Instruction<T>, &cfg::Jump<T>> {
    let blk = m.blocks.get(&i.blk).unwrap();
    if blk.body.len() == i.idx {
        Sum::Inr(&blk.jump_loc)
    } else {
        Sum::Inl(blk.body.get(i.idx).unwrap())
    }
}

fn get_children(m: &CfgMethod, i: InsnLoc) -> Vec<InsnLoc> {
    let blk = m.blocks.get(&i.blk).unwrap();
    if blk.body.len() == i.idx {
        match blk.jump_loc {
            Jump::Nowhere => vec![],
            Jump::Uncond(next_blk) => vec![InsnLoc {
                blk: next_blk,
                idx: 0,
            }],
            Jump::Cond {
                true_block,
                false_block,
                ..
            } => vec![
                InsnLoc {
                    blk: true_block,
                    idx: 0,
                },
                InsnLoc {
                    blk: false_block,
                    idx: 0,
                },
            ],
        }
    } else {
        vec![InsnLoc {
            blk: i.blk,
            idx: i.idx + 1,
        }]
    }
}

fn imm_var_sources<T: Copy>(iv: &ImmVar<T>) -> Vec<T> {
    match *iv {
        ImmVar::Var(a) => vec![a],
        ImmVar::Imm(_) => vec![],
    }
}

fn get_arg_sources<T: Copy>(arg: &Arg<T>) -> Vec<T> {
    match arg {
        Arg::VarArg(a) => imm_var_sources(a),
        Arg::StrArg(_) => vec![],
    }
}

fn get_insn_sources<T: Hash + Eq + Copy>(insn: &Instruction<T>) -> HashSet<T> {
    match insn {
        Instruction::LeftShift { source, .. } | Instruction::RightShift { source, .. } => {
            hashset! {*source}
        }
        Instruction::LoadParams { param } => hashset! {},
        Instruction::LoadString { dest, string } => hashset! {},
        Instruction::Push(x) => hashset! {*x},
        Instruction::Pop(_) => hashset! {},
        Instruction::StoreParam(_, a) => get_arg_sources(a).into_iter().collect(),
        Instruction::PhiExpr { .. } => panic!(),
        Instruction::ParMov(_) => panic!(),
        Instruction::LoadParam { .. } => hashset! {},
        Instruction::ArrayAccess { idx, .. } => imm_var_sources(idx).into_iter().collect(),
        Instruction::Call(_, args, _) => args.into_iter().flat_map(get_arg_sources).collect(),
        Instruction::Constant { .. } => hashset! {},
        Instruction::MoveOp { source, .. } => imm_var_sources(source).into_iter().collect(),
        Instruction::ThreeOp {
            source1, source2, ..
        } => imm_var_sources(source1)
            .into_iter()
            .chain(imm_var_sources(source2).into_iter())
            .collect(),
        Instruction::TwoOp { source1, .. } => imm_var_sources(source1).into_iter().collect(),
        Instruction::ArrayStore { source, idx, .. } => imm_var_sources(source)
            .into_iter()
            .chain(imm_var_sources(idx).into_iter())
            .collect(),
        Instruction::Spill { ord_var, .. } => hashset! {*ord_var},
        Instruction::Reload { .. } => hashset! {},
        Instruction::Ret(r) => match r {
            Some(ImmVar::Var(x)) => hashset! {*x},
            Some(ImmVar::Imm(_)) => hashset! {},
            None => hashset! {},
        },
    }
}

pub fn get_sources<T: Copy + Hash + Eq>(
    insn: &Sum<&cfg::Instruction<T>, &cfg::Jump<T>>,
) -> HashSet<T> {
    match insn {
        Sum::Inl(i) => get_insn_sources(i),
        Sum::Inr(j) => deadcode::get_jump_sources(j),
    }
}

pub fn get_dest<T: Copy + Eq + Hash>(
    insn: &Sum<&cfg::Instruction<T>, &cfg::Jump<T>>,
) -> HashSet<T> {
    match insn {
        Sum::Inl(i) => get_insn_dest(i),
        Sum::Inr(_) => hashset! {},
    }
}

pub fn get_insn_dest<T: Copy + Eq + Hash>(insn: &Instruction<T>) -> HashSet<T> {
    match insn {
        Instruction::LeftShift { dest, .. }
        | Instruction::RightShift { dest, .. }
        | Instruction::LoadString { dest, .. } => hashset! {*dest},
        Instruction::LoadParams { param } => param
            .iter()
            .map(|(param_num, var_name)| *var_name)
            .collect(),
        Instruction::Push(_) => hashset! {},
        Instruction::Pop(x) => hashset! {*x},
        Instruction::StoreParam(_, _) => hashset! {},
        Instruction::PhiExpr { dest, .. } => hashset! {*dest},
        Instruction::ParMov(_) => panic!(),
        Instruction::LoadParam { dest, .. }
        | Instruction::ArrayAccess { dest, .. }
        | Instruction::Constant { dest, .. }
        | Instruction::MoveOp { dest, .. }
        | Instruction::ThreeOp { dest, .. }
        | Instruction::TwoOp { dest, .. } => {
            hashset! {*dest}
        }
        Instruction::Call(_, _, dest) => match dest {
            Some(var) => hashset! {*var},
            None => hashset! {},
        },

        Instruction::ArrayStore { .. } => hashset! {},
        Instruction::Spill { .. } => hashset! {},
        Instruction::Reload { ord_var, .. } => hashset! {*ord_var},
        Instruction::Ret(_) => hashset! {},
    }
}

fn get_uses(m: &CfgMethod, x: VarLabel, i: InsnLoc) -> HashSet<InsnLoc> {
    let mut uses = HashSet::new();
    let mut seen = hashset! {};
    let mut next: Vec<_> = get_children(m, i);
    while let Some(v) = next.pop() {
        seen.insert(v);
        let insn = get_insn(m, v);
        if get_sources(&insn).contains(&x) {
            uses.insert(v);
        }
        if !get_dest(&insn).contains(&x) {
            next.append(
                &mut get_children(m, v)
                    .into_iter()
                    .filter(|i| !seen.contains(i))
                    .collect(),
            );
        }
    }
    uses
}

fn get_webs(m: &CfgMethod) -> Vec<Web> {
    let defs = get_defs(m);
    let mut webs = Vec::new();

    for (varname, def_set) in defs {
        // sets contains vec of def and set of uses
        let mut sets: Vec<(Vec<InsnLoc>, HashSet<InsnLoc>)> = def_set
            .iter()
            .map(|InsnLoc { blk: bid, idx: iid }| {
                (
                    vec![InsnLoc {
                        blk: *bid,
                        idx: *iid,
                    }],
                    get_uses(
                        &m,
                        varname,
                        InsnLoc {
                            blk: *bid,
                            idx: *iid,
                        },
                    ),
                )
            })
            .collect::<Vec<_>>();
        if sets.is_empty() {
            continue;
        }

        let mut new_sets = vec![];
        while let Some((mut curr_defs, curr_uses)) = sets.pop() {
            let mut to_merge = None;

            for (i, (defs, uses)) in sets.iter_mut().enumerate() {
                if !curr_uses.is_disjoint(uses) {
                    to_merge = Some((i, (defs.clone(), uses.clone())));
                    break;
                }
            }

            if let Some((i, (defs, uses))) = to_merge {
                sets.remove(i);
                let merged: HashSet<InsnLoc> = curr_uses.union(&uses).cloned().collect();
                curr_defs.extend(defs.clone());
                sets.push((curr_defs, merged));
            } else {
                new_sets.push((curr_defs, curr_uses));
            }
        }

        for (defs, uses) in new_sets {
            webs.push(Web {
                var: varname,
                defs,
                uses: uses.into_iter().collect(),
            });
        }
    }
    webs
}

pub fn all_insn_locs<T>(m: &cfg::CfgMethod<T>) -> HashSet<InsnLoc> {
    let mut ret = HashSet::new();
    for (bid, blk) in m.blocks.iter() {
        for i in 0..(blk.body.len() + 1) {
            ret.insert(InsnLoc { blk: *bid, idx: i });
        }
    }
    ret
}

fn insn_loc_graph_reversed(m: &CfgMethod) -> HashMap<InsnLoc, HashSet<InsnLoc>> {
    let mut ret = HashMap::new();
    for iloc in all_insn_locs(m) {
        ret.insert(iloc, HashSet::new());
    }
    for iloc in all_insn_locs(m) {
        for child in get_children(m, iloc) {
            ret.get_mut(&child).unwrap().insert(iloc);
        }
    }
    ret
}

fn reachable_from_defs(m: &CfgMethod, web: &Web) -> HashSet<InsnLoc> {
    let mut reachable = HashSet::new();
    let mut next: Vec<_> = web
        .defs
        .iter()
        .map(|i| get_children(m, *i))
        .flatten()
        .collect();
    let mut seen = HashSet::new();
    while let Some(v) = next.pop() {
        seen.insert(v);
        reachable.insert(v);
        let insn = get_insn(m, v);
        if !get_dest(&insn).contains(&web.var) {
            next.append(
                &mut get_children(m, v)
                    .into_iter()
                    .filter(|i| !seen.contains(i))
                    .collect(),
            );
        }
    }
    if let Some(def0) = web.defs.get(0) {
        if reachable.contains(def0) {
            reachable.remove(def0);
            println!("def0: {def0:?}");
            println!("how");
        }
    }
    reachable
}

fn reaches_a_use(m: &CfgMethod, web: &Web) -> HashSet<InsnLoc> {
    let mut reaches_a_use = HashSet::new();
    let g = insn_loc_graph_reversed(m);
    let mut next: Vec<_> = web.uses.clone();
    let mut seen = HashSet::new();
    while let Some(v) = next.pop() {
        seen.insert(v);
        reaches_a_use.insert(v);
        let insn = get_insn(m, v);
        if !get_dest(&insn).contains(&web.var) {
            next.append(
                &mut g
                    .get(&v)
                    .unwrap()
                    .into_iter()
                    .map(|i| *i)
                    .filter(|i| !seen.contains(i))
                    .collect(),
            );
        }
    }
    reaches_a_use
}

pub fn find_inter_instructions(m: &CfgMethod, web: &Web) -> HashSet<InsnLoc> {
    reaches_a_use(m, web)
        .intersection(&reachable_from_defs(m, web))
        .map(|i| *i)
        .collect()
}

fn distinct_pairs<'a, T>(l: &'a Vec<T>) -> Vec<(u32, &'a T, u32, &'a T)> {
    let mut ret = Vec::new();
    for i in 0..l.len() {
        for j in 0..l.len() {
            if i != j {
                ret.push((i as u32, l.get(i).unwrap(), j as u32, l.get(j).unwrap()));
            }
        }
    }
    ret
}

fn lower_calls_insn<T: Clone, U>(i: Instruction<T>, _: U) -> Vec<Instruction<T>> {
    if let Instruction::Call(name, args, dest) = i {
        let mut insns: Vec<_> = vec![];

        if args.len() % 2 == 1 && args.len() >= 6 {
            insns.push(Instruction::StoreParam(7, Arg::VarArg(ImmVar::Imm(0))))
        }

        insns.extend(
            args.iter()
                .enumerate()
                .skip(6)
                .map(|(n, arg)| Instruction::StoreParam(n as u16, arg.clone())),
        );

        insns.push(Instruction::Call(
            name,
            args.into_iter().take(6).map(|x| x.clone()).collect(),
            dest,
        ));

        insns
    } else {
        vec![i]
    }
}

fn reg_of_argnum(n: u32) -> Option<Reg> {
    match n {
        0 => Some(Reg::Rdi),
        1 => Some(Reg::Rsi),
        2 => Some(Reg::Rdx),
        3 => Some(Reg::Rcx),
        4 => Some(Reg::R8),
        5 => Some(Reg::R9),
        _ => None,
    }
}

fn dummy_with_same_type(
    fields: &mut HashMap<VarLabel, (CfgType, String)>,
    v: VarLabel,
) -> VarLabel {
    // 1000 avoids collision with global vars
    let dummy = (*fields.keys().max().unwrap() as usize + 1000) as u32;
    fields.insert(dummy, (*fields.get(&v).unwrap()).clone());
    dummy
}

fn corresponding_memvar(fields: &mut HashMap<VarLabel, (CfgType, String)>, r: Reg) -> VarLabel {
    // 1000 avoids collision with global vars
    let ret = (*fields.keys().max().unwrap_or(&0) as usize + 1000) as u32;
    fields.insert(ret, (CfgType::Scalar(Primitive::LongType), format!("{r}")));
    ret
}

/*
Change

LoadParam(x, 0)

into

LoadParam(dummy_var, 0)
Mov(dummy_var, x).

Similarly for StoreParam.  We do this because coloring will have the constraint that
LoadParam(v, 0) means that v has to be assigned the register rdi, etc.

However, we leave LoadParam(x, 5) and LoadParam(x, 2) alone, because the 5th arg reg is r9, and the 2nd arg reg is rdx, and we do not use either of these for coloring.

We also leave alone LoadParam(x, n) for n >= 6, since these guys go on the stack.

Also, if we see StoreParam("some_string", n), we leave that alone.


Also: for each dummy variable, add to reg_of_varlabel that dummy maps to the reg it should be given
 */
fn make_args_easy_to_color(
    i: Instruction<VarLabel>,
    regs_colored: &Vec<Reg>,
    args_to_dummy_vars: &HashMap<u32, u32>,
) -> Vec<Instruction<VarLabel>> {
    match i {
        Instruction::LoadParam { param, dest } => {
            if let Some(reg) = reg_of_argnum(param as u32) {
                if regs_colored.contains(&reg) {
                    let dummy = *args_to_dummy_vars.get(&(param as u32)).unwrap();
                    vec![
                        Instruction::LoadParam { param, dest: dummy },
                        Instruction::MoveOp {
                            source: ImmVar::Var(dummy),
                            dest,
                        },
                    ]
                } else {
                    vec![i]
                }
            } else {
                vec![i]
            }
        }
        Instruction::LoadParams { param } => {
            let mut instructions = vec![];
            let new_params = param
                .iter()
                .map(|(param_num, var_name)| {
                    (
                        *param_num,
                        *args_to_dummy_vars.get(&(*param_num as u32)).unwrap(),
                    )
                })
                .collect::<Vec<_>>();

            instructions.push(Instruction::LoadParams { param: new_params });
            param.iter().for_each(|(param_num, var_name)| {
                let dummy = *args_to_dummy_vars.get(&(*param_num as u32)).unwrap();
                instructions.push(Instruction::MoveOp {
                    source: ImmVar::Var(dummy),
                    dest: *var_name,
                });
            });
            instructions
        }
        Instruction::Call(func_name, params, ret_val) => {
            let mut instructions = vec![];
            let mut new_arguments = vec![];

            for (i, param) in params.iter().enumerate() {
                match param {
                    Arg::VarArg(v) => {
                        if i < 6 {
                            let dummy = args_to_dummy_vars.get(&(i as u32)).unwrap();
                            if let Some(reg) = reg_of_argnum(i as u32) {
                                if regs_colored.contains(&reg) {
                                    match v {
                                        ImmVar::Var(_) => {
                                            instructions.push(Instruction::MoveOp {
                                                source: v.clone(),
                                                dest: *dummy,
                                            });
                                        }
                                        ImmVar::Imm(i) => {
                                            instructions.push(Instruction::Constant {
                                                dest: *dummy,
                                                constant: *i,
                                            })
                                        }
                                    }
                                    new_arguments.push(Arg::VarArg(ImmVar::Var(*dummy)));
                                } else {
                                    new_arguments.push(param.clone());
                                }
                            } else {
                                new_arguments.push(param.clone());
                            }
                        } else {
                            panic!();
                            //instructions.push(Instruction::StoreParam(i as u16, param.clone()))
                        }
                    }
                    Arg::StrArg(s) => {
                        if i < 6 {
                            let dummy = args_to_dummy_vars.get(&(i as u32)).unwrap();
                            instructions.push(Instruction::LoadString {
                                dest: *dummy,
                                string: s.clone(),
                            });
                            new_arguments.push(Arg::VarArg(ImmVar::Var(*dummy)));
                        } else {
                            panic!();
                        }
                    }
                }
            }
            instructions.push(Instruction::Call(func_name, new_arguments, ret_val));
            instructions
        }

        Instruction::StoreParam(param, Arg::VarArg(ImmVar::Var(_))) => {
            println!("ctrlf: {i}");
            if param < 6 {
                panic!("bad");
            } else {
                vec![i]
            }
        }
        _ => vec![i],
    }
}

// returns a tuple: a list of the phi webs, and the interference graph, where webs are labelled by their indices in the list.
fn interference_graph(
    m: &CfgMethod,
    arg_reg_lookup: &HashMap<u32, Reg>,
) -> (Vec<Web>, HashMap<u32, HashSet<u32>>, HashMap<u32, Reg>) {
    let webs = get_webs(m);

    // Initialize graph
    let mut graph: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (i, _) in webs.iter().enumerate() {
        graph.insert(i as u32, HashSet::new());
    }

    // find webs defined over argument variables, and make sure they go into their correct argument

    let mut precoloring: HashMap<u32, Reg> = hashmap! {};
    for (i, Web { var, .. }) in webs.iter().enumerate() {
        match arg_reg_lookup.get(&var) {
            Some(reg) => {
                precoloring.insert(i as u32, *reg);
            }
            None => (),
        }
    }

    // find convex closures
    let convex_closures_of_webs = webs
        .iter()
        .map(|web| find_inter_instructions(m, web))
        .collect();

    // this should leave the dummy webs isolated since they do not interfere with anything
    for (i, ccwi, j, ccwj) in distinct_pairs(&convex_closures_of_webs) {
        if !ccwi.is_disjoint(ccwj) {
            graph.get_mut(&i).unwrap().insert(j);
        }
    }

    (webs, graph, precoloring)
}

fn remove_node(g: &mut HashMap<u32, HashSet<u32>>, v: u32) -> HashSet<u32> {
    let result = g.remove(&v).unwrap();
    for (_, es) in g {
        es.remove(&v);
    }
    result
}

fn color_not_in_set(regs_can_use: &Vec<Reg>, s: HashSet<Reg>) -> Reg {
    let all_colors: HashSet<Reg> = regs_can_use.into_iter().map(|x| *x).collect();
    *all_colors.difference(&s).collect::<Vec<_>>().pop().unwrap()
}

fn color(
    mut g: HashMap<u32, HashSet<u32>>,
    precoloring: HashMap<u32, Reg>,
    regs_can_use: &Vec<Reg>,
) -> Result<HashMap<u32, Reg>, Vec<u32>> {
    let mut color_later = Vec::new();

    loop {
        let mut to_remove = None;
        for (v, es) in g.iter() {
            let uncolored = es
                .iter()
                .filter(|e| precoloring.get(e) == None)
                .collect::<Vec<_>>()
                .len();
            let colors: HashSet<_> = es.iter().filter_map(|e| precoloring.get(e)).collect();
            if (uncolored + colors.len() < regs_can_use.len()) && precoloring.get(v) == None {
                to_remove = Some(*v);
                break;
            };
        }
        match to_remove {
            Some(v) => color_later.push((v, remove_node(&mut g, v))),
            None => break,
        }
    }
    if g.iter().all(|(v, _)| precoloring.contains_key(v)) {
        let mut colors = precoloring.clone();
        while let Some((v, es)) = color_later.pop() {
            colors.insert(
                v,
                color_not_in_set(
                    regs_can_use,
                    es.into_iter().map(|u| *colors.get(&u).unwrap()).collect(),
                ),
            );
        }
        // return an assignment of colors
        Ok(colors)
    } else {
        // return the nodes that could not be colored, sorted in order of how nice it'd be to color them
        println!("could not color this many nodes: {}", g.len());
        let to_color: HashSet<_> = g.keys().map(|x| *x).collect();

        let mut vec_to_color = to_color.into_iter().collect::<Vec<_>>();
        vec_to_color.sort_by_key(|x| g.get(x).unwrap().len() as u32);
        Err(vec_to_color)
    }
}

fn spill_web(m: cfg::CfgMethod<VarLabel>, web: Web) -> cfg::CfgMethod<VarLabel> {
    let mut new_method = m.clone();
    let Web { var, defs, uses } = web;

    for (bid, block) in m.blocks.iter() {
        let mut new_instructions = vec![];
        let mut new_block = block.clone();
        for (iid, instruction) in block.body.iter().enumerate() {
            let instr_loc = InsnLoc {
                blk: *bid,
                idx: iid,
            };

            if defs.contains(&instr_loc) {
                // define variable then spill
                new_instructions.push(instruction.clone());
                new_instructions.push(Instruction::Spill {
                    ord_var: var,
                    mem_var: var,
                });
            } else if uses.contains(&instr_loc) {
                new_instructions.push(Instruction::Reload {
                    ord_var: var,
                    mem_var: var,
                });
                new_instructions.push(instruction.clone());
            } else {
                new_instructions.push(instruction.clone());
            }

            if iid == block.body.len() - 1 {
                let potential_jump_insn_loc = InsnLoc {
                    blk: *bid,
                    idx: iid + 1,
                };

                if uses.contains(&potential_jump_insn_loc) {
                    new_instructions.push(Instruction::Reload {
                        ord_var: var,
                        mem_var: var,
                    });
                }
            }
        }
        new_block.body = new_instructions;
        new_method.blocks.insert(*bid, new_block);
    }

    return new_method;
}

fn make_argument_variables(m: &mut CfgMethod) -> HashMap<u32, u32> {
    let start = match m.fields.keys().max() {
        Some(max_key) => {
            if *max_key > 1000 {
                max_key + 1
            } else {
                1000
            }
        }
        None => 1000,
    };

    let mut arg_variables: HashMap<u32, u32> = hashmap! {};
    for i in 0..6 {
        m.fields.insert(
            start + i,
            (CfgType::Scalar(Primitive::LongType), format!("ARG{i}")),
        );

        arg_variables.insert(i, start + i);
    }

    arg_variables
}

fn is_trivial(w: &Web) -> bool {
    if w.uses.len() == 1 && w.defs.len() == 1 {
        let the_use = w.uses.clone().pop().unwrap();
        let the_def = w.defs.clone().pop().unwrap();
        the_use.blk == the_def.blk
            && (the_use.idx == the_def.idx + 1 || the_use.idx == the_def.idx + 2)
    } else {
        false
    }
}

fn max_degree(interfer_graph: HashMap<u32, HashSet<u32>>) -> u32 {
    let mut max_degree: u32 = 0;
    interfer_graph
        .values()
        .for_each(|c| max_degree = max(c.len() as u32, max_degree));
    max_degree
}

fn reg_alloc(
    m: &CfgMethod,
    all_regs: &Vec<Reg>,
    arg_var_to_reg: &HashMap<u32, Reg>,
) -> (CfgMethod, HashMap<u32, Reg>, Vec<Web>) {
    let mut new_method = m.clone();
    // try to color the graph
    loop {
        let (webs, interfer_graph, precoloring) = interference_graph(&new_method, arg_var_to_reg);
        let convex_closures_of_webs: Vec<HashSet<InsnLoc>> = webs
            .iter()
            .map(|web| find_inter_instructions(&new_method, web))
            .collect();

        match color(interfer_graph.clone(), precoloring, all_regs) {
            Ok(web_coloring) => {
                return (new_method.clone(), web_coloring, webs);
            }
            Err(things_to_spill) => {
                println!("failed to color");
                let mut will_spill: HashSet<u32> = HashSet::new();
                for i in all_insn_locs(&new_method) {
                    let bad_webs = webs
                        .iter()
                        .enumerate()
                        .zip(convex_closures_of_webs.iter())
                        .filter(|((web_num, _), ccw)| {
                            ccw.contains(&i) && !will_spill.contains(&(*web_num as u32))
                        })
                        .map(|((i, web), _)| (i as u32, web))
                        .collect();
                    let webs_to_remove: Vec<u32> = rank_webs(bad_webs, &interfer_graph)
                        .iter()
                        .map(|x| (*x).clone())
                        .collect();
                    let n = webs_to_remove.len();
                    for web in webs_to_remove
                        .iter()
                        .filter(|web_num| {
                            let web = webs.get(**web_num as usize).unwrap();
                            !is_trivial(&web) && !arg_var_to_reg.contains_key(&web.var)
                        })
                        .take(std::cmp::max(
                            (n as i32 - all_regs.len() as i32) as usize,
                            1 as usize,
                        ))
                        .collect::<Vec<_>>()
                    {
                        will_spill.insert(*web);
                    }
                }

                for web_num in will_spill {
                    let web = webs.get(web_num as usize).unwrap();
                    println!("spilling {web:?}");
                    //println!("method is {new_method}");
                    new_method = spill_web(new_method, web.clone());
                }

                // let mut spillable = None;
                // for web_to_spill in things_to_spill.into_iter().rev() {
                //     let web = webs.get(web_to_spill as usize).unwrap();
                //     if !is_trivial(&web)
                //         && !arg_var_to_reg.contains_key(&web.var)
                //         && web.var != u32::MAX
                //     {
                //         println!("arg_var_to_reg: {arg_var_to_reg:?}");
                //         println!("max degree: {}", max_degree(interfer_graph));
                //         println!(/*"spilling web number {web_to_spill}, */ " which is {web:?}");
                //         spillable = Some(web);
                //         break;
                //     }
                // }

                // match spillable {
                //     Some(spillable) => {
                //         // println!("spilling {spillable:?}");
                //         // println!("method is {new_method}");
                //         new_method = spill_web(&mut new_method, spillable.clone());
                //     }
                //     None => panic!("nothing to spill"),
                // }
            }
        }
    }
}

fn imm_map<T, U>(iv: ImmVar<T>, f: impl Fn(T) -> U) -> ImmVar<U> {
    match iv {
        ImmVar::Var(u) => ImmVar::Var(f(u)),
        ImmVar::Imm(i) => ImmVar::Imm(i),
    }
}

fn arg_map<T, U>(a: Arg<T>, f: impl Fn(T) -> U) -> Arg<U> {
    match a {
        Arg::StrArg(s) => Arg::StrArg(s),
        Arg::VarArg(v) => Arg::VarArg(imm_map(v, f)),
    }
}

use cfg::Instruction;
fn insn_map<T: Clone, U>(
    instr: cfg::Instruction<T>,
    src_fun: impl Fn(T) -> U,
    dst_fun: impl Fn(T) -> U,
) -> cfg::Instruction<U> {
    match instr {
        Instruction::LoadParams { param } => {
            let new_params = param
                .iter()
                .map(|(param_num, var_name)| (*param_num, dst_fun(var_name.clone())))
                .collect();

            Instruction::LoadParams { param: new_params }
        }

        Instruction::LoadString { dest, string } => Instruction::LoadString {
            dest: dst_fun(dest),
            string: string,
        },
        Instruction::LeftShift {
            dest,
            source,
            shift,
        } => Instruction::LeftShift {
            dest: dst_fun(dest),
            source: src_fun(source),
            shift: shift,
        },
        Instruction::RightShift {
            dest,
            source,
            shift,
        } => Instruction::RightShift {
            dest: dst_fun(dest),
            source: src_fun(source),
            shift: shift,
        },
        Instruction::Pop(x) => Instruction::Pop(dst_fun(x)),
        Instruction::Push(x) => Instruction::Push(src_fun(x)),
        cfg::Instruction::StoreParam(param, a) => {
            Instruction::StoreParam(param, arg_map(a, src_fun))
        }
        cfg::Instruction::ParMov(_) => panic!(),
        cfg::Instruction::ArrayAccess { dest, name, idx } => Instruction::ArrayAccess {
            dest: dst_fun(dest),
            name,
            idx: imm_map(idx, src_fun),
        },
        cfg::Instruction::ArrayStore { source, arr, idx } => Instruction::ArrayStore {
            source: imm_map(source, &src_fun),
            arr,
            idx: imm_map(idx, &src_fun),
        },
        Instruction::Call(string, args, opt_ret_val) => Instruction::Call(
            string,
            args.into_iter()
                .map(|a| arg_map(a, &src_fun))
                .collect::<Vec<_>>(),
            opt_ret_val.map(&dst_fun),
        ),
        Instruction::Constant { dest, constant } => Instruction::Constant {
            dest: dst_fun(dest),
            constant,
        },
        Instruction::LoadParam { dest, param } => Instruction::LoadParam {
            dest: dst_fun(dest),
            param,
        },
        Instruction::MoveOp { source, dest } => Instruction::MoveOp {
            source: imm_map(source, src_fun),
            dest: dst_fun(dest),
        },
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => Instruction::ThreeOp {
            source1: imm_map(source1, &src_fun),
            source2: imm_map(source2, &src_fun),
            dest: dst_fun(dest),
            op,
        },
        Instruction::TwoOp { source1, dest, op } => Instruction::TwoOp {
            source1: imm_map(source1, src_fun),
            dest: dst_fun(dest),
            op,
        },
        Instruction::PhiExpr { .. } => panic!(),
        Instruction::Ret(opt_ret_val) => Instruction::Ret(opt_ret_val.map(|x| imm_map(x, src_fun))),
        Instruction::Spill { ord_var, mem_var } => Instruction::Spill {
            ord_var: src_fun(ord_var),
            mem_var,
        },
        Instruction::Reload { ord_var, mem_var } => Instruction::Reload {
            ord_var: dst_fun(ord_var),
            mem_var,
        },
    }
}

fn jump_map<T, U>(jmp: cfg::Jump<T>, src_fun: impl Fn(T) -> U) -> cfg::Jump<U> {
    match jmp {
        cfg::Jump::Uncond(lbl) => cfg::Jump::Uncond(lbl),
        cfg::Jump::Nowhere => cfg::Jump::Nowhere,
        cfg::Jump::Cond {
            source,
            true_block,
            false_block,
        } => cfg::Jump::Cond {
            source: imm_map(source, src_fun),
            true_block,
            false_block,
        },
    }
}

// returns index of web, None if v is global
fn get_src_web(v: VarLabel, i: InsnLoc, webs: &Vec<Web>) -> Option<u32> {
    webs.iter()
        .enumerate()
        .find(|(_, Web { var, uses, .. })| *var == v && uses.contains(&i))
        .map(|(i, _)| i as u32)
}

fn get_dst_web(v: VarLabel, i: InsnLoc, webs: &Vec<Web>) -> Option<u32> {
    webs.iter()
        .enumerate()
        .find(|(_, Web { var, defs, .. })| *var == v && defs.contains(&i))
        .map(|(i, _)| i as u32)
}

fn src_reg(
    v: VarLabel,
    i: InsnLoc,
    webs: &Vec<Web>,
    web_to_reg: &HashMap<u32, Reg>,
) -> Sum<Reg, MemVarLabel> {
    // println!("v, i, webs: {v:?}, {i:?}, {webs:?}");
    match get_src_web(v, i, webs) {
        Some(j) => Sum::Inl(*web_to_reg.get(&j).unwrap()),
        None => Sum::Inr(v),
    }
}

fn dst_reg(
    v: VarLabel,
    i: InsnLoc,
    webs: &Vec<Web>,
    web_to_reg: &HashMap<u32, Reg>,
) -> Sum<Reg, MemVarLabel> {
    // println!("v, i, webs: {v:?}, {i:?}, {webs:?}");
    // println!("get_dst_web(v, i, webs): {:?}", get_dst_web(v, i, webs));
    match get_dst_web(v, i, webs) {
        Some(j) => Sum::Inl(*web_to_reg.get(&j).unwrap()),
        None => Sum::Inr(v),
    }
}

fn to_regs(
    m: cfg::CfgMethod<VarLabel>,
    web_to_reg: &HashMap<u32, Reg>,
    webs: &Vec<Web>,
) -> cfg::CfgMethod<Sum<Reg, MemVarLabel>> {
    let new_fields = all_mem_vars(&m);
    let new_blocks = m.blocks.into_iter().map(|(lbl, blk)| {
        (
            lbl,
            cfg::BasicBlock::<Sum<Reg, MemVarLabel>> {
                jump_loc: jump_map(blk.jump_loc, |v| {
                    src_reg(
                        v,
                        InsnLoc {
                            blk: lbl,
                            idx: blk.body.len(),
                        },
                        &webs,
                        &web_to_reg,
                    )
                }),
                body: blk
                    .body
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(idx, insn)| {
                        insn_map(
                            insn,
                            |v| src_reg(v, InsnLoc { blk: lbl, idx }, &webs, &web_to_reg),
                            |v| dst_reg(v, InsnLoc { blk: lbl, idx }, &webs, &web_to_reg),
                        )
                    })
                    .collect(),
            },
        )
    });
    cfg::CfgMethod::<Sum<Reg, MemVarLabel>> {
        name: m.name,
        num_params: m.num_params,
        blocks: new_blocks.collect(),
        fields: m
            .fields
            .into_iter()
            .filter(|(_, (t, _))| match t {
                CfgType::Scalar(_) => false,
                CfgType::Array(_, _) => true,
            })
            .chain(new_fields.into_iter())
            .collect(),
        return_type: m.return_type,
    }
}

fn all_mem_vars(m: &cfg::CfgMethod<VarLabel>) -> HashMap<u32, (CfgType, String)> {
    let mut ret = HashMap::new();
    for (_, blk) in m.blocks.iter() {
        for insn in blk.body.iter() {
            match insn {
                Instruction::Spill { mem_var, ord_var } => match m.fields.get(ord_var) {
                    Some(stuff) => {
                        ret.insert(*mem_var, stuff.clone());
                    }
                    None => {
                        println!("the variable {ord_var} cant be found");
                        println!("{}", m);
                        panic!();
                    }
                },
                _ => (),
            }
        }
    }
    ret
}

fn build_need_to_save(
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    m: &cfg::CfgMethod<Sum<Reg, MemVarLabel>>,
    web_to_reg: &HashMap<u32, Reg>,
) -> HashMap<InsnLoc, HashSet<Reg>> {
    all_insn_locs(m)
        .into_iter()
        .filter_map(|i| {
            if let Sum::Inl(Instruction::Call(_, _, _)) = get_insn(m, i) {
                let mut regs = HashSet::new();
                for ((webnum, web), ccw) in webs.iter().enumerate().zip(ccws) {
                    let child = InsnLoc {
                        blk: i.blk,
                        idx: i.idx + 1,
                    };
                    if ccw.contains(&i) && ccw.contains(&child) {
                        println!("webnum: {webnum}\n, web: {web:?}\n, loc: {i:?}");
                        println!(
                            "instruction: {}, reg: {}",
                            get_insn(m, i),
                            *web_to_reg.get(&(webnum as u32)).unwrap()
                        );
                        println!("ccw: {ccw:?}");
                        regs.insert(*web_to_reg.get(&(webnum as u32)).unwrap());
                    }
                }
                Some((i, regs))
            } else {
                None
            }
        })
        .collect()
}

// adds pushes and pops of caller-saved regs when necessary
fn push_and_pop(
    m: cfg::CfgMethod<Sum<Reg, MemVarLabel>>,
    caller_saved_regs: &Vec<Reg>,
    caller_saved_memvars: &HashMap<Reg, MemVarLabel>,
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    web_to_reg: &HashMap<u32, Reg>,
) -> cfg::CfgMethod<Sum<Reg, MemVarLabel>> {
    let need_to_save = build_need_to_save(webs, ccws, &m, web_to_reg);
    method_map(m, |i, loc| match need_to_save.get(&loc) {
        None => vec![i],
        Some(used) => {
            let regs = caller_saved_regs.into_iter().filter(|r| used.contains(r));
            let mut ret: Vec<_> = regs
                .clone()
                .map(|reg| Instruction::Spill {
                    ord_var: Sum::Inl(*reg),
                    mem_var: *caller_saved_memvars.get(reg).unwrap(),
                })
                .collect();
            ret.push(i);
            ret.append(
                &mut regs
                    .rev()
                    .map(|reg| Instruction::Reload {
                        ord_var: Sum::Inl(*reg),
                        mem_var: *caller_saved_memvars.get(reg).unwrap(),
                    })
                    .collect(),
            );
            ret
        }
    })
}

fn regalloc_method(m: cfg::CfgMethod<VarLabel>) -> cfg::CfgMethod<Sum<Reg, MemVarLabel>> {
    // callee-saved regs: RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15,
    let callee_saved_regs = vec![Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];
    let caller_saved_regs: Vec<Reg> =
        vec![Reg::Rsi, Reg::Rcx, Reg::R11, Reg::Rdi, Reg::R8, Reg::R10];
    let mut m = m.clone();

    let mut all_regs = callee_saved_regs.clone();
    all_regs.append(&mut caller_saved_regs.clone());

    let args_to_dummy_vars = make_argument_variables(&mut m);
    let reg_of_varname = vec![
        (0, Reg::Rdi),
        (1, Reg::Rsi),
        (2, Reg::Rdx),
        (3, Reg::Rcx),
        (4, Reg::R8),
        (5, Reg::R9),
    ]
    .into_iter()
    .filter(|(_, r)| caller_saved_regs.contains(&r))
    .map(|(arg_num, reg)| (*args_to_dummy_vars.get(&arg_num).unwrap(), reg))
    .collect::<HashMap<_, _>>();

    println!("here m is {m}");
    let m = method_map(m.clone(), |i, _| {
        make_args_easy_to_color(i, &all_regs, &args_to_dummy_vars)
    });

    println!("method after thing: {m}");
    println!("reg_of_varname: {reg_of_varname:?}");

    let (m, web_to_reg, webs) = reg_alloc(&m, &all_regs, &reg_of_varname);
    let ccws = webs
        .iter()
        .map(|web| find_inter_instructions(&m, web))
        .collect();
    println!("webs: {webs:?}");

    // println!("web_to_reg: {:?}", web_to_reg);
    // println!("before renaming: {spilled_method}");
    let mut m = to_regs(m, &web_to_reg, &webs);
    let caller_saved_memvars = caller_saved_regs
        .iter()
        .map(|reg| (*reg, corresponding_memvar(&mut m.fields, *reg)))
        .collect();
    let m = push_and_pop(
        m,
        &caller_saved_regs,
        &caller_saved_memvars,
        &webs,
        &ccws,
        &web_to_reg,
    );
    // println!("after renaming: {x}");
    m
}

fn method_map<T>(
    mut m: cfg::CfgMethod<T>,
    mut f: impl FnMut(Instruction<T>, InsnLoc) -> Vec<Instruction<T>>,
) -> cfg::CfgMethod<T> {
    let mut blocks = Vec::new();
    for (lbl, mut blk) in m.blocks.into_iter() {
        let mut body = Vec::new();
        for (idx, insn) in blk.body.into_iter().enumerate() {
            body.append(&mut f(insn, InsnLoc { blk: lbl, idx }));
        }
        blk.body = body;
        blocks.push((lbl, blk));
    }
    m.blocks = blocks.into_iter().collect();
    m
}

fn prog_map(
    mut p: cfg::CfgProgram<VarLabel>,
    f: impl Fn(cfg::CfgMethod<VarLabel>) -> cfg::CfgMethod<VarLabel>,
) -> cfg::CfgProgram<VarLabel> {
    p.methods = p.methods.into_iter().map(f).collect();

    p
}

pub fn regalloc_prog(p: cfg::CfgProgram<VarLabel>) -> cfg::CfgProgram<Sum<Reg, MemVarLabel>> {
    let p = prog_map(p, |m| method_map(m, lower_calls_insn));
    cfg::CfgProgram {
        externals: p.externals,
        global_fields: p.global_fields,
        methods: p.methods.into_iter().map(regalloc_method).collect(),
    }
}

// mod tests {
//     use super::*;
//     use ImmVar::Var;

//     #[test]
//     fn easy() {
//         let m = cfg::CfgMethod {
//             num_params: 0,
//             name: "".to_string(),
//             blocks: vec![(
//                 0,
//                 BasicBlock {
//                     jump_loc: Jump::Nowhere,
//                     body: vec![
//                         Instruction::Constant {
//                             dest: 1,
//                             constant: 17,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(2),
//                             dest: 3,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(1),
//                             dest: 2,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(1),
//                             dest: 3,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(1),
//                             dest: 1,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(1),
//                             dest: 2,
//                         },
//                         Instruction::MoveOp {
//                             source: Var(2),
//                             dest: 1,
//                         },
//                     ],
//                 },
//             )]
//             .into_iter()
//             .collect(),
//             fields: HashMap::new(),
//             return_type: None,
//         };
//         assert_eq!(
//             get_uses(&m, 1, InsnLoc { blk: 0, idx: 0 }),
//             hashset! {InsnLoc { blk: 0, idx: 3 }, InsnLoc { blk: 0, idx: 2 }, InsnLoc { blk: 0, idx: 4 }}
//         );
//         // println!("graph: {:?}", interference_graph(&mut m.clone()));
//         // println!("coloring: {:?}", reg_alloc(&mut m.clone(), 2));
//         // println!("{:?}", regalloc_method(m));
//     }
// }
