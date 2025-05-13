use crate::cfg::IsImmediate;
use crate::cfg_build::VarLabel;
use crate::{cfg, deadcode, parse, reg_asm, scan};
use cfg::{Arg, BlockLabel, CfgType, ImmVar, MemVarLabel};
use core::fmt;
use maplit::{hashmap, hashset};
use parse::Primitive;
use reg_asm::Reg;
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
type CfgMethod = cfg::CfgMethod<VarLabel>;
type Jump = cfg::Jump<VarLabel>;

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

#[derive(PartialEq, Debug, Eq, Hash, Clone, Copy)]
pub enum RegGlobMemVar {
    RegVar(Reg),
    GlobVar(MemVarLabel),
    MemVar(MemVarLabel),
}

impl fmt::Display for RegGlobMemVar {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            RegVar(r) => write!(f, "{}", r),
            GlobVar(v) => write!(f, "global_{}", v),
            MemVar(v) => write!(f, "memvar_{}", v),
        }
    }
}
use RegGlobMemVar::*;

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

fn get_children<T>(m: &cfg::CfgMethod<T>, i: InsnLoc) -> Vec<InsnLoc> {
    let blk = m.blocks.get(&i.blk).unwrap();
    if blk.body.len() == i.idx {
        match blk.jump_loc {
            cfg::Jump::Nowhere => vec![],
            cfg::Jump::Uncond(next_blk) => vec![InsnLoc {
                blk: next_blk,
                idx: 0,
            }],
            cfg::Jump::Cond {
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
        Instruction::LoadParams { .. } => hashset! {},
        Instruction::LoadString { .. } => hashset! {},
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
        Instruction::LoadParams { param } => param.iter().map(|(_, var_name)| *var_name).collect(),
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
            // println!("def0: {def0:?}");
            // println!("how");
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
                .map(|(param_num, _)| {
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
            if param < 6 {
                panic!("bad");
            } else {
                vec![i]
            }
        }
        _ => vec![i],
    }
}

fn ranges_disjoint(fst: &(usize, usize), snd: &(usize, usize)) -> bool {
    let (fst_lo, fst_hi) = fst;
    let (snd_lo, snd_hi) = snd;
    !(fst_lo <= snd_hi && snd_lo <= fst_hi)
}

fn ranges_all_disjoint(fst: &Vec<(usize, usize)>, snd: &Vec<(usize, usize)>) -> bool {
    fst.iter()
        .all(|r1| snd.iter().all(|r2| ranges_disjoint(r1, r2)))
}

fn ccws_disjoint(
    fst: &HashMap<BlockLabel, Vec<(usize, usize)>>,
    snd: &HashMap<BlockLabel, Vec<(usize, usize)>>,
) -> bool {
    for (blk, fst_ranges) in fst {
        match snd.get(&blk) {
            None => (),
            Some(snd_ranges) => {
                if !ranges_all_disjoint(fst_ranges, snd_ranges) {
                    return false;
                }
            }
        }
    }
    true
}

// returns a tuple: a list of the phi webs, and the interference graph, where webs are labelled by their indices in the list.
fn interference_graph(
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    arg_reg_lookup: &HashMap<u32, Reg>,
) -> (HashMap<u32, HashSet<u32>>, HashMap<u32, Reg>) {
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

    let better_ccws: Vec<HashMap<BlockLabel, Vec<(usize, usize)>>> = ccws
        .iter()
        .map(|ccw| {
            let mut thing: HashMap<BlockLabel, _> = HashMap::new();
            for InsnLoc { blk, idx } in ccw {
                match thing.get_mut(blk) {
                    None => {
                        thing.insert(*blk, vec![*idx]);
                    }
                    Some(ranges) => {
                        ranges.push(*idx);
                    }
                }
            }
            thing
                .into_iter()
                .map(|(blk, mut idcs)| {
                    idcs.sort();
                    let (mut lo, mut hi) = (None, None);
                    let mut ranges = vec![];
                    for idx in idcs {
                        match (lo, hi) {
                            (None, None) => (lo, hi) = (Some(idx), Some(idx)),
                            (Some(lo_val), Some(hi_val)) => {
                                if hi_val + 1 == idx {
                                    hi = Some(idx);
                                } else {
                                    ranges.push((lo_val, hi_val));
                                    (lo, hi) = (Some(idx), Some(idx));
                                }
                            }
                            _ => panic!(),
                        }
                    }
                    if let (Some(lo_val), Some(hi_val)) = (lo, hi) {
                        ranges.push((lo_val, hi_val));
                    }
                    (blk, ranges)
                })
                .collect()
        })
        .collect();

    // println!("better: {better_ccws:?}");

    for (num, i) in better_ccws.iter().enumerate() {
        for j in num + 1..better_ccws.len() {
            if !ccws_disjoint(i, better_ccws.get(j).unwrap()) {
                graph.get_mut(&(num as u32)).unwrap().insert(j as u32);
                graph.get_mut(&(j as u32)).unwrap().insert(num as u32);
            }
        }
    }
    (graph, precoloring)
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

// if a var is not in the returned map, it was not colored
fn color(
    mut g: HashMap<u32, HashSet<u32>>,
    precoloring: HashMap<u32, Reg>,
    regs_can_use: &Vec<Reg>,
) -> HashMap<u32, Reg> {
    let mut color_later = Vec::new();

    while g.len() > 0 {
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
            None => {
                // choose a guy to not color
                let mut uncolored: Vec<_> = g.keys().map(|x| *x).collect();
                uncolored.sort_by_key(|x| g.get(x).unwrap().len() as u32);
                remove_node(&mut g, uncolored.pop().unwrap());
            }
        }
    }
    let mut colors = precoloring.clone();
    while let Some((v, es)) = color_later.pop() {
        colors.insert(
            v,
            color_not_in_set(
                regs_can_use,
                es.into_iter()
                    .filter_map(|u| colors.get(&u).map(|x| *x))
                    .collect(),
            ),
        );
    }
    // return an assignment of colors
    colors
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

fn reg_alloc(
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    all_regs: &Vec<Reg>,
    arg_var_to_reg: &HashMap<u32, Reg>,
) -> HashMap<u32, Reg> {
    let (interfer_graph, precoloring) = interference_graph(webs, ccws, arg_var_to_reg);
    // let web_coloring = color(interfer_graph.clone(), precoloring, all_regs);
    // web_coloring
    HashMap::new()
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
) -> RegGlobMemVar {
    match get_src_web(v, i, webs) {
        Some(j) => match web_to_reg.get(&j) {
            Some(r) => RegVar(*r),
            None => MemVar(v),
        },
        None => GlobVar(v),
    }
}

fn dst_reg(
    v: VarLabel,
    i: InsnLoc,
    webs: &Vec<Web>,
    web_to_reg: &HashMap<u32, Reg>,
) -> RegGlobMemVar {
    match get_dst_web(v, i, webs) {
        Some(j) => match web_to_reg.get(&j) {
            Some(r) => RegVar(*r),
            None => MemVar(v),
        },
        None => GlobVar(v),
    }
}

fn to_regs(
    m: cfg::CfgMethod<VarLabel>,
    web_to_reg: &HashMap<u32, Reg>,
    webs: &Vec<Web>,
) -> cfg::CfgMethod<RegGlobMemVar> {
    let new_fields = all_mem_vars(&m);
    let new_blocks = m.blocks.into_iter().map(|(lbl, blk)| {
        (
            lbl,
            cfg::BasicBlock::<RegGlobMemVar> {
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

    let mem_webs = (0..webs.len()).filter(|i| web_to_reg.get(&(*i as u32)) == None);
    let mem_vars = mem_webs
        .map(|web_num| webs.get(web_num).unwrap().var)
        .collect::<HashSet<_>>();

    cfg::CfgMethod::<RegGlobMemVar> {
        name: m.name,
        num_params: m.num_params,
        blocks: new_blocks.collect(),
        fields: m
            .fields
            .into_iter()
            .filter(|(name, (t, _))| match t {
                CfgType::Scalar(_) => mem_vars.contains(name),
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
                        panic!();
                    }
                },
                _ => (),
            }
        }
    }
    ret
}

fn find_reload_spill_pair_at(
    reload_loc: &InsnLoc,
    m: &cfg::CfgMethod<RegGlobMemVar>,
) -> Option<(RegGlobMemVar, MemVarLabel, InsnLoc, InsnLoc)> {
    if let Sum::Inl(Instruction::Reload { ord_var, mem_var }) = get_insn(m, *reload_loc) {
        let mut curr = reload_loc.clone();
        curr.idx += 1;
        let target_reg: Reg;

        if let RegGlobMemVar::RegVar(reg) = ord_var {
            target_reg = *reg;
        } else {
            return None;
        }

        while curr.idx != 0 {
            if let Sum::Inl(i) = get_insn(m, curr) {
                if (*i
                    == Instruction::Spill {
                        ord_var: *ord_var,
                        mem_var: *mem_var,
                    })
                {
                    return Some((*ord_var, *mem_var, *reload_loc, curr));
                }

                if get_sources(&Sum::Inl(i)).contains(ord_var) {
                    return None;
                }
            }

            if let Some(child) = get_children(m, curr).get(0) {
                curr = *child;
            } else {
                break;
            }
        }
    }

    return None;
}

// returns vector of spill and reload instructions to skip
fn minimize_spills_and_reloads(m: &cfg::CfgMethod<RegGlobMemVar>) -> Vec<InsnLoc> {
    // map (register, mem_var pairs) to sequence of spill and reload pairs

    let mut to_remove = vec![];

    all_insn_locs(m)
        .iter()
        .for_each(|i| match find_reload_spill_pair_at(i, m) {
            Some((reload_var, mem_var, reload_loc, spill_loc)) => to_remove.push(spill_loc),
            None => (),
        });

    return to_remove;
}

fn prune_spills(m: &cfg::CfgMethod<RegGlobMemVar>) -> cfg::CfgMethod<RegGlobMemVar> {
    let to_spill = minimize_spills_and_reloads(m)
        .into_iter()
        .collect::<HashSet<_>>();

    let mut new_method = m.clone();

    for (bid, block) in m.blocks.iter() {
        let mut new_block = block.clone();
        let mut new_instructions = vec![];

        for (iid, instr) in block.body.iter().enumerate() {
            let insn_loc = InsnLoc {
                blk: *bid,
                idx: iid,
            };

            if to_spill.contains(&insn_loc) {
                continue;
            } else {
                new_instructions.push(instr.clone())
            }
        }

        new_block.body = new_instructions;

        new_method.blocks.insert(*bid, new_block);
    }

    new_method
}

fn prune_spills_and_reloads(m: &cfg::CfgMethod<RegGlobMemVar>) -> cfg::CfgMethod<RegGlobMemVar> {
    // iterate through blocks and simplify spills and stores
    let mut new_method = m.clone();

    println!("method before pruning: {new_method}");

    // iterate remove all spills that are never reloaded
    for (bid, block) in new_method.blocks.clone().into_iter() {
        let mut new_block = block.clone();
        let mut new_instructions: Vec<Instruction<RegGlobMemVar>> = vec![];

        // reg to memvar lookup
        let mut reg_to_mem_var: HashMap<Reg, MemVarLabel> = hashmap! {};

        // memvar to regs lookup
        let mut mem_var_to_reg: HashMap<MemVarLabel, Reg> = hashmap! {};

        for i in block.body.into_iter() {
            // if we come across a spill instruction, check if it is redundant
            //  - if the register is already associated with the target memvar, then remove instruction
            //  - if the register is not already associated with the memvar, then update reg_to_memvar, and remove the register from

            // if we come across a reload instruction, check if it is redundant
            //  - if the register already contains the target memvar, then remove instruciton
            //  - if the memvar has a register that is already associated with it in the block, then turn into a move instruction
            //  - else keep instruction, update reg_to_mem_var and memvar_to_reg

            // if we come across a move instruction
            //  - if the source contains a memvar, update reg_to_memvar so that the dest contains the same memvar
            //  - otherwise, evict the dest from reg_to_memvar and memvar_to_reg

            // for all other instructions
            //  - evict the the dest from reg_to_memvar and memvar_to_reg
            let mut keep_instruction = true;

            match i.clone() {
                Instruction::Spill { ord_var, mem_var } => {
                    match ord_var {
                        RegGlobMemVar::RegVar(reg) => {
                            match reg_to_mem_var.clone().get(&reg) {
                                Some(curr_mem_var_stored_by_reg)
                                    if *curr_mem_var_stored_by_reg == mem_var =>
                                {
                                    keep_instruction = false;
                                }
                                Some(curr_mem_var_stored_by_reg) => {
                                    // update mappings
                                    reg_to_mem_var.insert(reg, mem_var);
                                    mem_var_to_reg.insert(mem_var, reg);

                                    // remove reverse mapping if needed
                                    if mem_var_to_reg.get(curr_mem_var_stored_by_reg) == Some(&reg)
                                    {
                                        mem_var_to_reg.remove(curr_mem_var_stored_by_reg);
                                    }
                                }
                                None => {
                                    // create new mappings
                                    reg_to_mem_var.insert(reg, mem_var);
                                    mem_var_to_reg.insert(mem_var, reg);
                                }
                            }
                        }
                        _ => (),
                    }
                }
                Instruction::Reload { ord_var, mem_var } => {
                    match ord_var {
                        RegGlobMemVar::RegVar(curr_reg) => match reg_to_mem_var.get(&curr_reg) {
                            Some(curr_mem_var_stored_by_reg) => {
                                if *curr_mem_var_stored_by_reg == mem_var {
                                    // redundant instruction
                                    keep_instruction = false;
                                } else {
                                    // make current_reg point to memvar
                                    reg_to_mem_var.insert(curr_reg, mem_var);
                                    mem_var_to_reg.insert(mem_var, curr_reg);

                                    // check if there is a register that currently contains the
                                    if let Some(reg) = mem_var_to_reg.get(&mem_var) {
                                        keep_instruction = false;
                                        new_instructions.push(Instruction::MoveOp {
                                            source: ImmVar::Var(RegGlobMemVar::RegVar(*reg)),
                                            dest: ord_var,
                                        });
                                    }
                                }
                            }
                            None => {
                                // make current_reg point to memvar
                                reg_to_mem_var.insert(curr_reg, mem_var);
                                mem_var_to_reg.insert(mem_var, curr_reg);
                            }
                        },
                        _ => (),
                    }
                }
                Instruction::Call(..) => {
                    // can't guarantee some of the registers after a call
                    reg_to_mem_var.clear();
                    mem_var_to_reg.clear();
                }
                _ => {
                    let dests = get_insn_dest(&i);
                    for dest in dests {
                        // if dest is associated with some mem_var, evict it from the graphs
                        if let RegGlobMemVar::RegVar(dest_reg) = dest {
                            if let Some(mem_var) = reg_to_mem_var.remove(&dest_reg) {
                                if let Some(curr_reg) = mem_var_to_reg.get(&mem_var) {
                                    if *curr_reg == dest_reg {
                                        mem_var_to_reg.remove(&mem_var);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if keep_instruction {
                new_instructions.push(i.clone());
            }
        }

        new_block.body = new_instructions;
        new_method.blocks.insert(bid, new_block);
    }

    println!("Method after pruning {new_method}");
    new_method
}

fn build_need_to_save(
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    m: &cfg::CfgMethod<RegGlobMemVar>,
    web_to_reg: &HashMap<u32, Reg>,
) -> HashMap<InsnLoc, HashSet<Reg>> {
    all_insn_locs(m)
        .into_iter()
        .filter_map(|i| {
            if let Sum::Inl(Instruction::Call(_, _, _)) = get_insn(m, i) {
                let mut regs = HashSet::new();
                for ((webnum, _), ccw) in webs.iter().enumerate().zip(ccws) {
                    let child = InsnLoc {
                        blk: i.blk,
                        idx: i.idx + 1,
                    };
                    if ccw.contains(&i) && ccw.contains(&child) {
                        match web_to_reg.get(&(webnum as u32)) {
                            Some(reg) => {
                                regs.insert(*reg);
                            }
                            None => (),
                        }
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
    m: cfg::CfgMethod<RegGlobMemVar>,
    caller_saved_regs: &Vec<Reg>,
    caller_saved_memvars: &HashMap<Reg, MemVarLabel>,
    webs: &Vec<Web>,
    ccws: &Vec<HashSet<InsnLoc>>,
    web_to_reg: &HashMap<u32, Reg>,
) -> cfg::CfgMethod<RegGlobMemVar> {
    let need_to_save = build_need_to_save(webs, ccws, &m, web_to_reg);
    method_map(m, |i, loc| match need_to_save.get(&loc) {
        None => vec![i],
        Some(used) => {
            let regs = caller_saved_regs.into_iter().filter(|r| used.contains(r));
            let mut ret: Vec<_> = regs
                .clone()
                .map(|reg| Instruction::Spill {
                    ord_var: RegVar(*reg),
                    mem_var: *caller_saved_memvars.get(reg).unwrap(),
                })
                .collect();
            ret.push(i);
            ret.append(
                &mut regs
                    .rev()
                    .map(|reg| Instruction::Reload {
                        ord_var: RegVar(*reg),
                        mem_var: *caller_saved_memvars.get(reg).unwrap(),
                    })
                    .collect(),
            );
            ret
        }
    })
}

fn regalloc_method(mut m: cfg::CfgMethod<VarLabel>) -> cfg::CfgMethod<RegGlobMemVar> {
    // callee-saved regs: RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15,
    let callee_saved_regs = vec![Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];
    let caller_saved_regs = vec![Reg::Rsi, Reg::Rcx, Reg::R11, Reg::Rdi, Reg::R8, Reg::R10];
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

    // println!("here m is {m}");
    let m = method_map(m.clone(), |i, _| {
        make_args_easy_to_color(i, &all_regs, &args_to_dummy_vars)
    });

    let webs = get_webs(&m);
    let ccws: Vec<HashSet<_>> = webs
        .iter()
        .map(|web| find_inter_instructions(&m, web))
        .collect();

    // println!("method after thing: {m}");
    // println!("reg_of_varname: {reg_of_varname:?}");

    let web_to_reg = reg_alloc(&webs, &ccws, &all_regs, &reg_of_varname);

    // // println!("webs: {webs:?}");

    // println!("web_to_reg: {:?}", web_to_reg);
    // println!("before renaming: {spilled_method}");
    // let mut m = to_regs(m, &web_to_reg, &webs);
    // let caller_saved_memvars: HashMap<_, _> = caller_saved_regs
    //     .iter()
    //     .map(|reg| (*reg, corresponding_memvar(&mut m.fields, *reg)))
    //     .collect();

    // let m = push_and_pop(
    //     m,
    //     &caller_saved_regs,
    //     &caller_saved_memvars,
    //     &webs,
    //     &ccws,
    //     &web_to_reg,
    // );
    // println!("after renaming: {x}");
    //    m
    cfg::CfgMethod {
        name: "eh".to_string(),
        num_params: 0,
        blocks: HashMap::new(),
        fields: HashMap::new(),
        return_type: None,
    }
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

pub fn regalloc_prog(p: cfg::CfgProgram<VarLabel>) -> cfg::CfgProgram<RegGlobMemVar> {
    let p = prog_map(p, |m| method_map(m, lower_calls_insn));
    cfg::CfgProgram {
        externals: p.externals,
        global_fields: p.global_fields,
        methods: p.methods.into_iter().map(regalloc_method).collect(),
    }
}
