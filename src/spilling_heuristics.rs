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
use crate::regalloc::{find_inter_instructions, Web};
use std::cmp::max;

pub fn rank_webs(webs: Vec<Web>, interference_graph: &HashMap<u32, HashSet<u32>>) -> Vec<Web> {
    let mut webs_and_ratios = webs
        .iter()
        .enumerate()
        .map(|(i, web)| {
            let mut degree = interference_graph.get(&(i as u32)).unwrap().len();
            if degree == 0 {
                degree = 1;
            }
            (i, web.uses.len() / degree, web)
        })
        .collect::<Vec<_>>();

    // rank by convex closures
    webs_and_ratios.sort_by(|a, b| (a.1).cmp(&b.1));
    // webs_and_ratios.reverse();

    let sorted_webs = webs_and_ratios.iter().map(|c| c.2.clone()).collect();

    sorted_webs
}
