use crate::cfg_build::VarLabel;
use crate::{cfg, deadcode, parse, reg_asm, scan};
use cfg::{Arg, BlockLabel, CfgType, ImmVar, MemVarLabel};
use maplit::{hashmap, hashset};
use parse::Primitive;
use reg_asm::Reg;
use scan::Sum;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_float;
use std::hash::Hash;
type CfgMethod = cfg::CfgMethod<VarLabel>;
type Jump = cfg::Jump<VarLabel>;
use crate::regalloc::{find_inter_instructions, Web};
use std::cmp::max;

pub fn rank_webs(
    webs: Vec<(u32, &Web)>,
    interference_graph: &HashMap<u32, HashSet<u32>>,
) -> Vec<u32> {
    let mut webs_and_ratios = webs
        .iter()
        .map(|(i, web)| {
            let mut degree: f64 = interference_graph.get(i).unwrap().len() as f64;
            if degree == 0.0 {
                degree = 0.00001;
            }
            (i, web.uses.len() as f64 / degree, web)
        })
        .collect::<Vec<_>>();

    // rank by convex closures
    webs_and_ratios.sort_by(|a, b| (a.1).partial_cmp(&b.1).unwrap());
    // webs_and_ratios.reverse();

    let sorted_webs = webs_and_ratios.iter().map(|c| *c.0).collect();

    sorted_webs
}
