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
use std::num;

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

    // if let Some((i, min_spilling_cost, web)) = webs_and_ratios.get(0) {
    //     let mut webs_to_spill = vec![];
    //     let mut to_skip: HashSet<u32> = hashset! {};
    //     for (i, spilling_cost, web) in webs_and_ratios.into_iter() {
    //         // if web is within some percentage of the maximum and is not adjacent to the current webs to spill, then add it to webs to spill
    //         if (min_spilling_cost - spilling_cost).abs() / min_spilling_cost < 0.1
    //             && !to_skip.contains(i)
    //         {
    //             webs_to_spill.push(i);
    //             to_skip.extend(interference_graph.get(i).unwrap());
    //         }
    //     }
    // }
    // // find set of k webs that are not adjacent to each other and have similar spilling cost to the max.

    sorted_webs
}
