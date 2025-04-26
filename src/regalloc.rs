use std::collections::HashSet;

use crate::{deadcode::get_dest, ssa_construct::SSAVarLabel};

fn get_defs(m: CfgMethod<SSAVarLabel>) -> HashMap<VarLabel, HashSet<(block_label, instr_idx)>> {
    let mut defs: HashMap<VarLabel, HashSet<(block_label, instr_idx)>>;

    for (bid, block) in m.blocks {
        for (iid, instruction) in block.body {
            match get_dest(instruction) {
                Some(dest) => {
                    let mut curr_defs = defs.get_mut(dest).unwrap_or(vec![]);
                    curr_defs.push(dest);
                    def.insert(dest, curr_defs);
                }
            }
        }
    }
    defs
}

fn get_uses(m, block_label, instr_idx) -> HashSet<(block_label, instr_idx)> {

}

fn get_webs(m) -> HashSet<(VarLabel, HashSet<HashSet<(block_label, instr_idx))>>> {
    // find all defs
    let defs = get_defs(m);
    let webs = 
    // find all uses
    for (varname, def_set) in defs {
        // find all uses
        let mut sets = def_set.iter().map(|bid, iid| get_uses(m, bid, iid)).collect::<HashSet<_>>();
        if sets.len()==0 {
            continue
        }

        let mut new_sets = hashset!{};
        while sets.len() != 0{
            let curr = sets.next();
            sets.remove_curr();

            let found_overlap = false;
            for set in sets {
                // if curr intersects with set, remove this set and append to curr
                if curr.intersection(set).len()!= 0 { // overlapping uses
                    sets.remove(set);
                    sets.insert(curr.union(set));
                    found_overlap = true;
                    break
                }
            }

            if !found_overlap {
                new_sets.insert(curr);
            }

        }


        
    }

    // merge when possible

}

fn find_inter_instructions(m, HashSet<(block_label, instr_idx)>) -> {

}

fn interference_graph(m) -> {

}



