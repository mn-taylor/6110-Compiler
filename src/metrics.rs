use crate::cfg;
use crate::cfg::{Arg, BasicBlock, BlockLabel, Instruction, Jump};
use crate::cfg_build::{CfgMethod, VarLabel};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

pub fn num_intstructions<T>(m: &cfg::CfgMethod<T>) -> u32 {
    let mut num_instr = 0;
    for (_, block) in m.blocks.iter() {
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { dest, sources } => {}
                _ => num_instr += 1,
            }
        }
    }
    num_instr
}
