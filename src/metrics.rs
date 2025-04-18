use crate::cfg;
use crate::cfg::Instruction;

pub fn num_intstructions<T>(m: &cfg::CfgMethod<T>) -> u32 {
    let mut num_instr = 0;
    for (_, block) in m.blocks.iter() {
        for instruction in block.body.iter() {
            match instruction {
                Instruction::PhiExpr { .. } => {}
                _ => num_instr += 1,
            }
        }
    }
    num_instr
}
