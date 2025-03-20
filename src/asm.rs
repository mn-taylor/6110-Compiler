use crate::cfg::{BasicBlock, Instruction, Jump};
use crate::cfg_build::CfgMethod;
use crate::ir::Bop

enum Reg {
    Rax,
    Rbp,
    Rdx,
}

fn bop_to_asm(op: Bop)-> String {
    match op {
        Bop::MulBop(mulOp) => {mulOp.to_string()}
        Bop::AddBop(addOp) => {addOp.to_string()}
    }
}

fn asm_instruction(
    instr: Instruction,
    stack_lookup: HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let instructions: Vec<String> = match instr {}
    todo!();
}
