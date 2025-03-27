use crate::cfg::{Arg, BasicBlock, CfgType, Cmp, CmpType, Instruction, Jump, VarLabel};
use crate::cfg_build::CfgMethod;
use crate::ir::{Bop, UnOp};
use crate::scan::{AddOp, MulOp};
use std::fmt;

use std::collections::HashMap;

pub enum Reg {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    Rsp,
    Rbp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let reg_name = match self {
            Reg::Rax => "RAX",
            Reg::Rbx => "RBX",
            Reg::Rcx => "RCX",
            Reg::Rdx => "RDX",
            Reg::Rsi => "RSI",
            Reg::Rdi => "RDI",
            Reg::Rsp => "RSP",
            Reg::Rbp => "RBP",
            Reg::R8 => "R8",
            Reg::R9 => "R9",
            Reg::R10 => "R10",
            Reg::R11 => "R11",
            Reg::R12 => "R12",
            Reg::R13 => "R13",
            Reg::R14 => "R14",
            Reg::R15 => "R15",
        };
        write!(f, "{}", reg_name)
    }
}

fn convert_bop_to_asm(bop: Bop) -> String {
    match bop {
        Bop::MulBop(MulOp::Div) => "div",
        Bop::MulBop(MulOp::Mul) => "mul",
        Bop::MulBop(MulOp::Mod) => "mod",
        Bop::AddBop(AddOp::Add) => "add",
        Bop::AddBop(AddOp::Sub) => "sub",
        _ => panic!("Only operations allowed in cfg are multiplicative and additive"),
    }
    .to_string()
}

fn convert_cmp_to_cond_move(cmpt: CmpType) -> String {
    match cmpt {
        CmpType::Equal => "CMOVE".to_string(),
        CmpType::NotEqual => "CMOVNE".to_string(),
        CmpType::Greater => "CMOVG".to_string(),
        CmpType::GreaterEqual => "CMOVGE".to_string(),
        CmpType::Less => "CMOVL".to_string(),
        CmpType::LessEqual => "CMOVLE".to_string(),
    }
}

fn asm_instruction(
    stack_lookup: HashMap<VarLabel, (CfgType, u64)>,
    data: HashMap<String, String>,
    instr: Instruction,
) -> Vec<String> {
    match instr {
        Instruction::ThreeOp {
            source1: source1,
            source2: source2,
            dest: dest,
            op: op,
        } => {
            let (_, source1_offset) = stack_lookup.get(&source1).unwrap();
            let (_, source2_offset) = stack_lookup.get(&source2).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source1 = format!("mov {}, [{} + {}]", Reg::R9, Reg::Rbp, source1_offset);
            let get_source2 = format!("mov {}, [{} + {}]", Reg::R10, Reg::Rbp, source1_offset);
            let operate = format!("{} {}, {}", convert_bop_to_asm(op), Reg::R9, Reg::R10);
            let return_to_stack = format!("move [{} + {}], {}", Reg::Rbp, dest_offset, Reg::R9);

            return vec![get_source1, get_source2, operate, return_to_stack];
        }
        Instruction::TwoOp { source1, dest, op } => {
            let (_, source1_offset) = stack_lookup.get(&source1).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source1 = format!("MOV {}, [{} + {}]", Reg::Rax, Reg::Rbp, source1_offset);
            let operate = match op {
                UnOp::Not => {
                    format!("NOT {}", Reg::Rax)
                }
                UnOp::Neg => {
                    format!("NEG {}", Reg::Rax)
                }
                UnOp::IntCast => {
                    format!("MOV EAX, RAX")
                }
                UnOp::LongCast => {
                    format!("MOV RAX, RAX")
                }
            };
            let return_to_stack = format!("move [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);
            return vec![get_source1, operate, return_to_stack];
        }
        Instruction::MoveOp { source, dest } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source = format!("MOV {}, [{} + {}]", Reg::Rax, Reg::Rbp, source_offset);
            let return_to_stack = format!("move [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);

            return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let operate = format!("MOV QWORD [{} + {}], {}", Reg::Rbp, dest_offset, constant);

            return vec![operate];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let (_, arr_root) = stack_lookup.get(&name).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source = format!(
                "MOV {}, [{} + {}]",
                Reg::Rax,
                Reg::Rbp,
                *arr_root + 16 * idx as u64
            );
            let return_to_stack = format!("MOV [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);

            return vec![get_source, return_to_stack];
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let (_, arr_root) = stack_lookup.get(&arr).unwrap();

            let get_source = format!("MOV {}, [{} + {}]", Reg::Rax, Reg::Rbp, source_offset);
            let return_to_stack = format!(
                "MOV [{} + {}], {}",
                Reg::Rbp,
                arr_root + 16 * idx as u64,
                Reg::Rax
            );

            return vec![get_source, return_to_stack];
        }
        Instruction::Ret(ret_val) => {
            let mut instructions = vec![];
            match ret_val {
                Some(var) => {
                    let (_, var_offset) = stack_lookup.get(&var).unwrap();
                    instructions.push(format!("MOV {} [{} + {}]", Reg::Rax, Reg::Rbp, var_offset));
                }
                None => {}
            }
            instructions.push("RET".to_string());
            instructions
        }
        Instruction::Call(func_name, args, ret_dest) => {
            let mut instructions: Vec<String> = vec![];
            let mut argument_registers: Vec<Reg> =
                vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];

            // push arguments into registers and onto stack if needed
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Arg::VarArg(label) => {
                        let (_, source_offset) = stack_lookup.get(&label).unwrap();
                        let arg_reg = argument_registers.pop();
                        match arg_reg {
                            Some(reg) => {
                                instructions.push(format!(
                                    "MOV {} [{} + {}]",
                                    reg,
                                    Reg::Rbp,
                                    source_offset
                                ));
                            }
                            None => {
                                instructions.push(format!(
                                    "MOV {} [{} + {}]",
                                    Reg::Rax,
                                    Reg::Rbp,
                                    source_offset
                                ));
                                instructions.push(format!("PUSH {}", Reg::Rax));
                            }
                        }
                    }
                    Arg::StrArg(string) => {
                        let str_loc = data.get(string).unwrap();
                        let arg_reg = argument_registers.pop();

                        match arg_reg {
                            Some(reg) => {
                                instructions.push(format!("MOV {} {}", reg, str_loc));
                            }
                            None => {
                                instructions.push(format!("MOV {} {}", Reg::Rax, str_loc));
                                instructions.push(format!("PUSH {}", Reg::Rax));
                            }
                        }
                    }
                }
            }

            // call the function
            if func_name == "printf".to_string() {
                instructions.push(format!("XOR {}, {}", Reg::Rax, Reg::Rax));
            }
            instructions.push(format!("CALL {}", func_name));

            // store return value into temp
            match ret_dest {
                Some(dest) => {
                    let (_, dest_offset) = stack_lookup.get(&dest).unwrap();
                    instructions.push(format!(
                        "MOV [{} + {}], {}",
                        Reg::Rbp,
                        dest_offset,
                        Reg::Rax
                    ));
                }
                None => {}
            }

            return instructions;
        }
        Instruction::CondMove {
            cmp,
            cmp_type,
            dest,
            source,
        } => {
            let mut instructions: Vec<String> = vec![];
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            match cmp {
                Cmp::VarVar { source1, source2 } => {
                    let (_, source1_offset) = stack_lookup.get(&source1).unwrap();
                    let (_, source2_offset) = stack_lookup.get(&source2).unwrap();

                    instructions.push(format!(
                        "MOV {} [{} + {}]",
                        Reg::Rax,
                        Reg::Rbp,
                        source1_offset
                    ));
                    instructions.push(format!(
                        "MOV {} [{} + {}]",
                        Reg::R9,
                        Reg::Rbp,
                        source2_offset
                    ));

                    instructions.push(format!("CMP {}, {}", Reg::Rax, Reg::R9));
                }
                Cmp::VarImmediate { source, imm } => {
                    let (_, source_offset) = stack_lookup.get(&source).unwrap();
                    instructions.push(format!(
                        "MOV {} [{} + {}]",
                        Reg::Rax,
                        Reg::Rbp,
                        source_offset
                    ));

                    instructions.push(format!("CMP {}, {}", Reg::Rax, imm));
                }
            }

            instructions.push(format!(
                "{} {} {}",
                convert_cmp_to_cond_move(cmp_type),
                Reg::Rax,
                source
            ));

            instructions.push(format!(
                "MOV [{} + {}], {}",
                Reg::Rbp,
                dest_offset,
                Reg::Rax
            ));

            return instructions;
        }
    }
}
