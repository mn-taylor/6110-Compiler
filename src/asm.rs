use crate::cfg::{
    Arg, BasicBlock, CfgMethod, CfgProgram, CfgType, CmpType, Instruction, Jump, VarLabel,
};
use crate::ir::{Bop, UnOp};
use crate::scan::{AddOp, EqOp, MulOp, RelOp};
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

fn convert_rel_op_to_cmov_type(op: Bop) -> String {
    match op {
        Bop::RelBop(rop) => match rop {
            RelOp::Lt => "cmovl".to_string(),
            RelOp::Le => "cmovle".to_string(),
            RelOp::Gt => "cmovge".to_string(),
            RelOp::Ge => "cmovg".to_string(),
        },
        Bop::EqBop(eop) => match eop {
            EqOp::Eq => "cmove".to_string(),
            EqOp::Neq => "cmovne".to_string(),
        },
        _ => {
            panic!()
        }
    }
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

fn build_stack(
    all_fields: HashMap<VarLabel, (CfgType, String)>,
) -> (HashMap<VarLabel, (CfgType, u64)>, u64) {
    let mut offset: u64 = 0;
    let mut field = 0;
    let mut lookup: HashMap<VarLabel, (CfgType, u64)> = HashMap::new();

    loop {
        let res = all_fields.get(&field);
        match res {
            Some((typ, _)) => match typ {
                CfgType::Scalar(_) => {
                    lookup.insert(field, (typ.clone(), offset.clone()));
                    offset += 8;
                }
                CfgType::Array(_, len) => {
                    lookup.insert(field, (typ.clone(), offset.clone()));
                    offset += u64::from((len * 8) as u32);
                }
            },
            None => break,
        }

        field += 1;
    }

    (lookup, offset + (16 - offset % 16)) // keep stack 16 byte aligned
}

fn get_global_strings(p: &CfgProgram) -> HashMap<String, String> {
    let mut data_labels = HashMap::new();
    for method in p.methods.values() {
        for block in method.blocks.values() {
            for insn in block.body.iter() {
                match insn {
                    Instruction::Call(_, args, _) => {
                        for arg in args {
                            match arg {
                                Arg::StrArg(s) => {
                                    if !data_labels.contains_key(&s) {
                                        data_labels.insert(
                                            s.to_string(),
                                            format!("string{}", data_labels.len()),
                                        );
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
    }
    data_labels
}

pub fn asm_program(program: CfgProgram)-> Vec<Strings> {
    let CfgProgram{methods: Vec}
}

pub fn asm_method(method: CfgMethod, global_data: HashMap<String, String>) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];
    // set up stack frame
    instructions.push(format!("\tpush {}", Reg::Rbp));
    instructions.push(format!("\tmov {}, {}", Reg::Rbp, Reg::Rsp));

    let (offsets, total_offset) = build_stack(method.fields);

    // allocate space enough space on the stack
    instructions.push(format!("\tsub {}, {}", Reg::Rsp, total_offset));

    // read parameters from registers and/or stack
    let mut argument_registers: Vec<Reg> =
        vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];
    for (i, llname) in method.params.iter().enumerate() {
        let (_, dest) = offsets.get(llname).unwrap();
        if i < 6 {
            let reg = argument_registers.pop().unwrap();
            instructions.push(format!("\tmov [{} - {}], {}", Reg::Rbp, dest, reg));
        } else {
            instructions.push(format!(
                "\tmove {}, [{} + {}]",
                Reg::Rax,
                Reg::Rbp,
                16 + 16 * (6 - i) // check the math here
            ));
            instructions.push(format!("\tmov [{} - {}], {}", Reg::Rbp, dest, Reg::Rax));
        }
    }

    // assemble blocks
    let mut blocks: Vec<&BasicBlock> = method.blocks.values().collect::<Vec<_>>();
    blocks.sort_by_key(|c| c.block_id);

    for block in blocks {
        instructions.extend(asm_block(block, &offsets, &global_data, &method.name));
    }

    // make a label for end, that blocks which jump to Nowhere jump to.
    instructions.push(format!("{}end:", &method.name));

    // return the stack to original state
    instructions.push(format!("\tadd {}, {}", Reg::Rsp, total_offset));

    // ret instruction
    instructions.push(format!("\tret"));
    return instructions;
}

fn asm_block(
    b: &BasicBlock,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    data: &HashMap<String, String>,
    root: &String,
    // ^hopefully this parameter should not be necessary?
) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];

    // make label
    instructions.push(format!("{}{}:", root, b.block_id));

    // perform instructions
    for instruction in &b.body {
        instructions.extend(asm_instruction(stack_lookup, data, instruction.clone()));
    }

    // handle jumps
    match b.jump_loc {
        Jump::Uncond(block_id) => instructions.push(format!("jmp {}{}", root, block_id)),
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let get_source = format!("\tmov {}, [{} + {}]", Reg::R9, Reg::Rbp, source_offset);
            let compare = format!("\tcmp {}, {}", Reg::R9, 1);
            let true_jump = format!("\tje {}{}", root, true_block);
            let false_jump = format!("\tjmp {}{}", root, false_block);

            instructions.extend([get_source, compare, true_jump, false_jump]);
        }
        Jump::Nowhere => {
            instructions.push(format!("\tjmp {}end", root)); // this is the label that I'm thinking we should use.
        }
    }
    vec![]
}

fn asm_instruction(
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    data: &HashMap<String, String>,
    instr: Instruction,
) -> Vec<String> {
    match instr {
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => {
            let (_, source1_offset) = stack_lookup.get(&source1).unwrap();
            let (_, source2_offset) = stack_lookup.get(&source2).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source1 = format!("\tmov {}, [{} + {}]", Reg::R9, Reg::Rbp, source1_offset);
            let get_source2 = format!("\tmov {}, [{} + {}]", Reg::R10, Reg::Rbp, source1_offset);

            match op {
                Bop::AddBop(_) | Bop::MulBop(_) => {
                    let operate = format!("{} {}, {}", convert_bop_to_asm(op), Reg::R9, Reg::R10);
                    let return_to_stack =
                        format!("\tmov [{} + {}], {}", Reg::Rbp, dest_offset, Reg::R9);

                    return vec![get_source1, get_source2, operate, return_to_stack];
                }
                _ => {
                    let compare = format!("\tcmp {}, {}", Reg::R9, Reg::R10);
                    let initialize_dest = format!("\tmov {}, {}", Reg::R9, 0);
                    let cond_move =
                        format!("\t{} {}, {}", convert_rel_op_to_cmov_type(op), Reg::R9, 1);
                    let return_to_stack =
                        format!("\tmov [{} + {}], {}", Reg::Rbp, dest_offset, Reg::R9);
                    return vec![
                        get_source1,
                        get_source2,
                        compare,
                        initialize_dest,
                        cond_move,
                        return_to_stack,
                    ];
                }
            }
        }
        Instruction::TwoOp { source1, dest, op } => {
            let (_, source1_offset) = stack_lookup.get(&source1).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source1 = format!("\tmov {}, [{} + {}]", Reg::Rax, Reg::Rbp, source1_offset);
            let operate = match op {
                UnOp::Not => {
                    format!("\tnot {}", Reg::Rax)
                }
                UnOp::Neg => {
                    format!("\tneg {}", Reg::Rax)
                }
                UnOp::IntCast => {
                    format!("\tmov eax, rax")
                }
                UnOp::LongCast => {
                    format!("\tmov rax, rax")
                }
            };
            let return_to_stack = format!("\tmove [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);
            return vec![get_source1, operate, return_to_stack];
        }
        Instruction::MoveOp { source, dest } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source = format!("\tmov {}, [{} + {}]", Reg::Rax, Reg::Rbp, source_offset);
            let return_to_stack = format!("\tmov [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);

            return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let operate = format!("\tmov QWORD [{} + {}], {}", Reg::Rbp, dest_offset, constant);

            return vec![operate];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let (_, arr_root) = stack_lookup.get(&name).unwrap();
            let (_, dest_offset) = stack_lookup.get(&dest).unwrap();

            let get_source = format!(
                "\tmov {}, [{} + {}]",
                Reg::Rax,
                Reg::Rbp,
                *arr_root + 16 * idx as u64
            );
            let return_to_stack = format!("\tmov [{} + {}], {}", Reg::Rbp, dest_offset, Reg::Rax);

            return vec![get_source, return_to_stack];
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let (_, arr_root) = stack_lookup.get(&arr).unwrap();

            let get_source = format!("\tmov {}, [{} + {}]", Reg::Rax, Reg::Rbp, source_offset);
            let return_to_stack = format!(
                "\tmov [{} + {}], {}",
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
                    instructions.push(format!(
                        "\tmov {} [{} + {}]",
                        Reg::Rax,
                        Reg::Rbp,
                        var_offset
                    ));
                }
                None => {}
            }
            // instructions.push("RET".to_string());
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
                                    "\tmov {} [{} + {}]",
                                    reg,
                                    Reg::Rbp,
                                    source_offset
                                ));
                            }
                            None => {
                                instructions.push(format!(
                                    "\tmov {} [{} + {}]",
                                    Reg::Rax,
                                    Reg::Rbp,
                                    source_offset
                                ));
                                instructions.push(format!("\tpush {}", Reg::Rax));
                            }
                        }
                    }
                    Arg::StrArg(string) => {
                        let str_loc = data.get(string).unwrap();
                        let arg_reg = argument_registers.pop();

                        match arg_reg {
                            Some(reg) => {
                                instructions.push(format!("\tpush {} {}", reg, str_loc));
                            }
                            None => {
                                instructions.push(format!("\tmov {} {}", Reg::Rax, str_loc));
                                instructions.push(format!("\tpush {}", Reg::Rax));
                            }
                        }
                    }
                }
            }

            // call the function
            if func_name == "printf".to_string() {
                instructions.push(format!("\txor {}, {}", Reg::Rax, Reg::Rax));
            }
            instructions.push(format!("\tcall {}", func_name));

            // store return value into temp
            match ret_dest {
                Some(dest) => {
                    let (_, dest_offset) = stack_lookup.get(&dest).unwrap();
                    instructions.push(format!(
                        "\tmov [{} + {}], {}",
                        Reg::Rbp,
                        dest_offset,
                        Reg::Rax
                    ));
                }
                None => {}
            }

            instructions
        }
    }
}
