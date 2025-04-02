use crate::cfg::{
    Arg, BasicBlock, CfgMethod, CfgProgram, CfgType, CmpType, Instruction, Jump, VarLabel,
};
use crate::ir::{Bop, UnOp};
use crate::parse::Primitive;
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
            Reg::Rax => "rax",
            Reg::Rbx => "rbx",
            Reg::Rcx => "rcx",
            Reg::Rdx => "rdx",
            Reg::Rsi => "rsi",
            Reg::Rdi => "rdi",
            Reg::Rsp => "rsp",
            Reg::Rbp => "rbp",
            Reg::R8 => "r8",
            Reg::R9 => "r9",
            Reg::R10 => "r10",
            Reg::R11 => "r11",
            Reg::R12 => "r12",
            Reg::R13 => "r13",
            Reg::R14 => "r14",
            Reg::R15 => "r15",
        };
        write!(f, "%{}", reg_name)
    }
}

fn convert_bop_to_asm(bop: Bop) -> String {
    match bop {
        Bop::MulBop(MulOp::Div) => "idivq",
        Bop::MulBop(MulOp::Mul) => "imulq",
        Bop::MulBop(MulOp::Mod) => "mod",
        Bop::AddBop(AddOp::Add) => "addq",
        Bop::AddBop(AddOp::Sub) => "subq",
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

fn round_up(a: u64, factor: u64) -> u64 {
    let big_enough = a + (factor - 1);
    big_enough - big_enough % factor
}

fn build_stack(
    all_fields: HashMap<VarLabel, (CfgType, String)>,
) -> (HashMap<VarLabel, (CfgType, u64)>, u64) {
    let mut offset: u64 = 0;
    let mut lookup: HashMap<VarLabel, (CfgType, u64)> = HashMap::new();

    for (field, (typ, _)) in all_fields.iter() {
        match typ {
            CfgType::Scalar(_) => {
                lookup.insert(*field, (typ.clone(), offset.clone()));
                offset += 8;
            }
            CfgType::Array(_, len) => {
                lookup.insert(*field, (typ.clone(), offset.clone()));
                offset += u64::from((len * 8) as u32);
            }
        }
    }

    (lookup, round_up(offset, 16)) // keep stack 16 byte aligned
}

use std::collections::HashSet;

fn get_global_strings(p: &CfgProgram) -> Vec<String> {
    let mut all_strings = HashSet::new();
    for method in p.methods.iter() {
        for block in method.blocks.values() {
            for insn in block.body.iter() {
                match insn {
                    Instruction::Call(_, args, _) => {
                        for arg in args {
                            match arg {
                                Arg::StrArg(s) => {
                                    all_strings.insert(s.clone());
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
    all_strings.into_iter().collect()
}

pub fn asm_program(p: &CfgProgram) -> Vec<String> {
    let mut insns: Vec<String> = vec![];
    let glob_strings = get_global_strings(p);
    let glob_fields = &p.global_fields;

    insns.push(".data".to_string());
    for (varname, (typ, _)) in glob_fields {
        match typ {
            CfgType::Array(_, len) => {
                insns.push(format!("global{}:\n\t.zero {}", varname, len * 8))
            }
            CfgType::Scalar(_) => insns.push(format!("global{}:\n\t.zero 8", varname)), /*8 bytes*/
        }
    }
    for strin in glob_strings {
        // TODO this syntax is not right, I don't think .string is even a thing.
        insns.push(format!("global_{}:\n\t.string {}", strin, strin));
    }

    insns.push(".text".to_string());

    insns.push(format!("error_handler:"));
    insns.push(format!("\tmovl $0x2000001, %eax"));
    insns.push(format!("\tmovl $-1, %edi"));
    insns.push(format!("\tsyscall"));

    for method in p.methods.iter() {
        insns.extend(asm_method(method));
    }

    insns
}

pub fn asm_method(method: &CfgMethod) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];
    if method.name.as_str() == "main" {
        instructions.push(".globl _main".to_string());
        instructions.push("_main:".to_string());
        instructions.push("\tcall main".to_string());
        instructions.push(format!("\tmovq $0, {}", Reg::Rax));
        instructions.push("\tret".to_string());
    }

    // set up stack frame
    instructions.push(format!("{}:", method.name));

    instructions.push(format!("\tpushq {}", Reg::Rbp));
    instructions.push(format!("\tmovq {}, {}", Reg::Rsp, Reg::Rbp));

    let (offsets, total_offset) = build_stack(method.fields.clone());

    // allocate space enough space on the stack
    instructions.push(format!("\tsubq ${}, {}", total_offset, Reg::Rsp,));

    // read parameters from registers and/or stack
    let mut argument_registers: Vec<Reg> =
        vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];
    for (i, llname) in method.params.iter().enumerate() {
        if i < 6 {
            let reg = argument_registers.pop().unwrap();
            instructions.push(store_from_reg(reg, *llname, &offsets));
        } else {
            instructions.push(format!(
                "\tmovq {}({}), {}, ",
                16 + 8 * (i - 6),
                Reg::Rbp,
                Reg::Rax,
            ));
            instructions.push(store_from_reg(Reg::Rax, *llname, &offsets));
        }
    }

    instructions.push(format!("\tjmp {}0", method.name));

    // assemble blocks
    let mut blocks: Vec<&BasicBlock> = method.blocks.values().collect::<Vec<_>>();
    blocks.sort_by_key(|c| c.block_id);

    for block in blocks {
        instructions.extend(asm_block(
            block,
            &offsets,
            &method.name,
            method.return_type.clone(),
        ));
    }

    // make a label for end, that blocks which jump to Nowhere jump to.
    instructions.push(format!("{}end:", &method.name));

    // return the stack to original state
    instructions.push(format!("\taddq ${}, {}", total_offset, Reg::Rsp,));
    instructions.push(format!("\tpopq {}", Reg::Rbp));

    // ret instruction
    instructions.push(format!("\tret"));
    return instructions;
}

fn asm_block(
    b: &BasicBlock,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    root: &String,
    return_type: Option<Primitive>,
) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];

    // make label
    instructions.push(format!("{}{}:", root, b.block_id));

    // perform instructions
    for instruction in &b.body {
        instructions.extend(asm_instruction(stack_lookup, instruction.clone(), root));
    }

    // handle jumps
    match b.jump_loc {
        Jump::Uncond(block_id) => instructions.push(format!("\tjmp {}{}", root, block_id)),
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let get_source = format!("\tmovq -{}({}), {}", source_offset, Reg::Rbp, Reg::R9,);
            let compare = format!("\tcmp {}, {}", Reg::R9, 1);
            let true_jump = format!("\tje {}{}", root, true_block);
            let false_jump = format!("\tjmp {}{}", root, false_block);

            instructions.extend([get_source, compare, true_jump, false_jump]);
        }
        Jump::Nowhere => {
            // TODO this is correct only if the return type is void.
            // if it is not void, we should instead error here.  what an error looks like idk.
            if return_type.is_some() {
                instructions.push(format!("\tjmp error_handler"));
            } else {
                instructions.push(format!("\tjmp {}end", root));
            }
        }
    }
    instructions
}

fn load_into_reg(
    dest: Reg,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> String {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => format!("\tmovq -{}({}), {}", offset, Reg::Rbp, dest),
        None => format!("\tmovq global{}(%rip), {}", varname, dest),
    }
}

fn store_from_reg(
    src: Reg,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> String {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => format!("\tmovq {}, -{}({})", src, offset, Reg::Rbp),
        None => format!("\tmovq {}, global{}(%rip)", src, varname),
    }
}

fn asm_instruction(
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    instr: Instruction,
    root: &String,
) -> Vec<String> {
    match instr {
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => {
            let get_source1 = load_into_reg(Reg::R9, source1, stack_lookup);
            let get_source2 = load_into_reg(Reg::R10, source2, stack_lookup);

            match op {
                Bop::AddBop(_) | Bop::MulBop(_) => {
                    let operate = format!("\t{} {}, {}", convert_bop_to_asm(op), Reg::R9, Reg::R10);
                    let return_to_stack = store_from_reg(Reg::R10, dest, stack_lookup);

                    return vec![get_source1, get_source2, operate, return_to_stack];
                }
                _ => {
                    let compare = format!("\tcmp {}, {}", Reg::R9, Reg::R10);
                    let initialize_dest = format!("\tmovq {}, {}", 0, Reg::R9,);
                    let cond_move =
                        format!("\t{} {}, {}", convert_rel_op_to_cmov_type(op), Reg::R9, 1);
                    let return_to_stack = store_from_reg(Reg::R9, dest, stack_lookup);
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
            let get_source1 = load_into_reg(Reg::R9, source1, stack_lookup);
            let operate = match op {
                UnOp::Not => {
                    format!("\tnot {}", Reg::Rax)
                }
                UnOp::Neg => {
                    format!("\tneg {}", Reg::Rax)
                }
                UnOp::IntCast => {
                    format!("\tmovq eax, rax")
                }
                UnOp::LongCast => {
                    format!("\tmovq rax, rax")
                }
            };

            let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);
            return vec![get_source1, operate, return_to_stack];
        }
        Instruction::MoveOp { source, dest } => {
            let get_source = load_into_reg(Reg::Rax, source, stack_lookup);
            let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            let load_const = format!("\tmovq ${}, {}", constant, Reg::Rax);
            let restore_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            return vec![load_const, restore_to_stack];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let (_, arr_root) = stack_lookup.get(&name).unwrap();

            let get_source = format!(
                "\tmovq -{}({}), {}",
                arr_root + 16 * idx as u64,
                Reg::Rbp,
                Reg::Rax,
            );
            let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);
            return vec![get_source, return_to_stack];
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let (_, source_offset) = stack_lookup.get(&source).unwrap();
            let (_, arr_root) = stack_lookup.get(&arr).unwrap();

            let get_source = load_into_reg(Reg::Rax, source, stack_lookup);
            let return_to_stack = format!(
                "\tmovq {}, -{}({}) ",
                Reg::Rax,
                arr_root + 16 * idx as u64,
                Reg::Rbp
            );

            return vec![get_source, return_to_stack];
        }
        Instruction::Ret(ret_val) => {
            let mut instructions = vec![];
            match ret_val {
                Some(var) => {
                    let (_, var_offset) = stack_lookup.get(&var).unwrap();
                    instructions.push(format!(
                        "\tmovq -{}({}), {}",
                        var_offset,
                        Reg::Rbp,
                        Reg::Rax
                    ));
                }
                None => {}
            }
            instructions.push(format!("jmp {}end", root));
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
                                instructions.push(load_into_reg(reg, *label, stack_lookup))
                            }
                            None => {
                                instructions.push(load_into_reg(Reg::Rax, *label, stack_lookup));
                                instructions.push(format!("\tpushq {}", Reg::Rax));
                            }
                        }
                    }
                    Arg::StrArg(string) => {
                        let arg_reg = argument_registers.pop();

                        match arg_reg {
                            Some(reg) => {
                                instructions
                                    .push(format!("\tpushq global_{}(%rip) {}", string, reg));
                            }
                            None => {
                                instructions
                                    .push(format!("\tmovq global_{} {}", string, Reg::Rax,));
                                instructions.push(format!("\tpushq {}", Reg::Rax));
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
                        "\tmovq {}, -{}({})",
                        Reg::Rax,
                        dest_offset,
                        Reg::Rbp
                    ));
                }
                None => {}
            }

            instructions
        }
    }
}
