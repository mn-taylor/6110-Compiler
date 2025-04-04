use crate::cfg::{Arg, BasicBlock, CfgMethod, CfgProgram, CfgType, Instruction, Jump, VarLabel};
use crate::ir::{Bop, UnOp};
use crate::parse::Primitive;
use crate::scan::{format_str_for_output, AddOp, EqOp, MulOp, RelOp};
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
        Bop::MulBop(MulOp::Mod) => "modq",
        Bop::AddBop(AddOp::Add) => "addq",
        Bop::AddBop(AddOp::Sub) => "subq",
        _ => panic!("Only operations allowed in cfg are multiplicative and additive"),
    }
    .to_string()
}

fn convert_multipication(mop: MulOp, operand: Reg, optarget: Reg) -> Vec<String> {
    let mut instructions = vec![];
    match mop {
        MulOp::Mul => instructions.push(format!("\timulq {}, {}", operand, optarget)),
        MulOp::Mod => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rdx, optarget));
        }
        MulOp::Div => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, optarget));
        }
    };
    instructions
}

fn convert_rel_op_to_cmov_type(op: Bop) -> String {
    match op {
        Bop::RelBop(rop) => match rop {
            RelOp::Lt => "cmovl".to_string(),
            RelOp::Le => "cmovle".to_string(),
            RelOp::Gt => "cmovg".to_string(),
            RelOp::Ge => "cmovge".to_string(),
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
    let mut offset: u64 = 8;
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

fn get_global_strings(p: &CfgProgram) -> HashMap<String, usize> {
    let mut all_strings = HashMap::new();
    for method in p.methods.iter() {
        for block in method.blocks.values() {
            for insn in block.body.iter() {
                match insn {
                    Instruction::Call(_, args, _) => {
                        for arg in args {
                            match arg {
                                Arg::StrArg(s) => {
                                    if let None = all_strings.get(s) {
                                        all_strings.insert(s.clone(), all_strings.len());
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
    all_strings
}

pub fn asm_program(p: &CfgProgram, mac: bool) -> Vec<String> {
    let mut insns: Vec<String> = vec![];

    if !mac {
        insns.push(".section .note.GNU-stack,\"\",@progbits".to_string());
    }

    let external_funcs = &p.externals;
    let glob_strings = &get_global_strings(p);
    let glob_fields = &p.global_fields;

    insns.push(".data".to_string());
    for (varname, (typ, _)) in glob_fields {
        match typ {
            CfgType::Array(_, len) => {
                insns.push(format!("global_var{}:\n\t.zero {}", varname, len * 8))
            }
            CfgType::Scalar(_) => insns.push(format!("global_var{}:\n\t.zero 8", varname)), /*8 bytes*/
        }
    }

    // I think we drop newlines when there is only one string
    for (strin, label) in glob_strings {
        insns.push(format!(
            "global_str{}:\n\t.string \"{}\"",
            label,
            format_str_for_output(strin)
        ));
    }

    insns.push(".text".to_string());
    for external in external_funcs {
        if mac {
            insns.push(format!("\t.extern _{}", external));
        } else {
            insns.push(format!("\t.extern {}", external));
        }
    }

    insns.push(format!("error_handler:"));
    if mac {
        insns.push(format!("\tmovl $0x2000001, %eax"));
        insns.push(format!("\tmovl $-1, %edi"));
        insns.push(format!("\tsyscall"));
    } else {
        insns.push(format!("\tmovl $0x01, %eax"));
        insns.push(format!("\tmovl $-1, %ebx"));
        insns.push(format!("\tsyscall"));
    }

    for method in p.methods.iter() {
        insns.extend(asm_method(method, mac, external_funcs, glob_strings));
    }

    insns
}

pub fn asm_method(
    method: &CfgMethod,
    mac: bool,
    external_funcs: &Vec<String>,
    global_strings: &HashMap<String, usize>,
) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];
    if method.name == "main" {
        let name = format!("{}main", if mac { "_" } else { "" });
        instructions.push(format!(".globl {}", name));
        instructions.push(format!("{}:", name));
    } else {
        instructions.push(format!("{}:", method.name));
    }

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
            break;
        }
    }

    // read parameters off of the stack
    for (i, llname) in method.params.iter().enumerate() {
        if i >= 6 {
            instructions.push(format!(
                "\tmovq {}({}), {}",
                16 + 8 * (method.params.len() as i32 - 1 - i as i32),
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
            external_funcs,
            mac,
            global_strings,
        ));
    }

    // make a label for end, that blocks which jump to Nowhere jump to.
    instructions.push(format!("{}end:", &method.name));

    // return the stack to original state
    // instructions.push(format!("\taddq ${}, {}", total_offset, Reg::Rsp,));
    // instructions.push(format!("\tpopq {}", Reg::Rbp));
    // instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::Rsp));

    instructions.push("\tleave".to_string());

    if method.name == "main" {
        instructions.push(format!("\txorq {}, {}", Reg::Rax, Reg::Rax));
    }

    // ret instruction
    instructions.push(format!("\tret"));
    return instructions;
}

fn asm_block(
    b: &BasicBlock,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    root: &String,
    return_type: Option<Primitive>,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
) -> Vec<String> {
    let mut instructions: Vec<String> = vec![];

    // make label
    instructions.push(format!("{}{}:", root, b.block_id));

    // perform instructions
    for instruction in &b.body {
        instructions.extend(asm_instruction(
            stack_lookup,
            instruction.clone(),
            root,
            external_funcs,
            mac,
            global_strings,
        ));
        instructions.push("".to_string());
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
            let compare = format!("\tcmpq $1, {}", Reg::R9);
            let true_jump = format!("\tje {}{}", root, true_block);
            let false_jump = format!("\tjmp {}{}", root, false_block);

            instructions.extend([get_source, compare, true_jump, false_jump]);
        }
        Jump::Nowhere => {
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
        None => format!("\tmovq global_var{}(%rip), {}", varname, dest),
    }
}

fn load_into_reg_arr(
    dest: Reg,
    varname: VarLabel,
    index: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&varname) {
        Some((_, offset)) => {
            // movl offset(base, index, scale), destination

            // instructions.push(format!(
            //     "\tmovq -{}({}, {}, $8), {dest}",
            //     offset,
            //     Reg::Rbp,
            //     Reg::R9
            // ))
            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq -{offset}({}), {dest}", Reg::R10));
        }
        None => {
            // instructions.push(format!(
            //     "\tleaq (global_var{}(%rip), {}, $8), {}",
            //     varname,
            //     Reg::R9,
            //     Reg::R9
            // ));
            // instructions.push(format!("\tmovq 0({}), {}", Reg::R9, dest));
            // instructions.push(format!("\taddq {}, {}", Reg::R10, Reg::R9));

            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tleaq global_var{}(%rip), {}", varname, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq -0({}), {dest}", Reg::R10));
        }
    }
    instructions
}

fn store_from_reg_arr(
    src: Reg,
    arrname: VarLabel,
    index: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq {src}, -{offset}({})", Reg::R10));
        }
        None => {
            // instructions.push(format!(
            //    "\tleaq (global_var{}(%rip), {}, $16), {}",
            //    arrname,
            //    Reg::R9,
            //    Reg::R9
            // ));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))

            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq {src}, -0({}) ", Reg::R10));
            // instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            // instructions.push(format!("\taddq {}, {}", Reg::R10, Reg::R9));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))
        }
    }
    instructions
}

fn store_from_reg(
    src: Reg,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> String {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => format!("\tmovq {}, -{}({})", src, offset, Reg::Rbp),
        None => format!("\tmovq {}, global_var{}(%rip)", src, varname),
    }
}

fn asm_instruction(
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    instr: Instruction,
    root: &String,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
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
                Bop::MulBop(mop) => {
                    let mut instructions = vec![get_source1, get_source2];
                    instructions.extend(convert_multipication(mop, Reg::R10, Reg::R9));
                    instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
                    instructions
                }

                Bop::AddBop(_) => {
                    // NOTE: R10 (source1) is on lhs, R9 (source2) on rhs.
                    let operate = format!("\t{} {}, {}", convert_bop_to_asm(op), Reg::R10, Reg::R9);
                    let return_to_stack = store_from_reg(Reg::R9, dest, stack_lookup);

                    return vec![get_source1, get_source2, operate, return_to_stack];
                }
                _ => {
                    let compare = format!("\tcmpq {}, {}", Reg::R10, Reg::R9);
                    let initialize_dest = format!("\tmovq ${}, {}", 0, Reg::R9,);

                    let set_one = format!("\tmovq ${}, {}", 1, Reg::R11);
                    let cond_move = format!(
                        "\t{} {}, {}",
                        convert_rel_op_to_cmov_type(op),
                        Reg::R11,
                        Reg::R9
                    );
                    let return_to_stack = store_from_reg(Reg::R9, dest, stack_lookup);
                    return vec![
                        get_source1,
                        get_source2,
                        compare,
                        initialize_dest,
                        set_one,
                        cond_move,
                        return_to_stack,
                    ];
                }
            }
        }
        Instruction::TwoOp { source1, dest, op } => {
            let mut instructions = vec![];
            instructions.push(load_into_reg(Reg::Rax, source1, stack_lookup));
            match op {
                UnOp::Not => {
                    instructions.push(format!("\tnot {}", Reg::Rax));
                    instructions.push(format!("\txor $-2, {}", Reg::Rax));
                }
                UnOp::Neg => instructions.push(format!("\tneg {}", Reg::Rax)),
                UnOp::IntCast => instructions.push(format!("\tcdqe")),
                UnOp::LongCast => {
                    // format!("\tmovq rax, rax")
                }
            };

            instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            return instructions;
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
            let mut instructions = vec![];

            instructions.extend(load_into_reg_arr(Reg::Rax, name, idx, stack_lookup));
            instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            instructions
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let mut instructions: Vec<String> = vec![];
            instructions.push(load_into_reg(Reg::Rax, source, stack_lookup));
            instructions.extend(store_from_reg_arr(Reg::Rax, arr, idx, stack_lookup));
            instructions
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
            let mut pop_instrs: Option<u32> = None;
            let mut argument_registers: Vec<Reg> =
                vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];

            if (args.len() % 2 == 1) {
                instructions.push(format!("\tsubq $8, {}", Reg::Rsp));
            }

            // push arguments into registers and onto stack if needed
            for arg in args.iter() {
                match arg {
                    Arg::VarArg(label) => {
                        let arg_reg = argument_registers.pop();
                        match arg_reg {
                            Some(reg) => {
                                instructions.push(load_into_reg(reg, *label, stack_lookup))
                            }
                            None => {
                                instructions.push(load_into_reg(Reg::Rax, *label, stack_lookup));
                                instructions.push(format!("\tpushq {}", Reg::Rax));

                                if pop_instrs.is_none() {
                                    pop_instrs = Some(0);
                                }

                                pop_instrs = Some(pop_instrs.unwrap() + 1);
                            }
                        }
                    }
                    Arg::StrArg(string) => {
                        let arg_reg = argument_registers.pop();

                        match arg_reg {
                            Some(reg) => {
                                instructions.push(format!(
                                    "\tleaq global_str{}(%rip), {}",
                                    global_strings.get(string).unwrap(),
                                    reg
                                ));
                            }
                            None => {
                                instructions.push(format!(
                                    "\tmovq global_str{}, {}",
                                    global_strings.get(string).unwrap(),
                                    Reg::Rax,
                                ));
                                instructions.push(format!("\tpushq {}", Reg::Rax));
                            }
                        }
                    }
                }
            }

            // call the function
            if func_name == "printf".to_string() {
                instructions.push(format!("\txorq {}, {}", Reg::Rax, Reg::Rax));
            }

            if external_funcs.contains(&func_name) && mac {
                instructions.push(format!("\tcall _{}", func_name));
            } else {
                instructions.push(format!("\tcall {}", func_name));
            }

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

            match pop_instrs {
                Some(v) => {
                    if v % 2 == 1 {
                        instructions.push(format!("\taddq $8, {}", Reg::Rsp));
                    }
                }
                None => {}
            }

            instructions
        }
    }
}
