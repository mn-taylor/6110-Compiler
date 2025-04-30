use crate::cfg::{Arg, BasicBlock, CfgMethod, CfgProgram, CfgType, ImmVar, Instruction, Jump};
use crate::cfg_build::VarLabel;
use crate::ir::{Bop, UnOp};
use crate::parse::Primitive;
use crate::scan::{format_str_for_output, AddOp, EqOp, MulOp, RelOp};
use std::fmt;

use std::collections::HashMap;

#[derive(Clone, Copy)]
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

fn convert_aop_to_asm(aop: AddOp) -> String {
    match aop {
        AddOp::Sub => "subq",
        AddOp::Add => "addq",
    }
    .to_string()
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

//

fn convert_multipication_var_var(
    mop: MulOp,
    operand: Reg,
    optarget: Reg,
    dest: Reg,
) -> Vec<String> {
    let mut instructions = vec![];
    match mop {
        MulOp::Mul => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\timulq {}, {}", operand, Reg::Rax));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest))
        }
        MulOp::Mod => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rdx, dest));
        }
        MulOp::Div => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));
        }
    };
    instructions
}

fn convert_multipication_imm_var(
    mop: MulOp,
    operand: i64,
    optarget: Reg,
    dest: Reg,
) -> Vec<String> {
    let mut instructions = vec![];
    match mop {
        MulOp::Mul => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\timulq ${}, {}", operand, Reg::Rax));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));
        }
        MulOp::Mod => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tmovq ${}, {}", operand, Reg::R12));
            instructions.push(format!("\tidivq {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rdx, dest));
        }
        MulOp::Div => {
            instructions.push(format!("\tmovq {}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tmovq ${}, {}", operand, Reg::R9));
            instructions.push(format!("\tidivq {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));
        }
    };
    instructions
}

// the operand register is also the destination register
fn convert_multipication_var_imm(
    mop: MulOp,
    operand: Reg,
    optarget: i64,
    dest: Reg,
) -> Vec<String> {
    let mut instructions = vec![];
    match mop {
        MulOp::Mul => {
            instructions.push(format!("\tmovq ${}, {}", optarget, Reg::Rax));
            instructions.push(format!("\timulq ${}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));
        }
        MulOp::Mod => {
            instructions.push(format!("\tmovq ${}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rdx, dest));
        }
        MulOp::Div => {
            instructions.push(format!("\tmovq ${}, {}", optarget, Reg::Rax));
            instructions.push(format!("\tcqto"));
            instructions.push(format!("\tidivq {}", operand));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));
        }
    };
    instructions
}

fn flip_rel_op(op: Bop) -> Bop {
    match op {
        Bop::RelBop(rop) => Bop::RelBop(match rop {
            RelOp::Ge => RelOp::Lt,
            RelOp::Le => RelOp::Gt,
            RelOp::Gt => RelOp::Le,
            RelOp::Lt => RelOp::Ge,
        }),
        _ => op,
    }
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

    let mut fields = all_fields.iter().collect::<Vec<_>>();
    fields.sort_by(|a, b| {
        let (i, _) = a;
        let (j, _) = b;
        i.cmp(j)
    });

    for (field, (typ, _)) in fields {
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

fn get_global_strings(p: &CfgProgram<Reg>) -> HashMap<String, usize> {
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

pub fn asm_program(p: &CfgProgram<Reg>, mac: bool) -> Vec<String> {
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
    method: &CfgMethod<Reg>,
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
    // println!("offsets: {:?}", offsets);

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
    let mut blocks: Vec<&BasicBlock<Reg>> = method.blocks.values().collect::<Vec<_>>();
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
    b: &BasicBlock<Reg>,
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
    match &b.jump_loc {
        Jump::Uncond(block_id) => instructions.push(format!("\tjmp {}{}", root, block_id)),
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => {
            let jump_var = match source {
                ImmVar::Var(v)=> v.clone(),
                _=> panic!("Conditional Jump sources should never be immediates, since they can be simplified")
            };

            let compare = format!("\tcmpq $1, {}", jump_var);
            let true_jump = format!("\tje {}{}", root, true_block);
            let false_jump = format!("\tjmp {}{}", root, false_block);

            instructions.extend([compare, true_jump, false_jump]);
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
    index: Reg,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    instructions.push(format!("\tmovq {}, {}", index, Reg::R9));
    instructions.push(format!("\tsalq $3, {}", Reg::R9));
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&varname) {
        Some((_, offset)) => {
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq -{offset}({}), {dest}", Reg::R10));
        }
        None => {
            instructions.push(format!("\tleaq global_var{}(%rip), {}", varname, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq -0({}), {dest}", Reg::R10));
        }
    }
    instructions
}

fn load_into_reg_arr_imm(
    dest: Reg,
    varname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    // instructions.push(format!("\tmovq ${index}, {}", Reg::R9));
    match stack_lookup.get(&varname) {
        Some((_, offset)) => {
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq ${}, {}", index * 8, Reg::R10));
            instructions.push(format!("\tmovq -{offset}({}), {dest}", Reg::R10));
        }
        None => {
            instructions.push(format!("\tleaq global_var{}(%rip), {}", varname, Reg::R10));
            instructions.push(format!("\taddq ${}, {}", index * 8, Reg::R10));
            instructions.push(format!("\tmovq -0({}), {dest}", Reg::R10));
        }
    }
    instructions
}

fn store_from_reg_arr(
    src: Reg,
    arrname: VarLabel,
    index: Reg,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    instructions.push(format!("\tmovq {}, {}", index, Reg::R9));
    instructions.push(format!("\tsalq $3, {}", Reg::R9));
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
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

fn store_from_reg_arr_var_imm(
    src: Reg,
    arrname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    instructions.push(format!("\tmovq ${index}, {}", Reg::R9));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
            // instructions.push(format!("\tsubq ${}, {}", index * 8, Reg::R10));
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
            instructions.push(format!("\taddq ${}, {}", index * 8, Reg::R10));
            instructions.push(format!("\tmovq {src}, -0({}) ", Reg::R10));
            // instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            // instructions.push(format!("\taddq {}, {}", Reg::R10, Reg::R9));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))
        }
    }
    instructions
}

fn store_from_reg_arr_imm_var(
    imm: i64,
    arrname: VarLabel,
    index: Reg,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    instructions.push(format!("\tmovq {}, {}", index, Reg::R9));
    instructions.push(format!("\tsalq $3, {}", Reg::R9));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq ${imm}, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, -{offset}({})", Reg::R9, Reg::R10));
        }
        None => {
            instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq ${imm}, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, -0({}) ", Reg::R9, Reg::R10));
        }
    }
    instructions
}

fn store_from_reg_arr_imm_imm(
    imm: i64,
    arrname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    instructions.push(format!("\tmovq ${index}, {}", Reg::R9));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::R10));
            instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq ${imm}, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, -{offset}({})", Reg::R9, Reg::R10));
        }
        None => {
            instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            instructions.push(format!("\taddq {}, {}", Reg::R9, Reg::R10));
            instructions.push(format!("\tmovq ${imm}, {}", Reg::R9));
            instructions.push(format!("\tmovq {}, -0({}) ", Reg::R9, Reg::R10));
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

fn store_from_reg_imm(
    imm: i64,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> String {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => format!("\tmovq ${}, -{}({})", imm, offset, Reg::Rbp),
        None => format!("\tmovq ${}, global_var{}(%rip)", imm, varname),
    }
}

fn asm_add_op(
    source1: ImmVar<Reg>,
    source2: ImmVar<Reg>,
    dest: Reg,
    aop: AddOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                //var var
                instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                instructions.push(format!("\tmovq {}, {}", s2, Reg::R10));
                // instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                // instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));

                instructions.push(format!(
                    "\t{} {}, {}",
                    convert_aop_to_asm(aop),
                    Reg::R10,
                    Reg::R9
                ));

                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                // instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                instructions.push(format!(
                    "\t{} ${}, {}",
                    convert_aop_to_asm(aop),
                    i2,
                    Reg::R9
                ));
                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                //  instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(format!("\tmovq {}, {}", s2, Reg::R9));
                // instructions.push(load_into_reg(Reg::R9, s2, stack_lookup));
                match aop {
                    AddOp::Add => {
                        instructions.push(format!(
                            "\t{} ${}, {}",
                            convert_aop_to_asm(aop),
                            i1,
                            Reg::R9
                        ));
                    }
                    AddOp::Sub => {
                        instructions.push(format!("\tneg {}", Reg::R9));
                        instructions.push(format!(
                            "\t{} ${}, {}",
                            convert_aop_to_asm(AddOp::Add),
                            i1,
                            Reg::R9
                        ));
                    }
                }
                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(_) => panic!("Three Op with two immediate sources can be simplified"),
        },
    }

    instructions
}

fn asm_mul_op(
    source1: ImmVar<Reg>,
    source2: ImmVar<Reg>,
    dest: Reg,
    mop: MulOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => {
            match source2 {
                ImmVar::Var(s2) => {
                    // var-var
                    instructions.extend(convert_multipication_var_var(mop, s2, s1, dest));
                }
                ImmVar::Imm(i2) => {
                    // var_imm
                    instructions.extend(convert_multipication_imm_var(mop, i2, s1, dest));
                }
            }
        }
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // imm_var
                instructions.extend(convert_multipication_var_imm(mop, s2, i1, dest));
            }
            ImmVar::Imm(_) => {
                panic!("Three operand expression should be simplified");
            }
        },
    }
    instructions
}

fn asm_rel_op(
    source1: ImmVar<Reg>,
    source2: ImmVar<Reg>,
    dest: Reg,
    rop: Bop,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<String> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                instructions.push(format!("\tmovq {}, {}", s2, Reg::R10));

                instructions.push(format!("\tcmpq {}, {}", Reg::R10, Reg::R9));
                instructions.push(format!("\tmovq ${}, {}", 0, Reg::R9,));

                instructions.push(format!("\tmovq ${}, {}", 1, Reg::R10));
                instructions.push(format!(
                    "\t{} {}, {}",
                    convert_rel_op_to_cmov_type(rop),
                    Reg::R10,
                    Reg::R9
                ));
                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                // instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));

                instructions.push(format!("\tcmpq ${}, {}", i2, Reg::R9));

                instructions.push(format!("\tmovq ${}, {}", 0, Reg::R9,));

                instructions.push(format!("\tmovq ${}, {}", 1, Reg::R10));
                instructions.push(format!(
                    "\t{} {}, {}",
                    convert_rel_op_to_cmov_type(rop),
                    Reg::R10,
                    Reg::R9
                ));
                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                instructions.push(format!("\tmovq {}, {}", s2, Reg::R10));
                // instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));

                instructions.push(format!("\tcmpq ${i1}, {}", Reg::R10));
                instructions.push(format!("\tmovq ${}, {}", 0, Reg::R9,));

                instructions.push(format!("\tmovq ${}, {}", 1, Reg::R11));
                instructions.push(format!(
                    "\t{} {}, {}",
                    convert_rel_op_to_cmov_type(flip_rel_op(rop)),
                    Reg::R11,
                    Reg::R9
                ));
                instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
            }
            ImmVar::Imm(_) => {
                panic!("Three Op expression with two immediate sources can be simplified")
            }
        },
    }

    instructions
}

fn asm_instruction(
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    instr: Instruction<Reg>,
    root: &String,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
) -> Vec<String> {
    match instr {
        Instruction::PhiExpr { .. } => panic!(),
        Instruction::ParMov(_) => panic!(),
        Instruction::ThreeOp {
            source1,
            source2,
            dest,
            op,
        } => match op {
            Bop::MulBop(mop) => asm_mul_op(source1, source2, dest, mop, stack_lookup),

            Bop::AddBop(aop) => asm_add_op(source1, source2, dest, aop, stack_lookup),
            _ => asm_rel_op(source1, source2, dest, op, stack_lookup),
        },
        Instruction::TwoOp { source1, dest, op } => {
            // Will maintain the invariant that the source is always an s
            let s = match source1 {
                ImmVar::Var(v) => v,
                ImmVar::Imm(_) => panic!(),
            };

            let mut instructions = vec![];
            //  instructions.push(load_into_reg(Reg::Rax, s, stack_lookup));
            instructions.push(format!("\tmovq {}, {}", s, Reg::Rax));
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

            // instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest));

            return instructions;
        }
        Instruction::MoveOp { source, dest } => {
            // Maintain the invariant that the source is not a constant, when assembling
            let s: Reg = match source {
                ImmVar::Var(v) => v,
                ImmVar::Imm(_) => panic!(),
            };

            let mut instructions = vec![];
            instructions.push(format!("\tmovq {}, {}", s, dest));
            return instructions;
            // let get_source = load_into_reg(Reg::Rax, s, stack_lookup);
            // let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            // return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            return vec![format!("\tmovq ${}, {}", constant, dest)];
            // let load_const = format!("\tmovq ${}, {}", constant, Reg::Rax);
            // let restore_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            // return vec![load_const, restore_to_stack];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let mut instructions = vec![];

            match idx {
                ImmVar::Var(v) => {
                    instructions.extend(load_into_reg_arr(dest, name, v, stack_lookup));
                }
                ImmVar::Imm(i) => {
                    // todo!("write load into reg function for immediate indices");
                    instructions.extend(load_into_reg_arr_imm(dest, name, i, stack_lookup));
                }
            }

            // instructions.push(format!(""))
            // instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            instructions
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let mut instructions: Vec<String> = vec![];

            match source {
                ImmVar::Var(s) => match idx {
                    ImmVar::Var(index) => {
                        instructions.extend(store_from_reg_arr(s, arr, index, stack_lookup))
                    }
                    ImmVar::Imm(i) => {
                        instructions.extend(store_from_reg_arr_var_imm(s, arr, i, stack_lookup))
                    }
                },
                ImmVar::Imm(src) => match idx {
                    ImmVar::Var(index) => {
                        println!("here instead");
                        instructions.extend(store_from_reg_arr_imm_var(
                            src,
                            arr,
                            index,
                            stack_lookup,
                        ))
                    }
                    ImmVar::Imm(i) => {
                        print!("here");
                        instructions.extend(store_from_reg_arr_imm_imm(src, arr, i, stack_lookup))
                    }
                },
            }

            instructions
        }

        Instruction::Ret(ret_val) => {
            let mut instructions = vec![];
            match ret_val {
                Some(return_imm_var) => match return_imm_var {
                    ImmVar::Var(v) => {
                        instructions.push(format!("\tmovq {}, {}", v, Reg::Rax));
                        //instructions.push(load_into_reg(Reg::Rax, v, stack_lookup))
                    }
                    ImmVar::Imm(i) => instructions.push(format!("\tmovq ${i}, {}", Reg::Rax)),
                },
                None => {}
            }
            instructions.push(format!("jmp {}end", root));
            instructions
        }
        Instruction::Call(func_name, args, ret_dest) => {
            let mut instructions: Vec<String> = vec![];
            let mut argument_registers: Vec<Reg> =
                vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];

            if args.len() % 2 == 1 && args.len() >= 6 {
                instructions.push(format!("\tsubq $8, {}", Reg::Rsp));
            }

            // push arguments into registers and onto stack if needed
            for arg in args.iter() {
                match arg {
                    Arg::VarArg(label) => {
                        let arg_reg = argument_registers.pop();
                        match arg_reg {
                            Some(reg) => match label {
                                ImmVar::Var(v) => {
                                    instructions.push(format!("\tmovq {}, {}", v, reg));
                                    // instructions.push(load_into_reg(reg, *v, stack_lookup))
                                }
                                ImmVar::Imm(i) => {
                                    instructions.push(format!("\tmovq ${i}, {}", reg))
                                }
                            },
                            None => {
                                match label {
                                    ImmVar::Var(v) => {
                                        instructions.push(format!("\tmovq {}, {}", v, Reg::Rax));
                                        //instructions.push(load_into_reg(Reg::Rax, *v, stack_lookup))
                                    }
                                    ImmVar::Imm(i) => {
                                        instructions.push(format!("\tmovq ${i}, {}", Reg::Rax))
                                    }
                                }
                                instructions.push(format!("\tpushq {}", Reg::Rax));
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
                if func_name == "main".to_string() && mac {
                    instructions.push(format!("\tcall _{}", func_name));
                } else {
                    instructions.push(format!("\tcall {}", func_name));
                }
            }

            // store return value into temp
            match ret_dest {
                Some(dest) => instructions.push(format!("\tmovq {}, {}", Reg::Rax, dest)),
                None => {}
            }

            instructions
        }
        _ => panic!("Spills and reloads not implemented"),
    }
}
