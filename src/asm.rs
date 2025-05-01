use crate::cfg::{CfgType, ImmVar, Jump};
use crate::cfg_build::{Arg, BasicBlock, CfgMethod, CfgProgram, Instruction, VarLabel};
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
use Reg::*;

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

fn convert_aop_to_asm(aop: AddOp) -> &'static str {
    match aop {
        AddOp::Sub => "subq",
        AddOp::Add => "addq",
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

pub enum InsnArg {
    ImmArg(i64),
    RegArg(Reg),
    Offset(i64, Reg),
}
use InsnArg::*;

pub enum Insn {
    Special(String),
    ZeroArgs(&'static str),
    OneArg(&'static str, InsnArg),
    TwoArgs(&'static str, InsnArg, InsnArg),
}
use Insn::*;

impl fmt::Display for InsnArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            ImmArg(i) => write!(f, "${}", i),
            RegArg(r) => write!(f, "{}", r),
            Offset(o, r) => write!(f, "{}({})", o, r),
        }
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Special(s) => write!(f, "{}", s),
            ZeroArgs(s) => write!(f, "\t{}", s),
            OneArg(s, a) => write!(f, "\t{} {}", s, a),
            TwoArgs(s, a1, a2) => write!(f, "\t{} {}, {}", s, a1, a2),
        }
    }
}

trait ToInsnArg {
    fn to_insn_arg(self) -> InsnArg;
}

impl ToInsnArg for i64 {
    fn to_insn_arg(self) -> InsnArg {
        ImmArg(self)
    }
}

impl ToInsnArg for Reg {
    fn to_insn_arg(self) -> InsnArg {
        RegArg(self)
    }
}

impl ToInsnArg for (i64, Reg) {
    fn to_insn_arg(self) -> InsnArg {
        Offset(self.0, self.1)
    }
}

trait ToInsn {
    fn to_insn(self) -> Insn;
}

impl ToInsn for &'static str {
    fn to_insn(self) -> Insn {
        ZeroArgs(self)
    }
}

impl<T: ToInsnArg> ToInsn for (&'static str, T) {
    fn to_insn(self) -> Insn {
        OneArg(self.0, self.1.to_insn_arg())
    }
}

impl<T: ToInsnArg, U: ToInsnArg> ToInsn for (&'static str, T, U) {
    fn to_insn(self) -> Insn {
        TwoArgs(self.0, self.1.to_insn_arg(), self.2.to_insn_arg())
    }
}

fn insn<T: ToInsn>(i: T) -> Insn {
    i.to_insn()
}

fn convert_multipication_var_var(mop: MulOp, operand: Reg, optarget: Reg) -> Vec<Insn> {
    match mop {
        MulOp::Mul => vec![insn(("imulq", operand, optarget))],
        MulOp::Mod => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("idivq", operand)),
            insn(("movq", Rdx, optarget)),
        ],
        MulOp::Div => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("idivq", operand)),
            insn(("movq", Rax, optarget)),
        ],
    }
}

fn convert_multipication_imm_var(mop: MulOp, operand: i64, optarget: Reg) -> Vec<Insn> {
    match mop {
        MulOp::Mul => vec![insn(("imulq", operand, optarget))],
        MulOp::Mod => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("movq", operand, R9)),
            insn(("idivq", R9)),
            insn(("movq", Rdx, optarget)),
        ],
        MulOp::Div => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("movq", operand, R9)),
            insn(("idivq", R9)),
            insn(("movq", Rax, optarget)),
        ],
    }
}

// the operand register is also the destination register
fn convert_multipication_var_imm(mop: MulOp, operand_dest: Reg, optarget: i64) -> Vec<Insn> {
    match mop {
        MulOp::Mul => vec![insn(("imulq", optarget, operand_dest))],
        MulOp::Mod => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("idivq", operand_dest)),
            insn(("movq", Rdx, operand_dest)),
        ],
        MulOp::Div => vec![
            insn(("movq", optarget, Rax)),
            insn("cqto"),
            insn(("idivq", operand_dest)),
            insn(("movq", Rax, operand_dest)),
        ],
    }
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

fn convert_rel_op_to_cmov_type(op: Bop) -> &'static str {
    match op {
        Bop::RelBop(rop) => match rop {
            RelOp::Lt => "cmovl",
            RelOp::Le => "cmovle",
            RelOp::Gt => "cmovg",
            RelOp::Ge => "cmovge",
        },
        Bop::EqBop(eop) => match eop {
            EqOp::Eq => "cmove",
            EqOp::Neq => "cmovne",
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

pub fn asm_program(p: &CfgProgram, mac: bool) -> Vec<Insn> {
    let mut insns: Vec<Insn> = vec![];

    if !mac {
        insns.push(Special(
            ".section .note.GNU-stack,\"\",@progbits".to_string(),
        ));
    }

    let external_funcs = &p.externals;
    let glob_strings = &get_global_strings(p);
    let glob_fields = &p.global_fields;

    insns.push(Special(".data".to_string()));
    for (varname, (typ, _)) in glob_fields {
        match typ {
            CfgType::Array(_, len) => insns.push(Special(format!(
                "global_var{}:\n\t.zero {}",
                varname,
                len * 8
            ))),
            CfgType::Scalar(_) => insns.push(Special(format!("global_var{}:\n\t.zero 8", varname))), /*8 bytes*/
        }
    }

    // I think we drop newlines when there is only one string
    for (strin, label) in glob_strings {
        insns.push(Special(format!(
            "global_str{}:\n\t.string \"{}\"",
            label,
            format_str_for_output(strin)
        )));
    }

    insns.push(Special(".text".to_string()));
    for external in external_funcs {
        if mac {
            insns.push(Special(format!("\t.extern _{}", external)));
        } else {
            insns.push(Special(format!("\t.extern {}", external)));
        }
    }

    insns.push(Special("error_handler:".to_string()));
    if mac {
        insns.push(Special("\tmovl $0x2000001, %eax".to_string()));
        insns.push(Special("\tmovl $-1, %edi".to_string()));
        insns.push(Special("\tsyscall".to_string()));
    } else {
        insns.push(Special("\tmovl $0x01, %eax".to_string()));
        insns.push(Special("\tmovl $-1, %ebx".to_string()));
        insns.push(Special("\tsyscall".to_string()));
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
) -> Vec<Insn> {
    let mut instructions: Vec<Insn> = vec![];
    if method.name == "main" {
        let name = format!("{}main", if mac { "_" } else { "" });
        instructions.push(Special(format!(".globl {}", name)));
        instructions.push(Special(format!("{}:", name)));
    } else {
        instructions.push(Special(format!("{}:", method.name)));
    }

    instructions.push(insn(("pushq", Rbp)));
    instructions.push(insn(("movq", Rsp, Rbp)));

    let (offsets, total_offset) = build_stack(method.fields.clone());
    // println!("offsets: {:?}", offsets);

    // allocate space enough space on the stack
    instructions.push(insn(("subq", total_offset as i64, Rsp)));

    // read parameters from registers and/or stack
    let mut argument_registers: Vec<Reg> = vec![R9, R8, Rcx, Rdx, Rsi, Rdi];
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
            instructions.push(insn((
                "movq",
                (16 + 8 * (method.params.len() as i64 - 1 - i as i64), Rbp),
                Rax,
            )));
            instructions.push(store_from_reg(Reg::Rax, *llname, &offsets));
        }
    }

    instructions.push(Special(format!("\tjmp {}0", method.name)));

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
    instructions.push(Special(format!("{}end:", &method.name)));

    // return the stack to original state
    // instructions.push(format!("\taddq ${}, {}", total_offset, Reg::Rsp,));
    // instructions.push(format!("\tpopq {}", Reg::Rbp));
    // instructions.push(format!("\tmovq {}, {}", Reg::Rbp, Reg::Rsp));

    instructions.push(insn("leave"));

    if method.name == "main" {
        instructions.push(insn(("xorq", Rax, Rax)));
    }

    // ret instruction
    instructions.push(insn("ret"));
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
) -> Vec<Insn> {
    let mut instructions: Vec<Insn> = vec![];

    // make label
    instructions.push(Special(format!("{}{}:", root, b.block_id)));

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
        instructions.push(Special("".to_string()));
    }

    // handle jumps
    match &b.jump_loc {
        Jump::Uncond(block_id) => instructions.push(Special(format!("\tjmp {}{}", root, block_id))),
        Jump::Cond {
            source,
            true_block,
            false_block,
        } => {
            let jump_var = match source {
                ImmVar::Var(v)=> v.clone(),
                _=> panic!("Conditional Jump sources should never be immediates, since they can be simplified")
            };
            instructions.push(load_into_reg(R9, jump_var, stack_lookup));

            let compare = insn(("cmpq", 1, R9));
            let true_jump = Special(format!("\tje {}{}", root, true_block));
            let false_jump = Special(format!("\tjmp {}{}", root, false_block));

            instructions.extend([compare, true_jump, false_jump]);
        }
        Jump::Nowhere => {
            if return_type.is_some() {
                instructions.push(Special(format!("\tjmp error_handler")));
            } else {
                instructions.push(Special(format!("\tjmp {}end", root)));
            }
        }
    }
    instructions
}

fn load_into_reg(
    dest: Reg,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Insn {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => insn(("movq", (-(*offset as i64), Rbp), dest)),
        None => Special(format!("\tmovq global_var{}(%rip), {}", varname, dest)),
    }
}

fn load_into_reg_arr(
    dest: Reg,
    varname: VarLabel,
    index: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
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
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", R9, R10)));
            instructions.push(insn(("movq", (-(*offset as i64), R10), dest)));
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

            instructions.push(insn(("salq", 3, R9)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                varname, R10
            )));
            instructions.push(insn(("addq", R9, R10)));
            instructions.push(insn(("movq", (-0, R10), dest)));
        }
    }
    instructions
}

fn load_into_reg_arr_imm(
    dest: Reg,
    varname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    // instructions.push(format!("\tmovq ${index}, {}", Reg::R9));
    match stack_lookup.get(&varname) {
        Some((_, offset)) => {
            // movl offset(base, index, scale), destination

            // instructions.push(format!(
            //     "\tmovq -{}({}, {}, $8), {dest}",
            //     offset,
            //     Reg::Rbp,
            //     Reg::R9
            // ))
            // instructions.push(format!("\tsalq $3, {}", Reg::R9));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", index * 8, R10)));
            // instructions.push(format!("\tsubq {}, {}", Reg::R9, Reg::R10));
            instructions.push(insn(("movq", (-(*offset as i64), R10), dest)));
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

            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                varname, R10
            )));
            instructions.push(insn(("addq", index * 8, R10)));
            instructions.push(insn(("movq", (-0, R10), dest)));
        }
    }
    instructions
}

fn store_from_reg_arr(
    src: Reg,
    arrname: VarLabel,
    index: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", R9, R10)));
            instructions.push(insn(("movq", src, (-(*offset as i64), R10))));
        }
        None => {
            // instructions.push(format!(
            //    "\tleaq (global_var{}(%rip), {}, $16), {}",
            //    arrname,
            //    Reg::R9,
            //    Reg::R9
            // ));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))

            instructions.push(insn(("salq", 3, R9)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, R10
            )));
            instructions.push(insn(("addq", R9, R10)));
            instructions.push(insn(("movq", src, (-0, R10))));
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
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(insn(("movq", index, R9)));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", R9, R10)));
            // instructions.push(format!("\tsubq ${}, {}", index * 8, Reg::R10));
            instructions.push(insn(("movq", src, (-(*offset as i64), R10))));
        }
        None => {
            // instructions.push(format!(
            //    "\tleaq (global_var{}(%rip), {}, $16), {}",
            //    arrname,
            //    Reg::R9,
            //    Reg::R9
            // ));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))

            instructions.push(insn(("salq", 3, R9)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, R10
            )));
            instructions.push(insn(("addq", index * 8, R10)));
            instructions.push(insn(("movq", src, (-0, R10))));
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
    index: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(load_into_reg(R9, index, stack_lookup));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", R9, R10)));
            instructions.push(insn(("movq", imm, R9)));
            instructions.push(insn(("movq", R9, (-(*offset as i64), R10))));
        }
        None => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, R10
            )));
            instructions.push(insn(("addq", R9, R10)));
            instructions.push(insn(("movq", imm, R9)));
            instructions.push(insn(("movq", R9, (-0, R10))));
        }
    }
    instructions
}

fn store_from_reg_arr_imm_imm(
    imm: i64,
    arrname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    instructions.push(insn(("movq", index, R9)));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(insn(("movq", Rbp, R10)));
            instructions.push(insn(("subq", R9, R10)));
            instructions.push(insn(("movq", imm, R9)));
            instructions.push(insn(("movq", R9, (-(*offset as i64), R10))));
        }
        None => {
            instructions.push(insn(("salq", 3, R9)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, R10
            )));
            instructions.push(insn(("addq", R9, R10)));
            instructions.push(insn(("movq", imm, R9)));
            instructions.push(insn(("movq", R9, (-0, R10))));
        }
    }
    instructions
}

fn store_from_reg(
    src: Reg,
    varname: VarLabel,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Insn {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => insn(("movq", src, (-(*offset as i64), Rbp))),
        None => Special(format!("\tmovq {}, global_var{}(%rip)", src, varname)),
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
    source1: ImmVar<u32>,
    source2: ImmVar<u32>,
    dest: u32,
    aop: AddOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                //var var
                instructions.push(load_into_reg(R9, s1, stack_lookup));
                instructions.push(load_into_reg(R10, s2, stack_lookup));

                instructions.push(insn((convert_aop_to_asm(aop), R10, R9)));
                instructions.push(store_from_reg(R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                instructions.push(load_into_reg(R9, s1, stack_lookup));
                instructions.push(insn((convert_aop_to_asm(aop), i2, R9)));
                instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(load_into_reg(Reg::R9, s2, stack_lookup));
                match aop {
                    AddOp::Add => {
                        instructions.push(insn((convert_aop_to_asm(aop), i1, R9)));
                    }
                    AddOp::Sub => {
                        instructions.push(insn(("neg", R9)));
                        instructions.push(insn((convert_aop_to_asm(AddOp::Add), i1, R9)));
                    }
                }
                instructions.push(store_from_reg(R9, dest, stack_lookup));
            }
            ImmVar::Imm(_) => panic!("Three Op with two immediate sources can be simplified"),
        },
    }

    instructions
}

fn asm_mul_op(
    source1: ImmVar<u32>,
    source2: ImmVar<u32>,
    dest: u32,
    mop: MulOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => {
            match source2 {
                ImmVar::Var(s2) => {
                    // var-var
                    instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                    instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));
                    instructions.extend(convert_multipication_var_var(mop, Reg::R10, Reg::R9));
                    instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
                }
                ImmVar::Imm(i2) => {
                    // var_imm
                    instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                    instructions.extend(convert_multipication_imm_var(mop, i2, Reg::R9));
                    instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
                }
            }
        }
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // imm_var
                instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));
                instructions.extend(convert_multipication_var_imm(mop, Reg::R10, i1));
                instructions.push(store_from_reg(Reg::R10, dest, stack_lookup));
            }
            ImmVar::Imm(_) => {
                panic!("Three operand expression should be simplified");
            }
        },
    }
    instructions
}

fn asm_rel_op(
    source1: ImmVar<u32>,
    source2: ImmVar<u32>,
    dest: u32,
    rop: Bop,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));

                instructions.push(insn(("cmpq", R10, R9)));
                instructions.push(insn(("movq", 0, R9)));

                instructions.push(insn(("movq", 1, R11)));
                instructions.push(insn((convert_rel_op_to_cmov_type(rop), R11, R9)));
                instructions.push(store_from_reg(R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                instructions.push(load_into_reg(R9, s1, stack_lookup));

                instructions.push(insn(("cmpq", i2, R9)));

                instructions.push(insn(("movq", 0, R9)));

                instructions.push(insn(("movq", 1, R11)));
                instructions.push(insn((convert_rel_op_to_cmov_type(rop), R11, R9)));
                instructions.push(store_from_reg(R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                instructions.push(load_into_reg(R10, s2, stack_lookup));

                instructions.push(insn(("cmpq", i1, R10)));
                instructions.push(insn(("movq", 0, R9)));

                instructions.push(insn(("movq", 1, R11)));
                instructions.push(insn((
                    convert_rel_op_to_cmov_type(flip_rel_op(rop)),
                    R11,
                    R9,
                )));
                instructions.push(store_from_reg(R9, dest, stack_lookup));
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
    instr: Instruction,
    root: &String,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
) -> Vec<Insn> {
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
            instructions.push(load_into_reg(Reg::Rax, s, stack_lookup));
            match op {
                UnOp::Not => {
                    instructions.push(insn(("not", Rax)));
                    instructions.push(insn(("xor", -2, Rax)));
                }
                UnOp::Neg => instructions.push(insn(("neg", Rax))),
                UnOp::IntCast => instructions.push(insn("cdqe")),
                UnOp::LongCast => {
                    // format!("\tmovq rax, rax")
                }
            };

            instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            return instructions;
        }
        Instruction::MoveOp { source, dest } => {
            // Maintain the invariant that the source is not a constant, when assembling
            let s: u32 = match source {
                ImmVar::Var(v) => v,
                ImmVar::Imm(_) => panic!(),
            };

            let get_source = load_into_reg(Reg::Rax, s, stack_lookup);
            let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            let load_const = insn(("movq", constant, Reg::Rax));
            let restore_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            return vec![load_const, restore_to_stack];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let mut instructions = vec![];

            match idx {
                ImmVar::Var(v) => {
                    instructions.extend(load_into_reg_arr(Reg::Rax, name, v, stack_lookup));
                }
                ImmVar::Imm(i) => {
                    // todo!("write load into reg function for immediate indices");
                    instructions.extend(load_into_reg_arr_imm(Reg::Rax, name, i, stack_lookup));
                }
            }

            instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            instructions
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let mut instructions: Vec<Insn> = vec![];

            match source {
                ImmVar::Var(s) => {
                    instructions.push(load_into_reg(Rax, s, stack_lookup));
                    match idx {
                        ImmVar::Var(index) => instructions.extend(store_from_reg_arr(
                            Reg::Rax,
                            arr,
                            index,
                            stack_lookup,
                        )),
                        ImmVar::Imm(i) => {
                            print!("option 3");
                            instructions.extend(store_from_reg_arr_var_imm(
                                Reg::Rax,
                                arr,
                                i,
                                stack_lookup,
                            ))
                        }
                    }
                }
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
                    ImmVar::Var(v) => instructions.push(load_into_reg(Reg::Rax, v, stack_lookup)),
                    ImmVar::Imm(i) => instructions.push(insn(("movq", i, Rax))),
                },
                None => {}
            }
            instructions.push(Special(format!("jmp {}end", root)));
            instructions
        }
        Instruction::Call(func_name, args, ret_dest) => {
            let mut instructions: Vec<Insn> = vec![];
            let mut argument_registers: Vec<Reg> =
                vec![Reg::R9, Reg::R8, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi];

            if args.len() % 2 == 1 && args.len() >= 6 {
                instructions.push(insn(("subq", 8, Rsp)));
            }

            // push arguments into registers and onto stack if needed
            for arg in args.iter() {
                match arg {
                    Arg::VarArg(label) => {
                        let arg_reg = argument_registers.pop();
                        match arg_reg {
                            Some(reg) => match label {
                                ImmVar::Var(v) => {
                                    instructions.push(load_into_reg(reg, *v, stack_lookup))
                                }
                                ImmVar::Imm(i) => instructions.push(insn(("movq", *i, reg))),
                            },
                            None => {
                                match label {
                                    ImmVar::Var(v) => {
                                        instructions.push(load_into_reg(Rax, *v, stack_lookup))
                                    }
                                    ImmVar::Imm(i) => instructions.push(insn(("movq", *i, Rax))),
                                }
                                instructions.push(insn(("pushq", Rax)));
                            }
                        }
                    }
                    Arg::StrArg(string) => {
                        let arg_reg = argument_registers.pop();

                        match arg_reg {
                            Some(reg) => {
                                instructions.push(Special(format!(
                                    "\tleaq global_str{}(%rip), {}",
                                    global_strings.get(string).unwrap(),
                                    reg
                                )));
                            }
                            None => {
                                instructions.push(Special(format!(
                                    "\tmovq global_str{}, {}",
                                    global_strings.get(string).unwrap(),
                                    Rax,
                                )));
                                instructions.push(insn(("pushq", Rax)));
                            }
                        }
                    }
                }
            }

            // call the function
            if func_name == "printf".to_string() {
                instructions.push(insn(("xorq", Rax, Rax)));
            }

            if external_funcs.contains(&func_name) && mac {
                instructions.push(Special(format!("\tcall _{}", func_name)));
            } else {
                if func_name == "main".to_string() && mac {
                    instructions.push(Special(format!("\tcall _{}", func_name)));
                } else {
                    instructions.push(Special(format!("\tcall {}", func_name)));
                }
            }

            // store return value into temp
            match ret_dest {
                Some(dest) => instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup)),
                None => {}
            }

            instructions
        }
        _ => panic!("Spills and reloads not implemented"),
    }
}
