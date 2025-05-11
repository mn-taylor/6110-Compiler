use crate::cfg::{
    Arg, BasicBlock, CfgMethod, CfgProgram, CfgType, ImmVar, Instruction, Jump, MemVarLabel,
};
use crate::cfg_build::VarLabel;
use crate::ir::{Bop, UnOp};
use crate::parse::Primitive;
use crate::scan::{format_str_for_output, AddOp, EqOp, MulOp, RelOp, Sum};
use std::fmt;

use std::collections::HashMap;

#[derive(Hash, Eq, Clone, Copy, Debug, PartialEq)]
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

fn store_from_reg_var(
    src: Reg,
    varname: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Insn {
    match varname {
        Sum::Inl(reg) => insn(("movq", src, reg)),
        Sum::Inr(variable) => match stack_lookup.get(&variable) {
            Some((_, offset)) => insn(("movq", src, (-(*offset as i64), Rbp))),
            None => Special(format!("\tmovq {}, global_var{}(%rip)", src, variable)),
        },
    }
}

fn load_into_reg_var(
    dest: Reg,
    varname: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Insn {
    match varname {
        Sum::Inl(reg) => insn(("movq", reg, dest)),
        Sum::Inr(variable) => match stack_lookup.get(&variable) {
            Some((_, offset)) => insn(("movq", (-(*offset as i64), Rbp), dest)),
            None => Special(format!("\tmovq global_var{}(%rip), {}", variable, dest)),
        },
    }
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

fn convert_multipication_var_var(
    mop: MulOp,
    operand: Sum<Reg, MemVarLabel>,
    optarget: Sum<Reg, MemVarLabel>,
    dest: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];

    let operand = match operand {
        Sum::Inl(reg) => reg,
        Sum::Inr(_) => {
            instructions.push(load_into_reg_var(R9, operand, stack_lookup));
            R9
        }
    };

    let optarget = match optarget {
        Sum::Inl(reg) => reg,
        Sum::Inr(_) => {
            instructions.push(load_into_reg_var(Rax, optarget, stack_lookup));
            Rax
        }
    };

    match mop {
        MulOp::Mul => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn(("imulq", operand, Rax)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
        }
        MulOp::Mod => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("idivq", operand)));
            instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
        }
        MulOp::Div => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("idivq", operand)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
        }
    };
    instructions
}

fn convert_multipication_imm_var(
    mop: MulOp,
    operand: i64,
    optarget: Sum<Reg, MemVarLabel>,
    dest: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];

    let optarget = match optarget {
        Sum::Inl(reg) => reg,
        Sum::Inr(_) => {
            instructions.push(load_into_reg_var(Rax, optarget, stack_lookup));
            Rax
        }
    };
    match mop {
        MulOp::Mul => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn(("imulq", operand, Rax)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
        }
        MulOp::Mod => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("movq", operand, R9)));
            instructions.push(insn(("idivq", R9)));
            instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
        }
        MulOp::Div => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("movq", operand, R9)));
            instructions.push(insn(("idivq", R9)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
        }
    };
    instructions
}

// the operand register is also the destination register
fn convert_multipication_var_imm(
    mop: MulOp,
    operand: Sum<Reg, MemVarLabel>,
    optarget: i64,
    dest: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];

    let operand = match operand {
        Sum::Inl(reg) => reg,
        Sum::Inr(_) => {
            instructions.push(load_into_reg_var(Reg::R9, operand, stack_lookup));
            Reg::R9
        }
    };

    match mop {
        MulOp::Mul => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn(("imulq", operand, Rax)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
        }
        MulOp::Mod => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("idivq", operand)));
            instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
        }
        MulOp::Div => {
            instructions.push(insn(("movq", optarget, Rax)));
            instructions.push(insn("cqto"));
            instructions.push(insn(("idivq", operand)));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
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
        _ => panic!(),
    }
}

fn round_up(a: u64, factor: u64) -> u64 {
    let big_enough = a + (factor - 1);
    big_enough - big_enough % factor
}

fn build_stack(
    all_fields: HashMap<VarLabel, (CfgType, String)>,
    extra_offset: u64,
) -> (HashMap<VarLabel, (CfgType, u64)>, u64) {
    let mut offset: u64 = 8 + extra_offset;
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

fn get_global_strings(p: &CfgProgram<Sum<Reg, MemVarLabel>>) -> HashMap<String, usize> {
    let mut all_strings = HashMap::new();
    for method in p.methods.iter() {
        for block in method.blocks.values() {
            for insn in block.body.iter() {
                match insn {
                    Instruction::StoreParam(_, arg) => match arg {
                        Arg::StrArg(s) => {
                            if let None = all_strings.get(s) {
                                all_strings.insert(s.clone(), all_strings.len());
                            }
                        }
                        _ => (),
                    },
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

pub fn asm_program(p: &CfgProgram<Sum<Reg, MemVarLabel>>, mac: bool) -> Vec<Insn> {
    let mut insns = vec![];

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
            CfgType::Scalar(_) => insns.push(Special(format!("global_var{}:\n\t.zero 8", varname))),
            /*8 bytes*/
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

    insns.push(Special(format!("error_handler:")));
    if mac {
        insns.push(Special(format!("\tmovl $0x2000001, %eax")));
        insns.push(Special(format!("\tmovl $-1, %edi")));
        insns.push(Special(format!("\tsyscall")));
    } else {
        insns.push(Special(format!("\tmovl $0x01, %eax")));
        insns.push(Special(format!("\tmovl $-1, %ebx")));
        insns.push(Special(format!("\tsyscall")));
    }

    for method in p.methods.iter() {
        insns.extend(asm_method(method, mac, external_funcs, glob_strings));
    }
    insns
}

pub fn asm_method(
    method: &CfgMethod<Sum<Reg, MemVarLabel>>,
    mac: bool,
    external_funcs: &Vec<String>,
    global_strings: &HashMap<String, usize>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    if method.name == "main" {
        let name = format!("{}main", if mac { "_" } else { "" });
        instructions.push(Special(format!(".globl {}", name)));
        instructions.push(Special(format!("{}:", name)));
    } else {
        instructions.push(Special(format!("{}:", method.name)));
    }

    instructions.push(insn(("pushq", Rbp)));
    instructions.push(insn(("movq", Rsp, Rbp)));

    let (offsets, total_offset) = build_stack(method.fields.clone(), 40);
    // println!("offsets: {:?}", offsets);

    // allocate space enough space on the stack
    instructions.push(insn(("subq", total_offset as i64, Rsp)));
    // Should have an even number of regs here for stack alignment
    instructions.push(insn(("movq", Rbx, (-8, Rbp))));
    instructions.push(insn(("movq", R12, (-16, Rbp))));
    instructions.push(insn(("movq", R13, (-24, Rbp))));
    instructions.push(insn(("movq", R14, (-32, Rbp))));
    instructions.push(insn(("movq", R15, (-40, Rbp))));

    instructions.push(Special(format!("\tjmp {}0", method.name)));

    // assemble blocks
    let mut blocks: Vec<(&usize, &BasicBlock<Sum<Reg, MemVarLabel>>)> =
        method.blocks.iter().collect::<Vec<_>>();
    blocks.sort_by_key(|(id, _)| *id);

    for (id, block) in blocks {
        instructions.push(Special(format!("{}{}:", &method.name, id)));
        instructions.extend(asm_block(
            block,
            &offsets,
            &method.name,
            method.num_params,
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

    // keep in sync wiht prelude
    instructions.push(insn(("movq", (-8, Rbp), Rbx)));
    instructions.push(insn(("movq", (-16, Rbp), R12)));
    instructions.push(insn(("movq", (-24, Rbp), R13)));
    instructions.push(insn(("movq", (-32, Rbp), R14)));
    instructions.push(insn(("movq", (-40, Rbp), R15)));

    instructions.push(insn("leave"));

    if method.name == "main" {
        instructions.push(insn(("xorq", Rax, Rax)));
    }

    // ret instruction
    instructions.push(insn("ret"));
    return instructions;
}

fn asm_block(
    b: &BasicBlock<Sum<Reg, MemVarLabel>>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
    root: &String,
    num_params: u32,
    return_type: Option<Primitive>,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
) -> Vec<Insn> {
    let mut instructions = vec![];

    // perform instructions
    for instruction in &b.body {
        instructions.extend(asm_instruction(
            stack_lookup,
            instruction.clone(),
            root,
            num_params,
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
            let jump_var: Reg = match source {
                ImmVar::Var(v)=> {
                    match v{
                        Sum::Inl(reg)=> reg.clone(),
                        Sum::Inr(var) => {
                            instructions.push(load_into_reg(Reg::Rdx, *var, stack_lookup));
                            Reg::Rdx
                        }
                    }
                },
                _=> panic!("Conditional Jump sources should never be immediates, since they can be simplified")
            };

            let compare = insn(("cmpq", 1, jump_var));
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
    index: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(load_into_reg_var(Rdx, index, stack_lookup));
    // instructions.push(format!("\tmovq {}, {}", index, Reg::R9));
    instructions.push(insn(("salq", 3, Rdx)));
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    match stack_lookup.get(&varname) {
        Some((_, offset)) => {
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", Rdx, Rax)));
            instructions.push(insn(("movq", (-(*offset as i64), Rax), dest)));
        }
        None => {
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                varname, Rax
            )));
            instructions.push(insn(("addq", Rdx, Rax)));
            instructions.push(insn(("movq", (-0, Rax), dest)));
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
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", index * 8, Rax)));
            instructions.push(insn(("movq", (-(*offset as i64), Rax), dest)));
        }
        None => {
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                varname, Rax
            )));
            instructions.push(insn(("addq", index * 8, Rax)));
            instructions.push(insn(("movq", (-0, Rax), dest)));
        }
    }
    instructions
}

fn store_from_reg_arr(
    src: Sum<Reg, MemVarLabel>,
    arrname: VarLabel,
    index: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(load_into_reg_var(Rdx, index, stack_lookup));
    instructions.push(insn(("salq", 3, Rdx)));

    // let src = match src {
    //     Sum::Inl(reg) => reg,
    //     Sum::Inr(v) => {
    //         instructions.push(load_into_reg(R11, v, stack_lookup));
    //         Reg::R11
    //     }
    // };

    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", Rdx, Rax)));
            let src = match src {
                Sum::Inl(reg) => reg,
                Sum::Inr(v) => {
                    instructions.push(load_into_reg(Rdx, v, stack_lookup));
                    Rdx
                }
            };
            instructions.push(insn(("movq", src, (-(*offset as i64), Rax))));
        }
        None => {
            // instructions.push(format!(
            //    "\tleaq (global_var{}(%rip), {}, $16), {}",
            //    arrname,
            //    Reg::R9,
            //    Reg::R9
            // ));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))

            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, Rax
            )));
            instructions.push(insn(("addq", Rdx, Rax)));
            let src = match src {
                Sum::Inl(reg) => reg,
                Sum::Inr(v) => {
                    instructions.push(load_into_reg(Rdx, v, stack_lookup));
                    Rdx
                }
            };
            instructions.push(insn(("movq", src, (-0, Rax))));
            // instructions.push(format!("\tleaq global_var{}(%rip), {}", arrname, Reg::R10));
            // instructions.push(format!("\taddq {}, {}", Reg::R10, Reg::R9));
            // instructions.push(format!("\tmovq {src}, 0({})", Reg::R9))
        }
    }
    instructions
}

fn store_from_reg_arr_var_imm(
    src: Sum<Reg, MemVarLabel>,
    arrname: VarLabel,
    index: i64,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    instructions.push(insn(("movq", index, Rdx)));

    // let src = match src {
    //     Sum::Inl(reg) => reg,
    //     Sum::Inr(v) => {
    //         instructions.push(load_into_reg(R11, v, stack_lookup));
    //         R11
    //     }
    // };

    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, Rdx)));
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", Rdx, Rax)));
            let src = match src {
                Sum::Inl(reg) => reg,
                Sum::Inr(v) => {
                    instructions.push(load_into_reg(Rdx, v, stack_lookup));
                    Rdx
                }
            };
            instructions.push(insn(("movq", src, (-(*offset as i64), Rax))));
        }
        None => {
            instructions.push(insn(("salq", 3, Rdx)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, Rax
            )));
            instructions.push(insn(("addq", index * 8, Rax)));
            let src = match src {
                Sum::Inl(reg) => reg,
                Sum::Inr(v) => {
                    instructions.push(load_into_reg(Rdx, v, stack_lookup));
                    Rdx
                }
            };
            instructions.push(insn(("movq", src, (-0, Rax))));
        }
    }
    instructions
}

fn store_from_reg_arr_imm_var(
    imm: i64,
    arrname: VarLabel,
    index: Sum<Reg, MemVarLabel>,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    // instructions.push(load_into_reg(Reg::R9, index, stack_lookup));
    // instructions.push(format!("\tmovq {}, {}", index, Reg::R9));
    instructions.push(load_into_reg_var(Rdx, index, stack_lookup));
    instructions.push(insn(("salq", 3, Rdx)));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", Rdx, Rax)));
            instructions.push(insn(("movq", imm, Reg::Rdx)));
            instructions.push(insn(("movq", Rdx, (-(*offset as i64), Rax))));
        }
        None => {
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, Rax
            )));
            instructions.push(insn(("addq", Rdx, Rax)));
            instructions.push(insn(("movq", imm, Rdx)));
            instructions.push(insn(("movq", Rdx, (-0, Rax))));
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
    instructions.push(insn(("movq", index, Rdx)));
    match stack_lookup.get(&arrname) {
        Some((_, offset)) => {
            instructions.push(insn(("salq", 3, Rdx)));
            instructions.push(insn(("movq", Rbp, Rax)));
            instructions.push(insn(("subq", Rdx, Rax)));
            instructions.push(insn(("movq", imm, Rdx)));
            instructions.push(insn(("movq", Rdx, (-(*offset as i64), Rax))));
        }
        None => {
            instructions.push(insn(("salq", 3, Rdx)));
            instructions.push(Special(format!(
                "\tleaq global_var{}(%rip), {}",
                arrname, Rax
            )));
            instructions.push(insn(("addq", Rdx, Rax)));
            instructions.push(insn(("movq", imm, Rdx)));
            instructions.push(insn(("movq", Rdx, (-0, Rax))));
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
) -> Insn {
    match stack_lookup.get(&varname) {
        Some((_, offset)) => insn(("movq", imm, (-(*offset as i64), Rbp))),
        None => Special(format!("\tmovq ${}, global_var{}(%rip)", imm, varname)),
    }
}

fn asm_add_op(
    source1: ImmVar<Sum<Reg, MemVarLabel>>,
    source2: ImmVar<Sum<Reg, MemVarLabel>>,
    dest: Sum<Reg, MemVarLabel>,
    aop: AddOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                //var var
                instructions.push(load_into_reg_var(Rdx, s1, stack_lookup));
                instructions.push(load_into_reg_var(Rax, s2, stack_lookup));
                // instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                // instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));

                instructions.push(insn((convert_aop_to_asm(aop), Rax, Rdx)));

                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
                // instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                instructions.push(load_into_reg_var(Rdx, s1, stack_lookup));
                // instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                // instructions.push(load_into_reg(Reg::R9, s1, stack_lookup));
                instructions.push(insn((convert_aop_to_asm(aop), i2, Rdx)));
                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
                // instructions.push(format!("\tmovq {}, {}", Reg::R9, dest));
                //  instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(load_into_reg_var(Rdx, s2, stack_lookup));
                // instructions.push(format!("\tmovq {}, {}", s2, Reg::R9));
                // instructions.push(load_into_reg(Reg::R9, s2, stack_lookup));
                match aop {
                    AddOp::Add => {
                        instructions.push(insn((convert_aop_to_asm(aop), i1, Rdx)));
                    }
                    AddOp::Sub => {
                        instructions.push(insn(("neg", Rdx)));
                        instructions.push(insn((convert_aop_to_asm(AddOp::Add), i1, Rdx)));
                    }
                }
                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(_) => panic!("Three Op with two immediate sources can be simplified"),
        },
    }

    instructions
}

fn asm_mul_op(
    source1: ImmVar<Sum<Reg, MemVarLabel>>,
    source2: ImmVar<Sum<Reg, MemVarLabel>>,
    dest: Sum<Reg, MemVarLabel>,
    mop: MulOp,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => {
            match source2 {
                ImmVar::Var(s2) => {
                    // var-var
                    instructions.extend(convert_multipication_var_var(
                        mop,
                        s2,
                        s1,
                        dest,
                        stack_lookup,
                    ));
                }
                ImmVar::Imm(i2) => {
                    // var_imm
                    instructions.extend(convert_multipication_imm_var(
                        mop,
                        i2,
                        s1,
                        dest,
                        stack_lookup,
                    ));
                }
            }
        }
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // imm_var
                instructions.extend(convert_multipication_var_imm(
                    mop,
                    s2,
                    i1,
                    dest,
                    stack_lookup,
                ));
            }
            ImmVar::Imm(_) => {
                panic!("Three operand expression should be simplified");
            }
        },
    }
    instructions
}

fn asm_rel_op(
    source1: ImmVar<Sum<Reg, MemVarLabel>>,
    source2: ImmVar<Sum<Reg, MemVarLabel>>,
    dest: Sum<Reg, MemVarLabel>,
    rop: Bop,
    stack_lookup: &HashMap<VarLabel, (CfgType, u64)>,
) -> Vec<Insn> {
    let mut instructions = vec![];
    match source1 {
        ImmVar::Var(s1) => match source2 {
            ImmVar::Var(s2) => {
                instructions.push(load_into_reg_var(Rdx, s1, stack_lookup));
                instructions.push(load_into_reg_var(Rax, s2, stack_lookup));

                instructions.push(insn(("cmpq", Rax, Rdx)));
                instructions.push(insn(("movq", 0, Rdx)));

                instructions.push(insn(("movq", 1, Rax)));
                instructions.push(insn((convert_rel_op_to_cmov_type(rop), Rax, Rdx)));
                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
            ImmVar::Imm(i2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                // instructions.push(format!("\tmovq {}, {}", s1, Reg::R9));
                instructions.push(load_into_reg_var(Rdx, s1, stack_lookup));

                instructions.push(insn(("cmpq", i2, Rdx)));

                instructions.push(insn(("movq", 0, Rdx)));

                instructions.push(insn(("movq", 1, Rax)));
                instructions.push(insn((convert_rel_op_to_cmov_type(rop), Rax, Rdx)));
                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
                // instructions.push(store_from_reg(Reg::R9, dest, stack_lookup));
            }
        },
        ImmVar::Imm(i1) => match source2 {
            ImmVar::Var(s2) => {
                // cmp can only take immediates on the left argument so we must flip the relational operation
                instructions.push(load_into_reg_var(Rax, s2, stack_lookup));
                // instructions.push(load_into_reg(Reg::R10, s2, stack_lookup));

                instructions.push(insn(("cmpq", i1, Rax)));
                instructions.push(insn(("movq", 0, Reg::Rdx)));

                instructions.push(insn(("movq", 1, Rax)));
                instructions.push(insn((
                    convert_rel_op_to_cmov_type(flip_rel_op(rop)),
                    Rax,
                    Rdx,
                )));
                instructions.push(store_from_reg_var(Rdx, dest, stack_lookup));
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
    instr: Instruction<Sum<Reg, MemVarLabel>>,
    root: &String,
    num_params: u32,
    external_funcs: &Vec<String>,
    mac: bool,
    global_strings: &HashMap<String, usize>,
) -> Vec<Insn> {
    match instr {
        Instruction::PhiExpr { .. } | Instruction::ParMov(_) => panic!(),
        Instruction::Pop(var) => match var {
            Sum::Inl(reg) => vec![insn(("popq", reg))],
            Sum::Inr(_) => vec![],
        },
        Instruction::Push(var) => match var {
            Sum::Inl(reg) => vec![insn(("pushq", reg))],
            Sum::Inr(_) => vec![],
        },
        Instruction::Call(func_name, args, ret_dest) => {
            let mut instructions = vec![];
            assert!(args.len() <= 6);

            let argument_registers = vec![Rdi, Rsi, Rdx, Rcx, R8, R9];
            assert!(
                args == argument_registers[0..args.len()]
                    .into_iter()
                    .map(|reg| Arg::VarArg(ImmVar::Var(Sum::Inl(*reg))))
                    .collect::<Vec<_>>()
            );

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
                Some(dest) => instructions.push(store_from_reg_var(Rax, dest, stack_lookup)),
                None => {}
            }
            instructions
        }
        Instruction::StoreParam(param_num, arg) => {
            let mut instructions = vec![];
            let argument_registers = vec![Rdi, Rsi, Rdx, Rcx, R8, R9];

            match arg {
                Arg::VarArg(label) => {
                    let arg_reg: Option<Reg> = argument_registers.get(param_num as usize).copied();
                    match arg_reg {
                        Some(reg) => match label {
                            ImmVar::Var(v) => {
                                instructions.push(load_into_reg_var(reg, v.clone(), stack_lookup));
                                // instructions.push(format!("\tmovq {}, {}", v, reg));
                                // instructions.push(load_into_reg(reg, *v, stack_lookup))
                            }
                            ImmVar::Imm(i) => instructions.push(insn(("movq", i, reg))),
                        },
                        None => {
                            match label {
                                ImmVar::Var(v) => {
                                    instructions.push(load_into_reg_var(
                                        Rax,
                                        v.clone(),
                                        stack_lookup,
                                    ));
                                    // instructions.push(format!("\tmovq {}, {}", v, Rax));
                                    //instructions.push(load_into_reg(Rax, *v, stack_lookup))
                                }
                                ImmVar::Imm(i) => instructions.push(insn(("movq", i, Rax))),
                            }
                            instructions.push(insn(("pushq", Rax)));
                        }
                    }
                }
                Arg::StrArg(string) => {
                    let arg_reg = argument_registers.get(param_num as usize).copied();

                    match arg_reg {
                        Some(reg) => {
                            instructions.push(Special(format!(
                                "\tleaq global_str{}(%rip), {}",
                                global_strings.get(&string.to_string()).unwrap(),
                                reg
                            )));
                        }
                        None => {
                            instructions.push(Special(format!(
                                "\tmovq global_str{}, {}",
                                global_strings.get(&string.to_string()).unwrap(),
                                Rax,
                            )));
                            instructions.push(insn(("pushq", Rax)));
                        }
                    }
                }
            }

            instructions
        }
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
            instructions.push(load_into_reg_var(Reg::Rax, s, stack_lookup));
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

            // instructions.push(store_from_reg(Reg::Rax, dest, stack_lookup));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));

            return instructions;
        }
        Instruction::MoveOp { source, dest } => {
            // Maintain the invariant that the source is not a constant, when assembling
            let s = match source {
                ImmVar::Var(v) => v,
                ImmVar::Imm(_) => panic!(),
            };

            let mut instructions = vec![];
            instructions.push(load_into_reg_var(Rax, s, stack_lookup));
            instructions.push(store_from_reg_var(Rax, dest, stack_lookup));
            return instructions;
            // let get_source = load_into_reg(Reg::Rax, s, stack_lookup);
            // let return_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            // return vec![get_source, return_to_stack];
        }
        Instruction::Constant { dest, constant } => {
            return vec![
                insn(("movq", constant, Rax)),
                store_from_reg_var(Rax, dest, stack_lookup),
            ];

            // let load_const = format!("\tmovq ${}, {}", constant, Reg::Rax);
            // let restore_to_stack = store_from_reg(Reg::Rax, dest, stack_lookup);

            // return vec![load_const, restore_to_stack];
        }
        Instruction::ArrayAccess { dest, name, idx } => {
            let mut instructions = vec![];

            match idx {
                ImmVar::Var(v) => match dest {
                    Sum::Inl(reg) => {
                        instructions.extend(load_into_reg_arr(reg, name, v, stack_lookup))
                    }
                    Sum::Inr(_) => {
                        instructions.extend(load_into_reg_arr(Rax, name, v, stack_lookup));
                        instructions.push(store_from_reg_var(Rax, dest, stack_lookup))
                    }
                },
                ImmVar::Imm(i) => match dest {
                    Sum::Inl(reg) => {
                        instructions.extend(load_into_reg_arr_imm(reg, name, i, stack_lookup))
                    }
                    Sum::Inr(_) => {
                        instructions.extend(load_into_reg_arr_imm(Rax, name, i, stack_lookup));
                        instructions.push(store_from_reg_var(Rax, dest, stack_lookup))
                    }
                },
                //instructions.extend(load_into_reg_arr_imm(dest, name, i, stack_lookup));
            }

            // instructions.push(format!(""))
            instructions
        }
        Instruction::ArrayStore { source, arr, idx } => {
            let mut instructions = vec![];

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
                        instructions.push(load_into_reg_var(Reg::Rax, v.clone(), stack_lookup));
                        // instructions.push(format!("\tmovq {}, {}", v, Reg::Rax));
                        // instructions.push(load_into_reg(Reg::Rax, v, stack_lookup))
                    }
                    ImmVar::Imm(i) => instructions.push(insn(("movq", i, Rax))),
                },
                None => {}
            }
            instructions.push(Special(format!("\tjmp {}end", root)));
            instructions
        }
        Instruction::Spill { ord_var, mem_var } => {
            let ord_var = match ord_var {
                Sum::Inl(ord_var) => ord_var,
                Sum::Inr(_) => panic!("should be local"),
            };
            vec![store_from_reg(ord_var, mem_var, stack_lookup)]
        }
        Instruction::Reload { ord_var, mem_var } => {
            let ord_var = match ord_var {
                Sum::Inl(ord_var) => ord_var,
                Sum::Inr(_) => panic!("should be local"),
            };
            vec![load_into_reg(ord_var, mem_var, stack_lookup)]
        }
        Instruction::LoadParam { param, dest } => {
            // read parameters from registers and/or stack
            let argument_registers = vec![Rdi, Rsi, Rdx, Rcx, R8, R9];
            if param < 6 {
                let reg: Reg = *argument_registers.get(param as usize).unwrap();
                vec![store_from_reg_var(reg, dest, &stack_lookup)]
            } else {
                // read parameters off of the stack
                vec![
                    insn((
                        "movq",
                        (16 + 8 * (num_params as i64 - 1 - param as i64), Rbp),
                        Rax,
                    )),
                    store_from_reg_var(Rax, dest, &stack_lookup),
                ]
            }
        }
    }
}
