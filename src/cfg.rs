use crate::ir;
use crate::parse;
use crate::scan::Sum;
use crate::ssa_construct::SSAVarLabel;
use ir::{Bop, UnOp};
use parse::Primitive;
use std::collections::HashMap;
use std::fmt;

pub type BlockLabel = usize;

#[derive(Clone)]
pub struct CfgMethod<VarLabel> {
    pub name: String,
    pub params: Vec<u32>, // was var label but that was unnecessary
    pub blocks: HashMap<BlockLabel, BasicBlock<VarLabel>>,
    pub fields: HashMap<u32, (CfgType, String)>, // ws var label but that was unnecessary
    pub return_type: Option<Primitive>,
}

impl<VarLabel: fmt::Debug + fmt::Display> fmt::Display for CfgMethod<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Method: {}", self.name)?;
        // writeln!(f, "Fields: {:?}", self.fields)?;
        for block in self.blocks.values() {
            writeln!(f, "{}", block)?;
        }
        Ok(())
    }
}

pub struct CfgProgram<VarLabel> {
    pub externals: Vec<String>,
    pub methods: Vec<CfgMethod<VarLabel>>,
    pub global_fields: HashMap<VarLabel, (CfgType, String)>,
}

impl<VarLabel: fmt::Display + fmt::Debug> fmt::Display for CfgProgram<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Global fields:")?;
        for (lbl, (t, name)) in self.global_fields.iter() {
            writeln!(f, "{:?} {} (high-level  name {})", t, lbl, name)?;
        }
        writeln!(f, "Methods:")?;
        for method in self.methods.iter() {
            writeln!(f, "Method {}", method.name)?;
            write!(f, "{}", method)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct BasicBlock<VarLabel> {
    pub parents: Vec<usize>,
    pub block_id: BlockLabel,
    pub body: Vec<Instruction<VarLabel>>,
    pub jump_loc: Jump<VarLabel>,
}

impl<VarLabel: fmt::Debug + fmt::Display> fmt::Display for BasicBlock<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "------------------------- \n")?;
        write!(f, "BasicBlock {} \n", &self.block_id)?;
        writeln!(f, "  Parents: {:?}", self.parents)?;
        for instr in &self.body {
            writeln!(f, "| {} |", instr)?;
        }
        writeln!(f, "| {} |", self.jump_loc)?;
        writeln!(f, "-------------------------")
    }
}

#[derive(Clone)]
pub enum Jump<VarLabel> {
    Uncond(BlockLabel),
    Cond {
        source: ImmVar<VarLabel>,
        true_block: BlockLabel,
        false_block: BlockLabel,
    },
    Nowhere,
}

impl<VarLabel: fmt::Display> fmt::Display for Jump<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Jump::Uncond(block) => write!(f, "goto {}", block),
            Jump::Cond {
                source,
                true_block,
                false_block,
            } => write!(
                f,
                "if t{} then goto {} else goto {}",
                source, true_block, false_block
            ),
            Jump::Nowhere => write!(f, "nowhere"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OneMove<VarLabel> {
    pub src: VarLabel,
    pub dest: VarLabel,
}

#[derive(Clone, Debug)]
pub enum Instruction<VarLabel> {
    Spill {
        ord_var: VarLabel,
        mem_var: MemVarLabel,
    },
    Reload {
        ord_var: VarLabel,
        mem_var: MemVarLabel,
    },
    PhiExpr {
        dest: VarLabel,
        sources: Vec<(BlockLabel, Sum<VarLabel, MemVarLabel>)>,
    },
    MemPhiExpr {
        dest: MemVarLabel,
        sources: Vec<(BlockLabel, MemVarLabel)>,
    },
    ThreeOp {
        source1: ImmVar<VarLabel>,
        source2: ImmVar<VarLabel>,
        dest: VarLabel,
        op: Bop,
    },
    TwoOp {
        source1: ImmVar<VarLabel>,
        dest: VarLabel,
        op: UnOp,
    },
    MoveOp {
        source: ImmVar<VarLabel>,
        dest: VarLabel,
    },
    ParMov(Vec<OneMove<VarLabel>>),
    Constant {
        dest: VarLabel,
        constant: i64,
    },
    ArrayAccess {
        dest: VarLabel,
        name: VarLabel,
        idx: ImmVar<VarLabel>,
    },
    ArrayStore {
        source: ImmVar<VarLabel>,
        arr: VarLabel,
        idx: ImmVar<VarLabel>,
    },
    Ret(Option<ImmVar<VarLabel>>),
    Call(String, Vec<Arg<VarLabel>>, Option<VarLabel>),
}

impl<VarLabel: fmt::Display + fmt::Debug> fmt::Display for Instruction<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::PhiExpr { dest, sources } => {
                let sources_str = sources
                    .iter()
                    .map(|(block, var)| format!("({},  t{:?})", block, var))
                    .collect::<Vec<_>>()
                    .join(", ");

                // Final print: "dest = phi [block -> var, block -> var, ...]"
                write!(f, "t{} = phi ({})", dest, sources_str)
            }
            Instruction::ThreeOp {
                source1,
                source2,
                dest,
                op,
            } => write!(f, "t{} = t{} {} t{}", dest, source1, op, source2),
            Instruction::TwoOp { source1, dest, op } => {
                write!(f, "t{} <- {} t{}", dest, op, source1)
            }
            Instruction::MoveOp { source, dest } => write!(f, "t{} <- t{}", dest, source),
            Instruction::Constant { dest, constant } => write!(f, "t{} <- ${}", dest, constant),
            Instruction::ArrayAccess { dest, name, idx } => {
                write!(f, "t{} <- t{}[t{}]", dest, name, idx)
            }
            Instruction::ArrayStore { source, arr, idx } => {
                write!(f, "t{}[t{}] <- t{}", arr, idx, source)
            }
            Instruction::Ret(Some(var)) => write!(f, "return t{}", var),
            Instruction::Ret(None) => write!(f, "return"),
            Instruction::Call(name, args, Some(dest)) => write!(
                f,
                "t{} <- call {}(t{})",
                dest,
                name,
                args.iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Instruction::Call(name, args, None) => write!(
                f,
                "call {}({})",
                name,
                args.iter()
                    .map(|arg| format!("t{}", arg))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Instruction::ParMov(moves) => {
                for mov in moves {
                    write!(f, "t{} <- t{} || ", mov.dest, mov.src)?;
                }
                Ok(())
            }
            Instruction::Spill { ord_var, mem_var } => {
                write!(f, "Spill t{ord_var} into t{:?}", mem_var);
                Ok(())
            }
            Instruction::Reload { ord_var, mem_var } => {
                write!(f, "Reload t{:?} into t{ord_var}", mem_var);
                Ok(())
            }
            Instruction::MemPhiExpr { dest, sources } => {
                let sources_str = sources
                    .iter()
                    .map(|(block, var)| format!("({},  t{:?})", block, var))
                    .collect::<Vec<_>>()
                    .join(", ");

                // Final print: "dest = phi [block -> var, block -> var, ...]"
                write!(f, "t{:?} = phi ({})", dest, sources_str)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Arg<VarLabel> {
    VarArg(ImmVar<VarLabel>),
    StrArg(String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ImmVar<VarLabel> {
    Var(VarLabel),
    Imm(i64),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MemVarLabel {
    id: u32,
}

pub trait GetNameVer {
    fn get_name_ver(&self) -> Option<(u32, u32)>;
}

impl GetNameVer for ImmVar<SSAVarLabel> {
    fn get_name_ver(&self) -> Option<(u32, u32)> {
        match self {
            ImmVar::Var(v) => Some((v.name, v.version)),
            _ => panic!(),
        }
    }
}

pub trait IsImmediate {
    fn is_immediate(&self) -> bool;
}

impl<VarLabel> IsImmediate for ImmVar<VarLabel> {
    fn is_immediate(&self) -> bool {
        matches!(self, ImmVar::Imm(_))
    }
}

impl<VarLabel: fmt::Display> fmt::Display for ImmVar<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImmVar::Var(v) => write!(f, "{}", v),
            ImmVar::Imm(i) => write!(f, "{}", i),
        }
    }
}

impl<VarLabel: fmt::Display> fmt::Display for Arg<VarLabel> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Arg::VarArg(var) => write!(f, "{}", var),
            Arg::StrArg(s) => write!(f, "\"{}\"", s),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Type {
    Prim(Primitive),
    Arr(Primitive, i32),
    Func(Vec<Primitive>, Option<Primitive>),
    ExtCall,
}

#[derive(Debug, Clone)]
pub enum CfgType {
    Scalar(Primitive),
    Array(Primitive, i32),
}
