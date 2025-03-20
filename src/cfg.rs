use crate::ir;
use crate::parse;
use ir::{Bop, UnOp};
use parse::Primitive;
use std::fmt;

pub type BlockLabel = usize;
pub type VarLabel = u32;

// #[derive(Debug, Clone)]
pub struct BasicBlock {
    // parents: Vec<BasicBlock>,
    pub block_id: BlockLabel,
    pub body: Vec<Instruction>,
    pub jump_loc: Jump,
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "------------------------- \n");
        write!(f, "BasicBlock {} \n", &self.block_id);
        for instr in &self.body {
            writeln!(f, "| {} |", instr)?;
        }
        writeln!(f, "| {} |", self.jump_loc);
        writeln!(f, "-------------------------")
    }
}

pub enum Jump {
    Uncond(BlockLabel),
    Cond {
        source: VarLabel,
        true_block: BlockLabel,
        false_block: BlockLabel,
    },
    Nowhere,
}
impl fmt::Display for Jump {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Jump::Uncond(block) => write!(f, "goto {}", block),
            Jump::Cond {
                source,
                true_block,
                false_block,
            } => write!(
                f,
                "if {} then goto {} else goto {}",
                source, true_block, false_block
            ),
            Jump::Nowhere => write!(f, "nowhere"),
        }
    }
}

pub enum Instruction {
    ThreeOp {
        source1: VarLabel,
        source2: VarLabel,
        dest: VarLabel,
        op: Bop,
    },
    TwoOp {
        source1: VarLabel,
        dest: VarLabel,
        op: UnOp,
    },
    MoveOp {
        source: VarLabel,
        dest: VarLabel,
    },
    Constant {
        dest: VarLabel,
        constant: i64,
    },
    ArrayAccess {
        dest: VarLabel,
        name: VarLabel,
        idx: VarLabel,
    },
    ArrayStore {
        source: VarLabel,
        arr: VarLabel,
        idx: VarLabel,
    },
    Ret(Option<VarLabel>),
    Call(String, Vec<Arg>, Option<VarLabel>),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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
                "t{} <- call {}({})",
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
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

pub enum Arg {
    VarArg(VarLabel),
    StrArg(String),
}

impl fmt::Display for Arg {
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

#[derive(Debug, Clone)]
pub enum Var {
    Scalar { id: VarLabel },
    ArrIdx { arrname: VarLabel, idx: VarLabel },
}
