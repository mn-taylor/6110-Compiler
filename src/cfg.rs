use crate::ir;
use crate::parse;
use ir::{Bop, UnOp};
use parse::Primitive;

pub type BlockLabel = usize;
pub type VarLabel = u32;

// #[derive(Debug, Clone)]
pub struct BasicBlock {
    // parents: Vec<BasicBlock>,
    pub body: Vec<Instruction>,
    pub jump_loc: Jump,
}

pub enum Jump {
    Uncond(BlockLabel),
    Cond {
        source: Var,
        true_block: BlockLabel,
        false_block: BlockLabel,
    },
    Nowhere,
}
pub enum Type {
    Prim(Primitive),
    Arr(Primitive, i32),
    Func(Vec<Primitive>, Option<Primitive>),
    ExtCall,
}
pub enum Instruction {
    ThreeOp {
        source1: Var,
        source2: Var,
        dest: Var,
        op: Bop,
    },
    TwoOp {
        source1: Var,
        dest: Var,
        op: UnOp,
    },
    MoveOp {
        source: Var,
        dest: Var,
    },
    Constant {
        dest: Var,
        constant: i64,
    },
    Ret(Option<Var>),
    Call(String, Vec<Arg>, Option<Var>),
}

pub enum Arg {
    VarArg(Var),
    ArrArg(String),
    StrArg(String),
}

#[derive(Debug, Clone)]
pub enum Var {
    Scalar {
        id: u32,
        name: String,
        typ: Primitive,
    },
    ArrIdx {
        id: u32,
        name: String,
        idx: u32, // should change here
        typ: Primitive,
    },
}

impl Var {
    pub fn get_typ(&self) -> Primitive {
        match self {
            Self::Scalar {
                id: _,
                name: _,
                typ: t,
            } => t.clone(),
            Self::ArrIdx {
                id: _,
                name: _,
                idx: _,
                typ: t,
            } => t.clone(),
        }
    }
    pub fn get_id(&self) -> u32 {
        match self {
            Self::Scalar {
                id,
                name: _,
                typ: _,
            } => *id,
            Self::ArrIdx {
                id,
                name: _,
                idx: _,
                typ: _,
            } => *id,
        }
    }
}
