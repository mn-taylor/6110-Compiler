use crate::ir;
use crate::parse;
use ir::{Bop, UnOp};
use parse::Primitive;

pub struct BasicBlock<'a> {
    // parents: Vec<BasicBlock>,
    pub body: Vec<Instruction>,
    pub jump_loc: Jump<'a>,
}

pub enum Jump<'a> {
    Uncond(&'a BasicBlock<'a>),
    Cond {
        source: Var,
        true_block: &'a BasicBlock<'a>,
        false_block: &'a BasicBlock<'a>,
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

pub enum Var {
    Scalar {
        id: u32,
        name: String,
        typ: Primitive,
    },
    ArrIdx {
        id: u32,
        name: String,
        idx: Var,
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
}
