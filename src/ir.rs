use crate::parse;
use crate::scan;
use parse::Field;
use parse::Ident;
use parse::Param;
use parse::Type;
use parse::WithLoc;
use scan::Sum;
use scan::{AddOp, EqOp, MulOp, RelOp};

pub enum Bop {
    MulBop(MulOp),
    AddBop(AddOp),
    RelBop(RelOp),
    EqBop(EqOp),
    And,
    Or,
}

pub enum Literal {
    IntLit(i32),
    LongLit(i64),
    CharLit(char),
    BoolLit(bool),
}

pub enum Location {
    Var(WithLoc<Ident>),
    ArrayIndex(WithLoc<Ident>, Expr),
}

pub enum Stmt {
    AssignStmt(Ident, Expr),
    SelfAssign(Ident, Bop, Expr),
    // will represent ++, -- as SelfAssign
    If(Expr, Block, Scope, Option<(Block, Scope)>),
    For(Ident, Expr, Expr, Location, Ident, Expr, Block, Scope),
    While(Expr, Block, Scope),
    Return(Option<Expr>),
    Break,
    Continue,
}

pub enum UnOp {
    Neg,
    Not,
    IntCast,
    LongCast,
}

pub enum Expr {
    Bin(Box<Expr>, Bop, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Len(WithLoc<Ident>),
    Lit(WithLoc<Literal>),
    Loc(Box<Location>),
    Call(WithLoc<Ident>, Vec<parse::Arg>),
}

pub struct Program {
    fields: Vec<Field>,
    methods: Vec<Method>,
    imports: Vec<Ident>,
}

pub struct Scope {
    vars: Vec<(Ident, Type)>,
    parent: Option<Scope>,
}

pub struct Method {
    body: Block,
    params: Vec<Param>,
    scope: Scope,
}

type Block = Vec<Stmt>;
