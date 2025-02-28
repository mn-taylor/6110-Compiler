use crate::parse;
use crate::scan;
use parse::Field;
use parse::Ident;
use parse::Param;
use parse::Type;
use parse::WithLoc;
use scan::Sum;
use scan::{AddOp, EqOp, MulOp, RelOp};
use std::rc::Rc;
use crate::scan::IncrOp;
use crate::scan::AssignOp;

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

pub enum Arg {
    ExprArg(Expr),
    ExternArg(String),
}

pub enum Stmt {
    AssignStmt(Location, AssignExpr),
    Call(WithLoc<Ident>, Vec<Arg>),
    // SelfAssign(Ident, Bop, Expr),
    // will represent ++, -- as SelfAssign
    If(Expr, Block, Rc<Scope>, Option<(Block, Rc<Scope>)>),
    For {
        var_to_set: WithLoc<Ident>,
        initial_val: Expr,
        test: Expr,
        var_to_update: Location,
        update_val: AssignExpr,
        body: Block,
        scope: Rc<Scope>,
    },
    While(Expr, Block, Rc<Scope>),
    Return(Option<Expr>),
    Break,
    Continue,
}

pub enum AssignExpr {
    RegularAssign(AssignOp, Expr),
    IncrAssign(IncrOp),
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
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
    pub imports: Vec<Ident>,
}

pub struct Scope {
    pub vars: Vec<Field>,
    pub parent: Option<Rc<Scope>>,
}

pub struct Method {
    pub body: Block,
    pub params: Vec<Param>,
    pub scope: Rc<Scope>,
}

pub type Block = Vec<Stmt>;
