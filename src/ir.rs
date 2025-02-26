use crate::parse;
use crate::scan;
use parse::Ident;
use parse::Type;
use parse::WithLoc;
use parse::{AddOp, EqOp, MulOp, RelOp};
use scan::Sum;

enum Bop {
    MulBop(MulOp),
    AddBop(AddOp),
    RelBop(RelOp),
    EqBop(EqOp),
    And,
    Or,
}

enum Literal {
    IntLit(i32),
    LongLit(i64),
    CharLit(char),
    BoolLit(bool),
}

enum Location {
    Var(WithLoc<Ident>),
    ArrayIndex(WithLoc<Ident>, Expr),
}

enum Stmt {
    AssignStmt(Ident, Expr),
    SelfAssign(Ident, Bop, Expr),
    // will represent ++, -- as SelfAssign
    If(Expr, Block, Option<Block>),
    For(Ident, Expr, Expr, Location, Ident, Expr, Block),
    While(Expr, Block),
    Return(Option<Expr>),
    Break,
    Continue,
}

enum Expr {
    Bin(Expr, Bop, Expr),
    Unary(UnaryOp, Expr),
    Len(Ident),
    IntCast(Box<Expr>),
    LongCast(Box<Expr>),
    Loc(Box<Location>),
    Call(Ident, Vec<Arg>),
}

type ExprWithType = (Expr, Type);

struct GlobalScope {
    vars: Vec<(Ident, Type)>,
    parent: Sum<Box<LocalScope>, Box<GlobalScope>>,
    methods: Vec<Method>,
    exts: Vec<Ident>,
}

struct Method {
    body: Block,
    params: Vec<(Ident, Expr)>,
}

struct Block {
    vars: Vec<(String, Type)>,
    parent: Sum<Box<Block>, Box<GlobalScope>>,
    stmts: Vec<Stmt>,
}
