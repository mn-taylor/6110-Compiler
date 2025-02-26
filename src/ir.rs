use crate::parse;
use crate::scan;
use parse::Field;
use parse::Ident;
use parse::Param;
use parse::Type;
use parse::WithLoc;
use scan::Sum;
use scan::{AddOp, EqOp, MulOp, RelOp};

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
    If(Expr, Block, Scope, Option<(Block, Scope)>),
    For(Ident, Expr, Expr, Location, Ident, Expr, Block, Scope),
    While(Expr, Block, Scope),
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

struct Program {
    fields: Vec<Field>,
    methods: Vec<Method>,
    imports: Vec<Ident>,
}

// fn scope_lookup(p: Program, id: Ident) {

// }

struct Scope {
    vars: Vec<(Ident, Type)>,
    parent: Option<Scope>,
}

struct Method {
    body: Block,
    params: Vec<Param>,
    scope: Scope,
}

type Block = Vec<Stmt>;

// struct Block {
//     vars: Vec<Field>,
//     parent: Option<Box<Block>>,
//     stmts: Vec<Stmt>,
// }
