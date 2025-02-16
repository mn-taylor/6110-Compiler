use crate::scan::Token;
use std::iter::Peekable;

struct Ident {
    name: String,
}

enum Type {
    IntType,
    LongType,
    BoolType,
}

struct Int {
    val: String,
}

enum FieldDecl {
    ScalarDecl(Type, Ident),
    ArrayDecl(Type, Ident, Int),
}

struct Param {
    param_type: Type,
    name: Ident,
}

// allowing negation here is redundant, since negation appears in expr
enum Literal {
    DecInt(String),
    HexInt(String),
    DecLong(String),
    HexLong(String),
    Char(char),
    Bool(bool),
}

enum ArithOp {
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
}

enum RelOp {
    Lt,
    Gt,
    Le,
    Ge,
}

enum BoolOp {
    Eq,
    Neq,
    And,
    Or,
}

enum BinOp {
    Arith(ArithOp),
    Rel(RelOp),
    Bool(BoolOp),
}

enum Expr {
    Loc(Box<Location>),
    Call(Ident, Vec<Arg>),
    Lit(Literal),
    IntCast(Box<Expr>),
    LongCast(Box<Expr>),
    Len(Ident),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
}

enum Location {
    Var(Ident),
    ArrayIndex(Ident, Expr),
}

enum AssignOp {
    Eq,
    PlusEq,
    MinusEq,
    MulEq,
    DivEq,
    ModEq,
}

enum AssignExpr {
    RegularAssign(AssignOp, Expr),
    Increment,
    Decrement,
}

enum Arg {
    ExprArg(Expr),
    ExternArg(String),
}

// not done
enum Stmt {
    Assignment(Location, AssignExpr),
    Call(Ident, Vec<Arg>),
    If(Expr, Block, Block),
    For(Ident, Expr, Expr, Location, AssignExpr, Block),
    While(Expr, Block),
    Return(Option<Expr>),
    Break,
    Continue,
}

struct Block {
    fields: Vec<FieldDecl>,
    stmts: Vec<Stmt>,
}

struct MethodDecl {
    meth_type: Option<Type>,
    name: Ident,
    params: Vec<Param>,
    body: Block,
}

struct Program {
    imports: Vec<Ident>,
    fields: Vec<FieldDecl>,
    methods: Vec<MethodDecl>,
}

fn parse_program<T: Iterator<Item = Token>>(tokens: Peekable<T>) -> Program {}
