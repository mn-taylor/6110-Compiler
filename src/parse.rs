use crate::scan::Keyword;
use crate::scan::Symbol;
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

enum Field {
    Scalar(Type, Ident),
    Array(Type, Ident, Int),
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

enum AssignExpr {
    RegularAssign(AssignOp, Expr),
    Increment,
    Decrement,
}

enum Arg {
    ExprArg(Expr),
    ExternArg(String),
}

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
    fields: Vec<Field>,
    stmts: Vec<Stmt>,
}

struct Method {
    meth_type: Option<Type>,
    name: Ident,
    params: Vec<Param>,
    body: Block,
}

struct Program {
    imports: Vec<Ident>,
    fields: Vec<Field>,
    methods: Vec<Method>,
}

fn call_til_none<U, T: FnMut() -> Option<U>>(mut /*why*/ f: T) -> Vec<U> {
    let mut ret = Vec::new();
    while let Some(u) = f() {
        ret.push(u);
    }
    ret
}

fn parse_import<T: Clone + Iterator<Item = Token>>(tokens: &mut T) -> Option<Ident> {
    let tokens_clone = tokens.clone();
    if let (
        Some(Token::Key(Keyword::Import)),
        Some(Token::Ident(name)),
        Some(Token::Sym(Symbol::Semicolon)),
    ) = (tokens.next(), tokens.next(), tokens.next())
    {
        Some(Ident { name })
    } else {
        *tokens = tokens_clone;
        None
    }
}

fn parse_field<T: Iterator<Item = Token>>(tokens: &mut Peekable<T>) -> Option<FieldDecl> {
    let tokens_clone = tokens.clone();
    if let Some(Token::
}

fn parse_method<T: Iterator<Item = Token>>(tokens: &mut Peekable<T>) -> Option<MethodDecl> {
    panic!()
}

fn parse_program<T: Clone + Iterator<Item = Token>>(tokens: &mut T) -> Program {
    Program {
        imports: call_til_none(|| parse_import(tokens)),
        fields: call_til_none(|| parse_field(tokens)),
        methods: call_til_none(|| parse_method(tokens)),
    }
}
