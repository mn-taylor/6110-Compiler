use crate::parse;
use crate::parse::Param;
use crate::parse::WithLoc;
use crate::scan;
use crate::scan::AssignOp;
use crate::scan::IncrOp;
use parse::Field;
use parse::Ident;
use parse::Literal;
use parse::Primitive;
use scan::ErrLoc;
use scan::{AddOp, EqOp, MulOp, RelOp};
use std::fmt;

#[derive(PartialEq, Clone, Debug)]
pub enum Bop {
    MulBop(MulOp),
    AddBop(AddOp),
    RelBop(RelOp),
    EqBop(EqOp),
    And,
    Or,
}

impl fmt::Display for Bop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Bop::MulBop(op) => write!(f, "{}", op),
            Bop::AddBop(op) => write!(f, "{}", op),
            Bop::RelBop(op) => write!(f, "{}", op),
            Bop::EqBop(op) => write!(f, "{}", op),
            Bop::And => write!(f, "&&"),
            Bop::Or => write!(f, "||"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Location {
    Var(Ident),
    ArrayIndex(Ident, WithLoc<Expr>),
}

#[derive(Clone, Debug)]
pub enum Arg {
    ExprArg(WithLoc<Expr>),
    ExternArg(WithLoc<String>),
}

impl Arg {
    pub fn loc(&self) -> scan::ErrLoc {
        match self {
            Arg::ExprArg(a) => a.loc,
            Arg::ExternArg(a) => a.loc,
        }
    }
}

pub enum Stmt {
    AssignStmt(WithLoc<Location>, AssignExpr),
    Call(WithLoc<Ident>, Vec<Arg>),
    If(WithLoc<Expr>, Block, Option<Block>),
    For {
        var_to_set: WithLoc<Ident>,
        initial_val: WithLoc<Expr>,
        test: WithLoc<Expr>,
        var_to_update: WithLoc<Location>,
        update_val: AssignExpr,
        body: Block,
    },
    While(WithLoc<Expr>, Block),
    Return(ErrLoc, Option<WithLoc<Expr>>),
    Break(ErrLoc),
    Continue(ErrLoc),
}

#[derive(Debug)]
pub enum AssignExpr {
    RegularAssign(WithLoc<AssignOp>, WithLoc<Expr>),
    IncrAssign(WithLoc<IncrOp>),
}

#[derive(PartialEq, Clone, Debug)]
pub enum UnOp {
    Neg,
    Not,
    IntCast,
    LongCast,
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Not => write!(f, "!"),
            UnOp::IntCast => write!(f, "(int)"),
            UnOp::LongCast => write!(f, "(long)"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Bin(Box<WithLoc<Expr>>, Bop, Box<WithLoc<Expr>>),
    Unary(UnOp, Box<WithLoc<Expr>>),
    Len(WithLoc<Ident>),
    Lit(WithLoc<Literal>),
    Loc(Box<WithLoc<Location>>),
    Call(WithLoc<Ident>, Vec<Arg>),
}

pub struct Program {
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
    pub imports: Vec<WithLoc<Ident>>,
}

#[derive(Debug, PartialEq)]
pub enum Type {
    Prim(Primitive),
    Arr(Primitive),
    Func(Vec<Primitive>, Option<Primitive>),
    ExtCall,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Prim(t) => write!(f, "{}", t),
            Arr(t) => write!(f, "{} array", t),
            Func(_, _) => write!(f, "method"),
            ExtCall => write!(f, "external function"),
        }
    }
}

pub struct Block {
    pub fields: Vec<Field>,
    pub stmts: Vec<Stmt>,
}

pub struct Method {
    pub meth_type: Option<Primitive>,
    pub fields: Vec<Field>,
    pub stmts: Vec<Stmt>,
    pub params: Vec<Param>,
    pub name: WithLoc<Ident>,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "method {}", self.name.val)
    }
}

// scopes
use Type::*;

use std::collections::HashMap;
pub struct Scope<'a, T> {
    head: HashMap<&'a String, T>,
    tail: Option<&'a Scope<'a, T>>,
}

impl<'a, T> Scope<'a, T> {
    pub fn local_lookup(&self, id: &Ident) -> Option<&T> {
        self.head.get(&id.name)
    }

    pub fn lookup(&self, id: &Ident) -> Option<&T> {
        match self.head.get(&id.name) {
            Some(t) => Some(t),
            None => match self.tail {
                Some(s) => s.lookup(id),
                None => None,
            },
        }
    }

    pub fn new(
        local_scope: HashMap<&'a String, T>,
        parent_scope: Option<&'a Scope<'a, T>>,
    ) -> Scope<'a, T> {
        Scope {
            head: local_scope,
            tail: parent_scope,
        }
    }
}

pub trait Scoped {
    fn scope<'a>(&'a self, parent: &'a Scope<'a, Type>) -> Scope<'a, Type> {
        Scope::new(self.local_scope(), Some(parent))
    }
    fn local_scope<'a>(&'a self) -> HashMap<&'a String, Type>;
}

impl Scoped for Block {
    fn local_scope<'a>(&'a self) -> HashMap<&'a String, Type> {
        fields_scope(&self.fields).collect()
    }
}

impl Scoped for Method {
    fn local_scope<'a>(&'a self) -> HashMap<&'a String, Type> {
        params_scope(&self.params)
            .chain(fields_scope(&self.fields))
            .collect()
    }
}

fn fields_scope<'a>(fields: &'a [Field]) -> impl Iterator<Item = (&'a String, Type)> {
    fields.into_iter().map(|f| match f {
        Field::Scalar(t, id) => (&id.val.name, Prim(t.clone())),
        Field::Array(t, id, _) => (&id.val.name, Arr(t.clone())),
    })
}

fn methods_scope(methods: &[Method]) -> impl Iterator<Item = (&String, Type)> {
    methods.into_iter().map(|m| {
        (
            &m.name.val.name,
            Func(
                m.params.iter().map(|m| m.param_type.clone()).collect(),
                m.meth_type.clone(),
            ),
        )
    })
}

fn params_scope(params: &[Param]) -> impl Iterator<Item = (&String, Type)> {
    params
        .iter()
        .map(|p| (&p.name.val.name, Prim(p.param_type.clone())))
}

fn imports_scope(imports: &[WithLoc<Ident>]) -> impl Iterator<Item = (&String, Type)> {
    imports.iter().map(|id| (&id.val.name, ExtCall))
}

impl Program {
    pub fn scope_with_first_n_methods<'a>(
        self: &'a Program,
        num_methods: usize,
    ) -> Scope<'a, Type> {
        Scope::new(
            imports_scope(&self.imports)
                .chain(fields_scope(&self.fields))
                .chain(methods_scope(&self.methods[0..num_methods]))
                .collect(),
            None,
        )
    }
}
