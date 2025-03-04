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
use scan::{AddOp, EqOp, MulOp, RelOp};

#[derive(PartialEq)]
pub enum Bop {
    MulBop(MulOp),
    AddBop(AddOp),
    RelBop(RelOp),
    EqBop(EqOp),
    And,
    Or,
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
    If(Expr, Block, Option<Block>),
    For {
        var_to_set: WithLoc<Ident>,
        initial_val: Expr,
        test: Expr,
        var_to_update: Location,
        update_val: AssignExpr,
        body: Block,
    },
    While(Expr, Block),
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
    Call(WithLoc<Ident>, Vec<Arg>),
}

pub struct Program {
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
    pub imports: Vec<Ident>,
}

#[derive(Debug, PartialEq)]
pub enum Type {
    Prim(Primitive),
    Arr(Primitive),
    Func(Vec<Primitive>, Option<Primitive>),
    ExtCall,
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
    pub name: Ident,
}

// scopes
use Type::*;

use std::collections::HashMap;
pub struct Scope<'a> {
    head: HashMap<&'a String, Type>,
    tail: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    pub fn local_lookup(&self, id: &Ident) -> Option<&Type> {
        self.head.get(&id.name)
    }

    pub fn lookup(&self, id: &Ident) -> Option<&Type> {
        match self.head.get(&id.name) {
            Some(t) => Some(t),
            None => match self.tail {
                Some(s) => s.lookup(id),
                None => None,
            },
        }
    }

    fn new(
        local_scope: HashMap<&'a String, Type>,
        parent_scope: Option<&'a Scope<'a>>,
    ) -> Scope<'a> {
        Scope {
            head: local_scope,
            tail: parent_scope,
        }
    }
}

pub trait Scoped {
    fn scope<'a>(&'a self, parent: &'a Scope<'a>) -> Scope<'a> {
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
        Field::Scalar(t, id) => (&id.name, Prim(t.clone())),
        Field::Array(t, id, _) => (&id.name, Arr(t.clone())),
    })
}

fn methods_scope(methods: &[Method]) -> impl Iterator<Item = (&String, Type)> {
    methods.into_iter().map(|m| {
        (
            &m.name.name,
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
        .map(|p| (&p.name.name, Prim(p.param_type.clone())))
}

fn imports_scope(imports: &[Ident]) -> impl Iterator<Item = (&String, Type)> {
    imports.iter().map(|id| (&id.name, ExtCall))
}

impl Program {
    pub fn scope_with_first_n_methods<'a>(self: &'a Program, num_methods: usize) -> Scope<'a> {
        Scope::new(
            imports_scope(&self.imports)
                .chain(fields_scope(&self.fields))
                .chain(methods_scope(&self.methods[0..num_methods]))
                .collect(),
            None,
        )
    }
}
