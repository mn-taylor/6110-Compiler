use crate::parse;
use crate::parse::Param;
use crate::parse::WithLoc;
use crate::scan;
use crate::scan::AssignOp;
use crate::scan::IncrOp;
use parse::Ident;
use parse::Primitive;
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
    Arr(Primitive, i32),
    Func(Vec<Primitive>, Option<Primitive>),
    ExtCall,
}

pub trait Scoped {
    fn local_lookup(&self, id: &Ident) -> Option<Type>;
    fn scope(&self, parent: impl Fn(&Ident) -> Option<Type>) -> impl Fn(&Ident) -> Option<Type> {
        move |id| match Self::local_lookup(self, id) {
            Some(t) => Some(t),
            None => parent(id),
        }
    }
}

pub enum Field {
    Scalar(Primitive, Ident),
    Array(Primitive, Ident, i32),
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

impl Scoped for Block {
    fn local_lookup(&self, id: &Ident) -> Option<Type> {
        fields_lookup(&self.fields, id)
    }
}

impl Scoped for Method {
    fn local_lookup(&self, id: &Ident) -> Option<Type> {
        if let Some(t) = fields_lookup(&self.fields, id) {
            Some(t)
        } else {
            self.params
                .iter()
                .find(|other| other.name == *id)
                .map(|t| Prim(t.param_type.clone()))
        }
    }
}

fn fields_lookup(fields: &[Field], id: &Ident) -> Option<Type> {
    fields
        .iter()
        .find(|other| match other {
            Field::Scalar(_, name) => name == id,
            Field::Array(_, name, _) => name == id,
        })
        .map(|f| match f {
            Field::Scalar(t, _) => Prim(t.clone()),
            Field::Array(t, _, len) => Arr(t.clone(), *len),
        })
}

fn methods_lookup(methods: &[Method], id: &Ident) -> Option<Type> {
    methods.iter().find(|other| other.name == *id).map(|m| {
        Func(
            m.params.iter().map(|m| m.param_type.clone()).collect(),
            m.meth_type.clone(),
        )
    })
}

impl Program {
    pub fn local_scope_with_first_n_methods<'a>(
        self: &'a Program,
        num_methods: usize,
    ) -> impl 'a + Fn(&Ident) -> Option<Type> {
        move |id| {
            if let Some(t) = fields_lookup(&self.fields, id) {
                Some(t)
            } else if let Some(t) = methods_lookup(&self.methods[0..num_methods], id) {
                Some(t)
            } else {
                self.imports
                    .iter()
                    .find(|other| *other == id)
                    .map(|_| ExtCall)
            }
        }
    }
}
