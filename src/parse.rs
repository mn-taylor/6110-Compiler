use crate::scan::*;

// is this really not in the stdlib?
#[derive(Debug, PartialEq)]
pub enum Sum<A, B> {
    Inl(A),
    Inr(B),
}

use Sum::*;

#[derive(Debug, PartialEq)]
pub struct Ident {
    pub name: String,
}

// not sure about best way to handle current cloning of Type
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    IntType,
    LongType,
    BoolType,
}

#[derive(Debug, PartialEq)]
pub enum Field {
    Scalar(Type, Ident),
    Array(Type, Ident, Literal),
}

#[derive(Debug, PartialEq)]
pub struct Param {
    pub param_type: Type,
    pub name: Ident,
}

// allowing negation here would be redundant, since negation appears in expr
#[derive(Debug, PartialEq)]
pub enum Literal {
    DecInt(String),
    HexInt(String),
    DecLong(String),
    HexLong(String),
    Char(char),
    Bool(bool),
}

#[derive(Debug, PartialEq)]
pub enum AtomicExpr {
    Loc(Box<Location>),
    Call(Ident, Vec<Arg>),
    Lit(Literal),
    IntCast(Box<OrExpr>),
    LongCast(Box<OrExpr>),
    Len(Ident),
    Neg(Box<AtomicExpr>),
    Not(Box<AtomicExpr>),
    Ex(Box<OrExpr>),
}

use crate::scan::AddOp::*;
use crate::scan::Symbol::*;

fn parse_atomic_expr<'a, T: Clone + Iterator<Item = &'a Token>>(
    tokens: &mut T,
) -> Option<AtomicExpr> {
    // order matters here.  one that works is
    // Neg, Not, Ex, IntCast, LongCast, Len, Lit, Call, Loc
    let parse_neg = parse_concat(parse_one(exactly(Sym(AddSym(Sub)))), parse_atomic_expr);
    panic!()
}

#[derive(Debug, PartialEq)]
pub enum MulExpr {
    Atomic(AtomicExpr),
    Bin(AtomicExpr, MulOp, Box<MulExpr>),
}

#[derive(Debug, PartialEq)]
pub enum AddExpr {
    Mul(MulExpr),
    Bin(MulExpr, AddOp, Box<AddExpr>),
}

#[derive(Debug, PartialEq)]
pub enum RelExpr {
    Add(AddExpr),
    Bin(MulExpr, RelOp, Box<RelExpr>),
}

#[derive(Debug, PartialEq)]
pub enum EqExpr {
    Rel(RelExpr),
    Bin(RelExpr, EqOp, Box<EqExpr>),
}

#[derive(Debug, PartialEq)]
pub enum AndExpr {
    Eq(EqExpr),
    Bin(EqExpr, Box<AndExpr>),
}

#[derive(Debug, PartialEq)]
pub enum OrExpr {
    And(AndExpr),
    Bin(AndExpr, Box<OrExpr>),
}

#[derive(Debug, PartialEq)]
pub enum Location {
    Var(Ident),
    ArrayIndex(Ident, OrExpr),
}

#[derive(Debug, PartialEq)]
pub enum AssignExpr {
    RegularAssign(AssignOp, OrExpr),
    Increment,
    Decrement,
}

#[derive(Debug, PartialEq)]
pub enum Arg {
    ExprArg(OrExpr),
    ExternArg(String),
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    Assignment(Location, AssignExpr),
    Call(Ident, Vec<Arg>),
    If(OrExpr, Block, Block),
    For(Ident, OrExpr, OrExpr, Location, AssignExpr, Block),
    While(OrExpr, Block),
    Return(Option<OrExpr>),
    Break,
    Continue,
}

#[derive(Debug, PartialEq)]
pub struct Block {
    pub fields: Vec<Field>,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, PartialEq)]
pub struct Method {
    pub meth_type: Option<Type>,
    pub name: Ident,
    pub params: Vec<Param>,
    pub body: Block,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub imports: Vec<Ident>,
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
}

// there is no good reason for this to return an option, but i want it to be
// consistent with all of the other things that return options
fn parse_star<T, U>(f: impl Fn(&mut T) -> Option<U>) -> impl Fn(&mut T) -> Option<Vec<U>> {
    move |tokens| {
        let mut ret = Vec::new();
        while let Some(u) = f(tokens) {
            ret.push(u);
        }
        Some(ret)
    }
}

fn parse_concat<'a, T: Clone, U, V>(
    f: impl Fn(&mut T) -> Option<U>,
    g: impl Fn(&mut T) -> Option<V>,
) -> impl Fn(&mut T) -> Option<(U, V)> {
    move |tokens| {
        let tokens_clone = tokens.clone();
        if let Some(u) = f(tokens) {
            if let Some(v) = g(tokens) {
                return Some((u, v));
            }
        }
        *tokens = tokens_clone;
        None
    }
}

fn parse_or<T: Clone, U, V>(
    f: impl Fn(&mut T) -> Option<U>,
    g: impl Fn(&mut T) -> Option<V>,
) -> impl Fn(&mut T) -> Option<Sum<U, V>> {
    move |tokens| {
        if let Some(u) = f(tokens) {
            Some(Inl(u))
        } else if let Some(v) = g(tokens) {
            Some(Inr(v))
        } else {
            None
        }
    }
}

fn fst<A, B>(x: (A, B)) -> A {
    x.0
}

fn parse_comma_sep_list<'a, T: Clone + Iterator<Item = &'a Token>, U>(
    elt: impl Fn(&mut T) -> Option<U>,
) -> impl Fn(&mut T) -> Option<Vec<U>> {
    move |tokens| {
        let comma_sep_list = parse_concat(
            parse_star(parse_concat(&elt, parse_one(exactly(Sym(Misc(Comma)))))),
            &elt,
        );
        match comma_sep_list(tokens) {
            Some((us, u)) => {
                let mut us: Vec<U> = us.into_iter().map(fst).collect();
                us.push(u);
                Some(us)
            }
            None => None,
        }
    }
}

use crate::scan::MiscSymbol::*;

fn parse_one<'a, T: Clone + Iterator<Item = &'a Token>, U>(
    f: impl Fn(&Token) -> Option<U>,
) -> impl Fn(&mut T) -> Option<U> {
    move |tokens: &mut T| {
        let tokens_clone = tokens.clone();
        if let Some(next) = tokens.next() {
            if let Some(val) = f(next) {
                return Some(val);
            }
        }
        *tokens = tokens_clone;
        None
    }
}

use Literal::*;

// sad dthat output type is not intlit
fn int_lit(t: &Token) -> Option<Literal> {
    match t {
        DecLit(s) => Some(DecInt(s.to_string())),
        HexLit(s) => Some(HexInt(s.to_string())),
        _ => None,
    }
}

fn exactly(t: Token) -> impl Fn(&Token) -> Option<()> {
    move |token| {
        println!("{:?}, {:?}", token, t);
        if *token == t {
            Some(())
        } else {
            None
        }
    }
}

fn ident(token: &Token) -> Option<Ident> {
    match token {
        Ident(name) => Some(Ident {
            name: name.to_string(),
        }),
        _ => None,
    }
}

use crate::scan::Keyword::*;
use crate::scan::Symbol::*;
use crate::scan::Token::*;

fn parse_import<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Ident> {
    let parse_import = parse_concat(
        parse_one(exactly(Key(Import))),
        parse_concat(parse_one(ident), parse_one(exactly(Sym(Misc(Semicolon))))),
    );
    match parse_import(tokens) {
        Some(((), (name, ()))) => Some(name),
        None => None,
    }
}

use Type::*;

fn typ(token: &Token) -> Option<Type> {
    match token {
        Key(Int) => Some(IntType),
        Key(Long) => Some(LongType),
        Key(Keyword::Bool) => Some(BoolType),
        _ => None,
    }
}

use Field::*;

fn parse_field_decl<'a, T: Clone + Iterator<Item = &'a Token>>(
    tokens: &mut T,
) -> Option<Vec<Field>> {
    let parse_array_field_decl = parse_concat(
        parse_one(ident),
        parse_concat(
            parse_one(exactly(Sym(Misc(LBrack)))),
            parse_concat(parse_one(int_lit), parse_one(exactly(Sym(Misc(RBrack))))),
        ),
    );
    // subtle opportunity for bug: parse_array_field_decl needs to be on the left here
    let parse_scalar_or_arr = parse_or(parse_array_field_decl, parse_one(ident));
    let parse_field = parse_concat(
        parse_one(typ),
        parse_concat(
            parse_comma_sep_list(&parse_scalar_or_arr),
            parse_one(exactly(Sym(Misc(Semicolon)))),
        ),
    );
    match parse_field(tokens) {
        Some((t, (decls, ()))) => Some(
            decls
                .into_iter()
                .map(|decl| match decl {
                    Inr(Ident { name }) => Scalar(t.clone(), Ident { name }),
                    Inl((Ident { name }, ((), (lit, ())))) => Array(t.clone(), Ident { name }, lit),
                })
                .collect(),
        ),
        None => None,
    }
}

fn parse_method<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Method> {
    panic!()
}

pub fn parse_program<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Program {
    Program {
        imports: parse_star(parse_import)(tokens).unwrap(),
        fields: parse_star(parse_field_decl)(tokens)
            .unwrap()
            .into_iter()
            .flatten()
            .collect(),
        methods: parse_star(parse_method)(tokens).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_import() {
        assert_eq!(
            parse_import(
                &mut vec![Key(Import), Ident("blah".to_string()), Sym(Misc(Semicolon))].iter()
            ),
            Some(Ident {
                name: "blah".to_string()
            })
        );

        assert_eq!(
            parse_import(&mut vec![Key(Import), Ident("blah".to_string())].iter()),
            None
        );
    }

    #[test]
    fn one_field_decl() {
        assert_eq!(
            parse_field_decl(&mut vec![Key(Int), Ident("moo".to_string())].iter()),
            None
        );

        assert_eq!(
            parse_field_decl(
                &mut vec![Key(Int), Ident("moo".to_string()), Sym(Misc(Semicolon))].iter()
            ),
            Some(vec![Scalar(
                IntType,
                Ident {
                    name: "moo".to_string()
                }
            )])
        );

        assert_eq!(
            parse_field_decl(
                &mut vec![
                    Key(Int),
                    Ident("moo".to_string()),
                    Sym(Misc(Comma)),
                    Ident("br32".to_string()),
                    Sym(Misc(LBrack)),
                    DecLit("31231".to_string()),
                    Sym(Misc(RBrack)),
                    Sym(Misc(Semicolon))
                ]
                .iter()
            ),
            Some(vec![
                Scalar(
                    IntType,
                    Ident {
                        name: "moo".to_string()
                    }
                ),
                Array(
                    IntType,
                    Ident {
                        name: "br32".to_string()
                    },
                    DecInt("31231".to_string())
                )
            ])
        );
    }
}
