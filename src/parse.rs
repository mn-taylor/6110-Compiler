use crate::scan::*;

// is this really not in the stdlib?
#[derive(Clone, Debug, PartialEq)]
enum Sum<A, B> {
    Inl(A),
    Inr(B),
}

use Sum::*;

#[derive(Clone, Debug, PartialEq)]
struct Ident {
    name: String,
}

#[derive(Clone, Debug, PartialEq)]
enum Type {
    IntType,
    LongType,
    BoolType,
}

#[derive(Clone, Debug, PartialEq)]
struct Int {
    val: String,
}

#[derive(Clone, Debug, PartialEq)]
enum Field {
    Scalar(Type, Ident),
    Array(Type, Ident, Int),
}

#[derive(Clone, Debug, PartialEq)]
struct Param {
    param_type: Type,
    name: Ident,
}

// allowing negation here would be redundant, since negation appears in expr
#[derive(Clone, Debug, PartialEq)]
enum Literal {
    DecInt(String),
    HexInt(String),
    DecLong(String),
    HexLong(String),
    Char(char),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
enum Location {
    Var(Ident),
    ArrayIndex(Ident, Expr),
}

#[derive(Debug, PartialEq)]
enum AssignExpr {
    RegularAssign(AssignOp, Expr),
    Increment,
    Decrement,
}

#[derive(Clone, Debug, PartialEq)]
enum Arg {
    ExprArg(Expr),
    ExternArg(String),
}

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
struct Block {
    fields: Vec<Field>,
    stmts: Vec<Stmt>,
}

#[derive(Debug, PartialEq)]
struct Method {
    meth_type: Option<Type>,
    name: Ident,
    params: Vec<Param>,
    body: Block,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    imports: Vec<Ident>,
    fields: Vec<Field>,
    methods: Vec<Method>,
}

// there is no good reason for this to return an option, but i want it to be
// consistent with all of the other things that return options
fn parse_star<T, U>(
    mut f: impl FnMut(&mut T) -> Option<U>,
) -> impl FnMut(&mut T) -> Option<Vec<U>> {
    move |tokens| {
        let mut ret = Vec::new();
        while let Some(u) = f(tokens) {
            ret.push(u);
        }
        Some(ret)
    }
}

fn parse_concat<'a, T: Clone, U, V>(
    mut f: impl FnMut(&mut T) -> Option<U>,
    mut g: impl FnMut(&mut T) -> Option<V>,
) -> impl FnMut(&mut T) -> Option<(U, V)> {
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
    mut f: impl FnMut(&mut T) -> Option<U>,
    mut g: impl FnMut(&mut T) -> Option<V>,
) -> impl FnMut(&mut T) -> Option<Sum<U, V>> {
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
    elt: impl FnMut(&mut T) -> Option<U>,
    elt_again
) -> impl FnMut(&mut T) -> Option<Vec<U>> {
    move |tokens| {
        let mut comma_sep_list = parse_concat(
            parse_star(parse_concat(
                elt.clone(),
                parse_one(exactly(Sym(Misc(Semicolon)))),
            )),
            elt.clone(),
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
    mut f: impl FnMut(&Token) -> Option<U>,
) -> impl FnMut(&mut T) -> Option<U> {
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

fn exactly<'a>(t: Token) -> impl Fn(&Token) -> Option<()> {
    move |token| {
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
use crate::scan::MiscSymbol::*;
use crate::scan::Symbol::*;
use crate::scan::Token::*;

fn parse_import<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Ident> {
    let mut parse_import = parse_concat(
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

fn parse_field<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Field> {
    let mut parse_array_field_decl = parse_concat(
        parse_one(ident),
        parse_concat(
            parse_one(exactly(Sym(Misc(LBrack)))),
            parse_concat(parse_one(int_lit), parse_one(exactly(Sym(Misc(RBrack))))),
        ),
    );
    let mut parse_scalar_or_arr = parse_or(parse_one(ident), parse_array_field_decl);
    let mut parse_field = parse_concat(parse_one(typ), parse_comma_sep_list(parse_scalar_or_arr));
    panic!()
}

fn parse_method<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Method> {
    panic!()
}

pub fn parse_program<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Program {
    Program {
        imports: parse_star(parse_import)(tokens),
        fields: parse_star(parse_field)(tokens),
        methods: parse_star(parse_method)(tokens),
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
}
