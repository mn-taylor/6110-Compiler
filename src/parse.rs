use crate::scan::*;

// is this really not in the stdlib?
#[derive(Debug, PartialEq)]
enum Sum<A, B> {
    Inl(A),
    Inr(B),
}

use Sum::*;

#[derive(Debug, PartialEq)]
struct Ident {
    name: String,
}

// not sure about best way to handle current cloning of type
#[derive(Clone, Debug, PartialEq)]
enum Type {
    IntType,
    LongType,
    BoolType,
}

#[derive(Debug, PartialEq)]
enum Field {
    Scalar(Type, Ident),
    Array(Type, Ident, Literal),
}

#[derive(Debug, PartialEq)]
struct Param {
    param_type: Type,
    name: Ident,
}

// allowing negation here would be redundant, since negation appears in expr
#[derive(Debug, PartialEq)]
enum Literal {
    DecInt(String),
    HexInt(String),
    DecLong(String),
    HexLong(String),
    Char(char),
    Bool(bool),
}

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
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
