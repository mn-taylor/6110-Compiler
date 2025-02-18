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

fn bool_lit(t: &Token) -> Option<bool> {
    match t {
        Key(True) => Some(true),
        Key(False) => Some(false),
        _ => None,
    }
}

fn char_lit(t: &Token) -> Option<char> {
    match t {
        CharLit(c) => Some(*c),
        _ => None,
    }
}

fn parse_lit<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Literal> {
    // order doesn't much matter, just need to try long before int
    if let Some(b) = parse_one(bool_lit)(tokens) {
        return Some(Literal::Bool(b));
    }
    if let Some(c) = parse_one(char_lit)(tokens) {
        return Some(Char(c));
    }
    let parse_long = parse_concat(parse_one(long_lit), parse_one(exactly(Key(L))));
    if let Some((l, ())) = parse_long(tokens) {
        return Some(l);
    }
    parse_one(int_lit)(tokens)
}

#[derive(Debug, PartialEq)]
pub enum AtomicExpr {
    Loc(Box<Location>),
    Call(Ident, Vec<Arg>),
    Lit(Literal),
    IntCast(Box<OrExpr>),
    LongCast(Box<OrExpr>),
    LenEx(Ident),
    NegEx(Box<AtomicExpr>),
    NotEx(Box<AtomicExpr>),
    Ex(Box<OrExpr>),
}

use crate::scan::AddOp::*;
use crate::scan::MiscSymbol::*;
use crate::scan::Symbol::*;
use AtomicExpr::*;

fn parse_nothing<T>(_tokens: &mut T) -> Option<()> {
    Some(())
}

fn parse_method_call<'a, T: Clone + Iterator<Item = &'a Token>>(
    tokens: &mut T,
) -> Option<(Ident, Vec<Arg>)> {
    let parse_call = parse_concat(
        parse_one(ident),
        parse_concat(
            parse_one(exactly(Sym(Misc(LPar)))),
            parse_concat(
                parse_or(parse_comma_sep_list(parse_arg), parse_nothing),
                parse_one(exactly(Sym(Misc(RPar)))),
            ),
        ),
    );
    Some(match parse_call(tokens)? {
        (name, ((), (args, ()))) => {
            let args = match args {
                Inl(args) => args,
                Inr(()) => vec![],
            };
            (name, args)
        }
    })
}

impl Parse for AtomicExpr {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<AtomicExpr> {
        // order matters here.  one that works is
        // Neg, Not, Ex, IntCast, LongCast, Len, Lit, Call, Loc
        let parse_neg = parse_concat(parse_one(exactly(Sym(AddSym(Sub)))), Self::parse);
        let parse_not = parse_concat(parse_one(exactly(Sym(Misc(Not)))), Self::parse);
        let parse_ex = parse_concat(
            parse_one(exactly(Sym(Misc(LPar)))),
            parse_concat(OrExpr::parse, parse_one(exactly(Sym(Misc(RPar))))),
        );
        let parse_intcast = parse_concat(parse_one(exactly(Key(Int))), &parse_ex);
        let parse_longcast = parse_concat(parse_one(exactly(Key(Long))), &parse_ex);
        let parse_len = parse_concat(
            parse_one(exactly(Key(Len))),
            parse_concat(
                parse_one(exactly(Sym(Misc(LPar)))),
                parse_concat(parse_one(ident), parse_one(exactly(Sym(Misc(RPar))))),
            ),
        );
        // parse_lit defined elsewhere

        // parse_loc defined elsewhere
        if let Some(((), neg_exp)) = parse_neg(tokens) {
            Some(NegEx(Box::new(neg_exp)))
        } else if let Some(((), not_exp)) = parse_not(tokens) {
            Some(AtomicExpr::NotEx(Box::new(not_exp)))
        } else if let Some(((), (par_exp, ()))) = parse_ex(tokens) {
            Some(Ex(Box::new(par_exp)))
        } else if let Some(((), ((), (int_exp, ())))) = parse_intcast(tokens) {
            Some(IntCast(Box::new(int_exp)))
        } else if let Some(((), ((), (long_exp, ())))) = parse_longcast(tokens) {
            Some(LongCast(Box::new(long_exp)))
        } else if let Some(((), ((), (len_id, ())))) = parse_len(tokens) {
            Some(LenEx(len_id))
        } else if let Some(lit) = parse_lit(tokens) {
            Some(Lit(lit))
        } else if let Some((name, args)) = parse_method_call(tokens) {
            Some(Call(name, args))
        } else if let Some(loc) = parse_location(tokens) {
            Some(Loc(Box::new(loc)))
        } else {
            None
        }
    }
}

trait OfToken: Sized {
    fn of_token(t: &Token) -> Option<Self>;
}

trait Parse: Sized {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self>;
}

#[derive(Debug, PartialEq)]
pub enum BinExpr<OpType, AtomType> {
    Atomic(AtomType),
    Bin(AtomType, OpType, Box<BinExpr<OpType, AtomType>>),
}

impl<OpType: OfToken, AtomType: Parse> Parse for BinExpr<OpType, AtomType> {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self> {
        let bin_expr = parse_concat(
            AtomType::parse,
            parse_or(
                parse_concat(parse_one(OpType::of_token), Self::parse),
                parse_nothing,
            ),
        );
        Some(match bin_expr(tokens)? {
            (lhs, Inl((op, rhs))) => BinExpr::Bin(lhs, op, Box::new(rhs)),
            (atom, Inr(())) => BinExpr::Atomic(atom),
        })
    }
}

impl OfToken for AddOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(AddSym(s)) => Some(s.clone()), //really sboud implement copy
            _ => None,
        }
    }
}

impl OfToken for MulOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(MulSym(s)) => Some(s.clone()), //really sboud implement copy
            _ => None,
        }
    }
}

impl OfToken for RelOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(RelSym(s)) => Some(s.clone()), //really sboud implement copy
            _ => None,
        }
    }
}

impl OfToken for EqOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(EqSym(s)) => Some(s.clone()), //really sboud implement copy
            _ => None,
        }
    }
}

impl OfToken for AndOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(And) => Some(AndOp {}),
            _ => None,
        }
    }
}

impl OfToken for OrOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(Or) => Some(OrOp {}),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct AndOp {}

#[derive(Debug, PartialEq)]
pub struct OrOp {}

type MulExpr = BinExpr<MulOp, AtomicExpr>;
type AddExpr = BinExpr<AddOp, MulExpr>;
type RelExpr = BinExpr<RelOp, AddExpr>;
type EqExpr = BinExpr<EqOp, RelExpr>;
type AndExpr = BinExpr<AndOp, EqExpr>;
type OrExpr = BinExpr<OrOp, AndExpr>;

#[derive(Debug, PartialEq)]
pub enum Location {
    Var(Ident),
    ArrayIndex(Ident, OrExpr),
}

use Location::*;
fn parse_location<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Location> {
    let parse_location = parse_or(
        parse_concat(
            parse_one(ident),
            parse_concat(
                parse_one(exactly(Sym(Misc(LBrack)))),
                parse_concat(OrExpr::parse, parse_one(exactly(Sym(Misc(RBrack))))),
            ),
        ),
        parse_one(ident),
    );
    Some(match parse_location(tokens)? {
        Inl((id, ((), (idx, ())))) => ArrayIndex(id, idx),
        Inr(id) => Var(id),
    })
}

#[derive(Debug, PartialEq)]
pub enum AssignExpr {
    RegularAssign(AssignOp, OrExpr),
    Increment,
    Decrement,
}

impl Parse for AssignExpr {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self> {
        panic!()
    }
}

#[derive(Debug, PartialEq)]
pub enum Arg {
    ExprArg(OrExpr),
    ExternArg(String),
}

impl Parse for Arg {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self> {
        let arg = parse_or(OrExpr::parse, parse_one(str_lit));
        Some(match arg(tokens)? {
            Inl(a) => ExprArg(a),
            Inr(s) => ExternArg(s),
        })
    }
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

impl Parse for Stmt {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self> {
        // order doesnt' matter much
        let parse_ass = parse_concat(
            parse_location,
            parse_concat(AssignExpr::parse, parse_one(exactly(Sym(Misc(Semicolon))))),
        );
        let parse_if = parse_concat(
            parse_one(exactly(Key(If))),
            parse_concat(
                parse_one(exactly(Sym(Misc(LPar)))),
                parse_concat(
                    OrExpr::parse,
                    parse_concat(
                        parse_one(exactly(Sym(Misc(RPar)))),
                        parse_concat(
                            Block::parse,
                            parse_or(
                                parse_concat(parse_one(exactly(Key(Else))), Block::parse),
                                parse_nothing,
                            ),
                        ),
                    ),
                ),
            ),
        );
        let parse_for = parse_concat(
            parse_one(exactly(Key(For))),
            parse_concat(
                parse_one(exactly(Sym(Misc(LPar)))),
                parse_concat(
                    parse_one(ident),
                    parse_concat(
                        parse_one(exactly(Sym(Assign(AssignOp::Eq)))),
                        parse_concat(
                            OrExpr::parse,
                            parse_concat(
                                parse_one(exactly(Sym(Misc(Semicolon)))),
                                parse_concat(
                                    OrExpr::parse,
                                    parse_concat(
                                        parse_for_update,
                                        parse_concat(
                                            parse_one(exactly(Sym(Misc(RPar)))),
                                            Block::parse,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let parse_while = parse_concat(
            parse_one(exactly(Key(While))),
            parse_concat(
                parse_one(exactly(Sym(Misc(LPar)))),
                parse_concat(
                    OrExpr::parse,
                    parse_concat(parse_one(exactly(Sym(Misc(RPar)))), Block::parse),
                ),
            ),
        );
        let parse_return = parse_concat(
            parse_one(exactly(Key(Return))),
            parse_concat(
                parse_or(OrExpr::parse, parse_nothing),
                parse_one(exactly(Sym(Misc(Semicolon)))),
            ),
        );
        let parse_break = parse_concat(
            parse_one(exactly(Key(Break))),
            parse_one(exactly(Sym(Misc(Semicolon)))),
        );
        let parse_continue = parse_concat(
            parse_one(exactly(Key(Continue))),
            parse_one(exactly(Sym(Misc(Semicolon)))),
        );
    }
}

#[derive(Debug, PartialEq)]
pub struct Block {
    pub fields: Vec<Field>,
    pub stmts: Vec<Stmt>,
}

impl Parse for Block {
    fn parse<'a, T: Clone + Iterator<Item = &'a Token>>(tokens: &mut T) -> Option<Self> {
        panic!()
    }
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

fn long_lit(t: &Token) -> Option<Literal> {
    match t {
        DecLit(s) => Some(DecLong(s.to_string())),
        HexLit(s) => Some(HexLong(s.to_string())),
        _ => None,
    }
}

fn str_lit(t: &Token) -> Option<String> {
    match t {
        StrLit(s) => Some(s.to_string()),
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

    #[test]
    fn atomic_expr1() {
        assert_eq!(
            AtomicExpr::parse(&mut vec![CharLit('c')].iter()),
            Some(Lit(Char('c')))
        );
    }

    #[test]
    fn atomic_expr2() {
        assert_eq!(
            AtomicExpr::parse(&mut vec![Sym(Misc(Not)), CharLit('c')].iter()),
            Some(NotEx(Box::new(Lit(Char('c')))))
        );
    }
}
