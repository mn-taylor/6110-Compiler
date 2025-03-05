use crate::scan::*;
use std::fmt;
use Sum::*;

#[derive(PartialEq, Debug)]
pub struct Ident {
    pub name: String,
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Clone for Ident {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Primitive {
    IntType,
    LongType,
    BoolType,
}

impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            IntType => "int",
            LongType => "long",
            BoolType => "bool",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, PartialEq)]
pub enum Field {
    Scalar(Primitive, WithLoc<Ident>),
    Array(Primitive, WithLoc<Ident>, WithLoc<Literal>),
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar(t, id) => write!(f, "{} {}", t, id),
            Array(t, id, _) => write!(f, "{} array {}", t, id),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub param_type: Primitive,
    pub name: WithLoc<Ident>,
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
pub struct Block {
    pub fields: Vec<Field>,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, PartialEq)]
pub struct Method {
    pub meth_type: Option<Primitive>,
    pub name: WithLoc<Ident>,
    pub params: Vec<Param>,
    pub body: Block,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub imports: Vec<WithLoc<Ident>>,
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
}

#[derive(Debug, PartialEq)]
pub struct AndOp {}

#[derive(Debug, PartialEq)]
pub struct OrOp {}

#[derive(Debug, PartialEq)]
pub enum AtomicExpr {
    Loc(Box<WithLoc<Location>>),
    Call(WithLoc<Ident>, Vec<Arg>),
    Lit(WithLoc<Literal>),
    IntCast(Box<OrExpr>),
    LongCast(Box<OrExpr>),
    LenEx(WithLoc<Ident>),
    NegEx(Box<WithLoc<AtomicExpr>>),
    NotEx(Box<WithLoc<AtomicExpr>>),
    Ex(Box<OrExpr>),
}

#[derive(Debug, PartialEq)]
pub enum BinExpr<OpType, AtomType> {
    Atomic(AtomType),
    Bin(AtomType, OpType, Box<BinExpr<OpType, AtomType>>),
}

pub type MulExpr = BinExpr<MulOp, WithLoc<AtomicExpr>>;
pub type AddExpr = BinExpr<AddOp, MulExpr>;
pub type RelExpr = BinExpr<RelOp, AddExpr>;
pub type EqExpr = BinExpr<EqOp, RelExpr>;
pub type AndExpr = BinExpr<AndOp, EqExpr>;
pub type OrExpr = BinExpr<OrOp, AndExpr>;

#[derive(Debug, PartialEq)]
pub enum Location {
    Var(Ident),
    ArrayIndex(Ident, OrExpr),
}

#[derive(Debug, PartialEq)]
pub enum Arg {
    ExprArg(OrExpr),
    ExternArg(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct WithLoc<T> {
    pub val: T,
    pub loc: ErrLoc,
}

impl<T: fmt::Display> fmt::Display for WithLoc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} at line {}, col {}",
            self.val, self.loc.line, self.loc.col
        )
    }
}

impl<T> WithLoc<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> WithLoc<U> {
        WithLoc {
            val: f(self.val),
            loc: self.loc,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum AssignExpr {
    RegularAssign(AssignOp, OrExpr),
    IncrAssign(IncrOp),
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    Assignment(WithLoc<Location>, AssignExpr),
    Call(WithLoc<Ident>, Vec<Arg>),
    If(OrExpr, Block, Option<Block>),
    For {
        var_to_set: WithLoc<Ident>,
        initial_val: OrExpr,
        test: OrExpr,
        var_to_update: WithLoc<Location>,
        update_val: AssignExpr,
        body: Block,
    },
    While(OrExpr, Block),
    Return(Option<OrExpr>),
    Break,
    Continue,
}

pub trait TokenErrIter<'a>: Clone + Iterator<Item = &'a (Token, ErrLoc)> {}

impl<'a, T: Clone + Iterator<Item = &'a (Token, ErrLoc)>> TokenErrIter<'a> for T {}

impl OfToken for Literal {
    fn of_token(t: &Token) -> Option<Literal> {
        //bool, char, int, long
        match t {
            Key(True) => Some(Literal::Bool(true)),
            Key(False) => Some(Literal::Bool(false)),
            CharLit(c) => Some(Char(*c)),
            DecIntLit(s) => Some(DecInt(s.to_string())),
            HexIntLit(s) => Some(HexInt(s.to_string())),
            DecLongLit(s) => Some(DecLong(s.to_string())),
            HexLongLit(s) => Some(HexLong(s.to_string())),
            _ => None,
        }
    }
}

use crate::scan::AddOp::*;
use crate::scan::MiscSymbol::*;
use crate::scan::Symbol::*;
use AtomicExpr::*;

fn parse_nothing<T>(_tokens: &mut T) -> Option<()> {
    Some(())
}

fn parse_method_call<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<(WithLoc<Ident>, Vec<Arg>)> {
    let parse_call = parse_concat(
        WithLoc::<Ident>::parse,
        parse_concat(
            parse_one(exactly(Sym(Misc(LPar)))),
            parse_concat(
                parse_or(parse_comma_sep_list(Arg::parse), parse_nothing),
                parse_one(exactly(Sym(Misc(RPar)))),
            ),
        ),
    );
    let (name, ((), (args, ()))) = parse_call(tokens)?;
    Some((
        name,
        match args {
            Inl(args) => args,
            Inr(()) => vec![],
        },
    ))
}

impl Parse for WithLoc<AtomicExpr> {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<WithLoc<AtomicExpr>> {
        // order matters here.  one that works is
        // Neg, Not, Ex, IntCast, LongCast, Len, Lit, Call, Loc
        let parse_neg = parse_concat(parse_one(exactly_with_loc(Sym(AddSym(Sub)))), Self::parse);
        let parse_not = parse_concat(parse_one(exactly_with_loc(Sym(Misc(Not)))), Self::parse);
        let parse_ex = parse_concat(
            parse_one(exactly_with_loc(Sym(Misc(LPar)))),
            parse_concat(OrExpr::parse, parse_one(exactly(Sym(Misc(RPar))))),
        );
        let parse_intcast = parse_concat(parse_one(exactly_with_loc(Key(Int))), &parse_ex);
        let parse_longcast = parse_concat(parse_one(exactly_with_loc(Key(Long))), &parse_ex);
        let parse_len = parse_concat(
            parse_one(exactly_with_loc(Key(Len))),
            parse_concat(
                parse_one(exactly(Sym(Misc(LPar)))),
                parse_concat(WithLoc::<Ident>::parse, parse_one(exactly(Sym(Misc(RPar))))),
            ),
        );
        // parse_lit defined elsewhere

        // parse_loc defined elsewhere
        if let Some((loc, neg_exp)) = parse_neg(tokens) {
            Some(WithLoc {
                loc,
                val: NegEx(Box::new(neg_exp)),
            })
        } else if let Some((loc, not_exp)) = parse_not(tokens) {
            Some(WithLoc {
                loc,
                val: NotEx(Box::new(not_exp)),
            })
        } else if let Some((loc, (par_exp, ()))) = parse_ex(tokens) {
            Some(WithLoc {
                loc,
                val: Ex(Box::new(par_exp)),
            })
        } else if let Some((loc, (_, (int_exp, ())))) = parse_intcast(tokens) {
            Some(WithLoc {
                loc,
                val: IntCast(Box::new(int_exp)),
            })
        } else if let Some((loc, (_, (long_exp, ())))) = parse_longcast(tokens) {
            Some(WithLoc {
                loc,
                val: LongCast(Box::new(long_exp)),
            })
        } else if let Some((loc, ((), (len_id, ())))) = parse_len(tokens) {
            Some(WithLoc {
                loc,
                val: LenEx(len_id),
            })
        } else if let Some(lit) = WithLoc::<Literal>::parse(tokens) {
            Some(WithLoc {
                loc: lit.loc,
                val: Lit(lit),
            })
        } else if let Some((name, args)) = parse_method_call(tokens) {
            Some(WithLoc {
                loc: name.loc,
                val: Call(name, args),
            })
        } else if let Some(location) = WithLoc::<Location>::parse(tokens) {
            Some(WithLoc {
                loc: location.loc,
                val: Loc(Box::new(location)),
            })
        } else {
            None
        }
    }
}

trait OfToken: Sized {
    fn of_token(t: &Token) -> Option<Self>;
    fn of_tokenloc(t: &(Token, ErrLoc)) -> Option<Self> {
        let (t, _) = t;
        Self::of_token(t)
    }
    fn of_token_withloc(te: &(Token, ErrLoc)) -> Option<WithLoc<Self>> {
        let (t, e) = te;
        Some(WithLoc {
            val: Self::of_token(t)?,
            loc: e.clone(),
        })
    }
}

trait Parse: Sized {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self>;
    fn parse<'a, T: TokenErrIter<'a>>(tokens: &mut T) -> Option<Self> {
        // println!("trying to parse {}", std::any::type_name::<Self>());
        Self::parse_no_debug(tokens)
    }
}

impl<OpType: OfToken, AtomType: Parse> Parse for BinExpr<OpType, AtomType> {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let bin_expr = parse_concat(
            AtomType::parse,
            parse_or(
                parse_concat(parse_one(OpType::of_tokenloc), Self::parse),
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

impl Parse for WithLoc<Location> {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let parse_location = parse_or(
            parse_concat(
                WithLoc::<Ident>::parse,
                parse_concat(
                    parse_one(exactly(Sym(Misc(LBrack)))),
                    parse_concat(OrExpr::parse, parse_one(exactly(Sym(Misc(RBrack))))),
                ),
            ),
            WithLoc::<Ident>::parse,
        );
        Some(match parse_location(tokens)? {
            Inl((id, ((), (idx, ())))) => WithLoc {
                loc: id.loc,
                val: Location::ArrayIndex(id.val, idx),
            },
            Inr(id) => WithLoc {
                loc: id.loc,
                val: Location::Var(id.val),
            },
        })
    }
}

impl OfToken for AssignOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(Assign(op)) => Some(op.clone()),
            _ => None,
        }
    }
}

impl OfToken for IncrOp {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Sym(Incr(op)) => Some(op.clone()),
            _ => None,
        }
    }
}

impl<U: OfToken> Parse for U {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        parse_one(Self::of_tokenloc)(tokens)
    }
}

impl<U: OfToken> Parse for WithLoc<U> {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        parse_one(U::of_token_withloc)(tokens)
    }
}

use AssignExpr::*;
impl Parse for AssignExpr {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let ass_expr = parse_or(parse_concat(AssignOp::parse, OrExpr::parse), IncrOp::parse);
        Some(match ass_expr(tokens)? {
            Inl((op, expr)) => RegularAssign(op, expr),
            Inr(op) => IncrAssign(op),
        })
    }
}

impl Parse for Arg {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let arg = parse_or(OrExpr::parse, parse_one(str_lit));
        Some(match arg(tokens)? {
            Inl(a) => Arg::ExprArg(a),
            Inr(s) => Arg::ExternArg(s),
        })
    }
}

impl Parse for Stmt {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        // order doesnt' matter much
        let parse_ass = parse_concat(
            WithLoc::<Location>::parse,
            parse_concat(AssignExpr::parse, parse_one(exactly(Sym(Misc(Semicolon))))),
        );
        let parse_meth_stmt =
            parse_concat(parse_method_call, parse_one(exactly(Sym(Misc(Semicolon)))));
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
                    WithLoc::<Ident>::parse,
                    parse_concat(
                        parse_one(exactly(Sym(Assign(AssignOp::Eq)))),
                        parse_concat(
                            OrExpr::parse,
                            parse_concat(
                                parse_one(exactly(Sym(Misc(Semicolon)))),
                                parse_concat(
                                    OrExpr::parse,
                                    parse_concat(
                                        parse_one(exactly(Sym(Misc(Semicolon)))),
                                        parse_concat(
                                            WithLoc::<Location>::parse,
                                            parse_concat(
                                                AssignExpr::parse,
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
        if let Some((loc, (ass_ex, ()))) = parse_ass(tokens) {
            Some(Stmt::Assignment(loc, ass_ex))
        } else if let Some(((name, args), ())) = parse_meth_stmt(tokens) {
            Some(Stmt::Call(name, args))
        } else if let Some(((), ((), (exp, ((), (block, maybe_else_block)))))) = parse_if(tokens) {
            let maybe_block = match maybe_else_block {
                Inl(((), block)) => Some(block),
                Inr(()) => None,
            };
            Some(Stmt::If(exp, block, maybe_block))
        } else if let Some((
            (),
            (
                (),
                (
                    var_to_set,
                    (
                        (),
                        (
                            initial_val,
                            ((), (test, ((), (var_to_update, (update_val, ((), body)))))),
                        ),
                    ),
                ),
            ),
        )) = parse_for(tokens)
        {
            Some(Stmt::For {
                var_to_set,
                initial_val,
                test,
                var_to_update,
                update_val,
                body,
            })
        } else if let Some(((), ((), (test, ((), block))))) = parse_while(tokens) {
            Some(Stmt::While(test, block))
        } else if let Some(((), (maybe_expr, ()))) = parse_return(tokens) {
            Some(match maybe_expr {
                Inl(expr) => Stmt::Return(Some(expr)),
                Inr(()) => Stmt::Return(None),
            })
        } else if let Some(((), ())) = parse_break(tokens) {
            Some(Stmt::Break)
        } else if let Some(((), ())) = parse_continue(tokens) {
            Some(Stmt::Continue)
        } else {
            None
        }
    }
}

impl Parse for Block {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let block = parse_concat(
            parse_one(exactly(Sym(Misc(LBrace)))),
            parse_concat(
                parse_star(parse_field_decl),
                parse_concat(
                    parse_star(Stmt::parse),
                    parse_one(exactly(Sym(Misc(RBrace)))),
                ),
            ),
        );
        let ((), (field_decls, (stmts, ()))) = block(tokens)?;
        Some(Block {
            fields: field_decls.into_iter().flatten().collect(),
            stmts,
        })
    }
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

fn parse_concat<T: Clone, U, V>(
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

fn parse_comma_sep_list<'a, T: TokenErrIter<'a>, U>(
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

fn parse_one<'a, T: TokenErrIter<'a>, U>(
    f: impl Fn(&(Token, ErrLoc)) -> Option<U>,
) -> impl Fn(&mut T) -> Option<U> {
    move |tokens| {
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

fn of_token_withloc<T>(
    of_token: impl Fn(&Token) -> Option<T>,
) -> impl Fn(&(Token, ErrLoc)) -> Option<WithLoc<T>> {
    move |(t, e)| {
        Some(WithLoc {
            val: of_token(t)?,
            loc: e.clone(),
        })
    }
}

// sad dthat output type is not intlit
fn int_lit(t: &Token) -> Option<Literal> {
    match t {
        DecIntLit(s) => Some(DecInt(s.to_string())),
        HexIntLit(s) => Some(HexInt(s.to_string())),
        _ => None,
    }
}

fn str_lit(t: &(Token, ErrLoc)) -> Option<String> {
    let (t, _) = t;
    match t {
        StrLit(s) => Some(s.to_string()),
        _ => None,
    }
}

fn exactly_with_loc(t: Token) -> impl Fn(&(Token, ErrLoc)) -> Option<ErrLoc> {
    move |(token, e)| {
        // println!("trying to parse {:?}, finding {:?}", t, token);
        if *token == t {
            Some(*e)
        } else {
            None
        }
    }
}

fn exactly(t: Token) -> impl Fn(&(Token, ErrLoc)) -> Option<()> {
    move |(token, _)| {
        // println!("trying to parse {:?}, finding {:?}", t, token);
        if *token == t {
            Some(())
        } else {
            None
        }
    }
}

impl OfToken for Ident {
    fn of_token(t: &Token) -> Option<Ident> {
        match t {
            Ident(name) => Some(Ident {
                name: name.to_string(),
            }),
            _ => None,
        }
    }
}

use crate::scan::Keyword::*;
use crate::scan::Token::*;

fn parse_import<'a, T: TokenErrIter<'a>>(tokens: &mut T) -> Option<WithLoc<Ident>> {
    let parse_import = parse_concat(
        parse_one(exactly(Key(Import))),
        parse_concat(
            WithLoc::<Ident>::parse,
            parse_one(exactly(Sym(Misc(Semicolon)))),
        ),
    );
    match parse_import(tokens) {
        Some(((), (name, ()))) => Some(name),
        None => None,
    }
}

use Primitive::*;

impl OfToken for Primitive {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Key(Int) => Some(IntType),
            Key(Long) => Some(LongType),
            Key(Keyword::Bool) => Some(BoolType),
            _ => None,
        }
    }
}

impl OfToken for Option<Primitive> {
    fn of_token(t: &Token) -> Option<Self> {
        match t {
            Key(Void) => Some(None),
            _ => Primitive::of_token(t).map(Some),
        }
    }
}

use Field::*;

fn parse_field_decl<'a, T: TokenErrIter<'a>>(tokens: &mut T) -> Option<Vec<Field>> {
    let parse_array_field_decl = parse_concat(
        WithLoc::<Ident>::parse,
        parse_concat(
            parse_one(exactly(Sym(Misc(LBrack)))),
            parse_concat(
                parse_one(of_token_withloc(int_lit)),
                parse_one(exactly(Sym(Misc(RBrack)))),
            ),
        ),
    );
    // subtle opportunity for bug: parse_array_field_decl needs to be on the left here
    let parse_scalar_or_arr = parse_or(parse_array_field_decl, WithLoc::<Ident>::parse);
    let parse_field = parse_concat(
        Primitive::parse,
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
                    Inr(id) => Scalar(t.clone(), id),
                    Inl((id, ((), (lit, ())))) => Array(t.clone(), id, lit),
                })
                .collect(),
        ),
        None => None,
    }
}

impl Parse for Param {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Self> {
        let param = parse_concat(Primitive::parse, WithLoc::<Ident>::parse);
        let (param_type, name) = param(tokens)?;
        Some(Param { param_type, name })
    }
}

impl Parse for Method {
    fn parse_no_debug<'a>(tokens: &mut impl TokenErrIter<'a>) -> Option<Method> {
        let method = parse_concat(
            parse_one(Option::<Primitive>::of_tokenloc),
            parse_concat(
                WithLoc::<Ident>::parse,
                parse_concat(
                    parse_one(exactly(Sym(Misc(LPar)))),
                    parse_concat(
                        parse_or(parse_comma_sep_list(Param::parse), parse_nothing),
                        parse_concat(parse_one(exactly(Sym(Misc(RPar)))), Block::parse),
                    ),
                ),
            ),
        );
        let (meth_type, (name, ((), (maybe_params, ((), body))))) = method(tokens)?;
        let params = match maybe_params {
            Inl(params) => params,
            Inr(()) => Vec::new(),
        };
        Some(Method {
            meth_type,
            name,
            params,
            body,
        })
    }
}

pub fn parse_program<'a, T: TokenErrIter<'a>>(tokens: &mut T) -> Program {
    Program {
        imports: parse_star(parse_import)(tokens).unwrap(),
        fields: parse_star(parse_field_decl)(tokens)
            .unwrap()
            .into_iter()
            .flatten()
            .collect(),
        methods: parse_star(Method::parse)(tokens).unwrap(),
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
                    DecIntLit("31231".to_string()),
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
