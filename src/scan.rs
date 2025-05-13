use enum_iterator::all;
use enum_iterator::Sequence;
use std::fmt;
use std::string::ToString;

const FORM_FEED: char = 12u8 as char;

// is this really not in the stdlib?
#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub enum Sum<A, B> {
    Inl(A),
    Inr(B),
}

impl<A: fmt::Display, B: fmt::Display> fmt::Display for Sum<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Sum::Inl(a) => write!(f, "{}", a),
            Sum::Inr(b) => write!(f, "{}", b),
        }
    }
}

pub trait Finite {}

pub trait OfString {
    fn of_string(input: &str) -> Option<Self>
    where
        Self: Sized;
}

impl<T: ToString + Sequence + Finite> OfString for T {
    fn of_string(input: &str) -> Option<T> {
        all::<Self>().find(|val| val.to_string() == input)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum Keyword {
    Import,
    Void,
    Int,
    Long,
    Bool,
    If,
    Else,
    For,
    While,
    Return,
    Break,
    Continue,
    Len,
    True,
    False,
}

use Keyword::*;

impl Finite for Keyword {}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Import => "import",
            Void => "void",
            Int => "int",
            Long => "long",
            Bool => "bool",
            If => "if",
            For => "for",
            While => "while",
            Len => "len",
            Else => "else",
            Return => "return",
            Break => "break",
            Continue => "continue",
            True => "true",
            False => "false",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum AssignOp {
    Eq,
    PlusEq,
    MinusEq,
    MulEq,
    DivEq,
    ModEq,
}

impl AssignOp {
    pub fn is_arith(&self) -> bool {
        match self {
            Eq => false,
            PlusEq => true,
            MinusEq => true,
            MulEq => true,
            DivEq => true,
            ModEq => true,
        }
    }
}

use AssignOp::*;

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Eq => "=",
            PlusEq => "+=",
            MinusEq => "-=",
            MulEq => "*=",
            DivEq => "/=",
            ModEq => "%=",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum IncrOp {
    Increment,
    Decrement,
}

use IncrOp::*;
impl fmt::Display for IncrOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Increment => "++",
            Decrement => "--",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum MulOp {
    Mul,
    Div,
    Mod,
}

use MulOp::*;
impl fmt::Display for MulOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Mul => "*",
            Div => "/",
            Mod => "%",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum AddOp {
    Add,
    Sub,
}

use AddOp::*;
impl fmt::Display for AddOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Add => "+",
            Sub => "-",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum RelOp {
    Lt,
    Gt,
    Le,
    Ge,
}

impl fmt::Display for RelOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            RelOp::Lt => "<",
            RelOp::Gt => ">",
            RelOp::Le => "<=",
            RelOp::Ge => ">=",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum EqOp {
    Eq,
    Neq,
}

impl fmt::Display for EqOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            EqOp::Eq => "==",
            EqOp::Neq => "!=",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum MiscSymbol {
    Not,
    LPar,
    RPar,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Semicolon,
    Comma,
}

use MiscSymbol::*;

impl fmt::Display for MiscSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ret = match self {
            Not => "!",
            LPar => "(",
            RPar => ")",
            LBrack => "[",
            RBrack => "]",
            LBrace => "{",
            RBrace => "}",
            Semicolon => ";",
            Comma => ",",
        };
        write!(f, "{}", ret)
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum Symbol {
    MulSym(MulOp),
    AddSym(AddOp),
    RelSym(RelOp),
    EqSym(EqOp),
    And,
    Or,
    Assign(AssignOp),
    Incr(IncrOp),
    Misc(MiscSymbol),
}

impl Finite for Symbol {}

use Symbol::*;

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MulSym(b) => b.fmt(f),
            AddSym(b) => b.fmt(f),
            RelSym(b) => b.fmt(f),
            EqSym(b) => b.fmt(f),
            And => "&&".fmt(f),
            Or => "||".fmt(f),
            Assign(op) => op.fmt(f),
            Incr(op) => op.fmt(f),
            Misc(symb) => symb.fmt(f),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Key(Keyword),
    Ident(String),
    DecIntLit(String),
    HexIntLit(String),
    DecLongLit(String),
    HexLongLit(String),
    StrLit(String),
    CharLit(char),
    Sym(Symbol),
}

use Token::*;

pub fn format_char_for_output(c: char) -> String {
    match c {
        '"' => "\\\"".to_string(),
        '\'' => "\\\'".to_string(),
        '\\' => "\\\\".to_string(),
        '\t' => "\\t".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        FORM_FEED => "\\f".to_string(),
        _ => c.to_string(),
    }
}

pub fn format_str_for_output(s: &String) -> String {
    s.chars().map(format_char_for_output).collect()
}

impl Token {
    fn of_ident_or_keyword(word: String) -> Self {
        match Keyword::of_string(&word) {
            Some(k) => Self::Key(k),
            None => Self::Ident(word),
        }
    }

    pub fn format_for_output(&self) -> String {
        match self {
            Key(True) => "BOOLEANLITERAL true".to_string(),
            Key(False) => "BOOLEANLITERAL false".to_string(),
            Key(word) => word.to_string(),
            Ident(s) => format!("IDENTIFIER {}", s),
            DecIntLit(s) => format!("INTLITERAL {}", s),
            HexIntLit(s) => format!("INTLITERAL 0x{}", s),
            DecLongLit(s) => format!("LONGLITERAL {}L", s),
            HexLongLit(s) => format!("LONGLITERAL 0x{}L", s),
            StrLit(s) => format!(
                "STRINGLITERAL \"{}\"",
                s.to_string()
                    .chars()
                    .map(format_char_for_output)
                    .collect::<String>()
            ),
            CharLit(c) => format!("CHARLITERAL '{}'", format_char_for_output(*c)),
            Sym(s) => s.to_string(),
        }
    }
}

fn eat_while(
    input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>),
    test: fn(char) -> bool,
) -> String {
    let mut s = String::new();
    while let Some((_, c)) = input.clone().next() {
        if test(c) {
            s.push(c);
            input.next();
        } else {
            break;
        }
    }
    s
}

fn scan_char(input: &mut impl Iterator<Item = (ErrLoc, char)>) -> Result<char, String> {
    let first = match input.next() {
        Some((_, first)) => first,
        None => return Err("expected char but line ended".to_string()),
    };
    Ok(match first {
        '\\' => {
            let second = match input.next() {
                Some((_, second)) => second,
                None => return Err("cannot lex \\ as char".to_string()),
            };
            match second {
                '"' => '"',
                '\'' => '\'',
                '\\' => '\\',
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                'f' => FORM_FEED,
                _ => return Err(format!("cannot lex {}{} as char", first, second)),
            }
        }
        '\"' => return Err("cannot lex \" as char".to_string()),
        '\'' => return Err("cannt lex ' as char".to_string()),
        _ => {
            if 32 as char <= first && first <= 126 as char {
                first
            } else {
                return Err(format!("cannot lex uint_8 {} as char", first as u8));
            }
        }
    })
}

fn scan_char_lit(input: &mut impl Iterator<Item = (ErrLoc, char)>) -> Result<Token, String> {
    let first = input.next().map(snd);
    if first != Some('\'') {
        return Err(format!(
            "expected \' to begin char literal, got {:?}",
            first
        ));
    }
    let ret = scan_char(input)?;
    let last = input.next().map(snd);
    if last != Some('\'') {
        return Err(format!("expected \' to end char literal, got {:?}", last));
    }
    Ok(CharLit(ret))
}

fn scan_str_lit(
    input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>),
) -> Result<Token, String> {
    let mut ret = String::new();
    let first = input.next().map(snd).unwrap();
    assert_eq!(first, '\"');
    while let Some((_, c)) = input.clone().next() {
        if c == '\"' {
            input.next();
            return Ok(StrLit(ret));
        } else {
            ret.push(scan_char(input)?)
        }
    }
    Err("expected \" to end string literal, got end of line".to_string())
}

fn scan_dec_lit(input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>)) -> String {
    eat_while(input, |x| x.is_ascii_digit())
}

fn scan_hex_lit(input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>)) -> String {
    eat_while(input, |x| x.is_ascii_hexdigit())
}

// returns true if it found the end of the comment
fn scan_until_block_comment_end(input: &mut impl Iterator<Item = (ErrLoc, char)>) -> bool {
    let mut prev = None;
    for (_, c) in input {
        if c == '/' && prev == Some('*') {
            return true;
        }
        prev = Some(c);
    }
    false
}

fn is_alpha(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_alphanum(c: char) -> bool {
    is_alpha(c) || c.is_ascii_digit()
}

fn scan_ident_or_keyword(input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>)) -> String {
    eat_while(input, is_alphanum)
}

fn scan_integer_lit(input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>)) -> Token {
    let mut input_clone = input.clone();
    let val;
    let hex;
    if input_clone.next().map(snd) == Some('0') && input_clone.next().map(snd) == Some('x') {
        input.next();
        input.next();
        val = scan_hex_lit(input);
        hex = true;
    } else {
        val = scan_dec_lit(input);
        hex = false;
    }
    let long = input.clone().next().map(snd) == Some('L');
    if long {
        input.next();
    };
    match (hex, long) {
        (true, true) => Token::HexLongLit(val),
        (true, false) => Token::HexIntLit(val),
        (false, true) => Token::DecLongLit(val),
        (false, false) => Token::DecIntLit(val),
    }
}

fn snd<A, B>(x: (A, B)) -> B {
    let (_, b) = x;
    b
}

fn scan_sym(input: &mut (impl Clone + Iterator<Item = (ErrLoc, char)>)) -> Result<Token, String> {
    let mut sym = None;
    let mut input_clone = input.clone();
    let first = snd(input.next().unwrap());
    // first try to parse two-character symbol
    if let Some((_, second)) = input_clone.nth(1) {
        if let Some(s) = Symbol::of_string(&[first, second].iter().collect::<String>()) {
            sym = Some(s);
            // advance index again, since we scan two chars
            input.next();
        }
    }
    // if that didn't work, try to parse one-character symbol
    if sym.is_none() {
        sym = Symbol::of_string(&first.to_string());
    }
    match sym {
        Some(sym) => Ok(Sym(sym)),
        None => Err(format!("unknown symbol {}", first)),
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ErrLoc {
    pub line: u32,
    pub col: u32,
}

impl fmt::Display for ErrLoc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "line {}, col {}", self.line, self.col)
    }
}

pub fn scan(input: String) -> Vec<(Result<Token, String>, ErrLoc)> {
    let mut tokens = Vec::new();
    let mut scanning_block_comment = false;
    for (linenum, line) in input.lines().enumerate() {
        let mut line = line.chars().enumerate().map(|(col, c)| {
            (
                ErrLoc {
                    line: (linenum + 1) as u32,
                    col: (col + 1) as u32,
                },
                c,
            )
        });
        loop {
            let mut line_clone = line.clone();
            match line_clone.next() {
                None => break,
                Some((e, fst)) => {
                    if scanning_block_comment {
                        scanning_block_comment = !scan_until_block_comment_end(&mut line);
                        continue;
                    }

                    let second = line_clone.next().map(snd);
                    // lex ident or keyword
                    if is_alpha(fst) {
                        tokens.push((
                            Ok(Token::of_ident_or_keyword(scan_ident_or_keyword(&mut line))),
                            e,
                        ));
                    }
                    // lex int literal
                    else if fst.is_ascii_digit() {
                        tokens.push((Ok(scan_integer_lit(&mut line)), e));
                    } else if fst == '\'' {
                        tokens.push((scan_char_lit(&mut line), e));
                    } else if fst == '"' {
                        tokens.push((scan_str_lit(&mut line), e));
                    }
                    // lex comment
                    else if fst == '/' && second == Some('/') {
                        break;
                        // go to next line
                        // could also do line.last();
                    } else if fst == '/' && second == Some('*') {
                        line.next();
                        line.next();
                        scanning_block_comment = true;
                    }
                    // lex space
                    else if fst.is_ascii_whitespace() {
                        line.next();
                    }
                    // lex symbol
                    else {
                        tokens.push((scan_sym(&mut line), e));
                    }
                }
            }
        }
    }
    if scanning_block_comment {
        panic!("unterminated block comment");
    }
    tokens
}
