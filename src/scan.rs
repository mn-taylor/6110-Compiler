use enum_iterator::all;
use enum_iterator::Sequence;
use std::string::ToString;

const FORM_FEED: char = 12u8 as char;

// is this really not in the stdlib?
#[derive(Debug, PartialEq)]
pub enum Sum<A, B> {
    Inl(A),
    Inr(B),
}

pub trait Finite {}

pub trait OfString {
    fn of_string(input: &str) -> Option<Self>
    where
        Self: Sized;
}

impl<T: ToString + Sequence + Finite> OfString for T {
    fn of_string(input: &str) -> Option<T> {
        for val in all::<Self>() {
            if val.to_string() == input {
                return Some(val);
            }
        }
        return None;
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
    L,
}

use Keyword::*;

impl Keyword {
    fn to_string(&self) -> String {
        match self {
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
            L => "L",
        }
        .to_string()
    }

    fn of_string(word: &str) -> Option<Self> {
        for val in all::<Self>() {
            if val.to_string() == word {
                return Some(val);
            }
        }
        return None;
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

use AssignOp::*;

impl ToString for AssignOp {
    fn to_string(&self) -> String {
        match self {
            Eq => "=",
            PlusEq => "+=",
            MinusEq => "-=",
            MulEq => "*=",
            DivEq => "/=",
            ModEq => "%=",
        }
        .to_string()
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum IncrOp {
    Increment,
    Decrement,
}

use IncrOp::*;
impl ToString for IncrOp {
    fn to_string(&self) -> String {
        match self {
            Increment => "++",
            Decrement => "--",
        }
        .to_string()
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum MulOp {
    Mul,
    Div,
    Mod,
}

use MulOp::*;
impl ToString for MulOp {
    fn to_string(&self) -> String {
        match self {
            Mul => "*",
            Div => "/",
            Mod => "%",
        }
        .to_string()
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum AddOp {
    Add,
    Sub,
}

use AddOp::*;
impl ToString for AddOp {
    fn to_string(&self) -> String {
        match self {
            Add => "+",
            Sub => "-",
        }
        .to_string()
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum RelOp {
    Lt,
    Gt,
    Le,
    Ge,
}

impl ToString for RelOp {
    fn to_string(&self) -> String {
        match self {
            RelOp::Lt => "<",
            RelOp::Gt => ">",
            RelOp::Le => "<=",
            RelOp::Ge => ">=",
        }
        .to_string()
    }
}

#[derive(Debug, PartialEq, Sequence, Clone)]
pub enum EqOp {
    Eq,
    Neq,
}

impl ToString for EqOp {
    fn to_string(&self) -> String {
        match self {
            EqOp::Eq => "==",
            EqOp::Neq => "!=",
        }
        .to_string()
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

impl ToString for MiscSymbol {
    fn to_string(&self) -> String {
        match self {
            Not => "!",
            LPar => "(",
            RPar => ")",
            LBrack => "[",
            RBrack => "]",
            LBrace => "{",
            RBrace => "}",
            Semicolon => ";",
            Comma => ",",
        }
        .to_string()
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

impl ToString for Symbol {
    fn to_string(&self) -> String {
        match self {
            MulSym(b) => b.to_string(),
            AddSym(b) => b.to_string(),
            RelSym(b) => b.to_string(),
            EqSym(b) => b.to_string(),
            And => "&&".to_string(),
            Or => "||".to_string(),
            Assign(op) => op.to_string(),
            Incr(op) => op.to_string(),
            Misc(symb) => symb.to_string(),
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

impl Token {
    fn of_ident_or_keyword(word: String) -> Self {
        match Keyword::of_string(&word) {
            Some(k) => Self::Key(k),
            None => Self::Ident(word),
        }
    }

    pub fn format_for_output(&self) -> String {
        match self {
            Key(word) => word.to_string(),
            Ident(s) => format!("IDENTIFIER {}", s),
            DecIntLit(s) => format!("INTLITERAL {}", s),
            HexIntLit(s) => format!("INTLITERAL 0x{}", s),
            DecLongLit(s) => format!("LONGLITERAL {}L", s),
            HexLongLit(s) => format!("LONGLITERAL 0x{}L", s),
            StrLit(s) => format!("STRINGLITERAL \"{}\"", s),
            CharLit(c) => format!("CHARLITERAL '{}'", c),
            Sym(s) => s.to_string(),
        }
    }
}

use std::iter::Peekable;
use std::str::Chars;

fn eat_while(input: &mut Peekable<Chars>, test: fn(&char) -> bool) -> String {
    let mut s = String::new();
    loop {
        match input.peek() {
            Some(c) => {
                if test(c) {
                    s.push(*c);
                    input.next();
                } else {
                    break;
                }
            }
            None => break,
        }
    }
    return s;
}

fn scan_char(input: &mut Peekable<Chars>) -> char {
    let fst = input.next().unwrap(); // TODO error handling
    match fst {
        '\\' => {
            let snd = input.next().unwrap();
            match snd {
                '"' => '\"',
                '\'' => '\'',
                '\\' => '\\',
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                'f' => FORM_FEED,
                _ => panic!(),
            }
        }
        '\"' => panic!(),
        '\'' => panic!(),
        _ => fst,
    }
}

fn scan_char_lit(input: &mut Peekable<Chars>) -> char {
    assert_eq!(input.next(), Some('\''));
    let ret = scan_char(input);
    assert_eq!(input.next(), Some('\''));
    ret
}

fn scan_str_lit(input: &mut Peekable<Chars>) -> String {
    let mut ret = String::new();
    assert_eq!(input.next(), Some('\"'));
    while input.peek() != Some(&'\"') {
        ret.push(scan_char(input));
    }
    ret
}

fn scan_dec_lit(input: &mut Peekable<Chars>) -> String {
    eat_while(input, |x| x.is_ascii_digit())
}

fn scan_hex_lit(input: &mut Peekable<Chars>) -> String {
    eat_while(input, |x| x.is_ascii_hexdigit())
}

// returns true if it found the end of the comment
fn scan_until_block_comment_end(input: &mut Peekable<Chars>) -> bool {
    input.next();
    input.next();
    let mut prev = None;
    for c in input {
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

fn scan_ident_or_keyword(input: &mut Peekable<Chars>) -> String {
    eat_while(input, |x| is_alphanum(*x))
}

pub fn scan(input: String) -> Vec<Vec<Token>> {
    let mut all_tokens = Vec::new();
    let mut scanning_block_comment = false;
    for line in input.lines() {
        let mut line = line.chars().peekable();
        let mut tokens = Vec::new();
        loop {
            if scanning_block_comment {
                scanning_block_comment = scan_until_block_comment_end(&mut line);
                continue;
            }
            let mut line_clone = line.clone();
            match line_clone.next() {
                None => break,
                Some(fst) => {
                    // lex ident or keyword
                    if is_alpha(fst) {
                        tokens.push(Token::of_ident_or_keyword(scan_ident_or_keyword(&mut line)));
                    }
                    // lex int literal
                    // this is a horrible mess.  my fault of course, but also it would be so much cleaner if we could just treat L as a token...
                    // for that matter, why not 0x too?
                    else if fst.is_ascii_digit() {
                        let val;
                        let hex;
                        if fst == '0' && line_clone.next() == Some('x') {
                            val = scan_hex_lit(&mut line);
                            hex = true;
                        } else {
                            val = scan_dec_lit(&mut line);
                            hex = false;
                        }
                        let mut line_clone = line.clone();
                        if line_clone.next() == Some('L') {
                            line.next();
                            if hex {
                                tokens.push(Token::HexLongLit(val));
                            } else {
                                tokens.push(Token::DecLongLit(val));
                            }
                        } else {
                            if hex {
                                tokens.push(Token::HexIntLit(val));
                            } else {
                                tokens.push(Token::DecIntLit(val));
                            }
                        }
                    }
                    // lex comment
                    else if fst == '/' {
                        if line_clone.next() == Some('/') {
                            break;
                        // go to next line
                        // could also do line.last();
                        } else {
                            line.next();
                            line.next();
                            scanning_block_comment = true;
                        }
                    }
                    // lex space
                    else if fst.is_ascii_whitespace() {
                        line.next();
                    }
                    // lex symbol
                    else {
                        let mut sym = None;
                        // first try to parse two-character symbol
                        if let Some(snd) = line_clone.nth(1) {
                            if let Some(s) =
                                Symbol::of_string(&[fst, snd].iter().collect::<String>())
                            {
                                sym = Some(s);
                                line.next();
                                line.next();
                            }
                        }
                        // if that didn't work, try to parse one-character symbol
                        if let None = sym {
                            if let Some(s) = Symbol::of_string(&fst.to_string()) {
                                sym = Some(s);
                                line.next();
                            }
                        }
                        tokens.push(Token::Sym(sym.unwrap()));
                    }
                }
            }
        }
        all_tokens.push(tokens);
    }
    if scanning_block_comment {
        panic!("unterminated block comment");
    }
    all_tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;

    fn test(input: &str, output: Vec<Token>) {
        assert_eq!(scan(input.to_string()), vec![output]);
    }

    #[test]
    fn one_ident() {
        test("blah", vec![Ident("blah".to_string())]);
    }

    #[test]
    fn nothing() {
        test("", vec![]);
    }

    #[test]
    fn var_decl() {
        test(
            "int x = 5;",
            vec![
                Key(Int),
                Ident("x".to_string()),
                Sym(Assign(Eq)),
                DecIntLit("5".to_string()),
                Sym(Misc(Semicolon)),
            ],
        );
    }

    #[test]
    fn no() {}
}
