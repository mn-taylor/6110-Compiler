const FORM_FEED: char = 12u8 as char;

#[derive(Debug, PartialEq, Eq)]
pub enum Keyword {
    Import,
    Void,
    Int,
    Long,
    Bool,
    If,
    For,
    While,
    Return,
    Break,
    Continue,
    True,
    False,
    L,
}

use Keyword::*;

impl Keyword {
    // not statically checked for completeness :( keep me updated please
    fn all() -> Vec<Keyword> {
        vec![
            Import, Void, Int, Long, Bool, If, For, While, Return, Break, Continue, True, False, L,
        ]
    }

    fn to_string(&self) -> &str {
        match self {
            Import => "import",
            Void => "void",
            Int => "int",
            Long => "long",
            Bool => "bool",
            If => "if",
            For => "for",
            While => "while",
            Return => "return",
            Break => "break",
            Continue => "continue",
            True => "true",
            False => "false",
            L => "L",
        }
    }

    fn of_string(word: &str) -> Option<Keyword> {
        for val in Self::all() {
            if val.to_string() == word {
                return Some(val);
            }
        }
        return None;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Symbol {
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    Lt,
    Gt,
    Leq,
    Geq,
    Eq,
    Neq,
    And,
    Or,
    Not,
    LPar,
    RPar,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Assign,
    PlusAssign,
    MinusAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    PlusPlus,
    MinusMinus,
    Semicolon,
    Comma,
}

use Symbol::*;

impl Symbol {
    // not statically checked for completeness :( keep me updated please.
    fn all() -> Vec<Symbol> {
        vec![
            Plus,
            Minus,
            Mul,
            Div,
            Mod,
            Lt,
            Gt,
            Leq,
            Geq,
            Eq,
            Neq,
            And,
            Or,
            Not,
            LPar,
            RPar,
            LBrack,
            RBrack,
            LBrace,
            RBrace,
            Assign,
            PlusAssign,
            MinusAssign,
            MulAssign,
            DivAssign,
            ModAssign,
            PlusPlus,
            MinusMinus,
            Semicolon,
            Comma,
        ]
    }

    fn to_string(&self) -> &str {
        match self {
            Plus => "+",
            Minus => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Lt => "<",
            Gt => ">",
            Leq => "<=",
            Geq => ">=",
            Eq => "==",
            Neq => "!=",
            And => "&&",
            Or => "||",
            Not => "!",
            LPar => "(",
            RPar => ")",
            LBrack => "[",
            RBrack => "]",
            LBrace => "{",
            RBrace => "}",
            Assign => "=",
            PlusAssign => "+=",
            MinusAssign => "-=",
            MulAssign => "*=",
            DivAssign => "/=",
            ModAssign => "%=",
            PlusPlus => "++",
            MinusMinus => "--",
            Semicolon => ";",
            Comma => ",",
        }
    }

    fn of_string(word: &str) -> Option<Symbol> {
        for val in Self::all() {
            if val.to_string() == word {
                return Some(val);
            }
        }
        None
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Token {
    Key(Keyword),
    Ident(String),
    DecLit(String),
    HexLit(String),
    Sym(Symbol),
}

impl Token {
    fn of_ident_or_keyword(word: String) -> Self {
        match Keyword::of_string(&word) {
            Some(k) => Self::Key(k),
            None => Self::Ident(word),
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

pub fn scan(input: String) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut counter = 0;
    let mut scanning_block_comment = false;
    for line in input.lines() {
        let mut line = line.chars().peekable();
        loop {
            if scanning_block_comment {
                scanning_block_comment = scan_until_block_comment_end(&mut line);
                continue;
            }
            let mut line_clone = line.clone();
            match line_clone.next() {
                None => break,
                Some(fst) => {
                    println!("fst: {:?}", fst);

                    // lex ident or keyword
                    if is_alpha(fst) {
                        tokens.push(Token::of_ident_or_keyword(scan_ident_or_keyword(&mut line)));
                    }
                    // lex int literal
                    else if fst.is_ascii_digit() {
                        if fst == '0' && line_clone.next() == Some('x') {
                            tokens.push(Token::HexLit(scan_hex_lit(&mut line)))
                        } else {
                            tokens.push(Token::DecLit(scan_dec_lit(&mut line)))
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
                    counter += 1;
                    if counter >= 10 {
                        break;
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

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;

    fn test(input: &str, output: Vec<Token>) {
        assert_eq!(scan(input.to_string()), output);
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
                Sym(Assign),
                DecLit("5".to_string()),
                Sym(Semicolon),
            ],
        );
    }

    #[test]
    fn no() {}
}
