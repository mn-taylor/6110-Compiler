mod add;
mod utils;
const FORM_FEED: char = 12u8 as char;

fn get_writer(output: &Option<std::path::PathBuf>) -> Box<dyn std::io::Write> {
    match output {
        Some(path) => Box::new(std::fs::File::create(path.as_path()).unwrap()),
        None => Box::new(std::io::stdout()),
    }
}

fn main() {
    let args = utils::cli::parse();
    let _input = std::fs::read_to_string(&args.input).expect("Filename is incorrect.");

    if args.debug {
        eprintln!(
            "Filename: {:?}\nDebug: {:?}\nOptimizations: {:?}\nOutput File: {:?}\nTarget: {:?}",
            args.input, args.debug, args.opt, args.output, args.target
        );
    }

    // Use writeln!(writer, "template string") to write to stdout ot file.
    let _writer = get_writer(&args.output);
    match args.target {
        utils::cli::CompilerAction::Default => {
            panic!("Invalid target");
        }
        utils::cli::CompilerAction::Scan => {
            todo!("scan");
        }
        utils::cli::CompilerAction::Parse => {
            todo!("parse");
        }
        utils::cli::CompilerAction::Inter => {
            todo!("inter");
        }
        utils::cli::CompilerAction::Assembly => {
            todo!("assembly");
        }
    }
}

enum Keyword {
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

enum Symbol {
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

enum Token {
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

fn substring_before(input: &mut Peekable<Chars>, test: fn(&char) -> bool) -> String {
    let mut s = String::new();
    loop {
        match input.peek() {
            Some(c) => {
                if test(c) {
                    s.push(*c)
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
    panic!()
}

fn scan_hex_lit(input: &mut Peekable<Chars>) -> String {
    panic!()
}

fn scan_comment(input: &mut Peekable<Chars>) {
    panic!()
}

fn scan_block_comment(input: &mut Peekable<Chars>) {
    panic!()
}

fn is_alpha(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_alphanum(c: char) -> bool {
    is_alpha(c) || c.is_ascii_digit()
}

fn scan(input: String) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut input = input.chars().peekable();
    let mut input_clone = input.clone();
    if let Some(fst) = input.peek() {
        if fst.is_ascii_alphabetic() {
            let word = substring_before(&mut input, |x| !is_alphanum(*x));
            tokens.push(Token::of_ident_or_keyword(word));
        } else if *fst == '0' {
            if input_clone.nth(1) == Some('x') {
                tokens.push(Token::HexLit(scan_hex_lit(&mut input)))
            } else {
                tokens.push(Token::DecLit(scan_dec_lit(&mut input)))
            }
        } else if *fst == '/' {
            if input_clone.nth(1) == Some('/') {
                scan_comment(&mut input);
            } else {
                scan_block_comment(&mut input);
            }
        } else if fst.is_ascii_whitespace() {
            input.next();
        } else {
            let mut sym = None;
            // first try to parse two-character symbol
            if let Some(snd) = input_clone.nth(1) {
                sym = Symbol::of_string(&[*fst, snd].iter().collect::<String>())
            }
            // if that didn't work, try to parse one-character symbol
            if let None = sym {
                sym = Symbol::of_string(&fst.to_string());
            }
            tokens.push(Token::Sym(sym.unwrap()));
        }
    }
    tokens
}
