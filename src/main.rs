mod add;
mod utils;

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

enum Token {
    Key(Keyword),
    Ident(String),
}

impl Token {
    fn of_string(word: String) -> Self {
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

fn scan(input: String) {
    let mut tokens = Vec::new();
    let mut input = input.chars().peekable();
    if let Some(fst) = input.peek() {
        if fst.is_ascii_alphabetic() {
            let word = substring_before(&mut input, |x| *x == ' ');
            tokens.push(Token::of_string(word));
        }
    }
}
