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
    // not statically checked for completeness :(
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
    fn token_of_word(word: &str) {
	
    }
}

fn substring_before(input: &String, test: fn(char) -> bool) -> (&str, &str) {
    for (i, c) in input.chars().enumerate() {
        if test(c) {
            return (&input[..i], &input[i..]);
        }
    }
    return (&input[..], &input[input.len()..]);
}

fn scan(input: String) {
    let mut tokens = Vec::new();
    let mut input = input.chars();
    if let Some(fst) = input.next() {
        if fst.is_ascii_alphabetic() {
	    let (rest_of_word, input) = 
	}
    }
}
