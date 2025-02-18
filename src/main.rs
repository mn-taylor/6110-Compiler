mod utils;
use decaf_skeleton_rust::scan;

fn get_writer(output: &Option<std::path::PathBuf>) -> Box<dyn std::io::Write> {
    match output {
        Some(path) => Box::new(std::fs::File::create(path.as_path()).unwrap()),
        None => Box::new(std::io::stdout()),
    }
}

fn print_tokens(tokens: Vec<Vec<scan::Token>>) {
    for (line_num, tokens_in_line) in tokens.iter().enumerate() {
        for token in tokens_in_line {
            println!(
                "{} {}",
                line_num + 1,
                scan::Token::format_for_output(&token)
            );
        }
    }
}

fn main() {
    let args = utils::cli::parse();
    let input = std::fs::read_to_string(&args.input).expect("Filename is incorrect.");

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
            print_tokens(scan::scan(input));
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
