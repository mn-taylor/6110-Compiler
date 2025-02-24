mod utils;
use decaf_skeleton_rust::parse;
use decaf_skeleton_rust::scan;

fn get_writer(output: &Option<std::path::PathBuf>) -> Box<dyn std::io::Write> {
    match output {
        Some(path) => Box::new(std::fs::File::create(path.as_path()).unwrap()),
        None => Box::new(std::io::stdout()),
    }
}

fn write_tokens(
    mut writer: Box<dyn std::io::Write>,
    tokens: &Vec<(Result<scan::Token, String>, scan::ErrLoc)>,
) {
    for token in tokens {
        match token {
            (Ok(token), e) => writeln!(
                writer,
                "{} {}",
                e.line,
                scan::Token::format_for_output(&token)
            ),
            (Err(err), e) => writeln!(writer, "ERROR on line {}: {}", e.line, err),
        }
        .unwrap();
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
    let writer = get_writer(&args.output);
    match args.target {
        utils::cli::CompilerAction::Default => {
            panic!("Invalid target");
        }
        utils::cli::CompilerAction::Scan => {
            let tokens = scan::scan(input);
            write_tokens(writer, &tokens);
            tokens.iter().for_each(|x| match x {
                (Ok(_), _) => (),
                (Err(err), _) => panic!("{}", err),
            });
        }
        utils::cli::CompilerAction::Parse => {
            let tokens = scan::scan(input);
            let mut tokens = tokens.iter().map(|(x, _)| match x {
                Ok(x) => x,
                Err(_) => panic!("oops couldnt scan"),
            });
            // println!("{:?}", tokens.clone().collect::<Vec<_>>());
            parse::parse_program(&mut tokens);
            if tokens.collect::<Vec<_>>().len() > 0 {
                panic!("oops didnt parse everythign");
            }
        }
        utils::cli::CompilerAction::Inter => {
            todo!("inter");
        }
        utils::cli::CompilerAction::Assembly => {
            todo!("assembly");
        }
    }
}
