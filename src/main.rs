mod utils;
use decaf_skeleton_rust::parse;
use decaf_skeleton_rust::scan;
use decaf_skeleton_rust::ir_build;

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
            (Err(err), e) => writeln!(writer, "ERROR on line {} col {}: {}", e.line, e.col, err),
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
            let tokens = tokens
                .into_iter()
                .map(|x| match x {
                    (Ok(x), e) => (x, e),
                    (Err(_), _) => panic!("oops couldnt scan"),
                })
                .collect::<Vec<_>>();
            let mut tokens = tokens.iter();
            // println!("{:?}", tokens.clone().collect::<Vec<_>>());
            parse::parse_program(&mut tokens);
            if let Some(_) = tokens.next() {
                panic!("oops didnt parse everythign");
            }
        }
        utils::cli::CompilerAction::Inter => {
            let tokens = scan::scan(input);
            let tokens = tokens
                .into_iter()
                .map(|x| match x {
                    (Ok(x), e) => (x, e),
                    (Err(_), _) => panic!("oops couldnt scan"),
                })
                .collect::<Vec<_>>();
            let mut tokens = tokens.iter();
            // println!("{:?}", tokens.clone().collect::<Vec<_>>());
            let ast = parse::parse_program(&mut tokens);
            if let Some(_) = tokens.next() {
                panic!("oops didnt parse everythign");
            }
            ir_build::build_program(ast);
        }
        utils::cli::CompilerAction::Assembly => {
            todo!("assembly");
        }
    }
}
