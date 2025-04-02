mod utils;
use decaf_skeleton_rust::asm;
use decaf_skeleton_rust::cfg_build;
use decaf_skeleton_rust::ir_build;
use decaf_skeleton_rust::parse;
use decaf_skeleton_rust::scan;
use decaf_skeleton_rust::semantics;

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
                scan::Token::format_for_output(token)
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
    let mut writer = get_writer(&args.output);
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
            if tokens.next().is_some() {
                panic!("oops didnt parse everythign");
            }
        }
        utils::cli::CompilerAction::Inter => {
            let tokens_result = scan::scan(input);
            let mut tokens: Vec<(scan::Token, scan::ErrLoc)> = Vec::new();
            println!("\n****************** scan error messages: ******************");
            for t in tokens_result {
                match t {
                    (Ok(x), e) => {
                        tokens.push((x, e));
                    }
                    (Err(msg), e) => {
                        println!("{}: {}", e, msg);
                    }
                }
            }
            println!("**********************************************************");

            let ast;
            println!("\n***************** parsing error messages: *****************");
            match parse::parse_program_with_error_info(tokens.iter()) {
                Ok(prog) => ast = prog,
                Err((prog, (last_attempted_to_parse_token, loc))) => {
                    ast = prog;
                    writeln!(
                        writer,
                        "parser failed near token `{}` at {}",
                        last_attempted_to_parse_token.format_for_output(),
                        loc
                    )
                    .unwrap();
                }
            }
            println!("***********************************************************");

            let prog = ir_build::build_program(ast);
            let checked_prog = semantics::check_program(&prog);
            // just marking the start and end of the error msgs bc it looks ugly in the terminal
            println!("\n****************** semantic error messages: ******************");
            checked_prog
                .iter()
                .for_each(|(loc, s)| println!("{}: {}", loc, s));
            println!("**************************************************************");
            if !checked_prog.is_empty() {
                panic!("your program has semantic errors");
            }
        }
        utils::cli::CompilerAction::Assembly => {
            let tokens_result = scan::scan(input);
            let mut tokens: Vec<(scan::Token, scan::ErrLoc)> = Vec::new();
            println!("\n****************** scan error messages: ******************");
            for t in tokens_result {
                match t {
                    (Ok(x), e) => {
                        tokens.push((x, e));
                    }
                    (Err(msg), e) => {
                        println!("{}: {}", e, msg);
                    }
                }
            }
            println!("************************************");
            let mut tokens = tokens.iter();

            let ast = parse::parse_program(&mut tokens);
            if tokens.next().is_some() {
                panic!("oops didn't parse everything");
            }

            let prog = ir_build::build_program(ast);
            let checked_prog = semantics::check_program(&prog);
            // just marking the start and end of the error msgs bc it looks ugly in the terminal
            println!("\n****************** semantic error messages: ******************");
            checked_prog
                .iter()
                .for_each(|(loc, s)| println!("{}: {}", loc, s));
            println!("************************************");
            if !checked_prog.is_empty() {
                panic!("your program has semantic errors");
            }
            let p = cfg_build::lin_program(&prog);
            for l in asm::asm_program(&p) {
                println!("{}", l);
            }
            // println!("{}", cfg_build::lin_program(&prog));
        }
    }
}
