mod utils;

use decaf_skeleton_rust::asm;
use decaf_skeleton_rust::cfg_build;
use decaf_skeleton_rust::comsubelim;
use decaf_skeleton_rust::constprop;
use decaf_skeleton_rust::constprop::constant_propagation;
use decaf_skeleton_rust::deadcode;
use decaf_skeleton_rust::deadcode::dead_code_elimination;
use decaf_skeleton_rust::ir_build;
use decaf_skeleton_rust::metrics::num_intstructions;
use decaf_skeleton_rust::parse;
use decaf_skeleton_rust::scan;
use decaf_skeleton_rust::semantics;
use decaf_skeleton_rust::ssa_construct;
use decaf_skeleton_rust::ssa_destruct;

use decaf_skeleton_rust::copyprop;
use utils::cli::Optimization;

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
            args.input,
            args.debug,
            args.get_opts(),
            args.output,
            args.target
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
            println!("**********************************************************");

            let ast;
            println!("\n***************** parsing error messages: *****************");
            match parse::parse_program_with_error_info(tokens.iter()) {
                Ok(prog) => ast = prog,
                Err((prog, (last_attempted_to_parse_token, loc))) => {
                    ast = prog;
                    println!(
                        "parser failed near token `{}` at {}",
                        last_attempted_to_parse_token.format_for_output(),
                        loc
                    );
                }
            }
            println!("***********************************************************");

            let prog = ir_build::build_program(ast);
            if args.debug {
                // println!("{:?}", prog);
            }
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
            let mut p = cfg_build::lin_program(&prog);

            p.methods = p
                .methods
                .iter_mut()
                .map(|c| {
                    let mut metrics_string = vec![];
                    if args.debug {
                        println!("looking at method {}", c.name);
                        println!("method before ssa: \n{}", c);
                        metrics_string.push(format!(
                            " before ssa |  num_instructions: {}",
                            num_intstructions(c)
                        ));
                    }
                    let mut ssa_method = ssa_construct::construct(c);
                    if args.debug {
                        println!("method after ssa construction: \n{}", ssa_method);
                        metrics_string.push(format!(
                            "after ssa construction | num_instructions: {}",
                            num_intstructions(&ssa_method)
                        ));
                    }
                    if args.get_opts().contains(&Optimization::Dce) {
                        ssa_method = deadcode::dead_code_elimination(&mut ssa_method);
                    }
                    if args.debug && args.get_opts().contains(&Optimization::Dce) {
                        // println!("method after dead code elimination: \n{}", ssa_method);
                        metrics_string.push(format!(
                            "after dead code elimination | num_instructions: {}",
                            num_intstructions(&ssa_method)
                        ));
                    }

                    if args.get_opts().contains(&Optimization::Cp) {
                        ssa_method = copyprop::copy_propagation(&mut ssa_method);
                        ssa_method = deadcode::dead_code_elimination(&mut ssa_method);
                    }
                    if args.debug && args.get_opts().contains(&Optimization::Cp) {
                        // println!("method after copy propagation: \n{}", ssa_method);
                        metrics_string.push(format!(
                            "after copy propagation | num_instructions: {}",
                            num_intstructions(&ssa_method)
                        ));
                    }

                    if args.get_opts().contains(&Optimization::Cop) {
                        // ssa_method = constprop::constant_propagation(&mut ssa_method);
                        ssa_method = deadcode::dead_code_elimination(&mut ssa_method);
                    }
                    if args.debug && args.get_opts().contains(&Optimization::Cop) {
                        // println!("method after constant propagation: \n{}", ssa_method);
                        metrics_string.push(format!(
                            "after constant propagation | num_instructions: {}",
                            num_intstructions(&ssa_method)
                        ));
                    }

                    if args.get_opts().contains(&Optimization::Cse) {
                        ssa_method = comsubelim::eliminate_common_subexpressions(&mut ssa_method);
                        ssa_method = copyprop::copy_propagation(&mut ssa_method);
                        ssa_method = deadcode::dead_code_elimination(&mut ssa_method);
                    }
                    if args.debug && args.get_opts().contains(&Optimization::Cse) {
                        println!("method after CSE: \n{}", ssa_method);
                        metrics_string.push(format!(
                            "after CSE propagation | num_instructions: {}",
                            num_intstructions(&ssa_method)
                        ));
                    }

                    ssa_destruct::split_crit_edges(&mut ssa_method);

                    if args.debug {
                        // println!("method after splitting edges: \n{ssa_method}");
                    }
                    let mut result = ssa_destruct::destruct(&mut ssa_method);
                    ssa_destruct::seq_method(&mut result);
                    if args.debug {
                        println!("method after ssa destruction: \n{}", result);
                        metrics_string.push(format!(
                            "after de-ssa | num_instructions: {}",
                            num_intstructions(&result)
                        ));
                    }

                    if args.debug {
                        println!("Optimization Metrics for {}", c.name);
                        metrics_string.iter().for_each(|c| println!("{}", c));
                    }

                    result
                })
                .collect::<Vec<_>>();

            // if args.debug {
            //     println!("{}", p);
            // }

            for l in asm::asm_program(&p, args.mac) {
                writeln!(writer, "{}", l).unwrap();
            }
        }
    }
}
