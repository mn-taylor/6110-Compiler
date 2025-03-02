use std::iter::zip;
use crate::ir;


fn check_program(program: ir::Program) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    // check for duplicates within fields and method name and imports

    // check main exists

    for (i, method) in program.methods.into_iter().enumerate() {
        check_method(methods, &errors, program_scope(program, i));
    }

    errors
}

fn program_scope(program: ir::Program, i: usize) -> impl Fn(Ident) -> Option<Type> {
    if let Some(t) = fields_lookup(&self.fields, id) {
        Some(t)
    } else if let Some(t) = methods_lookup(&program.methods[0..i], id) {
        Some(t)
    } else {
        self.imports
            .iter()
            .find(|other| *other == id)
            .map(|_| ExtCall)
    }
}

fn check_method(method: ir::Method, errors: &Vec<String>, scope: ir::Scope) {
    descriptions = preprocess(method.params).extend(preprocess(method.fields));
    check_duplicates(descriptions, &errors);
    for stmt in method.stmts {
        check_stmt(stmt, &errors, method.scope(), false, method.meth_type);
    }
    errors
}

fn check_duplicates(ids: Vec<(Ident, String)>, errors: &Vec<String>) {}

fn check_stmt(stmt: ir::Stmt, errors: &Vec<String>, scope: impl Fn(Ident) -> Option<Type>, in_loop: bool, return_type: Option<Primitive>) {
    match stmt {
        Stmt::AssignStmt(loc, assign_expr) => {
            let left_type = check_location(loc, &errors);
            let right_type = check_assign_expr(assign_expr, left_type, &errors);
        }
        Stmt::Call(loc_info, args) => {
            match scope(loc_info.val) {
                Some(ir::Type::Func(params, _)) => {
                    if args.len() != params.len(){
                        let error_message = format!("{} expected {} many arguments and got {}", loc_info, params.len(), args.len());
                    }
                    let arg_param_types = zip(args.iter().map(|arg| check_arg(arg, &errors)), params);
                    for (arg_type, param_type) in arg_param_types{
                        if arg_type != param_type{
                            let error_message = format!("Mismatched types, function definition expected argument of type {} and got {}", param_type, arg_type);
                            errors.push(error_message)
                        }
                    }

                }
                Some(ir::Type::ExtCall) => {args.map(|arg| check_arg(arg, &errors))}
            }
        }
        Stmt::If(condition, if_block, else_block)=>{
            if ir::Type(BoolType) != check_expr(condtion, &errors){
                let error_message = format!("if condition must be boolean expression");
                errors.push(error_message);
            }

            check_block(if_block, &errors, scope);
            else_block.map(|block| check_block(block, &errors,scope));
        }
        Stmt::For{
            var_to_set: var_to_set,
            initial_val: intial_val,
            test: test,
            var_to_update: var_to_update,
            update_val: update_val,
            body: body,
        } => {
            // var_to_set is int
            if ir::Type::Primitive(IntType) != scope(var_to_set){
                let error_message = format!("loop variable must be declared as int");
                errors.push(error_message);
            }

            // initial_val is int
            let initial_val_type = check_expr(inital_val, &errors, scope);
            if ir::Type::Primitive(IntType) != initial_val_type{
                let error_message = format!("loop variable must be initalized as an int, got {} type", inital_val_type);
                errors.push(error_message);
            }

            // test is bool
            let condition_type = check_expr(test, &errors, scope)
            if ir::Type::Primitive(Bool) != condition_type {
                let error_message = format!("loop condition must be of type bool, got {}", condition_type);
                errors.push(error_message);
            }

            // var_to_update
            let loop_update = ir::Stmt::AssignStmt(var_to_update, update_val);
            check_stmt(loop_update, &errors, scope,true, return_type);// checks that the types are the same and variables have been defined.

            // check block
            check_block(body, &errors, scope, true, return);
        }
        ir::Stmt::While(condition, body)=>{
            let condition_type = check_expr(condition, &errors, scope);
            if ir::Type::Primitive(Bool) != condition_type {
                let error_message = format!("loop condition must be of type bool, got {}", condition_type);
                errors.push(error_message);
            } 

            check_block(body, &errors, scope, true, return_type);
        }
        ir::Stmt::Return(some_return_val)=>{
            if let Some(return_val) = some_return_val{
                let return_val_type = check_expr(return_val, &errors, scope);
                if return_val_type != return_type {
                    let error_message = format!("Method should return type {}, got {}", return_type, return_val_type);
                    errors.push(error_message);
                }
            }else{
                if return_type != None{
                    let error_message = format!("Method should return {}, got None", return_type);
                }
            }
        }
        ir::Break => {
            if !in_loop {
                let error_message = "Break statement only allowed inside of loops".to_string();
                errors.push(error_message);
            }
        }
        ir::Continue => {
            if !in_loop {
                let error_message = "Continue statement only allowed inside of loops".to_string();
                errors.push(error_message);
            }
        }


    }
}

//
fn check_block(block: ir::Block, errors: &Vec<String>, scope: impl Fn(Ident) -> Option<Type>, in_loop:bool, return_type: Option<Primitive>) -> {
   let block_scope = block.lookup(parent_scope);

   for stmt in block.stmts{
        check_stmt(stmt, &errors, block_scope, in_loop, return_type);
   }
}

fn check_expr(expr: ir::Expr, errors: &Vec<String>, scope: impl Fn(Ident) -> Option<Type>) -> {}

// Should call check expr, if argument is none then should report an error message
fn check_arg(arg: ir::Arg, error: &Vec<String>)->Sum<Option<Type>, String>{

}

fn check_assign_expr(assign_expr: ir::AssignExpr, left_type: Type, errors: &Vec<String>) {}
