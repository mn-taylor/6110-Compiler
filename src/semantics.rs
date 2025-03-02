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
        check_stmt(stmt, &errors, method.scope());
    }
    errors
}

fn check_duplicates(ids: Vec<(Ident, String)>, errors: &Vec<String>) {}

fn check_stmt(stmt: ir::Stmt, errors: &Vec<String>, scope: impl Fn(Ident) -> Option<Type>) {
    match stmt {
        Stmt::AssignStmt(loc, assign_expr) => {
            let left_type = check_location(loc, &errors);
            let right_type = check_assign_expr(assign_expr, left_type, &errors);
        }
        Stmt::Call(loc_info, args) => {
            match scope(loc_info.val) {
                Some(Type::Func(params, _)) => params,
                Some(Type::ExtCall) => ,
                _=>
            }
        }
    }
}

fn check_assign_expr(assign_expr: ir::AssignExpr, left_type: Type, errors: &Vec<String>) {}
