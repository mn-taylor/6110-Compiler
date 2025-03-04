use crate::ir;
use crate::parse;
use crate::scan;
use ir::*;
use parse::WithLoc;
use parse::{Field, Ident, Literal, Param, Primitive};
use scan::AssignOp;
use std::iter::zip;

pub fn check_program(program: &Program) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    // check for duplicates within fields and method name and imports
    // TODO: for array fields, check that the length is a valid int.
    let import_ids = program
        .imports
        .iter()
        .map(|c| (c, format!("External function {}", c.name)));
    let method_ids = program
        .methods
        .iter()
        .map(|method| (&method.name, format!("Method {}", method.name.name)));
    let field_ids = program.fields.iter().map(Field::describe);

    check_duplicates(
        import_ids
            .chain(method_ids)
            .chain(field_ids)
            .collect::<Vec<_>>(),
        &mut errors,
    );

    // check main exists

    let mut has_void_main = false;
    for (i, method) in program.methods.iter().enumerate() {
        if method.name.name == "main".to_string() && method.meth_type == None {
            has_void_main = true;
        }

        check_method(&method, &mut errors, &program.scope_with_first_n_methods(i))
    }

    if !has_void_main {
        let error_message = "Main method not defined";
        errors.push(error_message.to_string());
    }

    errors
}

fn check_method(method: &Method, errors: &mut Vec<String>, scope: &Scope) {
    let descriptions = method
        .params
        .iter()
        .map(Param::describe)
        .chain(method.fields.iter().map(Field::describe));
    check_duplicates(descriptions.collect(), errors);
    let method_scope = method.scope(scope);
    for stmt in method.stmts.iter() {
        check_stmt(&stmt, errors, &method_scope, false, &method.meth_type);
    }
}

trait Describe {
    fn describe(&self) -> (&Ident, String);
}

impl Describe for Param {
    fn describe(&self) -> (&Ident, String) {
        (&self.name, format!("parameter {}", self.name.name))
    }
}

impl Describe for Field {
    fn describe(&self) -> (&Ident, String) {
        match self {
            Field::Scalar(_, id) | Field::Array(_, id, _) => (id, format!("field {}", id.name)),
        }
    }
}

use std::collections::HashMap;
fn check_duplicates(ids: Vec<(&Ident, String)>, errors: &mut Vec<String>) {
    let mut ident_to_descr: HashMap<&String, String> = HashMap::new();
    for (id, descr) in ids {
        match ident_to_descr.get(&id.name) {
            Some(other_descr) => {
                errors.push(format!(
                    "Duplicate identifier: {descr} conflicts with {other_descr}"
                ));
            }
            None => {
                ident_to_descr.insert(&id.name, descr);
            }
        }
    }
}

fn check_location(
    location: &Location,
    errors: &mut Vec<String>,
    scope: &Scope,
) -> Option<Primitive> {
    let id = match location {
        Location::Var(id) => id,
        Location::ArrayIndex(id, _) => id,
    };
    match scope.lookup(&id.val) {
        None => {
            errors.push(format!(
                "Could not find identifier {:?} at {:?}",
                id.val, id.loc
            ));
            None
        }
        Some(Type::Prim(p)) => Some(p.clone()),
        Some(x) => {
            errors.push(format!("Expected primitive, got {:?}", x));
            None
        }
    }
}

fn check_types(
    expected_type: &[&Primitive],
    actual_type: &Option<Primitive>,
    errors: &mut Vec<String>,
) -> bool {
    if let Some(actual_type) = actual_type {
        if !expected_type
            .iter()
            .any(|exp_type| *exp_type == actual_type)
        {
            let expected_str = expected_type
                .iter()
                .map(|t| format!("{:?}", t))
                .collect::<Vec<_>>()
                .join(" or ");
            errors.push(format!(
                "Mismatched Types: Expected {:?}, got {:?}",
                expected_str, actual_type
            ));
            return false;
        }
    }
    return true;
}

fn check_literal(
    literal_expr: &Literal,
    negated: bool,
    errors: &mut Vec<String>,
) -> Option<Primitive> {
    let maybe_negate = |s: &String| {
        if negated {
            format!("-{}", s)
        } else {
            s.to_string()
        }
    };
    match literal_expr {
        Literal::DecInt(s) => {
            let s = maybe_negate(s);
            if s.parse::<i32>().is_err() {
                let error_message = format!("Integer out of bounds, got {s}");
                errors.push(error_message);
                None
            } else {
                Some(Primitive::IntType)
            }
        }
        Literal::HexInt(s) => {
            let s = maybe_negate(&format!("0x{}", s));
            if s.parse::<i32>().is_err() {
                let error_message = format!("Integer out of bounds, got 0x{s}");
                errors.push(error_message);
                None
            } else {
                Some(Primitive::IntType)
            }
        }
        Literal::DecLong(s) => {
            let s = maybe_negate(s);
            if s.parse::<i64>().is_err() {
                let error_message = format!("Long out of bounds, got {s}");
                errors.push(error_message);
                None
            } else {
                Some(Primitive::LongType)
            }
        }
        Literal::HexLong(s) => {
            let s = maybe_negate(&format!("0x{}", s));
            if s.parse::<i64>().is_err() {
                let error_message = format!("Long out of bounds, got 0x{s}");
                errors.push(error_message);
                None
            } else {
                Some(Primitive::LongType)
            }
        }
        Literal::Bool(_) => Some(Primitive::BoolType),
        Literal::Char(_) => Some(Primitive::IntType),
    }
}

// want to factor this out so as not to implement it for both exprs and stmts
fn check_call(
    id: &WithLoc<Ident>,
    args: &Vec<Arg>,
    errors: &mut Vec<String>,
    scope: &Scope,
) -> Option<Primitive> {
    match scope.lookup(&id.val) {
        Some(Type::Func(params, return_type)) => {
            if args.len() != params.len() {
                errors.push(format!(
                    "{:?} expected {} many arguments and got {}",
                    id.val,
                    params.len(),
                    args.len()
                ));
            }
            for (arg, param_type) in zip(args, params) {
                let arg_type = check_arg(arg, errors, scope);
                check_types(&[&param_type], &arg_type, errors);
            }
            return return_type.clone();
        }
        Some(Type::ExtCall) => {
            for arg in args {
                check_arg(arg, errors, &scope);
            }
            return Some(Primitive::IntType);
        }
        Some(x) => {
            errors.push(format!("{:?} should have been function, was {:?}", id, x));
            None
        }
        None => {
            errors.push(format!("{:?} not found", id));
            None
        }
    }
}

fn expect_val<T: PartialEq>(
    expected: T,
    err_msg: impl FnOnce(T) -> String,
) -> impl FnOnce(T) -> Result<(), String> {
    move |got| {
        if got == expected {
            Ok(())
        } else {
            Err(err_msg(got))
        }
    }
}

fn assert_eq<T: PartialEq>(expected: T, got: T, errors: &mut Vec<String>, err_msg: String) {
    let _ = expect_val(expected, move |_| err_msg)(got).map_err(|err| errors.push(err));
}

fn check_stmt(
    stmt: &Stmt,
    errors: &mut Vec<String>,
    scope: &Scope,
    in_loop: bool,
    return_type: &Option<Primitive>,
) {
    match stmt {
        Stmt::AssignStmt(loc, assign_expr) => {
            let left_type = check_location(loc, errors, scope);
            check_assign_expr(assign_expr, errors, scope, left_type);
        }
        Stmt::Call(id, args) => {
            check_call(id, args, errors, scope);
        }
        Stmt::If(condition, if_block, else_block) => {
            let cond_type = check_expr(condition, errors, scope);
            check_types(&[&Primitive::BoolType], &cond_type, errors);
            check_block(if_block, errors, scope, in_loop, return_type);
            if let Some(else_block) = else_block {
                check_block(else_block, errors, scope, in_loop, return_type);
            }
        }
        Stmt::For {
            var_to_set,
            initial_val,
            test,
            var_to_update,
            update_val,
            body,
        } => {
            // var_to_set is int
            let var_to_set_type = scope.lookup(&var_to_set.val);
            assert_eq(
                Some(&Type::Prim(Primitive::IntType)),
                var_to_set_type,
                errors,
                format!(
                    "Loop variable must be declared as int, not {:?}",
                    var_to_set_type
                ),
            );
            // initial_val is int
            check_regular_assign(
                &AssignOp::Eq,
                initial_val,
                errors,
                scope,
                Some(Primitive::IntType),
            );

            // test is bool
            let cond_type = check_expr(test, errors, scope);
            check_types(&[&Primitive::BoolType], &cond_type, errors);
            // var_to_update
            let left_type = check_location(var_to_update, errors, scope);
            check_assign_expr(update_val, errors, scope, left_type);

            // body
            check_block(body, errors, scope, true, return_type);
        }
        Stmt::While(condition, body) => {
            let cond_type = check_expr(condition, errors, scope);
            check_types(&[&Primitive::BoolType], &cond_type, errors);

            check_block(body, errors, scope, true, return_type);
        }
        Stmt::Return(return_val) => {
            if let Some(return_val) = return_val {
                if let None = return_type {
                    errors.push(format!("Function should not return a value"));
                }
                let return_val_type = check_expr(return_val, errors, scope);
                if let Some(return_type) = return_type {
                    check_types(&[return_type], &return_val_type, errors);
                }
            }
        }
        Stmt::Break => {
            if !in_loop {
                let error_message = "Break statement only allowed inside of loops".to_string();
                errors.push(error_message);
            }
        }
        Stmt::Continue => {
            if !in_loop {
                let error_message = "Continue statement only allowed inside of loops".to_string();
                errors.push(error_message);
            }
        }
    }
}

fn check_block(
    block: &Block,
    errors: &mut Vec<String>,
    scope: &Scope,
    in_loop: bool,
    return_type: &Option<Primitive>,
) {
    let block_scope = block.scope(scope);

    for stmt in block.stmts.iter() {
        check_stmt(stmt, errors, &block_scope, in_loop, return_type);
    }
}
fn guess_right_type(left_type_option: Option<Primitive>, bop: &Bop) -> Option<Primitive> {
    if let Some(left_type) = left_type_option {
        match left_type {
            Primitive::IntType => {
                if *bop != Bop::Or && *bop != Bop::And {
                    return Some(Primitive::IntType);
                } else {
                    return None;
                }
            }
            Primitive::LongType => {
                if *bop != Bop::Or && *bop != Bop::And {
                    return Some(Primitive::LongType);
                } else {
                    return None;
                }
            }
            Primitive::BoolType => {
                if *bop == Bop::And || *bop == Bop::Or {
                    return Some(Primitive::BoolType);
                } else {
                    return None;
                }
            }
        }
    } else {
        return None;
    }
}

fn convert_bop_to_primitive(bop: &Bop, right_type: Primitive) -> Primitive {
    match bop {
        Bop::MulBop(_) => right_type,
        Bop::AddBop(_) => right_type,
        _ => Primitive::BoolType,
    }
}

fn check_expr(expr: &Expr, errors: &mut Vec<String>, scope: &Scope) -> Option<Primitive> {
    match expr {
        Expr::Bin(left_expr, bop, right_expr) => {
            // check left expression
            let left_type = check_expr(left_expr, errors, scope);
            let potential_right_type = guess_right_type(left_type, bop);
            if let Some(expected_right_type) = potential_right_type {
                let right_type = check_expr(right_expr, errors, scope);
                if check_types(&[&expected_right_type], &right_type, errors) {
                    return Some(convert_bop_to_primitive(bop, expected_right_type));
                }
            }
            None
        }
        Expr::Unary(op, expr) => {
            if matches!(op, UnOp::Neg) {
                if let Expr::Lit(lit_with_loc) = expr.as_ref() {
                    return check_literal(&lit_with_loc.val, true, errors);
                }
            }
            let expr_type = check_expr(expr, errors, scope);
            match op {
                UnOp::Neg | UnOp::IntCast | UnOp::LongCast => {
                    check_types(
                        &[&Primitive::IntType, &Primitive::LongType],
                        &expr_type,
                        errors,
                    );
                    return expr_type; // if type isn't int or long, this is gonna be wrong but doesnt matter since thats an error anyway
                }
                UnOp::Not => {
                    if check_types(&[&Primitive::BoolType], &expr_type, errors) {
                        return Some(Primitive::BoolType);
                    } else {
                        None
                    }
                }
            }
        }
        Expr::Len(with_loc) => {
            //  check that arg of len is an array
            match scope.lookup(&with_loc.val) {
                Some(Type::Arr(prim_type)) => return Some(prim_type.clone()),
                Some(_) => {
                    let error_message =
                        format!("Identifier {} not defined as an array", with_loc.val.name);
                    errors.push(error_message);
                    None
                }
                None => {
                    let error_message =
                        format!("Identifier {} has not been defined", with_loc.val.name);
                    errors.push(error_message);
                    None
                }
            }
        }
        Expr::Lit(with_loc) => check_literal(&with_loc.val, false, errors),
        Expr::Loc(loc) => check_location(loc, errors, scope),
        Expr::Call(with_loc, args) => check_call(with_loc, args, errors, scope),
    }
}

fn check_arg(arg: &Arg, errors: &mut Vec<String>, scope: &Scope) -> Option<Primitive> {
    match arg {
        Arg::ExprArg(expr) => check_expr(expr, errors, scope),
        Arg::ExternArg(_) => None,
    }
}

fn check_regular_assign(
    op: &AssignOp,
    rhs: &Expr,
    errors: &mut Vec<String>,
    scope: &Scope,
    lhs_type: Option<Primitive>,
) {
    if let Some(t) = lhs_type {
        if op.is_arith() && !(t == Primitive::IntType || t == Primitive::LongType) {
            errors.push(format!(
                "Assignment operator {} incompatible with type {:?}",
                op, t
            ));
        }
    }
    check_expr(rhs, errors, scope);
}

// left_type is Option<Type> because if left side failed to typecheck, we still want to sanity-check rhs but do not want to complain about its type
fn check_assign_expr(
    assign_expr: &AssignExpr,
    errors: &mut Vec<String>,
    scope: &Scope,
    lhs_type: Option<Primitive>,
) {
    match assign_expr {
        AssignExpr::RegularAssign(op, rhs) => {
            check_regular_assign(op, rhs, errors, scope, lhs_type)
        }
        AssignExpr::IncrAssign(op) => match lhs_type {
            Some(Primitive::IntType) | Some(Primitive::LongType) | None => (),
            Some(x) => errors.push(format!(
                "Did not expect increment operator {:?} to be applied to type {:?}",
                op, x
            )),
        },
    }
}
