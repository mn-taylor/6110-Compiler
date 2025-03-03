use crate::ir;
use crate::parse;
use crate::scan;
use ir::*;
use parse::WithLoc;
use parse::{Ident, Param, Primitive};
use scan::AssignOp;
use std::iter::zip;

// idk if we want this, but could be nice maybe
trait Scope: Fn(&Ident) -> Option<Type> {}
impl<T: Fn(&Ident) -> Option<Type>> Scope for T {}

pub fn check_program(program: &Program) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    // check for duplicates within fields and method name and imports

    // check main exists

    for (i, method) in program.methods.iter().enumerate() {
        check_method(
            &method,
            &mut errors,
            program.local_scope_with_first_n_methods(i),
        );
    }

    errors
}

fn check_method(method: &Method, errors: &mut Vec<String>, scope: impl Scope) {
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
                    "duplicate identifier: {descr} conflicts with {other_descr}"
                ));
            }
            None => {
                ident_to_descr.insert(&id.name, descr);
            }
        }
    }
}

// returns None if location invalid.  nothing to do with void.  locations cannot be void.
// i think using None to mean void was a bad idea.  it is confusing.  should have enum VoidOrType
fn check_location(
    location: &Location,
    errors: &mut Vec<String>,
    scope: impl Scope,
) -> Option<Primitive> {
    let id = match location {
        Location::Var(id) => id,
        Location::ArrayIndex(id, _) => id,
    };
    match scope(&id.val) {
        None => {
            errors.push(format!(
                "could not find identifier {:?} at {:?}",
                id.val, id.loc
            ));
            None
        }
        Some(Type::Prim(p)) => Some(p),
        Some(x) => {
            errors.push(format!("expected primitive, got {:?}", x));
            None
        }
    }
}

// want to factor this out so as not to implement it for both exprs and stmts
fn check_call(id: &WithLoc<Ident>, args: &Vec<Arg>, errors: &mut Vec<String>, scope: impl Scope) {
    match scope(&id.val) {
        Some(Type::Func(params, _)) => {
            if args.len() != params.len() {
                errors.push(format!(
                    "{:?} expected {} many arguments and got {}",
                    id.val,
                    params.len(),
                    args.len()
                ));
            }
            for (arg, param_type) in zip(args, params) {
                check_arg(arg, errors, |typ| {
                    if typ == param_type {
                        Ok(())
                    } else {
                        Err(format!(
                            "expected arg of type {:?}, got {:?}",
                            param_type, typ
                        ))
                    }
                });
            }
        }
        Some(Type::ExtCall) => {
            for arg in args {
                check_arg(arg, errors, |_| Ok(()));
            }
        }
        Some(x) => errors.push(format!("{:?} should have been function, was {:?}", id, x)),
        None => errors.push(format!("{:?} not found", id)),
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
    scope: impl Scope,
    in_loop: bool,
    return_type: &Option<Primitive>,
) {
    match stmt {
        Stmt::AssignStmt(loc, assign_expr) => {
            let left_type = check_location(loc, errors, &scope);
            check_assign_expr(assign_expr, errors, &scope, left_type);
        }
        Stmt::Call(id, args) => check_call(id, args, errors, scope),
        Stmt::If(condition, if_block, else_block) => {
            let err_msg = |wrongt| format!("expected if cond to be bool but it was {:?}", wrongt);
            check_expr(
                condition,
                errors,
                &scope,
                expect_val(Primitive::BoolType, err_msg),
            );
            check_block(if_block, errors, &scope, in_loop, return_type);
            if let Some(else_block) = else_block {
                check_block(else_block, errors, &scope, in_loop, return_type);
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
            let var_to_set_type = scope(&var_to_set.val);
            assert_eq(
                &Some(Type::Prim(Primitive::IntType)),
                &var_to_set_type,
                errors,
                format!(
                    "loop variable must be declared as int, not {:?}",
                    var_to_set_type
                ),
            );
            // initial_val is int
            check_regular_assign(
                &AssignOp::Eq,
                initial_val,
                errors,
                &scope,
                Some(Primitive::IntType),
            );

            // test is bool
            check_expr(
                test,
                &errors,
                &scope,
                expect_val(Primitive::BoolType, |wrongt| {
                    format!("loop condition must be of type bool, got {:?}", wrongt)
                }),
            );
            // var_to_update
            let left_type = check_location(var_to_update, errors, &scope);
            check_assign_expr(update_val, errors, &scope, left_type);

            // body
            check_block(body, errors, &scope, true, return_type);
        }
        Stmt::While(condition, body) => {
            check_expr(
                condition,
                errors,
                &scope,
                expect_val(Primitive::BoolType, |wrongt| {
                    format!("loop condition must be of type bool, got {:?}", wrongt)
                }),
            );

            check_block(body, errors, scope, true, return_type);
        }
        Stmt::Return(return_val) => {
            if let Some(return_val) = return_val {
                let type_ok = |typ| match return_type {
                    Some(return_type) => {
                        if *return_type == typ {
                            Ok(())
                        } else {
                            Err(format!(
                                "Method should return type {:?}, got {:?}",
                                return_type, typ
                            ))
                        }
                    }
                    None => Err(format!("Method should return void, got {:?}", return_type)),
                };

                check_expr(return_val, &errors, scope, type_ok);
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

//
fn check_block(
    block: &Block,
    errors: &mut Vec<String>,
    scope: impl Scope,
    in_loop: bool,
    return_type: &Option<Primitive>,
) {
    let block_scope = block.scope(scope);

    for stmt in block.stmts.iter() {
        check_stmt(stmt, errors, &block_scope, in_loop, return_type);
    }
}

// note: we always know what type an expression should be, so we can pass in expected_typ rather than returning something.
fn check_expr(
    expr: &Expr,
    errors: &Vec<String>,
    scope: impl Scope,
    expected_type: impl FnOnce(Primitive) -> Result<(), String>,
) {
    todo!()
}

fn check_arg(
    arg: &Arg,
    errors: &mut Vec<String>,
    expected_type: impl Fn(Primitive) -> Result<(), String>,
) {
    todo!()
}

fn check_regular_assign(
    op: &AssignOp,
    rhs: &Expr,
    errors: &mut Vec<String>,
    scope: impl Scope,
    lhs_type: Option<Primitive>,
) {
    let expected_type = move |typ| {
        if let Some(lhs_type) = lhs_type {
            if typ == lhs_type {
                Ok(())
            } else {
                Err(format!("expected {:?}, got {:?}", lhs_type, typ))
            }
        }
        // this is probably unnecessarily fancy but idk
        else if op.is_arith() && !(typ == Primitive::IntType || typ == Primitive::LongType) {
            Err(format!(
                "expected int or long type to go along with op {:?}",
                op
            ))
        } else {
            Ok(())
        }
    };
    check_expr(rhs, errors, scope, expected_type);
}

// left_type is Option<Type> because if left side failed to typecheck, we still want to sanity-check rhs but do not want to complain about its type
fn check_assign_expr(
    assign_expr: &AssignExpr,
    errors: &mut Vec<String>,
    scope: impl Scope,
    lhs_type: Option<Primitive>,
) {
    match assign_expr {
        AssignExpr::RegularAssign(op, rhs) => {
            check_regular_assign(op, rhs, errors, scope, lhs_type)
        }
        AssignExpr::IncrAssign(op) => match lhs_type {
            Some(Primitive::IntType) | Some(Primitive::LongType) | None => (),
            Some(x) => errors.push(format!(
                "did not expect increment operator {:?} to be applied to type {:?}",
                op, x
            )),
        },
    }
}
