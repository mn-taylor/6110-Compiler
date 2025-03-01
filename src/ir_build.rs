use crate::ir;
use crate::parse;
use crate::scan;
use ir::Expr;
use ir::Expr::*;
use std::rc::Rc;

use ir::{Method, Program, Scope};
pub fn build_program(program: parse::Program) -> Program {
    Program {
        imports: program.imports,
        methods: program.methods.into_iter().map(build_method).collect(),
        fields: program.fields,
    }
}

fn build_method(method: parse::Method) -> Method {
    let fields = method
        .params
        .clone()
        .into_iter()
        .map(|param| parse::Field::Scalar(param.param_type, param.name));

    let method_scope = Rc::new(Scope {
        vars: fields.collect(),
        parent: None,
    });

    let ir_block = build_block(method.body, Rc::clone(&method_scope));

    Method {
        meth_type: method.meth_type,
        body: ir_block,
        params: method.params,
        scope: method_scope,
    }
}

// fn simplify_assignment(
//     location: parse::Location,
//     assignment_expr: parse::AssignExpr
// ) -> (Rc<ir::Location>, ir::Expr) {
//     match assignment_expr {
//         parse::AssignExpr::RegularAssign(op, expr) => {
//             let location_expr = parse::AtomicExpr::Loc(Box::new(location));

//             let simple_expr = match op {
//                 scan::AssignOp::Eq => ir_right_expr,
//                 scan::AssignOp::PlusEq =>
//                     ir::Expr::Bin(Box::new(ir_location_expr), ir::Bop::AddBop(scan::Add), Box::new(ir_right_expr)),
//                 scan::AssignOp::MinusEq =>
//                     ir::Expr::Bin(Box::new(ir_location_expr), ir::Bop::AddBop(scan::Sub), Box::new(ir_right_expr)),
//                 scan::AssignOp::MulEq =>
//                     ir::Expr::Bin(Box::new(ir_location_expr), ir::Bop::MulBop(scan::Mul), Box::new(ir_right_expr)),
//                 scan::AssignOp::DivEq =>
//                     ir::Expr::Bin(Box::new(ir_location_expr), ir::Bop::MulBop(scan::Div), Box::new(ir_right_expr)),
//                 scan::AssignOp::ModEq =>
//                     ir::Expr::Bin(Box::new(ir_location_expr), ir::Bop::MulBop(scan::Mod), Box::new(ir_right_expr)),
//             };

//             return (Rc::clone(&ir_location), simple_expr)
//         }
//         _ => panic!("Unsupported assignment expression"),
//     }
// }

fn build_for(ast_for: parse::Stmt, parent_scope: Rc<Scope>) -> ir::Stmt {
    match ast_for {
        parse::Stmt::For {
            var_to_set: withloc_idx,
            initial_val: initial_value,
            test: condition,
            var_to_update: location,
            update_val: assignment_expr,
            body,
        } => {
            let for_scope = Rc::new(Scope {
                vars: vec![],
                parent: Some(parent_scope),
            });

            let ir_initial_value = build_expr(initial_value);
            let ir_condition = build_expr(condition);
            let identifier = build_location(location);
            let assignment = build_assign_expr(assignment_expr);
            let block = build_block(body, Rc::clone(&for_scope));

            ir::Stmt::For {
                var_to_set: withloc_idx,
                initial_val: ir_initial_value,
                test: ir_condition,
                var_to_update: identifier,
                update_val: assignment,
                body: block,
                scope: for_scope,
            }
        }
        _ => {
            panic!("should not get here")
        }
    }
}

fn build_while(ast_while: parse::Stmt, parent_scope: Rc<ir::Scope>) -> ir::Stmt {
    let while_scope = Rc::new(Scope {
        vars: vec![],
        parent: Some(parent_scope),
    });

    match ast_while {
        parse::Stmt::While(while_condition, block) => {
            let ir_while_condition = build_expr(while_condition);
            let ir_block = build_block(block, Rc::clone(&while_scope));

            // confused if we should be making a new scope explicitly or it should be done in build block
            ir::Stmt::While(ir_while_condition, ir_block, while_scope)
        }
        _ => {
            panic!("should not get here")
        }
    }
}

use ir::Bop;

trait ToExpr {
    fn to_expr(self) -> Expr;
}

trait ToBop {
    fn to_op(self) -> Bop;
}

impl ToBop for scan::AddOp {
    fn to_op(self) -> Bop {
        Bop::AddBop(self)
    }
}

impl ToBop for scan::MulOp {
    fn to_op(self) -> Bop {
        Bop::MulBop(self)
    }
}

impl ToBop for scan::RelOp {
    fn to_op(self) -> Bop {
        Bop::RelBop(self)
    }
}

impl ToBop for scan::EqOp {
    fn to_op(self) -> Bop {
        Bop::EqBop(self)
    }
}

impl ToBop for parse::AndOp {
    fn to_op(self) -> Bop {
        Bop::And
    }
}

impl ToBop for parse::OrOp {
    fn to_op(self) -> Bop {
        Bop::Or
    }
}

use parse::BinExpr;
impl<O: ToBop, A: ToExpr> ToExpr for BinExpr<O, A> {
    fn to_expr(self) -> Expr {
        match self {
            BinExpr::Atomic(e) => A::to_expr(e),
            BinExpr::Bin(a1, o1, rhs1) => match *rhs1 {
                BinExpr::Atomic(a2) => Bin(
                    Box::new(A::to_expr(a1)),
                    O::to_op(o1),
                    Box::new(A::to_expr(a2)),
                ),
                BinExpr::Bin(a2, o2, rhs2) => Bin(
                    Box::new(Bin(
                        Box::new(A::to_expr(a1)),
                        O::to_op(o1),
                        Box::new(A::to_expr(a2)),
                    )),
                    O::to_op(o2),
                    Box::new(Self::to_expr(*rhs2)),
                ),
            },
        }
    }
}

use ir::UnOp::*;
use parse::AtomicExpr;
impl ToExpr for AtomicExpr {
    fn to_expr(self) -> Expr {
        match self {
            AtomicExpr::Loc(l) => Loc(Box::new(build_location(*l))),
            AtomicExpr::Call(id, args) => Call(id, args),
            AtomicExpr::Lit(lit) => Lit(lit.map(|l| build_literal(l, false))),
            AtomicExpr::IntCast(e) => Unary(IntCast, Box::new(build_expr(*e))),
            AtomicExpr::LongCast(e) => Unary(LongCast, Box::new(build_expr(*e))),
            AtomicExpr::LenEx(id) => Len(id),
            AtomicExpr::NegEx(e) => match *e {
                AtomicExpr::Lit(l) => Lit(l.map(|l| build_literal(l, true))),
                _ => Unary(Neg, Box::new(Self::to_expr(*e))),
            },
            AtomicExpr::NotEx(e) => Unary(Not, Box::new(Self::to_expr(*e))),
            AtomicExpr::Ex(e) => build_expr(*e),
        }
    }
}

fn build_expr(e: parse::OrExpr) -> Expr {
    parse::OrExpr::to_expr(e)
}

use ir::Literal;
use ir::Literal::*;
// TODO: handle errors somehow
fn build_literal(lit: parse::Literal, negated: bool) -> Literal {
    let maybe_neg = |s| if negated { format!("-{}", s) } else { s };
    match lit {
        parse::Literal::DecInt(s) => IntLit(maybe_neg(s).parse().unwrap()),
        parse::Literal::HexInt(s) => IntLit(i32::from_str_radix(&maybe_neg(s), 16).unwrap()),
        parse::Literal::DecLong(s) => LongLit(maybe_neg(s).parse().unwrap()),
        parse::Literal::HexLong(s) => LongLit(i64::from_str_radix(&maybe_neg(s), 16).unwrap()),
        parse::Literal::Char(c) => {
            if negated {
                panic!()
            } else {
                CharLit(c)
            }
        }
        parse::Literal::Bool(b) => {
            if negated {
                panic!()
            } else {
                BoolLit(b)
            }
        }
    }
}

use ir::Location;
use ir::Location::*;
fn build_location(l: parse::Location) -> Location {
    match l {
        parse::Location::Var(id) => Var(id),
        parse::Location::ArrayIndex(id, idx) => ArrayIndex(id, build_expr(idx)),
    }
}

fn build_assignment(assignment: parse::Stmt) -> ir::Stmt {
    match assignment {
        parse::Stmt::Assignment(loc, assign_expr) => {
            ir::Stmt::AssignStmt(build_location(loc), build_assign_expr(assign_expr))
        }
        _ => {
            panic!("should not get here")
        }
    }
}

fn build_assign_expr(assign_expr: parse::AssignExpr) -> ir::AssignExpr {
    use crate::parse::AssignExpr;
    match assign_expr {
        AssignExpr::RegularAssign(assign_op, expr) => {
            ir::AssignExpr::RegularAssign(assign_op, build_expr(expr))
        }
        AssignExpr::IncrAssign(inc_op) => ir::AssignExpr::IncrAssign(inc_op),
    }
}

fn build_call(call: parse::Stmt) -> ir::Stmt {
    match call {
        parse::Stmt::Call(loc_info, args) => {
            let new_args = args.into_iter().map(|arg| match arg {
                parse::Arg::ExprArg(expr) => ir::Arg::ExprArg(build_expr(expr)),
                parse::Arg::ExternArg(str) => ir::Arg::ExternArg(str),
            });
            ir::Stmt::Call(loc_info, new_args.collect())
        }
        _ => {
            panic!("should not get here")
        }
    }
}

fn build_if(if_stmt: parse::Stmt, scope_ptr: Rc<ir::Scope>) -> ir::Stmt {
    let if_block_scope = Rc::new(ir::Scope {
        vars: Vec::new(),
        parent: Some(Rc::clone(&scope_ptr)),
    });

    match if_stmt {
        parse::Stmt::If(expr, if_block, else_block) => {
            let new_else_block = match else_block {
                Some(block) => {
                    let else_block_scope = Rc::new(ir::Scope {
                        vars: Vec::new(),
                        parent: Some(Rc::clone(&scope_ptr)),
                    });
                    Some((
                        build_block(block, Rc::clone(&else_block_scope)),
                        Rc::clone(&else_block_scope),
                    ))
                }
                _ => None,
            };

            ir::Stmt::If(
                build_expr(expr),
                build_block(if_block, Rc::clone(&if_block_scope)),
                Rc::clone(&if_block_scope),
                new_else_block,
            )
        }
        _ => {
            panic!("should not get here")
        }
    }
}

fn build_block(block: parse::Block, scope: Rc<ir::Scope>) -> ir::Block {
    let new_scope = ir::Scope {
        vars: block.fields,
        parent: Some(scope),
    };

    let rc_new_scope = Rc::new(new_scope);

    use crate::parse::Stmt;
    block
        .stmts
        .into_iter()
        .map(|stmt| match stmt {
            Stmt::Assignment(_, _) => build_assignment(stmt),
            Stmt::Call(_, _) => build_call(stmt),
            Stmt::If(_, _, _) => build_if(stmt, Rc::clone(&rc_new_scope)),
            Stmt::For { .. } => build_for(stmt, Rc::clone(&rc_new_scope)),
            Stmt::While(_, _) => build_while(stmt, Rc::clone(&rc_new_scope)),
            Stmt::Return(expr) => ir::Stmt::Return(expr.map(build_expr)),
            Stmt::Break => ir::Stmt::Break,
            Stmt::Continue => ir::Stmt::Continue,
        })
        .collect()
}
