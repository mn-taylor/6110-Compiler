use crate::ir;
use crate::parse;
use crate::scan;
use ir::Block;
use ir::Expr;
use ir::Expr::*;
use ir::Field;
use ir::{Method, Program};

pub fn build_program(program: parse::Program) -> Program {
    Program {
        imports: program.imports,
        methods: program.methods.into_iter().map(build_method).collect(),
        fields: program.fields.into_iter().map(build_field).collect(),
    }
}

fn build_field(field: parse::Field) -> Field {
    match field {
        parse::Field::Scalar(t, id) => Field::Scalar(t, id),
        parse::Field::Array(t, id, len) => {
            if let IntLit(len) = build_literal(len, false) {
                Field::Array(t, id, len)
            } else {
                panic!()
            }
        }
    }
}

fn build_method(method: parse::Method) -> Method {
    Method {
        name: method.name.val,
        meth_type: method.meth_type,
        body: build_block(method.body),
        params: method.params,
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

fn build_assign_expr(assign_expr: parse::AssignExpr) -> ir::AssignExpr {
    use crate::parse::AssignExpr;
    match assign_expr {
        AssignExpr::RegularAssign(assign_op, expr) => {
            ir::AssignExpr::RegularAssign(assign_op, build_expr(expr))
        }
        AssignExpr::IncrAssign(inc_op) => ir::AssignExpr::IncrAssign(inc_op),
    }
}

fn build_stmt(stmt: parse::Stmt) -> ir::Stmt {
    use crate::parse::Stmt;
    match stmt {
        Stmt::Assignment(loc, assign_expr) => {
            ir::Stmt::AssignStmt(build_location(loc), build_assign_expr(assign_expr))
        }
        Stmt::Call(id, args) => ir::Stmt::Call(
            id,
            args.into_iter()
                .map(|arg| match arg {
                    parse::Arg::ExprArg(expr) => ir::Arg::ExprArg(build_expr(expr)),
                    parse::Arg::ExternArg(str) => ir::Arg::ExternArg(str),
                })
                .collect(),
        ),
        Stmt::If(cond, if_block, else_block) => ir::Stmt::If(
            build_expr(cond),
            build_block(if_block),
            else_block.map(build_block),
        ),
        Stmt::For {
            var_to_set,
            initial_val,
            test,
            var_to_update,
            update_val,
            body,
        } => ir::Stmt::For {
            var_to_set,
            initial_val: build_expr(initial_val),
            test: build_expr(test),
            var_to_update: build_location(var_to_update),
            update_val: build_assign_expr(update_val),
            body: build_block(body),
        },
        Stmt::While(cond, body) => ir::Stmt::While(build_expr(cond), build_block(body)),
        Stmt::Return(expr) => ir::Stmt::Return(expr.map(build_expr)),
        Stmt::Break => ir::Stmt::Break,
        Stmt::Continue => ir::Stmt::Continue,
    }
}

fn build_block(block: parse::Block) -> ir::Block {
    Block {
        stmts: block.stmts.into_iter().map(build_stmt).collect(),
        fields: block.fields.into_iter().map(build_field).collect(),
    }
}
