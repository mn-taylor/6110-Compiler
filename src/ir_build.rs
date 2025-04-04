use crate::ir;
use crate::parse;
use crate::scan;
use ir::Expr::*;
use ir::*;

pub fn build_program(program: parse::Program) -> Program {
    Program {
        imports: program.imports,
        methods: program.methods.into_iter().map(build_method).collect(),
        fields: program.fields,
    }
}

fn build_method(method: parse::Method) -> Method {
    Method {
        name: method.name,
        meth_type: method.meth_type,
        fields: method.body.fields,
        stmts: method.body.stmts.into_iter().map(build_stmt).collect(),
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
    fn to_expr(self) -> WithLoc<Expr>;
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

fn to_expr_helper<O: ToBop, A: ToExpr>(
    lhs: WithLoc<Expr>,
    lhs_op: Bop,
    rhs: BinExpr<O, A>,
) -> WithLoc<Expr> {
    match rhs {
        BinExpr::Atomic(a) => WithLoc {
            loc: lhs.loc,
            val: Bin(Box::new(lhs), lhs_op, Box::new(A::to_expr(a))),
        },
        BinExpr::Bin(a1, o, rhs) => to_expr_helper(
            WithLoc {
                loc: lhs.loc,
                val: Bin(Box::new(lhs), lhs_op, Box::new(A::to_expr(a1))),
            },
            O::to_op(o),
            *rhs,
        ),
    }
}

use parse::BinExpr;
impl<O: ToBop, A: ToExpr> ToExpr for BinExpr<O, A> {
    fn to_expr(self) -> WithLoc<Expr> {
        match self {
            BinExpr::Atomic(a) => A::to_expr(a),
            BinExpr::Bin(a1, o, rhs) => to_expr_helper(A::to_expr(a1), O::to_op(o), *rhs),
        }
    }
}

use ir::UnOp::*;
use parse::AtomicExpr;
use parse::WithLoc;
fn atomic_expr_to_expr(e: AtomicExpr) -> Expr {
    match e {
        AtomicExpr::Loc(l) => Loc(Box::new(WithLoc::map(*l, build_location))),
        AtomicExpr::Call(id, args) => Call(id, args.into_iter().map(build_arg).collect()),
        AtomicExpr::Lit(lit) => Lit(lit),
        AtomicExpr::IntCast(e) => Unary(IntCast, Box::new(build_expr(*e))),
        AtomicExpr::LongCast(e) => Unary(LongCast, Box::new(build_expr(*e))),
        AtomicExpr::LenEx(id) => Len(id),
        AtomicExpr::NegEx(e) => Unary(Neg, Box::new(WithLoc::<AtomicExpr>::to_expr(*e))),
        AtomicExpr::NotEx(e) => Unary(Not, Box::new(WithLoc::<AtomicExpr>::to_expr(*e))),
        AtomicExpr::Ex(e) => build_expr(*e).val, // here we choose that ((x)) has location of the leftmost paren
    }
}

impl ToExpr for WithLoc<AtomicExpr> {
    fn to_expr(self) -> WithLoc<Expr> {
        WithLoc {
            loc: self.loc,
            val: atomic_expr_to_expr(self.val),
        }
    }
}

fn build_expr(e: parse::OrExpr) -> WithLoc<Expr> {
    parse::OrExpr::to_expr(e)
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

fn build_arg<'a>(arg: parse::Arg) -> Arg {
    match arg {
        parse::Arg::ExprArg(expr) => ir::Arg::ExprArg(build_expr(expr)),
        parse::Arg::ExternArg(str) => ir::Arg::ExternArg(str),
    }
}

fn build_stmt(stmt: parse::Stmt) -> ir::Stmt {
    use crate::parse::Stmt;
    match stmt {
        Stmt::Assignment(loc, assign_expr) => {
            ir::Stmt::AssignStmt(loc.map(build_location), build_assign_expr(assign_expr))
        }
        Stmt::Call(id, args) => ir::Stmt::Call(id, args.into_iter().map(build_arg).collect()),
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
            var_to_update: var_to_update.map(build_location),
            update_val: build_assign_expr(update_val),
            body: build_block(body),
        },
        Stmt::While(cond, body) => ir::Stmt::While(build_expr(cond), build_block(body)),
        Stmt::Return(loc, expr) => ir::Stmt::Return(loc, expr.map(build_expr)),
        Stmt::Break(loc) => ir::Stmt::Break(loc),
        Stmt::Continue(loc) => ir::Stmt::Continue(loc),
    }
}

fn build_block(block: parse::Block) -> ir::Block {
    Block {
        stmts: block.stmts.into_iter().map(build_stmt).collect(),
        fields: block.fields,
    }
}
