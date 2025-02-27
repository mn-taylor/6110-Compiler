use crate::ir;
use crate::parse;
use crate::scan;
use ir::Expr;
use ir::Expr::*;

fn build_program(program: parse::Program) {
    // define global scope
    // call functions to build imports, fields, and methods,

    Program {
        imports: program.imports,
        methods: program.methods.map(build_method),
        fields: program.fields,
    }
}

fn build_method(method: parse::Method) {
    let method_scope = Scope {
        vars: method.params,
        parent: None,
    };

    let ir_block = build_block(method.block, method_scope);

    Method {
        block: ir_block,
        params: method.params,
        method_scope: method_scope,
    }
}

fn build_for(ast_for: parse::For, parent_scope: Scope) {
    match ast_for {
        parse::Stmt::For {
            withloc_idex,
            initial_value,
            condition,
            location,
            assignment_expr,
        } => {
            let for_scope = Scope {
                vars: [(withloc_idex, initial_value)],
                parent: scope,
            };

            let ir_initial_value = build_expr(inital_value);
            let ir_condition = build_expr(condition);
            let identifier = build_expr(location);
            let assigment = build_assign(assignment_expr);
            let block = build_block(block);

            return ir::Stmt::For {
                var_to_set: withloc_idx,
                initial_val: inital_value,
                test: ir_condition,
                var_to_update: identifier,
                update_val: assigment,
                body: block,
                scope: for_scope,
            };
        }

        _ => {}
    }
    return None;
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
            AtomicExpr::Lit(lit) => Lit(lit.map(build_literal)),
            AtomicExpr::IntCast(e) => Unary(IntCast, Box::new(parse::OrExpr::to_expr(*e))),
            AtomicExpr::LongCast(e) => Unary(LongCast, Box::new(parse::OrExpr::to_expr(*e))),
            AtomicExpr::LenEx(id) => Len(id),
            AtomicExpr::NegEx(e) => Unary(Neg, Box::new(Self::to_expr(*e))),
            AtomicExpr::NotEx(e) => Unary(Not, Box::new(Self::to_expr(*e))),
            AtomicExpr::Ex(e) => parse::OrExpr::to_expr(*e),
        }
    }
}

use ir::Literal;
fn build_literal(lit: parse::Literal) -> Literal {
    panic!();
}

use ir::Location;
use ir::Location::*;
fn build_location(l: parse::Location) -> Location {
    match l {
        parse::Location::Var(id) => Var(id),
        parse::Location::ArrayIndex(id, idx) => ArrayIndex(id, parse::OrExpr::to_expr(idx)),
    }
}
