use crate::ir;
use crate::parse;
use crate::scan;
use ir::Expr;
use ir::Expr::*;

use ir::{Method, Program, Scope};
fn build_program(program: parse::Program) -> Program {
    Program {
        imports: program.imports,
        methods: program.methods.into_iter().map(build_method).collect(),
        fields: program.fields,
    }
}

fn build_method(method: parse::Method) -> Method {
    let method_scope = Scope {
        vars: method.params,
        parent: None,
    };

    let ir_block = build_block(method.block, method_scope);

    Method {
        block: ir_block,
        params: method.params,
        scope: method_scope,
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
        parse::Literal::HexLong(s) => LongLit(maybe_neg(s).parse().unwrap()),
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
            use crate::parse::AssignExpr;
            match assign_expr {
                AssignExpr::RegularAssign(assign_op, expr) => return ir::Stmt::AssignStmt(build_location(loc), AssignExpr::RegularAssign(assign_op, build_expr(expr))),
                AssignExpr::IncrAssign(inc_op) => return ir::Stmt::AssignStmt(build_location(loc), assign_expr),
            }
        },
        _=>{}
    }   
}

fn build_call(call: parse::Stmt) -> ir::Stmt {
    match call {
        parse::Stmt::Call(loc_info, args) => { 
            let mut new_args: Vec<parse::Arg> = Vec::new();
            for arg in args {
                match arg {
                    parse::Arg::ExprArg(expr)=>new_args.push(parse::Arg::ExprArg(build_expr(expr))),
                    parse::Arg::ExternArg(str)=>new_args.push(arg)
                }
            }
            return ir::Stmt::Call(loc_info, new_args);
        },
        _=>{}
    }
}

fn build_if(if_stmt: parse::Stmt, scope: ir::Scope) -> ir::Stmt {
    let if_block_scope = ir::Scope {
        vars: Vec::new(),
        parent: scope,
    };

    match if_stmt {
        parse::Stmt::If(expr, if_block, else_block) => {
            let mut new_else_block: Option<(ir::Block, ir::Scope)>;
            match else_block {
                Some(block) => {
                    let else_block_scope = ir::Scope {
                        vars: Vec::new(),
                        parent: scope,
                    };

                    new_else_block = Some((build_block(block, else_block_scope), else_block_scope));
                },
                _=> new_else_block = None
            }

            return ir::Stmt::If (build_expr(expr), build_block(if_block, if_block_scope), if_block_scope, new_else_block);
        },
        _=>{}
    }
}

fn build_block(block: parse::Block, scope: ir::Scope) -> ir::Block {
    let statements: Vec<ir::Stmt>;

    for field in block.fields {
        scope.vars.push(field);
    }

    use crate::parse::Stmt;
    for stmt in block.stmts {
        match stmt {
            Stmt::Assignment(loc, assign_expr) => statements.push(build_assignment(stmt)),
            Stmt::Call(loc_info, args) => statements.push(build_call(stmt)),
            Stmt::If(expr, block,else_block) => statements.push(build_if(stmt, scope)),
            Stmt::For(loc_info, expr1, expr2, loc, assign_expr, block) => statements.push(build_for(stmt, scope)),
            Stmt::While(expr, block) => statements.push(build_while(stmt)),
            Stmt::Return(expr) => statements.push(build_return(stmt)),
            Stmt::Break => statements.push(stmt),
            Stmt::Continue => statements.push(stmt),
        }
    }

    statements
}