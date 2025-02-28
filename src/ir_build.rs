use crate::ir;
use crate::parse;
use crate::scan;
use ir::Expr;
use ir::Expr::*;
use std::rc::Rc;

use ir::{Method, Program, Scope};
fn build_program(program: parse::Program) -> Program {
    Program {
        imports: program.imports,
        methods: program.methods.into_iter().map(build_method).collect(),
        fields: program.fields,
    }
}

fn build_method(method: parse::Method) -> Method {
    let fields: Vec<parse::Field> = Vec::new();

    for param in method.params {
        fields.push(parse::Field::Scalar(param.param_type, param.name));
    }

    let mut method_scope = Scope {
        vars: fields,
        parent: None,
    };

    let ir_block = build_block(method.body, method_scope);

    Method {
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


fn build_for(ast_for: parse::Stmt, parent_scope: Rc<Scope>)-> ir::Stmt {
    match ast_for {
        parse::Stmt::For {
            var_to_set: withloc_idx,
            initial_val: initial_value,
            test: condition,
            var_to_update: location,
            update_val: assignment_expr,
            body: body,
        } => {
            let for_scope = Rc::new(Scope {
                vars: vec![],
                parent:Some(parent_scope),
            });

            let ir_initial_value = build_expr(initial_value);
            let ir_condition = build_expr(condition);
            let identifier = build_location(location);
            let assignment = build_assign_expr(assignment_expr);
            let block = build_block(body, Rc::clone(&for_scope));

            return ir::Stmt::For {
                var_to_set: withloc_idx,
                initial_val: ir_initial_value,
                test: ir_condition,
                var_to_update: identifier,
                update_val: assignment,
                body: block,
                scope: for_scope,
            };
        },
        _=> {panic!("should not get here")}
    }
}


fn build_while(while_condition: parse::OrExpr, while_block: parse::Block, parent_scope: ir::Scope)->ir::Stmt{
    let while_scope = Rc::new(Scope {
        vars: vec![],
        parent: Some(Rc::new(parent_scope)),
    });
    
    let ir_while_condition = build_expr(while_condition);
    let ir_block = build_block(while_block, Rc::clone(&while_scope));

    // confused if we should be making a new scope explicitly or it should be done in build block
    return ir::Stmt::While(ir_while_condition, ir_block, while_scope);
}

fn build_return(ast_return: parse::Stmt)->ir::Stmt{
    match ast_return {
        parse::Stmt::Return(expr)=>{
            if let Some(return_val) = expr{
                let ir_return_val = build_expr(return_val);
                return ir::Stmt::Return(Some(ir_return_val));
            }else {
                return ir::Stmt::Return(None);
            }

        },
        _=>{panic!("should not get here")}
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
        parse::Stmt::Assignment(loc, assign_expr ) => {
            return ir::Stmt::AssignStmt(build_location(loc), build_assign_expr(assign_expr));
        },
        _=>{ panic!("should not get here") }
    }
}

fn build_assign_expr(assign_expr: parse::AssignExpr) -> ir::AssignExpr {
    use crate::parse::AssignExpr;
    match assign_expr {
        AssignExpr::RegularAssign(assign_op, expr) => return ir::AssignExpr::RegularAssign(assign_op, build_expr(expr)),
        AssignExpr::IncrAssign(inc_op) => return ir::AssignExpr::IncrAssign(inc_op),
    }
}

fn build_call(call: parse::Stmt) -> ir::Stmt {
    match call {
        parse::Stmt::Call(loc_info, args) => { 
            let mut new_args: Vec<ir::Arg> = Vec::new();
            for arg in args {
                match arg {
                    parse::Arg::ExprArg(expr)=>new_args.push(ir::Arg::ExprArg(build_expr(expr))),
                    parse::Arg::ExternArg(str)=>new_args.push(ir::Arg::ExternArg(str))
                }
            }
            return ir::Stmt::Call(loc_info, new_args);
        },
        _=>{ panic!("should not get here") }
    }
}

fn build_if(if_stmt: parse::Stmt, scope_ptr: Rc<ir::Scope>) -> ir::Stmt {
    // let scope_ptr = Rc::new(scope);
    let if_block_scope = Rc::new(ir::Scope {
        vars: Vec::new(),
        parent: Some(Rc::clone(&scope_ptr)),
    });

    match if_stmt {
        parse::Stmt::If(expr, if_block, else_block) => {
            let new_else_block: Option<(ir::Block, Rc<ir::Scope>)>;
            match else_block {
                Some(block) => {
                    let else_block_scope = Rc::new(ir::Scope {
                        vars: Vec::new(),
                        parent: Some(Rc::clone(&scope_ptr)),
                    });
                    new_else_block = Some((build_block(block, Rc::clone(&else_block_scope)), Rc::clone(&else_block_scope)));
                },
                _=> new_else_block = None
            }

            return ir::Stmt::If (build_expr(expr), build_block(if_block, Rc::clone(&if_block_scope)), Rc::clone(&if_block_scope), new_else_block);
        },
        _=>{ panic!("should not get here") }
    }
}

fn build_block(block: parse::Block, scope: Rc<ir::Scope>) -> ir::Block {
    let mut statements: Vec<ir::Stmt>;

    let mut new_scope = ir::Scope {
        vars: Vec::new(),
        parent: Some(scope),
    }

    for field in block.fields {
        new_scope.vars.push(field);
    }

    use crate::parse::Stmt;
    for stmt in block.stmts {
        match stmt {
            Stmt::Assignment(loc, assign_expr) => statements.push(build_assignment(stmt)),
            Stmt::Call(loc_info, args) => statements.push(build_call(stmt)),
            Stmt::If(expr, block,else_block) => statements.push(build_if(stmt, new_scope)),
            Stmt::For{
                var_to_set: loc,
                initial_val: expr1,
                test: expr2,
                var_to_update: var,
                update_val: assign_expr,
                body: block,
            }=> statements.push(build_for(stmt,new_scope)),
            Stmt::While(expr, block) => statements.push(build_while(stmt)),
            Stmt::Return(expr) => statements.push(build_return(stmt)),
            Stmt::Break => statements.push(ir::Stmt::Break),
            Stmt::Continue => statements.push(ir::Stmt::Continue),
        }
    }

    statements
}
