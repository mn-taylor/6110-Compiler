use crate::ir::Expr;
use crate::ir::Expr::*;
use crate::ir::*;
use crate::parse;
use crate::parse::BinExpr;

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
        Stmt::For(withloc_idex, initial_value, condition, location, assignment_expr) => {
            let for_scope = Scope {
                vars: [(withloc_idex, initial_value)],
                parent: scope,
            };

            let ir_initial_value = build_expr(inital_value);
            let ir_condition = build_expr(condition);
            let identifier = build_expr(location);
            let assigment = build_assign(assignment_expr);
            let block = build_block(block);

            return ir::Stmt::For(
                withloc_idx,
                inital_value,
                ir_condition,
                identifier,
                assigment,
                block,
                for_scope,
            );
        }

        _ => {}
    }
    return None;
}

trait ToExpr {
    fn to_expr(x: Self) -> Expr;
}

trait ToOp {
    fn to_op(x: Self) -> Op;
}

impl<O: ToOp, A: ToExpr> ToExpr for BinExpr<O, A> {
    fn to_expr(e: Self) {
        match e {
            BinExpr::Atomic(e) => A::to_expr(e),
            BinExpr::Bin(a1, o1, rhs1) => match *rhs1 {
                BinExpr::Atomic(a2) => Bin(
                    Box::new(A::to_expr(a1)),
                    O::to_op(o1),
                    Box::new(A::to_expr(a2)),
                ),
                BinExpr::Bin(a2, o2, rhs2) => Bin(
                    Box::new(Bin(a1, O::to_op(o1), a2)),
                    O::to_op(o2),
                    Box::new(A::to_expr(rhs2)),
                ),
            },
        }
    }
}

fn expr_of_bin_expr<AtomType, OpType>(expr: parse::BinExpr<OpType, AtomType>) -> Expr {}
