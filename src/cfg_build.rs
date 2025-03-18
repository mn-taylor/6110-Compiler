use crate::{
    cfg,
    ir::{self, Program},
    parse,
    scan::{self, IncrOp},
};
use cfg::{Arg, BasicBlock, Instruction, Jump, Type, Var};
use ir::{AssignExpr, Block, Bop, Expr, Location, Method, UnOp};
use parse::{Field, Literal, Primitive, WithLoc};
use scan::{AddOp, AssignOp, ErrLoc, MulOp};

fn new_noop<'a>(st: &mut State<'a>) -> BasicBlock<'static> {
    let ret = BasicBlock {
        body: vec![],
        jump_loc: Jump::Nowhere,
    };
    st.all_blocks.push(&ret);
    &ret
}

type Scope<'a> = ir::Scope<'a, (Type, u32)>;

struct State<'a> {
    break_loc: Option<&'a BasicBlock<'a>>,
    continue_loc: Option<&'a BasicBlock<'a>>,
    last_name: u32,
    all_blocks: Vec<BasicBlock<'a>>,
    all_fields: Vec<Field>,
    // all_arrays: Vec<String>,
}

fn gen_name(st: &mut State) -> u32 {
    st.last_name += 1;
    st.last_name
}

fn gen_temp(typ: Primitive, st: &mut State) -> Var {
    st.last_name += 1;
    Var::Scalar {
        id: gen_name(st),
        typ,
        name: "temp".to_string(),
    }
}

fn lin_program<'a>(program: Program) -> State<'a> {
    let mut st: State = State {
        break_loc: None,
        continue_loc: None,
        last_name: 0,
        all_blocks: vec![],
        all_fields: vec![],
    };

    // Get the local scope from the program
    let scope: Scope<'a> = program.local_scope(&mut st);

    // Process methods
    for method in program.methods {
        let _ = lin_method(method, &mut st, &scope);
    }

    st
}

fn lin_method<'a>(
    method: Method,
    st: &'a mut State<'a>,
    scope: &Scope,
) -> (&'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    let fst = new_noop(st);
    let mut last = fst;
    let method_scope = method.scope(scope, st);

    for s in method.stmts {
        let (start, end) = lin_stmt(s, st, &method_scope);
        last.jump_loc = Jump::Uncond(start);
        last = end;
    }

    return (fst, last);
}

fn lin_branch<'a>(
    true_branch: &'a BasicBlock,
    false_branch: &'a BasicBlock,
    cond: Expr,
    st: &'a mut State<'a>,
    scope: &Scope,
) -> &'a BasicBlock<'a> /*start*/
{
    match cond {
        Expr::Bin(e1, Bop::And, e2) => {
            let e2start = lin_branch(true_branch, false_branch, e2.val, st, scope);
            let e1start = lin_branch(e2start, false_branch, e1.val, st, scope);
            return e1start;
        }
        Expr::Bin(e1, Bop::Or, e2) => {
            let e2start = lin_branch(true_branch, false_branch, e2.val, st, scope);
            let e1start = lin_branch(true_branch, e2start, e1.val, st, scope);
            return e1start;
        }
        _ => {
            let (t, tstart, tend) = lin_expr(cond, st, scope);
            tend.jump_loc = Jump::Cond {
                source: t,
                true_block: true_branch,
                false_block: false_branch,
            };
            return tstart;
        }
    }
}

// will call lin_branch to deal with bool exprs
fn lin_expr<'a>(
    e: Expr,
    st: &'a mut State<'a>,
    scope: &Scope,
) -> (Var, &'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    let start = new_noop(st);
    match e {
        Expr::Bin(e1, op, e2) => {
            match op {
                Bop::And | Bop::Or => {
                    let end = new_noop(st);
                    let temp = gen_temp(Primitive::BoolType, st);
                    let true_branch = BasicBlock {
                        body: vec![Instruction::Constant {
                            dest: temp,
                            constant: 1,
                        }],
                        jump_loc: Jump::Uncond(end),
                    }; //block that sets temp = true and jumps to end;
                    let false_branch = BasicBlock {
                        body: vec![Instruction::Constant {
                            dest: temp,
                            constant: 1,
                        }],
                        jump_loc: Jump::Uncond(end),
                    }; //block taht  sets temp = fasle and jumpts to end;
                    st.all_blocks.push(true_branch);
                    st.all_blocks.push(false_branch);
                    let start = lin_branch(&true_branch, &false_branch, e, st, scope);
                    return (temp, start, end);
                }
                _ => {
                    let (t1, t1start, t1end) = lin_expr(e1.val, st, scope);
                    let (t2, t2start, t2end) = lin_expr(e2.val, st, scope);
                    t1end.jump_loc = Jump::Uncond(&t2start);
                    let t3 = gen_temp(infer_type(t1.get_typ(), op), st);
                    let mut end = BasicBlock {
                        body: vec![Instruction::ThreeOp {
                            source1: t1,
                            source2: t2,
                            dest: t3,
                            op: op,
                        }],
                        jump_loc: Jump::Nowhere,
                    };
                    st.all_blocks.push(end);
                    t2end.jump_loc = Jump::Uncond(&end);
                    return (t3, &t1start, &mut end);
                }
            }
        }
        Expr::Unary(op, e) => {
            let (t1, t1start, t1end) = lin_expr(e.val, st, scope);
            let t2 = gen_temp(infer_unary_type(t1.get_typ(), op), st);
            let end = BasicBlock {
                body: vec![Instruction::TwoOp {
                    source1: t1,
                    dest: t2,
                    op,
                }],
                jump_loc: Jump::Nowhere,
            };
            st.all_blocks.push(end);
            return (t1, &t1start, &end);
        }
        Expr::Len(id) => match scope.lookup(&id.val) {
            Some((Type::Arr(_, len), _)) => {
                let t = gen_temp(Primitive::IntType, st);
                let blk = BasicBlock {
                    body: vec![Instruction::Constant {
                        dest: t,
                        constant: *len as i64,
                    }],
                    jump_loc: Jump::Nowhere,
                };
                st.all_blocks.push(blk);
                (t, &blk, &blk)
            }
            Some(_) => panic!("can only take len of array"),
            None => panic!("array identifier not found"),
        },
        Expr::Lit(lit) => {
            let (typ, val) = lin_literal(lit.val);
            let t = gen_temp(typ, st);
            let end = BasicBlock {
                body: vec![Instruction::Constant {
                    dest: t,
                    constant: val,
                }],
                jump_loc: Jump::Nowhere,
            };
            st.all_blocks.push(end);
            return (t, &end, &end);
        }
        ir::Expr::Loc(loc) => {
            return lin_location(loc.val, st, scope);
        }
        ir::Expr::Call(id, args) => {
            let func_name = id.val.name;

            let start = new_noop(st);
            let mut prev_block = start;
            let mut temp_args: Vec<Arg> = vec![];
            for arg in args {
                match arg {
                    ir::Arg::ExprArg(WithLoc { val: e1, loc: _ }) => {
                        let (t, tstart, tend) = lin_expr(e1, st, scope);
                        let cfg_arg = Arg::VarArg(t);
                        temp_args.push(cfg_arg);

                        prev_block.jump_loc = Jump::Uncond(tstart);
                        prev_block = tend;
                    }
                    ir::Arg::ExternArg(WithLoc {
                        val: string,
                        loc: _,
                    }) => {
                        let cfg_arg = Arg::StrArg(string);
                        temp_args.push(cfg_arg);
                    }
                }
            }

            let ret_val = match scope.lookup(&id.val) {
                Some((Type::Prim(t), _)) => gen_temp(t.clone(), st),
                _ => panic!("Should not get here. function calls within expression must have non-void return type"),
            };
            let call_instr = cfg::Instruction::Call(func_name, temp_args, Some(ret_val));
            let mut end = cfg::BasicBlock {
                body: vec![call_instr],
                jump_loc: Jump::Nowhere,
            };
            st.all_blocks.push(end);
            return (ret_val, start, &mut end);
        }
    }
}

fn lin_block<'a>(
    b: Block,
    st: &'a mut State,
    scope: &Scope,
) -> (&'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    let fst = new_noop(st);
    let mut last = fst;
    let block_scope = b.scope(scope, st);

    for s in b.stmts {
        let (start, end) = lin_stmt(s, st, &block_scope);
        last.jump_loc = Jump::Uncond(start);
        last = end;
    }

    return (fst, last);
}

fn infer_unary_type(typ: Primitive, op: UnOp) -> Primitive {
    match op {
        UnOp::Neg => typ,
        UnOp::Not => Primitive::BoolType,
        UnOp::IntCast => Primitive::IntType,
        UnOp::LongCast => Primitive::LongType,
    }
}

fn infer_type(typ: Primitive, op: Bop) -> Primitive {
    match op {
        Bop::MulBop(_) | Bop::AddBop(_) => typ,
        _ => Primitive::BoolType,
    }
}

fn link<'a>(
    start1: &'a BasicBlock,
    end1: &'a BasicBlock,
    start2: &'a BasicBlock,
    end2: &'a BasicBlock,
) -> (&'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    end1.jump_loc = Jump::Uncond(start2);
    (start1, end2)
}

fn lin_stmt<'a>(
    s: ir::Stmt,
    st: &'a mut State,
    scope: &Scope,
) -> (&'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    match s {
        ir::Stmt::AssignStmt(loc, assign_expr) => {
            let (target, target_start, target_end): (Var, &BasicBlock<'_>, &BasicBlock<'_>) =
                lin_location(loc.val, st, scope);
            let (tstart, tend) = lin_assign_expr(target, assign_expr, st, scope);

            link(target_start, target_end, tstart, tend)
        }
        ir::Stmt::Break(_) => match st.break_loc {
            Some(break_block) => (break_block, break_block),
            _ => panic!("should not get here"),
        },
        ir::Stmt::Continue(_) => match st.continue_loc {
            Some(continue_block) => (continue_block, continue_block),
            _ => panic!("Should not get here"),
        },
        ir::Stmt::Call(id, args) => {
            let func_name = id.val.name;

            let start = new_noop(st);
            let mut prev_block = start;
            let mut temp_args: Vec<Arg> = vec![];
            for arg in args {
                match arg {
                    ir::Arg::ExprArg(WithLoc { val: e1, loc: _ }) => {
                        let (t, tstart, tend) = lin_expr(e1, st, scope);
                        let cfg_arg = Arg::VarArg(t);
                        temp_args.push(cfg_arg);

                        prev_block.jump_loc = Jump::Uncond(tstart);
                        prev_block = tend;
                    }
                    ir::Arg::ExternArg(WithLoc {
                        val: string,
                        loc: _,
                    }) => {
                        let cfg_arg = Arg::StrArg(string);
                        temp_args.push(cfg_arg);
                    }
                }
            }

            // lookup method and get type
            let ret_val = match scope.lookup(&id.val) {
                Some((Type::Prim(t), _)) => Some(gen_temp(t.clone(), st)),
                _ => None,
            };

            let call_instr = cfg::Instruction::Call(func_name, temp_args, ret_val);
            let end = cfg::BasicBlock {
                body: vec![call_instr],
                jump_loc: Jump::Nowhere,
            };
            st.all_blocks.push(end);

            return (&start, &end);
        }
        ir::Stmt::If(WithLoc { val: expr, loc: _ }, if_block, else_block) => {
            let (if_start, if_end) = lin_block(if_block, st, scope);
            let (else_start, else_end) = match else_block {
                Some(block) => lin_block(block, st, scope),
                None => {
                    let noop = new_noop(st);
                    (noop, noop)
                }
            };

            let start = lin_branch(if_start, else_start, expr, st, scope);

            let end = new_noop(st);
            if_end.jump_loc = Jump::Uncond(&end);
            else_end.jump_loc = Jump::Uncond(&end);

            (start, end)
        }
        ir::Stmt::While(
            WithLoc {
                val: condition,
                loc: _,
            },
            block,
        ) => {
            let continue_target = new_noop(st);
            let end = new_noop(st);

            // Update break/continue locations
            let old_break_loc = st.break_loc;
            let old_continue_loc = st.continue_loc;
            st.break_loc = Some(&end);
            st.continue_loc = Some(&continue_target);

            let (while_block_start, while_block_end) = lin_block(block, st, scope);

            // Restore old break/continue locations
            st.break_loc = old_break_loc;
            st.continue_loc = old_continue_loc;

            let while_condition = lin_branch(while_block_start, &end, condition, st, scope);

            continue_target.jump_loc = Jump::Uncond(while_condition);
            while_block_end.jump_loc = Jump::Uncond(while_condition);

            return (while_block_start, end);
        }
        ir::Stmt::For {
            var_to_set: var_to_set,
            initial_val: initial_val,
            test: test,
            var_to_update: var_to_update,
            update_val: update_val,
            body: body,
        } => {
            let (loop_var, loop_start, loop_end) =
                lin_location(Location::Var(var_to_set.val), st, scope);
            let (loop_init_start, loop_init_end) = lin_assign_expr(
                loop_var,
                AssignExpr::RegularAssign(
                    WithLoc {
                        val: AssignOp::Eq,
                        loc: ErrLoc { line: 0, col: 0 },
                    },
                    initial_val,
                ),
                st,
                scope,
            );
            loop_end.jump_loc = Jump::Uncond(loop_init_start);

            let end = new_noop(st);

            // Modify state so that we point to continue and break to the right places.
            let continue_target = new_noop(st);

            // update break and continue locations
            let old_break_loc = st.break_loc;
            let old_continue_loc = st.continue_loc;
            st.break_loc = Some(&end);
            st.continue_loc = Some(&continue_target);

            let (body_start, body_end) = lin_block(body, st, scope);

            // Restore old break and continue locations
            st.break_loc = old_break_loc;
            st.continue_loc = old_continue_loc;

            // change to handle increments and decrements better
            let (update_var, loop_update, update_loc_end) =
                lin_location(var_to_update.val, st, scope);
            let (update_start, update_end) = lin_assign_expr(update_var, update_val, st, scope);

            update_loc_end.jump_loc = Jump::Uncond(update_start);

            body_end.jump_loc = Jump::Uncond(&loop_update);

            let condition_start = lin_branch(body_start, &end, test.val, st, scope);
            continue_target.jump_loc = Jump::Uncond(loop_update);
            loop_init_end.jump_loc = Jump::Uncond(condition_start);
            loop_update.jump_loc = Jump::Uncond(condition_start);

            return (&loop_start, &end);
        }
        ir::Stmt::Return(_, ret_val) => match ret_val {
            Some(expr) => {
                let (t, tstart, tend) = lin_expr(expr.val, st, scope);
                let ret_instr = Instruction::Ret(Some(t));
                tend.body.push(ret_instr);
                (tstart, tend)
            }
            None => {
                let ret_instr = Instruction::Ret(None);
                let ret_block = BasicBlock {
                    body: vec![ret_instr],
                    jump_loc: Jump::Nowhere,
                };
                (&ret_block, &ret_block)
            }
        },
    }
}

fn convert_assign_op(o: AssignOp) -> Option<Bop> {
    match o {
        AssignOp::Eq => None,
        AssignOp::PlusEq => Some(Bop::AddBop(AddOp::Add)),
        AssignOp::MinusEq => Some(Bop::AddBop(AddOp::Sub)),
        AssignOp::MulEq => Some(Bop::MulBop(MulOp::Mul)),
        AssignOp::DivEq => Some(Bop::MulBop(MulOp::Div)),
        AssignOp::ModEq => Some(Bop::MulBop(MulOp::Mod)),
    }
}

fn convert_incr_op(o: IncrOp) -> Bop {
    match o {
        IncrOp::Decrement => Bop::AddBop(AddOp::Sub),
        IncrOp::Increment => Bop::AddBop(AddOp::Add),
    }
}

fn lin_assign_expr<'a>(
    target: Var,
    assign_expr: AssignExpr,
    st: &'a mut State,
    scope: &Scope,
) -> (&'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    let start: &BasicBlock;
    let mut end: BasicBlock;
    match assign_expr {
        AssignExpr::RegularAssign(op, rhs) => {
            let (t1, t1start, t1end) = lin_expr(rhs.val, st, scope);
            start = t1start;

            let op_ = convert_assign_op(op.val);
            let instr = match op_ {
                None => Instruction::MoveOp {
                    source: t1,
                    dest: target,
                },
                Some(binop) => Instruction::ThreeOp {
                    source1: target,
                    source2: t1,
                    dest: target,
                    op: binop,
                },
            };
            end = BasicBlock {
                body: vec![instr],
                jump_loc: Jump::Nowhere,
            };
        }
        AssignExpr::IncrAssign(op) => {
            let t1 = gen_temp(Primitive::IntType, st);
            let start = BasicBlock {
                body: vec![Instruction::Constant {
                    dest: t1,
                    constant: 1,
                }],
                jump_loc: Jump::Uncond(&end),
            };
            st.all_blocks.push(start);
            end = BasicBlock {
                body: vec![Instruction::ThreeOp {
                    source1: target,
                    source2: t1,
                    dest: target,
                    op: convert_incr_op(op.val),
                }],
                jump_loc: Jump::Nowhere,
            };
        }
    }
    st.all_blocks.push(end);
    return (start, &end);
}

fn lin_literal(lit: Literal) -> (Primitive, i64) {
    match lit {
        Literal::DecInt(s) => (Primitive::IntType, s.parse::<i64>().unwrap()),
        Literal::HexInt(s) => (
            Primitive::IntType,
            i64::from_str_radix(s.as_str(), 16).unwrap(),
        ),
        Literal::DecLong(s) => (
            Primitive::LongType,
            i64::from_str_radix(s.as_str(), 10).unwrap(),
        ),
        Literal::HexLong(s) => (
            Primitive::LongType,
            i64::from_str_radix(s.as_str(), 16).unwrap(),
        ),
        Literal::Bool(bool) => match bool {
            true => (Primitive::BoolType, 1),
            false => (Primitive::BoolType, 0),
        },
        Literal::Char(c) => (Primitive::IntType, c as i64),
    }
}

fn lin_location<'a>(
    loc: Location,
    st: &mut State<'a>,
    scope: &Scope,
) -> (Var, &'a BasicBlock<'a>, &'a BasicBlock<'a>) {
    match loc {
        Location::Var(name) => match scope.lookup(&name) {
            Some((Type::Prim(typ), id)) => {
                let blk = new_noop(st);
                return (
                    Var::Scalar {
                        id: *id,
                        name: name.name,
                        typ: *typ,
                    },
                    blk,
                    blk,
                );
            }
            Some(_) => panic!("location should have primitive type"),
            None => panic!("location not found"),
        },
        Location::ArrayIndex(name, idx) => match scope.lookup(&name) {
            Some((Type::Arr(typ, _), id)) => {
                let (idx_val, tstart, tend) = lin_expr(idx.val, st, scope);
                return (
                    Var::ArrIdx {
                        id: *id,
                        name: name.name,
                        idx: idx_val,
                        typ: *typ,
                    },
                    tstart,
                    tend,
                );
            }
            Some(_) => panic!("array should be an array"),
            None => panic!("array name not found"),
        },
    }
}

use std::collections::HashMap;

pub trait Scoped {
    fn scope<'a>(&'a self, parent: &'a Scope<'a>, st: &'a mut State<'a>) -> Scope<'a> {
        Scope::new(self.local_scope(st), Some(parent))
    }
    fn local_scope<'a>(&'a self, st: &'a mut State<'a>) -> HashMap<&'a String, (Type, u32)>;
}

fn imports_scope(
    imports: &Vec<WithLoc<parse::Ident>>,
) -> impl Iterator<Item = (&String, (Type, u32))> {
    let default: u32 = 0;
    imports
        .iter()
        .map(move |import| (&import.val.name, (Type::ExtCall, default)))
}

fn outer_methods_scope(methods: &Vec<Method>) -> impl Iterator<Item = (&String, (Type, u32))> {
    methods.iter().map(|method| {
        let default: u32 = 0;
        (
            &method.name.val.name,
            (
                Type::Func(
                    method
                        .params
                        .iter()
                        .map(|param| param.param_type.clone())
                        .collect::<Vec<_>>(),
                    method.meth_type.clone(),
                ),
                default,
            ),
        )
    })
}

fn program_scope<'a>(
    program: &'a Program,
    st: &'a mut State<'a>,
) -> impl Iterator<Item = (&'a String, (Type, u32))> {
    imports_scope(&program.imports)
        .chain(outer_methods_scope(&program.methods))
        .chain(fields_scope(&program.fields, st))
}

fn method_scope<'a>(
    method: &'a Method,
    st: &'a mut State<'a>,
) -> impl Iterator<Item = (&'a String, (Type, u32))> {
    let mut params: Vec<(&String, Type)> = method
        .params
        .iter()
        .map(|param| (&param.name.val.name, Type::Prim(param.param_type.clone())))
        .collect::<Vec<_>>();

    let fields = &mut method
        .fields
        .iter()
        .map(|field| match field {
            Field::Scalar(t, id) => (&id.val.name, Type::Prim(t.clone())),
            Field::Array(t, id, len) => (
                &id.val.name,
                Type::Arr(t.clone(), lin_literal(len.val.clone()).1 as i32),
            ),
        })
        .collect::<Vec<_>>();

    params.append(fields);
    params.into_iter().map(|combination| {
        let (id, t) = combination;
        (id, (t, gen_name(st)))
    })
}

fn fields_scope<'a>(
    fields: &'a Vec<Field>,
    st: &'a mut State<'a>,
) -> impl Iterator<Item = (&'a String, (Type, u32))> {
    fields.into_iter().map(|f| match f {
        Field::Scalar(t, id) => (&id.val.name, (Type::Prim(t.clone()), gen_name(st))),
        Field::Array(t, id, len) => (
            &id.val.name,
            (
                Type::Arr(t.clone(), lin_literal(len.val.clone()).1 as i32),
                gen_name(st),
            ),
        ),
    })
}

impl Program {
    fn local_scope<'a>(&'a self, st: &'a mut State<'a>) -> Scope<'a> {
        Scope::new(program_scope(self, st).collect(), None)
    }
}

impl Scoped for Method {
    fn local_scope<'a>(&'a self, st: &'a mut State<'a>) -> HashMap<&'a String, (Type, u32)> {
        method_scope(&self, st).collect()
    }
}

impl Scoped for Block {
    fn local_scope<'a>(&'a self, st: &'a mut State<'a>) -> HashMap<&'a String, (Type, u32)> {
        fields_scope(&self.fields, st).collect()
    }
}
