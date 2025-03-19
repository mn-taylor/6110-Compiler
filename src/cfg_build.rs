use crate::{
    cfg,
    ir::{self, Program},
    parse,
    scan::{self, IncrOp},
};
use cfg::{Arg, BasicBlock, BlockLabel, Instruction, Jump, Type, Var, VarLabel};
use ir::{AssignExpr, Block, Bop, Expr, Location, Method, Stmt, UnOp};
use parse::{Field, Literal, Primitive, WithLoc};
use scan::{AddOp, AssignOp, MulOp};

type Scope<'a> = ir::Scope<'a, (Type, u32)>;

struct State {
    break_loc: Option<BlockLabel>,
    continue_loc: Option<BlockLabel>,
    last_name: VarLabel,
    all_blocks: HashMap<BlockLabel, BasicBlock>,
    all_fields: Vec<Field>,
}

impl State {
    fn add_block(&mut self, b: BasicBlock) -> BlockLabel {
        let id = self.all_blocks.len();
        let not_already_in = self.all_blocks.insert(id, b).is_none();
        assert!(not_already_in, "should be generating a unique label (you should not be calling this function after removing some blocks)");
        id
    }

    fn get_block(&mut self, lbl: BlockLabel) -> &mut BasicBlock {
        self.all_blocks.get_mut(&lbl).unwrap()
    }
}

fn new_noop(st: &mut State) -> BlockLabel {
    st.add_block(BasicBlock {
        body: vec![],
        jump_loc: Jump::Nowhere,
    })
}

fn gen_name(st: &mut State) -> VarLabel {
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

fn lin_program(program: &Program) -> State {
    let mut st: State = State {
        break_loc: None,
        continue_loc: None,
        last_name: 0,
        all_blocks: HashMap::new(),
        all_fields: vec![],
    };

    // Get the local scope from the program
    let scope = program.scope(None, &mut st);

    // Process methods
    for method in &program.methods {
        lin_method(method, &mut st, &scope);
    }
    st
}

fn lin_method(method: &Method, st: &mut State, scope: &Scope) -> (BlockLabel, BlockLabel) {
    let fst = new_noop(st);
    let mut last = fst;
    let method_scope = method.scope(Some(scope), st);

    for s in method.stmts.iter() {
        let (start, end) = lin_stmt(s, st, &method_scope);
        st.get_block(last).jump_loc = Jump::Uncond(start);
        last = end;
    }

    return (fst, last);
}

fn lin_branch(
    true_branch: BlockLabel,
    false_branch: BlockLabel,
    cond: &Expr,
    st: &mut State,
    scope: &Scope,
) -> BlockLabel /*start*/
{
    match cond {
        Expr::Bin(e1, Bop::And, e2) => {
            let e2start = lin_branch(true_branch, false_branch, &e2.val, st, scope);
            let e1start = lin_branch(e2start, false_branch, &e1.val, st, scope);
            return e1start;
        }
        Expr::Bin(e1, Bop::Or, e2) => {
            let e2start = lin_branch(true_branch, false_branch, &e2.val, st, scope);
            let e1start = lin_branch(true_branch, e2start, &e1.val, st, scope);
            return e1start;
        }
        _ => {
            let (t, tstart, tend) = lin_expr(cond, st, scope);
            st.get_block(tend).jump_loc = Jump::Cond {
                source: t,
                true_block: true_branch,
                false_block: false_branch,
            };
            return tstart;
        }
    }
}

// will call lin_branch to deal with bool exprs
fn lin_expr(e: &Expr, st: &mut State, scope: &Scope) -> (Var, BlockLabel, BlockLabel) {
    match e {
        Expr::Bin(e1, op, e2) => {
            match op {
                Bop::And | Bop::Or => {
                    let end = new_noop(st);
                    let temp = gen_temp(Primitive::BoolType, st);
                    let true_branch = st.add_block(BasicBlock {
                        body: vec![Instruction::Constant {
                            dest: temp.clone(),
                            constant: 1,
                        }],
                        jump_loc: Jump::Uncond(end),
                    }); //block that sets temp = true and jumps to end;
                    let false_branch = st.add_block(BasicBlock {
                        body: vec![Instruction::Constant {
                            dest: temp.clone(),
                            constant: 1,
                        }],
                        jump_loc: Jump::Uncond(end),
                    }); //block taht  sets temp = fasle and jumpts to end;
                    let start = lin_branch(true_branch, false_branch, e, st, scope);
                    return (temp, start, end);
                }
                _ => {
                    let (t1, t1start, t1end) = lin_expr(&e1.val, st, scope);
                    let (t2, t2start, t2end) = lin_expr(&e2.val, st, scope);
                    st.get_block(t1end).jump_loc = Jump::Uncond(t2start);
                    let t3 = gen_temp(infer_type(t1.get_typ(), op), st);
                    let end = st.add_block(BasicBlock {
                        body: vec![Instruction::ThreeOp {
                            source1: t1,
                            source2: t2,
                            dest: t3.clone(),
                            op: *op,
                        }],
                        jump_loc: Jump::Nowhere,
                    });
                    st.get_block(t2end).jump_loc = Jump::Uncond(end);
                    return (t3, t1start, end);
                }
            }
        }
        Expr::Unary(op, e) => {
            let (t1, t1start, _t1end) = lin_expr(&e.val, st, scope);
            let t2 = gen_temp(infer_unary_type(t1.get_typ(), op), st);
            let end = st.add_block(BasicBlock {
                body: vec![Instruction::TwoOp {
                    source1: t1.clone(),
                    dest: t2,
                    op: *op,
                }],
                jump_loc: Jump::Nowhere,
            });
            return (t1, t1start, end);
        }
        Expr::Len(id) => match scope.lookup(&id.val) {
            Some((Type::Arr(_, len), _)) => {
                let t = gen_temp(Primitive::IntType, st);
                let blk = st.add_block(BasicBlock {
                    body: vec![Instruction::Constant {
                        dest: t.clone(),
                        constant: *len as i64,
                    }],
                    jump_loc: Jump::Nowhere,
                });
                (t, blk, blk)
            }
            Some(_) => panic!("can only take len of array"),
            None => panic!("array identifier not found"),
        },
        Expr::Lit(lit) => {
            let (typ, val) = lin_literal(lit.val.clone());
            let t = gen_temp(typ, st);
            let end = st.add_block(BasicBlock {
                body: vec![Instruction::Constant {
                    dest: t.clone(),
                    constant: val,
                }],
                jump_loc: Jump::Nowhere,
            });
            return (t, end, end);
        }
        ir::Expr::Loc(loc) => {
            return lin_location(loc.val, st, scope);
        }
        ir::Expr::Call(id, args) => {
            let func_name = id.val.name.clone();

            let start = new_noop(st);
            let mut prev_block = start;
            let mut temp_args: Vec<Arg> = vec![];
            for arg in args {
                match arg {
                    ir::Arg::ExprArg(WithLoc { val: e1, loc: _ }) => {
                        let (t, tstart, tend) = lin_expr(e1, st, scope);
                        let cfg_arg = Arg::VarArg(t);
                        temp_args.push(cfg_arg);

                        st.get_block(prev_block).jump_loc = Jump::Uncond(tstart);
                        prev_block = tend;
                    }
                    ir::Arg::ExternArg(WithLoc {
                        val: string,
                        loc: _,
                    }) => {
                        let cfg_arg = Arg::StrArg(string.to_string());
                        temp_args.push(cfg_arg);
                    }
                }
            }

            let ret_val = match scope.lookup(&id.val) {
                Some((Type::Prim(t), _)) => gen_temp(t.clone(), st),
                _ => panic!("Should not get here. function calls within expression must have non-void return type"),
            };
            let call_instr = cfg::Instruction::Call(func_name, temp_args, Some(ret_val.clone()));
            let end = st.add_block(BasicBlock {
                body: vec![call_instr],
                jump_loc: Jump::Nowhere,
            });
            return (ret_val, start, end);
        }
    }
}

fn lin_block(b: &Block, st: &mut State, scope: &Scope) -> (BlockLabel, BlockLabel) {
    let fst = new_noop(st);
    let mut last = fst;
    let block_scope = b.scope(Some(scope), st);

    for s in b.stmts.iter() {
        let (start, end) = lin_stmt(s, st, &block_scope);
        st.get_block(last).jump_loc = Jump::Uncond(start);
        last = end;
    }

    return (fst, last);
}

fn infer_unary_type(typ: Primitive, op: &UnOp) -> Primitive {
    match op {
        UnOp::Neg => typ,
        UnOp::Not => Primitive::BoolType,
        UnOp::IntCast => Primitive::IntType,
        UnOp::LongCast => Primitive::LongType,
    }
}

fn infer_type(typ: Primitive, op: &Bop) -> Primitive {
    match op {
        Bop::MulBop(_) | Bop::AddBop(_) => typ,
        _ => Primitive::BoolType,
    }
}

fn link<'a>(
    start1: BlockLabel,
    end1: BlockLabel,
    start2: BlockLabel,
    end2: BlockLabel,
    st: &mut State,
) -> (BlockLabel, BlockLabel) {
    st.get_block(end1).jump_loc = Jump::Uncond(start2);
    (start1, end2)
}

fn lin_stmt(s: &Stmt, st: &mut State, scope: &Scope) -> (BlockLabel, BlockLabel) {
    match s {
        ir::Stmt::AssignStmt(loc, assign_expr) => {
            let (target, target_start, target_end) = lin_location(loc.val, st, scope);
            let (tstart, tend) = lin_assign_expr(target, assign_expr, st, scope);
            link(target_start, target_end, tstart, tend, st)
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
            let func_name = id.val.name.clone();

            let start = new_noop(st);
            let mut prev_block = start;
            let mut temp_args: Vec<Arg> = vec![];
            for arg in args {
                match arg {
                    ir::Arg::ExprArg(WithLoc { val: e1, loc: _ }) => {
                        let (t, tstart, tend) = lin_expr(e1, st, scope);
                        let cfg_arg = Arg::VarArg(t);
                        temp_args.push(cfg_arg);

                        st.get_block(prev_block).jump_loc = Jump::Uncond(tstart);
                        prev_block = tend;
                    }
                    ir::Arg::ExternArg(WithLoc {
                        val: string,
                        loc: _,
                    }) => {
                        let cfg_arg = Arg::StrArg(string.to_string());
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
            let end = st.add_block(BasicBlock {
                body: vec![call_instr],
                jump_loc: Jump::Nowhere,
            });

            return (start, end);
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
            st.get_block(if_end).jump_loc = Jump::Uncond(end);
            st.get_block(else_end).jump_loc = Jump::Uncond(end);

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
            st.break_loc = Some(end);
            st.continue_loc = Some(continue_target);

            let (while_block_start, while_block_end) = lin_block(block, st, scope);

            // Restore old break/continue locations
            st.break_loc = old_break_loc;
            st.continue_loc = old_continue_loc;

            let while_condition = lin_branch(while_block_start, end, condition, st, scope);

            st.get_block(continue_target).jump_loc = Jump::Uncond(while_condition);
            st.get_block(while_block_end).jump_loc = Jump::Uncond(while_condition);

            return (while_block_start, end);
        }
        ir::Stmt::For {
            var_to_set,
            initial_val,
            test,
            var_to_update,
            update_val,
            body,
        } => {
            let (loop_var, loop_start, loop_end) =
                lin_location(Location::Var(var_to_set.val.clone()), st, scope);
            let (loop_init_start, loop_init_end) =
                lin_reg_assign(loop_var, &AssignOp::Eq, &initial_val.val, st, scope);
            st.get_block(loop_end).jump_loc = Jump::Uncond(loop_init_start);

            let end = new_noop(st);

            // Modify state so that we point to continue and break to the right places.
            let continue_target = new_noop(st);

            // update break and continue locations
            let old_break_loc = st.break_loc;
            let old_continue_loc = st.continue_loc;
            st.break_loc = Some(end);
            st.continue_loc = Some(continue_target);

            let (body_start, body_end) = lin_block(body, st, scope);

            // Restore old break and continue locations
            st.break_loc = old_break_loc;
            st.continue_loc = old_continue_loc;

            // change to handle increments and decrements better
            let (update_var, loop_update, update_loc_end) =
                lin_location(var_to_update.val, st, scope);
            let (update_start, _update_end) = lin_assign_expr(update_var, update_val, st, scope);

            st.get_block(update_loc_end).jump_loc = Jump::Uncond(update_start);

            st.get_block(body_end).jump_loc = Jump::Uncond(loop_update);

            let condition_start = lin_branch(body_start, end, &test.val, st, scope);
            st.get_block(continue_target).jump_loc = Jump::Uncond(loop_update);
            st.get_block(loop_init_end).jump_loc = Jump::Uncond(condition_start);
            st.get_block(loop_update).jump_loc = Jump::Uncond(condition_start);

            return (loop_start, end);
        }
        ir::Stmt::Return(_, ret_val) => match ret_val {
            Some(expr) => {
                let (t, tstart, tend) = lin_expr(&expr.val, st, scope);
                let ret_instr = Instruction::Ret(Some(t));
                st.get_block(tend).body.push(ret_instr);
                (tstart, tend)
            }
            None => {
                let ret_instr = Instruction::Ret(None);
                let ret_block = st.add_block(BasicBlock {
                    body: vec![ret_instr],
                    jump_loc: Jump::Nowhere,
                });
                (ret_block, ret_block)
            }
        },
    }
}

fn convert_assign_op(o: &AssignOp) -> Option<Bop> {
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

fn lin_reg_assign(
    target: Var,
    op: &AssignOp,
    rhs: &Expr,
    st: &mut State,
    scope: &Scope,
) -> (BlockLabel, BlockLabel) {
    // t1end being unused is surely a bug
    let (t1, t1start, _t1end) = lin_expr(rhs, st, scope);

    let op_ = convert_assign_op(op);
    let instr = match op_ {
        None => Instruction::MoveOp {
            source: t1.clone(),
            dest: target.clone(),
        },
        Some(binop) => Instruction::ThreeOp {
            source1: target.clone(),
            source2: t1,
            dest: target,
            op: binop,
        },
    };
    let end = st.add_block(BasicBlock {
        body: vec![instr],
        jump_loc: Jump::Nowhere,
    });
    (t1start, end)
}

fn lin_assign_expr(
    target: Var,
    assign_expr: &AssignExpr,
    st: &mut State,
    scope: &Scope,
) -> (BlockLabel, BlockLabel) {
    match assign_expr {
        AssignExpr::RegularAssign(op, rhs) => lin_reg_assign(target, &op.val, &rhs.val, st, scope),
        AssignExpr::IncrAssign(op) => {
            let t1 = gen_temp(Primitive::IntType, st);
            let end = st.add_block(BasicBlock {
                body: vec![Instruction::ThreeOp {
                    source1: target.clone(),
                    source2: t1.clone(),
                    dest: target,
                    op: convert_incr_op(op.val.clone()),
                }],
                jump_loc: Jump::Nowhere,
            });
            let start = st.add_block(BasicBlock {
                body: vec![Instruction::Constant {
                    dest: t1,
                    constant: 1,
                }],
                jump_loc: Jump::Uncond(end),
            });
            (start, end)
        }
    }
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

fn lin_location(loc: Location, st: &mut State, scope: &Scope) -> (Var, BlockLabel, BlockLabel) {
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
                let (idx_val, tstart, tend) = lin_expr(&idx.val, st, scope);
                return (
                    Var::ArrIdx {
                        id: *id,
                        name: name.name,
                        idx: idx_val.get_id(),
                        typ: typ.clone(),
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
    fn scope<'a>(&'a self, parent: Option<&'a Scope>, st: &mut State) -> Scope<'a> {
        Scope::new(self.local_scope(st), parent)
    }
    fn local_scope<'a>(&'a self, st: &mut State) -> HashMap<&'a String, (Type, u32)>;
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

fn method_scope<'a, 'b>(
    method: &'a Method,
    st: &'b mut State,
) -> impl Iterator<Item = (&'a String, (Type, u32))> + use<'a, 'b> {
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

fn fields_scope<'a, 'b>(
    fields: &'a Vec<Field>,
    st: &'b mut State,
) -> impl Iterator<Item = (&'a String, (Type, u32))> + use<'a, 'b> {
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

impl Scoped for Program {
    fn local_scope<'a>(&'a self, st: &mut State) -> HashMap<&'a String, (Type, u32)> {
        imports_scope(&self.imports)
            .chain(outer_methods_scope(&self.methods))
            .chain(fields_scope(&self.fields, st))
            .collect()
    }
}

impl Scoped for Method {
    fn local_scope<'a>(&'a self, st: &mut State) -> HashMap<&'a String, (Type, u32)> {
        method_scope(&self, st).collect()
    }
}

impl Scoped for Block {
    fn local_scope<'a>(&'a self, st: &mut State) -> HashMap<&'a String, (Type, u32)> {
        fields_scope(&self.fields, st).collect()
    }
}
