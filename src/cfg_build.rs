use crate::{
    cfg,
    ir::{self, Program},
    parse,
    scan::{self, IncrOp},
};
use cfg::{BlockLabel, CfgType, Jump, Type};
use ir::{AssignExpr, Block, Bop, Expr, Location, Method, Stmt, UnOp};
use parse::{Field, Literal, Primitive, WithLoc};
use scan::{AddOp, AssignOp, MulOp};

type Scope<'a> = ir::Scope<'a, (Type, u32)>;
pub type VarLabel = u32;
pub type BasicBlock = cfg::BasicBlock<VarLabel>;
pub type Arg = cfg::Arg<VarLabel>;
pub type Instruction = cfg::Instruction<VarLabel>;
pub type CfgMethod = cfg::CfgMethod<VarLabel>;
pub type CfgProgram = cfg::CfgProgram<VarLabel>;

struct State {
    break_loc: Option<BlockLabel>,
    continue_loc: Option<BlockLabel>,
    last_name: VarLabel,
    all_blocks: HashMap<BlockLabel, BasicBlock>,
    all_fields: HashMap<VarLabel, (CfgType, String /*high-level name*/)>,
    global_fields: HashMap<VarLabel, (CfgType, String /*high-level name*/)>,
}

impl State {
    fn add_block(&mut self, mut b: BasicBlock) -> BlockLabel {
        let id = self.all_blocks.len();
        b.block_id = id;
        let not_already_in = self.all_blocks.insert(id, b).is_none();
        assert!(not_already_in, "should be generating a unique label (you should not be calling this function after removing some blocks)");
        id
    }

    fn get_block(&mut self, lbl: BlockLabel) -> &mut BasicBlock {
        self.all_blocks.get_mut(&lbl).unwrap()
    }

    // should only be given scalar-type var as input
    fn type_of(&self, v: VarLabel) -> Primitive {
        match &self.all_fields.get(&v) {
            Some((CfgType::Scalar(t), _)) => t.clone(),
            Some((CfgType::Array(_, _), _)) => panic!(),
            None => {
                match &self.global_fields.get(&v) {
                    Some((CfgType::Scalar(t), _)) => t.clone(),
                    Some((CfgType::Array(_, _), _)) => panic!(),
                    None => {
                        panic!("Couldn't find low-level variable {}", v)
                    }
                }
                // println!("{:?}", self.all_fields);
                //  panic!("Couldn't find low-level variable {}", v)
            }
        }
    }

    fn gen_temp(&mut self, t: Primitive) -> VarLabel {
        gen_temp(t, &mut self.last_name, &mut self.all_fields)
    }
}

fn collapse_jumps(blks: &mut HashMap<BlockLabel, BasicBlock>, prune: bool) {
    get_parents(blks);

    let mut lbls_set: HashSet<BlockLabel> = blks.keys().map(|x| *x).collect();
    // for each parent, see if we can glue parent with its child
    // note that we completely screw up parent pointers, but it doesn't matter,
    // because the *number* of parents a block has is all that matters.
    while let Some(parent_lbl) = lbls_set.iter().next() {
        let parent_lbl = *parent_lbl;
        let parent = blks.get(&parent_lbl).unwrap().clone();

        match parent.jump_loc {
            Jump::Uncond(child_lbl) => {
                let child = blks.get_mut(&child_lbl).unwrap();
                if parent_lbl != child_lbl && child.parents.len() == 1 {
                    // glue parent with child
                    let mut parent_and_child = parent;
                    parent_and_child.body.append(&mut child.body);
                    parent_and_child.jump_loc = child.jump_loc.clone();
                    blks.insert(parent_lbl, parent_and_child);
                    // remove child
                    blks.remove(&child_lbl);
                    lbls_set.remove(&child_lbl);
                } else {
                    lbls_set.remove(&parent_lbl);
                }
            }
            _ => {
                lbls_set.remove(&parent_lbl);
            }
        }
    }
    // fix screwed-up parents
    get_parents(blks);
}

// might be prettier to have parents separate from cfg.  fewer things to worry about.  debatable.
fn get_parents(blocks: &mut HashMap<BlockLabel, BasicBlock>) {
    let mut parents = HashMap::new();
    for lbl in blocks.keys() {
        parents.insert(*lbl, vec![]);
    }
    for (parent_label, parent) in blocks.iter() {
        let children = match parent.jump_loc {
            Jump::Nowhere => vec![],
            Jump::Uncond(c1) => vec![c1],
            Jump::Cond {
                source: _,
                true_block,
                false_block,
            } => vec![true_block, false_block],
        };
        for child in children {
            parents.get_mut(&child).unwrap().push(*parent_label);
        }
    }
    for (child, child_parents) in parents {
        blocks.get_mut(&child).unwrap().parents = child_parents;
    }
}

fn type_to_prim(t: Type) -> Primitive {
    match t {
        Type::Prim(p) => p,
        Type::Arr(p, _) => p,
        Type::Func(_, Some(p)) => p,
        Type::ExtCall => Primitive::IntType,
        _ => panic!("Should be called on definite types"),
    }
}

fn new_noop(st: &mut State) -> BlockLabel {
    st.add_block(BasicBlock {
        parents: vec![],
        block_id: 0,
        body: vec![],
        jump_loc: Jump::Nowhere,
    })
}

fn gen_var(
    t: CfgType,
    high_name: String,
    last_name: &mut VarLabel,
    all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
) -> VarLabel {
    // println!("{}, {}", high_name, last_name);
    *last_name += 1;
    let name = *last_name;
    all_fields.insert(name, (t, high_name));
    // println!("{:?}", all_fields);
    name
}

fn gen_temp(
    t: Primitive,
    last_name: &mut VarLabel,
    all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
) -> VarLabel {
    gen_var(
        CfgType::Scalar(t),
        "temp".to_string(),
        last_name,
        all_fields,
    )
}

pub fn lin_program(program: &Program) -> CfgProgram {
    let mut global_fields = HashMap::new();
    let mut last_name = 0;
    let scope = program.scope(None, &mut last_name, &mut global_fields);
    let last_name = last_name;
    let methods = program
        .methods
        .iter()
        .map(|method| lin_method(method, last_name, &scope, &mut global_fields))
        .collect();

    CfgProgram {
        externals: program
            .imports
            .iter()
            .map(|c| c.val.name.clone())
            .collect::<Vec<_>>(),
        methods: methods,
        global_fields: global_fields,
    }
}

fn block_with_instr(instr: Instruction) -> BasicBlock {
    BasicBlock {
        parents: vec![],
        block_id: 0,
        body: vec![instr],
        jump_loc: Jump::Nowhere,
    }
}

pub fn lin_method(
    method: &Method,
    last_name: VarLabel,
    scope: &Scope,
    global_fields: &mut HashMap<VarLabel, (CfgType, String)>,
) -> CfgMethod {
    let mut st: State = State {
        break_loc: None,
        continue_loc: None,
        last_name,
        all_blocks: HashMap::new(),
        all_fields: HashMap::new(),
        global_fields: global_fields.clone(),
    };

    let fst: usize = new_noop(&mut st);
    let mut last = fst;
    let method_scope = method.scope(Some(&scope), &mut st.last_name, &mut st.all_fields);

    for s in method.stmts.iter() {
        let (start, end) = lin_stmt(s, &mut st, &method_scope);
        st.get_block(last).jump_loc = Jump::Uncond(start);
        last = end;
    }

    // get low_level_names of all parameters so we can lookup stack offsets at start of method when we asm
    let params = method
        .params
        .iter()
        .map(|c| {
            let (_, ll_name) = method_scope.lookup(&c.name.val).unwrap();
            *ll_name
        })
        .collect::<Vec<_>>();

    collapse_jumps(&mut st.all_blocks, true);
    // get_parents(&mut st.all_blocks);
    CfgMethod {
        name: method.name.val.to_string(),
        blocks: st.all_blocks,
        fields: st.all_fields,
        params: params,
        return_type: method.meth_type.clone(),
    }
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
            lin_branch(e2start, false_branch, &e1.val, st, scope) // e1start
        }
        Expr::Bin(e1, Bop::Or, e2) => {
            let e2start = lin_branch(true_branch, false_branch, &e2.val, st, scope);
            lin_branch(true_branch, e2start, &e1.val, st, scope) // e1start
        }
        _ => {
            let (t, tstart, tend) = lin_expr(cond, st, scope);

            st.get_block(tend).jump_loc = Jump::Cond {
                source: t,
                true_block: true_branch,
                false_block: false_branch,
            };
            tstart
        }
    }
}

// will call lin_branch to deal with bool exprs
fn lin_expr(e: &Expr, st: &mut State, scope: &Scope) -> (VarLabel, BlockLabel, BlockLabel) {
    match e {
        Expr::Bin(e1, op, e2) => match op {
            Bop::And | Bop::Or => {
                let end = new_noop(st);
                let temp = st.gen_temp(Primitive::BoolType);
                let true_branch = st.add_block(BasicBlock {
                    parents: vec![],
                    block_id: 0,
                    body: vec![Instruction::Constant {
                        dest: temp.clone(),
                        constant: 1,
                    }],
                    jump_loc: Jump::Uncond(end),
                });
                let false_branch = st.add_block(BasicBlock {
                    parents: vec![],
                    block_id: 0,
                    body: vec![Instruction::Constant {
                        dest: temp.clone(),
                        constant: 0,
                    }],
                    jump_loc: Jump::Uncond(end),
                });

                print!("{}", end);

                let start = lin_branch(true_branch, false_branch, e, st, scope);
                (temp, start, end)
            }
            _ => {
                let (t1, t1start, t1end) = lin_expr(&e1.val, st, scope);
                let (t2, t2start, t2end) = lin_expr(&e2.val, st, scope);
                st.get_block(t1end).jump_loc = Jump::Uncond(t2start);

                let t3 = st.gen_temp(infer_type(st.type_of(t1), op));
                let end = st.add_block(block_with_instr(Instruction::ThreeOp {
                    source1: t1,
                    source2: t2,
                    dest: t3,
                    op: op.clone(),
                }));
                st.get_block(t2end).jump_loc = Jump::Uncond(end);
                (t3, t1start, end)
            }
        },
        Expr::Unary(op, e) => {
            let (t1, t1start, t1end) = lin_expr(&e.val, st, scope);
            let t2 = st.gen_temp(infer_unary_type(st.type_of(t1), op));
            let end = st.add_block(block_with_instr(Instruction::TwoOp {
                source1: t1,
                dest: t2,
                op: op.clone(),
            }));
            st.get_block(t1end).jump_loc = Jump::Uncond(end);
            (t2, t1start, end)
        }
        Expr::Len(id) => match scope.lookup(&id.val) {
            Some((Type::Arr(_, len), _)) => {
                let t = st.gen_temp(Primitive::IntType);
                let blk = st.add_block(block_with_instr(Instruction::Constant {
                    dest: t.clone(),
                    constant: *len as i64,
                }));
                (t, blk, blk)
            }
            Some(_) => panic!("can only take len of array"),
            None => panic!("array identifier not found"),
        },
        Expr::Lit(lit) => {
            let (typ, val) = lin_literal(lit.val.clone());
            let t = st.gen_temp(typ);
            let end = st.add_block(block_with_instr(Instruction::Constant {
                dest: t,
                constant: val,
            }));
            (t, end, end)
        }
        ir::Expr::Loc(loc) => {
            return lin_location(&loc.val, st, scope);
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

            let ret_val_type = match scope.lookup(&id.val) {
                Some((Type::Func(_, p), _)) => p.clone().unwrap(),
                Some((Type::ExtCall, _)) => Primitive::IntType,
                _ => panic!("could not find function name"),
            };
            let ret_val = st.gen_temp(ret_val_type);
            let call_instr = Instruction::Call(func_name, temp_args, Some(ret_val));
            let end = st.add_block(block_with_instr(call_instr));
            st.get_block(prev_block).jump_loc = Jump::Uncond(end);
            (ret_val, start, end)
        }
    }
}

fn lin_block(b: &Block, st: &mut State, scope: &Scope) -> (BlockLabel, BlockLabel) {
    let fst = new_noop(st);
    let mut last = fst;
    let block_scope = b.scope(Some(scope), &mut st.last_name, &mut st.all_fields);

    for s in b.stmts.iter() {
        let (start, end) = lin_stmt(s, st, &block_scope);
        st.get_block(last).jump_loc = Jump::Uncond(start);
        last = end;
    }

    (fst, last)
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
            lin_assign_to_loc(&loc.val, assign_expr, st, scope)
        }
        ir::Stmt::Break(_) => match st.break_loc {
            Some(break_block) => {
                let start = new_noop(st);
                st.get_block(start).jump_loc = Jump::Uncond(break_block);
                (start, new_noop(st))
            }
            _ => panic!("should not get here"),
        },
        ir::Stmt::Continue(_) => match st.continue_loc {
            Some(continue_block) => {
                let start = new_noop(st);
                st.get_block(start).jump_loc = Jump::Uncond(continue_block);
                (start, new_noop(st))
            }
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
                Some((Type::Prim(t), _)) => Some(st.gen_temp(t.clone())),
                _ => None,
            };

            let call_instr = Instruction::Call(func_name, temp_args, ret_val);
            let end = st.add_block(block_with_instr(call_instr));
            st.get_block(prev_block).jump_loc = Jump::Uncond(end);
            (start, end)
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
            (continue_target, end)
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
                lin_location(&Location::Var(var_to_set.val.clone()), st, scope);
            let (loop_init_start, loop_init_end) =
                lin_reg_assign(loop_var, &AssignOp::Eq, &initial_val.val, st, scope);
            st.get_block(loop_end).jump_loc = Jump::Uncond(loop_init_start);

            let end = new_noop(st);

            // Modify state so that we point to continue and break to the right places.

            // change to handle increments and decrements better
            let (update_var, loop_update, update_loc_end) =
                lin_location(&var_to_update.val, st, scope);
            let (update_start, update_end) = lin_assign_expr(update_var, update_val, st, scope);

            st.get_block(update_loc_end).jump_loc = Jump::Uncond(update_start);

            // update break and continue locations
            let old_break_loc = st.break_loc;
            let old_continue_loc = st.continue_loc;
            st.break_loc = Some(end);
            st.continue_loc = Some(loop_update);

            let (body_start, body_end) = lin_block(body, st, scope);

            // Restore old break and continue locations
            st.break_loc = old_break_loc;
            st.continue_loc = old_continue_loc;

            st.get_block(body_end).jump_loc = Jump::Uncond(loop_update);

            let condition_start = lin_branch(body_start, end, &test.val, st, scope);

            st.get_block(loop_init_end).jump_loc = Jump::Uncond(condition_start);
            st.get_block(update_end).jump_loc = Jump::Uncond(condition_start);

            (loop_start, end)
        }
        ir::Stmt::Return(_, ret_val) => match ret_val {
            Some(expr) => {
                let (t, tstart, tend) = lin_expr(&expr.val, st, scope);
                let ret_instr = Instruction::Ret(Some(t));
                st.get_block(tend).body.push(ret_instr);
                (tstart, tend)
            }
            None => {
                let ret_block = st.add_block(block_with_instr(Instruction::Ret(None)));
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
    target_id: VarLabel,
    op: &AssignOp,
    rhs: &Expr,
    st: &mut State,
    scope: &Scope,
) -> (BlockLabel, BlockLabel) {
    let (t1, t1start, t1end) = lin_expr(rhs, st, scope);

    let op_ = convert_assign_op(op);
    let instr = match op_ {
        None => Instruction::MoveOp {
            source: t1,
            dest: target_id,
        },
        Some(binop) => Instruction::ThreeOp {
            source1: target_id,
            source2: t1,
            dest: target_id,
            op: binop,
        },
    };
    let end = st.add_block(block_with_instr(instr));
    st.get_block(t1end).jump_loc = Jump::Uncond(end);
    (t1start, end)
}

// should work even when loc is array idx
fn lin_assign_to_loc(
    loc: &Location,
    val_to_assign: &AssignExpr,
    st: &mut State,
    scope: &Scope,
) -> (BlockLabel, BlockLabel) {
    match loc {
        Location::Var(id) => {
            let ll_name = scope.lookup(id).unwrap().1;
            lin_assign_expr(ll_name, val_to_assign, st, scope)
        }
        Location::ArrayIndex(id, idx) => {
            let (t, ll_arr_name) = scope.lookup(id).unwrap();
            let (idx, idx_start, idx_end) = lin_expr(&idx.val, st, scope);
            let temp = st.gen_temp(type_to_prim(t.clone()));
            let load_block = st.add_block(block_with_instr(Instruction::ArrayAccess {
                dest: temp,
                name: *ll_arr_name,
                idx,
            }));
            // what are these naming conventions????!
            let (ass_start, ass_end) = lin_assign_expr(temp, val_to_assign, st, scope);
            let store_block = st.add_block(block_with_instr(Instruction::ArrayStore {
                source: temp,
                arr: *ll_arr_name,
                idx,
            }));
            let (start, end) = link(idx_start, idx_end, load_block, load_block, st);
            let (start, end) = link(start, end, ass_start, ass_end, st);
            let (start, end) = link(start, end, store_block, store_block, st);
            (start, end)
        }
    }
}

// assumes target is just a scalar
fn lin_assign_expr(
    target: VarLabel,
    assign_expr: &AssignExpr,
    st: &mut State,
    scope: &Scope,
) -> (BlockLabel, BlockLabel) {
    match assign_expr {
        AssignExpr::RegularAssign(op, rhs) => lin_reg_assign(target, &op.val, &rhs.val, st, scope),
        AssignExpr::IncrAssign(op) => {
            let t1 = st.gen_temp(Primitive::IntType);

            // can just do the instructions in sequence in one block.
            let end = st.add_block(block_with_instr(Instruction::ThreeOp {
                source1: target,
                source2: t1,
                dest: target,
                op: convert_incr_op(op.val.clone()),
            }));
            let start = st.add_block(block_with_instr(Instruction::Constant {
                dest: t1,
                constant: 1,
            }));
            st.get_block(start).jump_loc = Jump::Uncond(end);
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
        Literal::DecLong(s) => (Primitive::LongType, str::parse(s.as_str()).unwrap()),
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

// Call for *getting the value*, not for setting
fn lin_location(
    loc: &Location,
    st: &mut State,
    scope: &Scope,
) -> (VarLabel, BlockLabel, BlockLabel) {
    match loc {
        Location::Var(name) => match scope.lookup(&name) {
            Some((Type::Prim(_), id)) => {
                let blk = new_noop(st);
                (*id, blk, blk)
            }
            Some(_) => panic!("location should have primitive type"),
            None => {
                println!("actually didn't find it");
                panic!("location not found")
            }
        },
        Location::ArrayIndex(name, idx) => match scope.lookup(&name) {
            Some((Type::Arr(typ, _), id)) => {
                let (idx_val, tstart, tend) = lin_expr(&idx.val, st, scope);

                let dest = st.gen_temp(typ.clone());
                let instr = Instruction::ArrayAccess {
                    dest: dest,
                    name: *id,
                    idx: idx_val,
                };

                let block = st.add_block(block_with_instr(instr));

                st.get_block(tend).jump_loc = Jump::Uncond(block);

                (dest, tstart, block)
            }
            Some(_) => panic!("array should be an array"),
            None => panic!("array name not found"),
        },
    }
}

use std::collections::{HashMap, HashSet};

trait Scoped {
    fn scope<'a>(
        &'a self,
        parent: Option<&'a Scope>,
        last_name: &mut VarLabel,
        all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
    ) -> Scope<'a> {
        Scope::new(self.local_scope(last_name, all_fields), parent)
    }
    fn local_scope<'a>(
        &'a self,
        last_name: &mut VarLabel,
        all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
    ) -> HashMap<&'a String, (Type, u32)>;
}

fn imports_scope(
    imports: &Vec<WithLoc<parse::Ident>>,
) -> impl Iterator<Item = (&String, (Type, u32))> {
    imports
        .iter()
        .map(move |import| (&import.val.name, (Type::ExtCall, 0 /*never used*/)))
}

fn outer_methods_scope(methods: &Vec<Method>) -> impl Iterator<Item = (&String, (Type, u32))> {
    methods.iter().map(|method| {
        let param_types = method
            .params
            .iter()
            .map(|param| param.param_type.clone())
            .collect();
        let hl_name = &method.name.val.name;
        let ret_type = method.meth_type.clone();
        (hl_name, (Type::Func(param_types, ret_type), 0))
    })
}

fn method_scope<'a, 'b>(
    method: &'a Method,
    last_name: &'b mut VarLabel,
    all_fields: &'b mut HashMap<VarLabel, (CfgType, String)>,
) -> impl Iterator<Item = (&'a String, (Type, u32))> + use<'a, 'b> {
    let params = method
        .params
        .iter()
        .map(|param| (&param.name.val.name, Type::Prim(param.param_type.clone())));

    let params_scope = params
        .map(|(id, t)| {
            let p = type_to_prim(t.clone());
            let name = gen_var(CfgType::Scalar(p), id.to_string(), last_name, all_fields);
            (id, (t.clone(), name))
        })
        .collect::<Vec<_>>();

    params_scope
        .into_iter()
        .chain(fields_scope(&method.fields, last_name, all_fields))
}

fn fields_scope<'a, 'b>(
    fields: &'a Vec<Field>,
    last_name: &'b mut VarLabel,
    all_fields: &'b mut HashMap<VarLabel, (CfgType, String)>,
) -> impl Iterator<Item = (&'a String, (Type, VarLabel))> + use<'a, 'b> {
    fields.into_iter().map(|f| match f {
        Field::Scalar(t, id) => {
            let name = gen_var(
                CfgType::Scalar(t.clone()),
                id.val.name.clone(),
                last_name,
                all_fields,
            );
            (&id.val.name, (Type::Prim(t.clone()), name))
        }
        Field::Array(t, id, len) => {
            let name = gen_var(
                CfgType::Array(t.clone(), lin_literal(len.val.clone()).1 as i32),
                id.val.name.clone(),
                last_name,
                all_fields,
            );
            let typ = Type::Arr(t.clone(), lin_literal(len.val.clone()).1 as i32);
            (&id.val.name, (typ, name))
        }
    })
}

impl Scoped for Program {
    fn local_scope<'a>(
        &'a self,
        last_name: &mut VarLabel,
        all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
    ) -> HashMap<&'a String, (Type, VarLabel)> {
        imports_scope(&self.imports)
            .chain(outer_methods_scope(&self.methods))
            .chain(fields_scope(&self.fields, last_name, all_fields))
            .collect()
    }
}

impl Scoped for Method {
    fn local_scope<'a>(
        &'a self,
        last_name: &mut VarLabel,
        all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
    ) -> HashMap<&'a String, (Type, u32)> {
        method_scope(&self, last_name, all_fields).collect()
    }
}

impl Scoped for Block {
    fn local_scope<'a>(
        &'a self,
        last_name: &mut VarLabel,
        all_fields: &mut HashMap<VarLabel, (CfgType, String)>,
    ) -> HashMap<&'a String, (Type, u32)> {
        fields_scope(&self.fields, last_name, all_fields).collect()
    }
}
