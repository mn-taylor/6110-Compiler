fn noop() -> BasicBlock {
    todo!()
}

struct State {
    break_loc: Option<&BasicBlock>,
    continue_loc: Option<&BasicBlock>,
    last_name: u32,
    all_blocks: Vec<BasicBlock>,
    scope: LowScope
}

fn gen_temp_name(typ: Type, st: &mut State) -> Var {
    st.last_name += 1;
    Scalar {
	id: st.last_name,
	typ,
	name: "temp",
    }
}

fn gen_name_from_hl_name(high_level_name: String, typ: Type, st: &mut State) -> Var {
    st.last_name += 1;
    st.scope.put(high_level_name, (typ, st.last_name));
    Scalar {
        id: st.last_name,
	typ: st.scope.lookup_type(high_level_name),
        name: high_level_name,
    }
}

fn lin_branch(true_branch: &BasicBlock, false_branch: &BasicBlock, cond: Expr) -> &BasicBlock/*start*/ {
    match cond {
	Expr::Bin(e1, And, e2) => {
	    let e2start = lin_branch(true_branch, false_branch, e2);
	    let e1start = lin_branch(e2start, false_branch, e1);
	    return e1start;
	}
	Expr::Bin(e1, Or, e2) => {
	    let e2start = lin_branch(true_branch, false_branch, e2);
	    let e1start = lin_branch(true_branch, e2start, e1);
	    return e1start;
	}
	_ => {
	    let (t, tstart, tend) = lin_expr(cond, st);
	    tend.jump_loc = cond {
		source: t,
		true_block: true_branch,
		false_block: false_branch
	    };
	    return tstart;
	}
    }
}

// will call lin_branch to deal with bool exprs
fn lin_expr(e: Expr, st: &mut State) -> (Var, &BasicBlock, &BasicBlock) {
    let start = noop();
    match e {
	Expr::Bin(e1, op, e2) => {
	    match op {
		And | Or => {
		    let end = noop();
		    let temp = gen_temp(st);
		    let true_branch = block that sets temp = true and jumps to end;
		    let false_branch = block taht  sets temp = fasle and jumpts to end;
		    st.all_blocks.push(true_branch);
		    st.all_blocks.push(false_branch);
		    let start = lin_branch(&true_branch, &false_branch, e);
		    return (temp, start, end);
		}
		_=>{
		    let (t1, t1start, t1end) = lin_expr(e1, st);
		    let (t2, t2start, t2end) = lin_expr(e2, st);
		    t1end.jump = uncond(&t2start);
		    let t3 = gen_temp(infer_type(t1.typ, op), st);
		    let end = BasicBlock {
			body: vec![threeOp {
			    source1: t1,
			    source2: t2,
			    dest: t3,
			    op: op
			}],
			jump_loc: nowhere,
		    };
		    st.all_blocks.push(end);
		    t2end.jump = uncond(&end);
		   return (t3, &t1start, &end);
		}
	    }
	}
    }
}

fn lin_block(b: Block, s: State) {
    let fst = noop();
    let fst = &fst;
    let last = &fst;
    for s in b.stmts {
        let (start, end) = lin_stmt(s, all_blocks);
        last.jump_loc = uncond { start };
        last = end;
    }
}

fn lin_stmt(s: Stmt, all_blocks: Vec<BasicBlock>) -> (&BasicBlock, &BasicBlock) {}
