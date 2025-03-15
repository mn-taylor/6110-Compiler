struct BasicBlock {
    // parents: Vec<BasicBlock>,
    body: Vec<Instructions>,
    jump_loc: Jump,
}

enum Jump {
    uncond(&BasicBlock),
    cond {
        source: Var,
        true_block: &BasicBlock,
        false_block: &BasicBlock,
    },
    nowhere,
}

enum Instructions {
    threeOp {
        source1: Var,
        source2: Var,
        dest: Var,
        op: BinOp,
    },
    twoOp {
        source1: Var,
        dest: Var,
        op: UnOp,
    },
    constant {
        dest: Var,
        constant: i64,
    },
    ret(Option<Var>),
    call(String, Vec<Arg>),
}

enum Arg {
    VarArg(Var),
    StrArg(String),
}

enum Var {
    Scalar {
        id: u32,
        name: String,
        typ: Primitive,
    },
    ArrIdx {
        id: u32,
        name: String,
        idx: i32,
        typ: Primitive,
    },
}
