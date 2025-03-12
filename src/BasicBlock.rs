struct BasicBlock {
    parents: Vec<BasicBlock>,
    children: Vec<BasicBlock>,
    body: Vec<Instructions>,
}

enum Instructuins {
    threeOp {
        source1: Source,
        source2: Source,
        dest: Source,
        op: BinOp,
    },
    twoOp {
        source1: Source,
        dest: Source,
        op: UnOp,
    },
    oneOp {
        dest: Source,
        constant: i64,
    },
    uncoditionJump {
        label: BasicBlock,
    },
    conditionalJump {
        source: Source,
        jumptype: JumpType,
        ifblock: BasicBlock,
        elseblock: BasicBlock,
    },
}

struct Source {
    name: string,
}
