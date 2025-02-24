// use crate::parse::Type;
// use crate::scan::Sum;

// enum Literal {
//     IntLit(i32),
//     LongLit(i64),
//     CharLit(char),
//     BoolLit(bool),
// }

// enum Stmt {
//     AssignStmt(Var, Expr),
//     SelfAssign(Var, Op, Expr),
//     // about same as in parser
// }

// enum Expr {
//     Bin(Expr, Op, Expr),
//     Unary(UnaryOp, Expr),
//     Len(Ident),
//     IntCast(Box<Expr>),
//     LongCast(Box<Expr>),
//     Loc(Box<Location>),
//     Call(Ident, Vec<Arg>),
// }

// type ExprWithType = (Expr, Type);

// struct GlobalScope {
//     vars: Vec<(Ident, Type)>,
//     parent: Sum<Box<LocalScope>, Box<GlobalScope>>,
//     methods: Vec<Method>,
//     exts: Vec<Ident>,
// }

// struct Method {
//     body: Block,
//     params: Vec<(Ident, Expr)>,
// }

// struct Block {
//     vars: Vec<(String, Type)>,
//     parent: Sum<Box<Block>, Box<GlobalScope>>,
//     stmts: Vec<Stmt>,
// }
