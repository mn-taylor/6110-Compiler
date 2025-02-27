use crate::ir::*;
use crate::parse;

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


fn build_for(ast_for: parse::For, parent_scope: Scope){
    match ast_for{
        Stmt::For(withloc_idex, initial_value, condition, location, assignment_expr)=>{
            let for_scope = Scope{
                vars: [(withloc_idex, initial_value)],
                parent: scope
            };

            let ir_initial_value = build_expr(inital_value);
            let ir_condition = build_expr(condition);
            let identifier = build_expr(location);
            let assigment = build_assign(assignment_expr);
            let block = build_block(block);
            
            return ir::Stmt::For(withloc_idx, inital_value, ir_condition, identifier, assigment, block, for_scope);
        }

        _=> {}
    }
    return None
}