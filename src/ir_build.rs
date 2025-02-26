use crate::ir::*;

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
