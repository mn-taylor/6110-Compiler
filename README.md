# Decaffineated C Compiler
A tiny C compiler implemented in Rust. It lexes, parses, type‑checks, lowers to a simple SSA‑like IR, performs a few classic optimizations, and emits x86‑64 (System V) or via an integrated code generator and linker shim.

