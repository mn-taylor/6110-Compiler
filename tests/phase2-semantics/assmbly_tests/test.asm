.data
global1:
        .zero 8
.text
error_handler:
        movl $0x2000001, %eax
        movl $-1, %edi
        syscall
.globl _main
_main:
        call main
        movq $0, %rax
        ret
main:
        pushq %rbp
        movq %rsp, %rbp
        subq $16, %rsp
        jmp main0
main0:
        movq $5, %rax
        movq %rax, -0(%rbp)
        movq -0(%rbp), %rax

        call printf
        movq %rax, global1(%rip)


        jmp mainend
mainend:
        addq $16, %rsp
        popq %rbp
        ret 