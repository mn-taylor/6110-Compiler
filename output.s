.data
global_str0:
	.string "%d\n"
.text
	.extern _printf
error_handler:
	movl $0x2000001, %eax
	movl $-1, %edi
	syscall
.globl _main
_main:
	pushq %rbp
	movq %rsp, %rbp
	subq $48, %rsp
	jmp main0
main0:
	movq $1, %rax
	movq %rax, -8(%rbp)

	movq $1, %rax
	movq %rax, -32(%rbp)

	movq -8(%rbp), %r9
	movq -32(%rbp), %r10
	subq %r10, %r9
	movq %r9, -16(%rbp)

	movq $2, %rax
	movq %rax, -0(%rbp)

	movq -16(%rbp), %r9
	movq -0(%rbp), %r10
	subq %r10, %r9
	movq %r9, -24(%rbp)

	leaq global_str0(%rip), %rdi
	movq -24(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	jmp mainend
mainend:
	addq $48, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
