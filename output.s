.data
global_var1:
	.zero 40
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
	subq $64, %rsp
	jmp main0
main0:
	movq $0, %rax
	movq %rax, -40(%rbp)

	movq -40(%rbp), %r9
	leaq (global_var1(%rip), %r9, $16), %r9
	movq 0(%r9), %rax
	movq %rax, -32(%rbp)

	movq $5, %rax
	movq %rax, -24(%rbp)

	movq -24(%rbp), %rax
	movq %rax, -32(%rbp)

	movq -32(%rbp), %rax
	movq -40(%rbp), %r9
	leaq (global_var1(%rip), %r9, $16), %r9
	movq %rax, 0(%r9)

	movq $0, %rax
	movq %rax, -56(%rbp)

	movq -56(%rbp), %r9
	leaq (global_var1(%rip), %r9, $16), %r9
	movq 0(%r9), %rax
	movq %rax, -16(%rbp)

	movq $1, %rax
	movq %rax, -48(%rbp)

	movq -16(%rbp), %r9
	movq -48(%rbp), %r10
	addq %r9, %r10
	movq %r10, -16(%rbp)

	movq -16(%rbp), %rax
	movq -56(%rbp), %r9
	leaq (global_var1(%rip), %r9, $16), %r9
	movq %rax, 0(%r9)

	movq $0, %rax
	movq %rax, -8(%rbp)

	movq -8(%rbp), %r9
	leaq (global_var1(%rip), %r9, $16), %r9
	movq 0(%r9), %rax
	movq %rax, -0(%rbp)

	leaq global_str0(%rip), %rdi
	movq -0(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	jmp mainend
mainend:
	addq $64, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
