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
	subq $80, %rsp
	jmp main0
main0:
	movq $0, %rax
	movq %rax, -32(%rbp)

	movq -32(%rbp), %rax
	movq %rax, -16(%rbp)

	jmp main21
main4:
	jmp mainend
main5:
	movq $1, %rax
	movq %rax, -24(%rbp)

	movq -16(%rbp), %r9
	movq -24(%rbp), %r10
	addq %r10, %r9
	movq %r9, -16(%rbp)

	jmp main21
main8:
	movq $2, %rax
	movq %rax, -48(%rbp)

	movq -16(%rbp), %r9
	movq -48(%rbp), %r10
	modq %r10, %r9
	movq %r9, -0(%rbp)

	movq $0, %rax
	movq %rax, -40(%rbp)

	movq -0(%rbp), %r9
	movq -40(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -56(%rbp)

	movq -56(%rbp), %r9
	cmpq $1, %r9
	je main9
	jmp main11
main9:
	jmp main5
main11:
	leaq global_str0(%rip), %rdi
	movq -16(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	jmp main5
main21:
	movq $10, %rax
	movq %rax, -64(%rbp)

	movq -16(%rbp), %r9
	movq -64(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovl %r11, %r9
	movq %r9, -8(%rbp)

	movq -8(%rbp), %r9
	cmpq $1, %r9
	je main8
	jmp main4
mainend:
	addq $80, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
