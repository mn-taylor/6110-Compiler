.data
global_str0:
	.string "%d"
.text
	.extern _printf
error_handler:
	movl $0x2000001, %eax
	movl $-1, %edi
	syscall
bar:
	pushq %rbp
	movq %rsp, %rbp
	subq $64, %rsp
	movq %rdi, -0(%rbp)
	jmp bar0
bar0:
	movq $0, %rax
	movq %rax, -8(%rbp)

	movq -8(%rbp), %rax
	movq %rax, -24(%rbp)

	jmp bar16
bar4:
	jmp barend
bar8:
	movq $1, %rax
	movq %rax, -32(%rbp)

	movq -0(%rbp), %r9
	movq -32(%rbp), %r10
	subq %r10, %r9
	movq %r9, -40(%rbp)

	movq -40(%rbp), %rax
	movq %rax, -0(%rbp)

	leaq global_str0(%rip), %rdi
	movq -0(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $1, %rax
	movq %rax, -16(%rbp)

	movq -24(%rbp), %r9
	movq -16(%rbp), %r10
	addq %r10, %r9
	movq %r9, -24(%rbp)

	jmp bar16
bar16:
	movq -24(%rbp), %r9
	movq -0(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovl %r11, %r9
	movq %r9, -48(%rbp)

	movq -48(%rbp), %r9
	cmpq $1, %r9
	je bar8
	jmp bar4
barend:
	addq $64, %rsp
	popq %rbp
	ret
.globl _main
_main:
	pushq %rbp
	movq %rsp, %rbp
	subq $16, %rsp
	jmp main0
main0:
	movq $10, %rax
	movq %rax, -0(%rbp)

	movq -0(%rbp), %rdi
	call bar

	jmp mainend
mainend:
	addq $16, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
