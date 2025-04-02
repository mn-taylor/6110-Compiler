.data
global_0:
	.string "0"
.text
	.extern _printf
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
	subq $0, %rsp
	jmp main0
main0:
	leaq global_0(%rip), %rdi
	xor %rax, %rax
	call _printf
	jmp mainend
mainend:
	addq $0, %rsp
	popq %rbp
	ret
