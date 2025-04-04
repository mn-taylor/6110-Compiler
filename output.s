.data
global_str2:
	.string "- assoc: result should be 10, is: %d\n"
global_str7:
	.string "*,- assoc: result should be 2, is: %d\n"
global_str0:
	.string "min int operation: result should be -2147483648, is: %d\n"
global_str1:
	.string "paren assoc: result should be 100, is: %d\n"
global_str4:
	.string "- * assoc: result should be 80, is: %d\n"
global_str3:
	.string "-,+ assoc: result should be 90, is: %d\n"
global_str5:
	.string "result should be 46, is: %d\n"
global_str6:
	.string "*, -, uses var: result should be 11, is: %d\n"
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
	subq $352, %rsp
	jmp main0
main0:
	movq $2147483647, %rax
	movq %rax, -24(%rbp)

	movq -24(%rbp), %rax
	neg %rax
	movq %rax, -264(%rbp)

	movq $1, %rax
	movq %rax, -208(%rbp)

	movq -264(%rbp), %r9
	movq -208(%rbp), %r10
	subq %r10, %r9
	movq %r9, -168(%rbp)

	movq -168(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str0(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $100, %rax
	movq %rax, -320(%rbp)

	movq $50, %rax
	movq %rax, -232(%rbp)

	movq -320(%rbp), %r9
	movq -232(%rbp), %r10
	subq %r10, %r9
	movq %r9, -296(%rbp)

	movq $2, %rax
	movq %rax, -216(%rbp)

	movq -296(%rbp), %r9
	movq -216(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -32(%rbp)

	movq -32(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str1(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $100, %rax
	movq %rax, -160(%rbp)

	movq $50, %rax
	movq %rax, -336(%rbp)

	movq -160(%rbp), %r9
	movq -336(%rbp), %r10
	subq %r10, %r9
	movq %r9, -312(%rbp)

	movq $40, %rax
	movq %rax, -128(%rbp)

	movq -312(%rbp), %r9
	movq -128(%rbp), %r10
	subq %r10, %r9
	movq %r9, -176(%rbp)

	movq -176(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str2(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $100, %rax
	movq %rax, -280(%rbp)

	movq $50, %rax
	movq %rax, -16(%rbp)

	movq -280(%rbp), %r9
	movq -16(%rbp), %r10
	subq %r10, %r9
	movq %r9, -144(%rbp)

	movq $40, %rax
	movq %rax, -56(%rbp)

	movq -144(%rbp), %r9
	movq -56(%rbp), %r10
	addq %r10, %r9
	movq %r9, -304(%rbp)

	movq -304(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str3(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $100, %rax
	movq %rax, -88(%rbp)

	movq $5, %rax
	movq %rax, -72(%rbp)

	movq $4, %rax
	movq %rax, -272(%rbp)

	movq -72(%rbp), %r9
	movq -272(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -0(%rbp)

	movq -88(%rbp), %r9
	movq -0(%rbp), %r10
	subq %r10, %r9
	movq %r9, -192(%rbp)

	movq -192(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str4(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $10, %rax
	movq %rax, -136(%rbp)

	movq $5, %rax
	movq %rax, -328(%rbp)

	movq -136(%rbp), %r9
	movq -328(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -120(%rbp)

	movq $4, %rax
	movq %rax, -8(%rbp)

	movq -120(%rbp), %r9
	movq -8(%rbp), %r10
	subq %r10, %r9
	movq %r9, -184(%rbp)

	movq -184(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str5(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $3, %rax
	movq %rax, -40(%rbp)

	movq -40(%rbp), %rax
	movq %rax, -288(%rbp)

	movq -288(%rbp), %r9
	movq -288(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -240(%rbp)

	movq $0, %rax
	movq %rax, -64(%rbp)

	movq $2, %rax
	movq %rax, -48(%rbp)

	movq -64(%rbp), %r9
	movq -48(%rbp), %r10
	subq %r10, %r9
	movq %r9, -80(%rbp)

	movq -240(%rbp), %r9
	movq -80(%rbp), %r10
	subq %r10, %r9
	movq %r9, -96(%rbp)

	movq -96(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str6(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $3, %rax
	movq %rax, -104(%rbp)

	movq $4, %rax
	movq %rax, -200(%rbp)

	movq -104(%rbp), %r9
	movq -200(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -112(%rbp)

	movq $2, %rax
	movq %rax, -152(%rbp)

	movq $5, %rax
	movq %rax, -224(%rbp)

	movq -152(%rbp), %r9
	movq -224(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -248(%rbp)

	movq -112(%rbp), %r9
	movq -248(%rbp), %r10
	subq %r10, %r9
	movq %r9, -256(%rbp)

	movq -256(%rbp), %rax
	movq %rax, -288(%rbp)

	leaq global_str7(%rip), %rdi
	movq -288(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	jmp mainend
mainend:
	addq $352, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
