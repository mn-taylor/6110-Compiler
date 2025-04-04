.data
global_var1:
	.zero 8
global_str8:
	.string "if failed on a && proc\n"
global_str9:
	.string "sum from 0 to 99 is %d, should be 4950\n"
global_str12:
	.string "returned a variable, value is %d, should be 81\n"
global_str13:
	.string "returned a constant, value is %d, should be 1\n"
global_str5:
	.string "if failed on !a || proc\n"
global_str10:
	.string "hello world from main\n"
global_str1:
	.string "short circuit failed in or\n"
global_str3:
	.string "successfully did a short-circuited or.\n"
global_str0:
	.string "args: %d+%d+%d+%d+%d\n"
global_str11:
	.string "some values (13,14): %d %d\n"
global_str6:
	.string "successfully did a short-circuited and.\n"
global_str2:
	.string "short circuit failed in and\n"
global_str4:
	.string "if failed on !a && proc\n"
global_str7:
	.string "if failed on a || proc\n"
.text
	.extern _printf
error_handler:
	movl $0x2000001, %eax
	movl $-1, %edi
	syscall
test_long_args:
	pushq %rbp
	movq %rsp, %rbp
	subq $96, %rsp
	movq %rdi, -16(%rbp)
	movq %rsi, -48(%rbp)
	movq %rdx, -80(%rbp)
	movq %rcx, -88(%rbp)
	movq %r8, -56(%rbp)
	movq %r9, -64(%rbp)
	jmp test_long_args0
test_long_args0:
	leaq global_str0(%rip), %rdi
	movq -16(%rbp), %rsi
	movq -48(%rbp), %rdx
	movq -80(%rbp), %rcx
	movq -88(%rbp), %r8
	movq -56(%rbp), %r9
	xorq %rax, %rax
	call _printf

	movq -16(%rbp), %r9
	movq -48(%rbp), %r10
	addq %r10, %r9
	movq %r9, -24(%rbp)

	movq -80(%rbp), %r9
	movq -88(%rbp), %r10
	addq %r10, %r9
	movq %r9, -72(%rbp)

	movq -56(%rbp), %r9
	movq -64(%rbp), %r10
	addq %r10, %r9
	movq %r9, -32(%rbp)

	movq -72(%rbp), %r9
	movq -32(%rbp), %r10
	addq %r10, %r9
	movq %r9, -40(%rbp)

	movq -24(%rbp), %r9
	movq -40(%rbp), %r10
	addq %r10, %r9
	movq %r9, -8(%rbp)

	movq -8(%rbp), %rax
jmp test_long_argsend

	jmp error_handler
test_long_argsend:
	addq $96, %rsp
	popq %rbp
	ret
test_short_circuit:
	pushq %rbp
	movq %rsp, %rbp
	subq $64, %rsp
	movq %rdi, -32(%rbp)
	movq %rsi, -8(%rbp)
	jmp test_short_circuit0
test_short_circuit0:
	movq $1, %rax
	movq %rax, -24(%rbp)

	movq -8(%rbp), %r9
	movq -24(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -40(%rbp)

	movq -40(%rbp), %r9
	cmpq $1, %r9
	je test_short_circuit1
	jmp test_short_circuit12
test_short_circuit1:
	movq $1, %rax
	movq %rax, -16(%rbp)

	movq -32(%rbp), %r9
	movq -16(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -56(%rbp)

	movq -56(%rbp), %r9
	cmpq $1, %r9
	je test_short_circuit2
	jmp test_short_circuit5
test_short_circuit2:
	leaq global_str2(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp test_short_circuit11
test_short_circuit5:
	leaq global_str1(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp test_short_circuit11
test_short_circuit11:
	jmp test_short_circuit16
test_short_circuit12:
	jmp test_short_circuit16
test_short_circuit16:
	movq $1, %rax
	movq %rax, -48(%rbp)

	movq -48(%rbp), %rax
jmp test_short_circuitend

	jmp error_handler
test_short_circuitend:
	addq $64, %rsp
	popq %rbp
	ret
.globl _main
_main:
	pushq %rbp
	movq %rsp, %rbp
	subq $288, %rsp
	jmp main0
main0:
	leaq global_str10(%rip), %rdi
	xorq %rax, %rax
	call _printf

	movq $13, %rax
	movq %rax, -72(%rbp)

	movq -72(%rbp), %rax
	movq %rax, global_var1(%rip)

	movq $14, %rax
	movq %rax, -168(%rbp)

	movq -168(%rbp), %rax
	movq %rax, -224(%rbp)

	leaq global_str11(%rip), %rdi
	movq global_var1(%rip), %rsi
	movq -224(%rbp), %rdx
	xorq %rax, %rax
	call _printf

	movq global_var1(%rip), %rdi
	movq global_var1(%rip), %rsi
	movq global_var1(%rip), %rdx
	movq -224(%rbp), %rcx
	movq -224(%rbp), %r8
	movq -224(%rbp), %r9
	call test_long_args
	movq %rax, -264(%rbp)

	movq -264(%rbp), %rax
	movq %rax, -136(%rbp)

	leaq global_str12(%rip), %rdi
	movq -136(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $0, %rax
	movq %rax, -80(%rbp)

	movq $1, %rax
	movq %rax, -48(%rbp)

	movq -80(%rbp), %rdi
	movq -48(%rbp), %rsi
	call test_short_circuit
	movq %rax, -16(%rbp)

	movq -16(%rbp), %rax
	movq %rax, -104(%rbp)

	leaq global_str13(%rip), %rdi
	movq -104(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $1, %rax
	movq %rax, -184(%rbp)

	movq -184(%rbp), %rax
	movq %rax, -104(%rbp)

	movq -104(%rbp), %r9
	cmpq $1, %r9
	je main39
	jmp main36
main33:
	leaq global_str6(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main44
main36:
	leaq global_str8(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main44
main39:
	movq $1, %rax
	movq %rax, -256(%rbp)

	movq $1, %rax
	movq %rax, -8(%rbp)

	movq -256(%rbp), %rdi
	movq -8(%rbp), %rsi
	call test_short_circuit
	movq %rax, -160(%rbp)

	movq -160(%rbp), %r9
	cmpq $1, %r9
	je main33
	jmp main36
main44:
	movq $1, %rax
	movq %rax, -24(%rbp)

	movq -104(%rbp), %r9
	movq -24(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -112(%rbp)

	movq -112(%rbp), %r9
	cmpq $1, %r9
	je main51
	jmp main48
main45:
	leaq global_str4(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main58
main48:
	leaq global_str6(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main58
main51:
	movq $1, %rax
	movq %rax, -192(%rbp)

	movq $0, %rax
	movq %rax, -240(%rbp)

	movq -192(%rbp), %rdi
	movq -240(%rbp), %rsi
	call test_short_circuit
	movq %rax, -32(%rbp)

	movq -32(%rbp), %r9
	cmpq $1, %r9
	je main45
	jmp main48
main58:
	movq -104(%rbp), %r9
	cmpq $1, %r9
	je main59
	jmp main65
main59:
	leaq global_str3(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main70
main62:
	leaq global_str7(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main70
main65:
	movq $0, %rax
	movq %rax, -176(%rbp)

	movq $0, %rax
	movq %rax, -200(%rbp)

	movq -176(%rbp), %rdi
	movq -200(%rbp), %rsi
	call test_short_circuit
	movq %rax, -88(%rbp)

	movq -88(%rbp), %r9
	cmpq $1, %r9
	je main59
	jmp main62
main70:
	movq $1, %rax
	movq %rax, -96(%rbp)

	movq -104(%rbp), %r9
	movq -96(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -144(%rbp)

	movq -144(%rbp), %r9
	cmpq $1, %r9
	je main71
	jmp main77
main71:
	leaq global_str3(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main84
main74:
	leaq global_str5(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main84
main77:
	movq $0, %rax
	movq %rax, -216(%rbp)

	movq $1, %rax
	movq %rax, -208(%rbp)

	movq -216(%rbp), %rdi
	movq -208(%rbp), %rsi
	call test_short_circuit
	movq %rax, -152(%rbp)

	movq -152(%rbp), %r9
	cmpq $1, %r9
	je main71
	jmp main74
main84:
	movq $0, %rax
	movq %rax, -120(%rbp)

	movq -120(%rbp), %rax
	movq %rax, -136(%rbp)

	movq $100, %rax
	movq %rax, -56(%rbp)

	movq -56(%rbp), %rax
	movq %rax, -224(%rbp)

	movq $0, %rax
	movq %rax, -248(%rbp)

	movq -248(%rbp), %rax
	movq %rax, -128(%rbp)

	jmp main101
main92:
	leaq global_str9(%rip), %rdi
	movq -136(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	jmp mainend
main96:
	movq -136(%rbp), %r9
	movq -128(%rbp), %r10
	addq %r10, %r9
	movq %r9, -40(%rbp)

	movq -40(%rbp), %rax
	movq %rax, -136(%rbp)

	movq $1, %rax
	movq %rax, -272(%rbp)

	movq -128(%rbp), %r9
	movq -272(%rbp), %r10
	addq %r10, %r9
	movq %r9, -128(%rbp)

	jmp main101
main101:
	movq -128(%rbp), %r9
	movq -224(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovl %r11, %r9
	movq %r9, -64(%rbp)

	movq -64(%rbp), %r9
	cmpq $1, %r9
	je main96
	jmp main92
mainend:
	addq $288, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
