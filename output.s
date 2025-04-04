.data
global_str3:
	.string "false or true is correct\n"
global_str16:
	.string "false or false is incorrect\n"
global_str4:
	.string "a == a is correct\n"
global_str8:
	.string "a == b is incorrect\n"
global_str11:
	.string "true and false is incorrect\n"
global_str14:
	.string "true or true is correct\n"
global_str5:
	.string "10 + 20 is %d (30)\n"
global_str7:
	.string "10 * 20 is %d (200)\n"
global_str2:
	.string "false and false is incorrect\n"
global_str0:
	.string "a != a is incorrect\n"
global_str1:
	.string "c > b is incorrect\n"
global_str10:
	.string "true or false is correct\n"
global_str12:
	.string "true and true is correct\n"
global_str6:
	.string "10 - 20 is %d (-10)\n"
global_str15:
	.string "c >= b is correct\n"
global_str9:
	.string "a != b is correct\n"
global_str13:
	.string "false and true is incorrect\n"
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
	subq $304, %rsp
	jmp main0
main0:
	movq $10, %rax
	movq %rax, -264(%rbp)

	movq $20, %rax
	movq %rax, -216(%rbp)

	movq -264(%rbp), %r9
	movq -216(%rbp), %r10
	addq %r10, %r9
	movq %r9, -224(%rbp)

	movq -224(%rbp), %rax
	movq %rax, -128(%rbp)

	leaq global_str5(%rip), %rdi
	movq -128(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $10, %rax
	movq %rax, -232(%rbp)

	movq $20, %rax
	movq %rax, -24(%rbp)

	movq -232(%rbp), %r9
	movq -24(%rbp), %r10
	subq %r10, %r9
	movq %r9, -48(%rbp)

	movq -48(%rbp), %rax
	movq %rax, -128(%rbp)

	leaq global_str6(%rip), %rdi
	movq -128(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $10, %rax
	movq %rax, -80(%rbp)

	movq $20, %rax
	movq %rax, -248(%rbp)

	movq -80(%rbp), %r9
	movq -248(%rbp), %r10
	imulq %r10, %r9
	movq %r9, -88(%rbp)

	movq -88(%rbp), %rax
	movq %rax, -128(%rbp)

	leaq global_str7(%rip), %rdi
	movq -128(%rbp), %rsi
	xorq %rax, %rax
	call _printf

	movq $1, %rax
	movq %rax, -160(%rbp)

	movq -160(%rbp), %rax
	movq %rax, -128(%rbp)

	movq $2, %rax
	movq %rax, -56(%rbp)

	movq -56(%rbp), %rax
	movq %rax, -72(%rbp)

	movq $2, %rax
	movq %rax, -32(%rbp)

	movq -32(%rbp), %rax
	movq %rax, -152(%rbp)

	movq -152(%rbp), %r9
	movq -72(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovge %r11, %r9
	movq %r9, -0(%rbp)

	movq -0(%rbp), %r9
	cmpq $1, %r9
	je main28
	jmp main31
main28:
	leaq global_str15(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main35
main31:
	jmp main35
main35:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -240(%rbp)

	movq -240(%rbp), %r9
	cmpq $1, %r9
	je main36
	jmp main39
main36:
	leaq global_str4(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main43
main39:
	jmp main43
main43:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -136(%rbp)

	movq -136(%rbp), %r9
	cmpq $1, %r9
	je main44
	jmp main47
main44:
	leaq global_str0(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main51
main47:
	jmp main51
main51:
	movq -128(%rbp), %r9
	movq -72(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -96(%rbp)

	movq -96(%rbp), %r9
	cmpq $1, %r9
	je main52
	jmp main55
main52:
	leaq global_str8(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main59
main55:
	jmp main59
main59:
	movq -128(%rbp), %r9
	movq -72(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -256(%rbp)

	movq -256(%rbp), %r9
	cmpq $1, %r9
	je main60
	jmp main63
main60:
	leaq global_str9(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main67
main63:
	jmp main67
main67:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -104(%rbp)

	movq -104(%rbp), %r9
	cmpq $1, %r9
	je main72
	jmp main71
main68:
	leaq global_str12(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main78
main71:
	jmp main78
main72:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -16(%rbp)

	movq -16(%rbp), %r9
	cmpq $1, %r9
	je main68
	jmp main71
main78:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -176(%rbp)

	movq -176(%rbp), %r9
	cmpq $1, %r9
	je main83
	jmp main82
main79:
	leaq global_str13(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main89
main82:
	jmp main89
main83:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -272(%rbp)

	movq -272(%rbp), %r9
	cmpq $1, %r9
	je main79
	jmp main82
main89:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -288(%rbp)

	movq -288(%rbp), %r9
	cmpq $1, %r9
	je main94
	jmp main93
main90:
	leaq global_str11(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main100
main93:
	jmp main100
main94:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -8(%rbp)

	movq -8(%rbp), %r9
	cmpq $1, %r9
	je main90
	jmp main93
main100:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -280(%rbp)

	movq -280(%rbp), %r9
	cmpq $1, %r9
	je main105
	jmp main104
main101:
	leaq global_str2(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main111
main104:
	jmp main111
main105:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -168(%rbp)

	movq -168(%rbp), %r9
	cmpq $1, %r9
	je main101
	jmp main104
main111:
	movq -152(%rbp), %r9
	movq -72(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovg %r11, %r9
	movq %r9, -144(%rbp)

	movq -144(%rbp), %r9
	cmpq $1, %r9
	je main112
	jmp main115
main112:
	leaq global_str1(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main119
main115:
	jmp main119
main119:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -192(%rbp)

	movq -192(%rbp), %r9
	cmpq $1, %r9
	je main120
	jmp main124
main120:
	leaq global_str14(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main130
main123:
	jmp main130
main124:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -184(%rbp)

	movq -184(%rbp), %r9
	cmpq $1, %r9
	je main120
	jmp main123
main130:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -112(%rbp)

	movq -112(%rbp), %r9
	cmpq $1, %r9
	je main131
	jmp main135
main131:
	leaq global_str3(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main141
main134:
	jmp main141
main135:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -64(%rbp)

	movq -64(%rbp), %r9
	cmpq $1, %r9
	je main131
	jmp main134
main141:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmove %r11, %r9
	movq %r9, -120(%rbp)

	movq -120(%rbp), %r9
	cmpq $1, %r9
	je main142
	jmp main146
main142:
	leaq global_str10(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main152
main145:
	jmp main152
main146:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -200(%rbp)

	movq -200(%rbp), %r9
	cmpq $1, %r9
	je main142
	jmp main145
main152:
	movq -128(%rbp), %r9
	movq -128(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -208(%rbp)

	movq -208(%rbp), %r9
	cmpq $1, %r9
	je main153
	jmp main157
main153:
	leaq global_str16(%rip), %rdi
	xorq %rax, %rax
	call _printf

	jmp main163
main156:
	jmp main163
main157:
	movq -72(%rbp), %r9
	movq -152(%rbp), %r10
	cmpq %r10, %r9
	movq $0, %r9
	movq $1, %r11
	cmovne %r11, %r9
	movq %r9, -40(%rbp)

	movq -40(%rbp), %r9
	cmpq $1, %r9
	je main153
	jmp main156
main163:
	jmp mainend
mainend:
	addq $304, %rsp
	popq %rbp
	xorq %rax, %rax
	ret
