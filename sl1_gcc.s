	.file	"sl1.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	addi	sp,sp,-144
	.cfi_def_cfa_offset 144
	sd	ra,136(sp)
	sd	s0,128(sp)
	sd	s1,120(sp)
	sd	s2,112(sp)
	sd	s3,104(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	call	getint@plt
	mv	s1,a0
	call	getint@plt
	mv	s0,a0
	call	starttime@plt
	ble	s1,zero,.L21
	sd	s8,64(sp)
	.cfi_offset 24, -80
	li	s8,1441792
	addi	s8,s8,-1792
	mul	a5,s1,s8
	sd	s5,88(sp)
	.cfi_offset 21, -56
	li	s5,4096
	addi	s5,s5,-1696
	sd	s6,80(sp)
	sd	s7,72(sp)
	sd	s4,96(sp)
	sd	s9,56(sp)
	sd	s11,40(sp)
	sd	s10,48(sp)
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 20, -48
	.cfi_offset 25, -88
	.cfi_offset 27, -104
	.cfi_offset 26, -96
	mul	s6,s1,s5
	sd	a5,8(sp)
	sext.w	a5,s1
	sd	a5,16(sp)
	slli	a5,s1,32
	srli	s7,a5,30
	lla	a5,x
	lla	s4,y
	sd	a5,24(sp)
	mv	s3,a5
	li	s9,0
	li	s11,1
.L3:
	li	s10,0
.L5:
	add	a4,s10,s3
	li	a5,0
.L4:
	mv	s2,a5
	sw	s11,0(a4)
	addiw	a5,a5,1
	addi	a4,a4,4
	bne	s1,a5,.L4
	add	a0,s10,s4
	mv	a2,s7
	li	a1,0
	add	s10,s10,s5
	call	memset@plt
	bne	s10,s6,.L5
	ld	a5,8(sp)
	add	s9,s9,s8
	add	s4,s4,s8
	add	s3,s3,s8
	bne	s9,a5,.L3
	ble	s2,s11,.L19
	ld	a5,16(sp)
	li	a1,-1441792
	addi	a1,a1,1792
	addiw	a5,a5,-3
	slli	a5,a5,32
	srli	a5,a5,32
	mul	a2,a5,a1
	li	a3,-2879488
	addi	a3,a3,-512
	li	a4,8192
	li	s4,4096
	addi	a4,a4,-992
	li	t4,2879488
	addi	s4,s4,-1688
	li	t0,-4096
	li	t6,4096
	mul	a5,a5,s5
	li	s3,1441792
	add	s5,a2,a3
	lla	a3,x
	addi	t4,t4,512
	li	a0,0
	add	s4,a3,s4
	addi	t0,t0,1692
	addi	t6,t6,-1696
	mv	s6,a1
	add	t2,a5,a4
	addi	s3,s3,-1792
.L7:
	li	t3,-4096
	li	t1,4096
	sub	t5,s4,a1
	addi	t3,t3,1696
	addi	t1,t1,704
.L11:
	lw	s7,-8(t5)
	add	a3,t5,t0
	mv	a4,t5
	li	a2,1
.L8:
	add	a5,a4,a1
	add	s8,a5,a0
	add	a5,a5,t4
	lw	s9,-4(s8)
	lw	a5,-4(a5)
	add	s8,t1,a4
	lw	s11,0(a3)
	add	s8,s8,t3
	lw	s10,-4(s8)
	addw	a5,a5,s9
	add	s8,a3,a1
	lw	s9,0(a4)
	addw	a5,a5,s11
	add	s8,s8,a0
	lw	s8,-4(s8)
	addw	a5,a5,s10
	addw	a5,a5,s7
	addw	a5,a5,s9
	addw	a5,a5,s8
	divw	s7,a5,s0
	addiw	a2,a2,1
	addi	a4,a4,4
	addi	a3,a3,4
	sw	s7,-8(a4)
	bne	a2,s2,.L8
	add	t1,t1,t6
	add	t5,t5,t6
	sub	t3,t3,t6
	bne	t2,t1,.L11
	add	a1,a1,s6
	add	a0,a0,s3
	add	t4,t4,s3
	bne	a1,s5,.L7
	ld	a5,16(sp)
	ld	s4,96(sp)
	.cfi_restore 20
	ld	s5,88(sp)
	.cfi_restore 21
	ld	s6,80(sp)
	.cfi_restore 22
	ld	s7,72(sp)
	.cfi_restore 23
	ld	s8,64(sp)
	.cfi_restore 24
	ld	s9,56(sp)
	.cfi_restore 25
	ld	s10,48(sp)
	.cfi_restore 26
	ld	s11,40(sp)
	.cfi_restore 27
	addiw	s0,a5,-2
.L10:
	call	stoptime@plt
	lla	a1,x
	mv	a0,s1
	call	putarray@plt
	srliw	a1,s1,31
	addw	a1,a1,s1
	li	s2,1441792
	addi	s2,s2,608
	sraiw	a1,a1,1
	mul	a1,a1,s2
	ld	s3,24(sp)
	mv	a0,s1
	mul	s0,s0,s2
	add	a1,s3,a1
	call	putarray@plt
	mv	a0,s1
	add	a1,s3,s0
	call	putarray@plt
	ld	ra,136(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,128(sp)
	.cfi_restore 8
	ld	s1,120(sp)
	.cfi_restore 9
	ld	s2,112(sp)
	.cfi_restore 18
	ld	s3,104(sp)
	.cfi_restore 19
	li	a0,0
	addi	sp,sp,144
	.cfi_def_cfa_offset 0
	jr	ra
.L21:
	.cfi_restore_state
	lla	a5,x
	sd	a5,24(sp)
	li	s0,0
	j	.L10
.L19:
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	.cfi_offset 26, -96
	.cfi_offset 27, -104
	ld	s4,96(sp)
	.cfi_restore 20
	ld	s5,88(sp)
	.cfi_restore 21
	ld	s6,80(sp)
	.cfi_restore 22
	ld	s7,72(sp)
	.cfi_restore 23
	ld	s8,64(sp)
	.cfi_restore 24
	ld	s9,56(sp)
	.cfi_restore 25
	ld	s10,48(sp)
	.cfi_restore 26
	ld	s11,40(sp)
	.cfi_restore 27
	li	s0,0
	j	.L10
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	y
	.globl	x
	.bss
	.align	3
	.type	y, @object
	.size	y, 864000000
y:
	.zero	864000000
	.type	x, @object
	.size	x, 864000000
x:
	.zero	864000000
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
