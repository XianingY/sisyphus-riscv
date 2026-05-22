.global main
getNumPos:
.Lbb0:
  mv t1, a1
  mv t0, a0
  li a0, 0
  j .Lbb1
.Lbb2:
  sraiw a1, t0, 31
  addiw a0, a0, 1
  andi a1, a1, 15
  addw a1, t0, a1
  sraiw a1, a1, 4
  mv t0, a1
.Lbb1:
  blt a0, t1, .Lbb2
.Lbb3:
  slli a0, t0, 1
  srli a0, a0, 60
  add a0, t0, a0
  andi a0, a0, -16
  subw a0, t0, a0
  ret 


radixSort:
.Lbb4:
  addi sp, sp, -288
  sd s0, 280(sp)
  sd s1, 272(sp)
  sd s2, 264(sp)
  sd s3, 256(sp)
  sd s4, 248(sp)
  sd s5, 240(sp)
  sd s6, 232(sp)
  sd s7, 224(sp)
  sd s8, 216(sp)
  sd s9, 208(sp)
  sd s11, 200(sp)
  sd ra, 192(sp)
  mv s0, a3
  mv s5, a2
  mv s8, a1
  mv s7, a0
  mv s6, sp
  mv s11, sp
  mv s9, sp
  li s1, 0
  sw s1, 0(s6)
  sw s1, 4(s6)
  sw s1, 8(s6)
  sw s1, 12(s6)
  sw s1, 16(s6)
  sw s1, 20(s6)
  sw s1, 24(s6)
  sw s1, 28(s6)
  sw s1, 32(s6)
  sw s1, 36(s6)
  sw s1, 40(s6)
  sw s1, 44(s6)
  sw s1, 48(s6)
  sw s1, 52(s6)
  sw s1, 56(s6)
  sw s1, 60(s6)
  sw s1, 64(s11)
  sw s1, 68(s11)
  sw s1, 72(s11)
  sw s1, 76(s11)
  sw s1, 80(s11)
  sw s1, 84(s11)
  sw s1, 88(s11)
  sw s1, 92(s11)
  sw s1, 96(s11)
  sw s1, 100(s11)
  sw s1, 104(s11)
  sw s1, 108(s11)
  sw s1, 112(s11)
  sw s1, 116(s11)
  sw s1, 120(s11)
  sw s1, 124(s11)
  sw s1, 128(s9)
  sw s1, 132(s9)
  sw s1, 136(s9)
  sw s1, 140(s9)
  sw s1, 144(s9)
  sw s1, 148(s9)
  sw s1, 152(s9)
  sw s1, 156(s9)
  sw s1, 160(s9)
  sw s1, 164(s9)
  sw s1, 168(s9)
  sw s1, 172(s9)
  sw s1, 176(s9)
  sw s1, 180(s9)
  sw s1, 184(s9)
  sw s1, 188(s9)
  li a0, -1
  beq s7, a0, .Lbb5
  j .Lbb6
.Lbb7:
  mv s4, s5
  j .Lbb8
.Lbb6:
  addiw a0, s5, 1
  bge a0, s0, .Lbb5
  j .Lbb7
.Lbb9:
  slliw a0, s4, 2
  add a0, s8, a0
  lw a0, 0(a0)
  mv a1, s7
  call getNumPos
  mv s2, a0
  slliw a0, s4, 2
  add a0, s8, a0
  lw a0, 0(a0)
  mv a1, s7
  call getNumPos
  slliw a0, a0, 2
  add a0, s9, a0
  lw a0, 128(a0)
  addiw a0, a0, 1
  slliw a1, s2, 2
  add a1, s9, a1
  sw a0, 128(a1)
  addiw s4, s4, 1
.Lbb8:
  blt s4, s0, .Lbb9
.Lbb10:
  lw a1, 128(s9)
  li a0, 1
  li s11, 16
  sw s5, 0(s6)
  addw a1, s5, a1
  sw a1, 64(s11)
  j .Lbb11
.Lbb12:
  slliw a5, a0, 2
  slliw a4, a0, 2
  slliw a3, a0, 2
  slliw a2, a0, 2
  slliw a1, a0, 2
  addiw a0, a0, 1
  add a5, s11, a5
  lw a5, 60(a5)
  add a4, s6, a4
  add a3, s6, a3
  add a2, s9, a2
  add a1, s11, a1
  sw a5, 0(a4)
  lw a3, 0(a3)
  lw a2, 128(a2)
  addw a2, a3, a2
  sw a2, 64(a1)
.Lbb11:
  blt a0, s11, .Lbb12
.Lbb13:
  mv s3, s1
.Lbb14:
  blt s3, s11, .Lbb15
.Lbb16:
  lw a0, 128(s9)
  sw s5, 0(s6)
  addw a0, s5, a0
  sw a0, 64(s11)
  mv s0, s1
  j .Lbb17
.Lbb15:
  slliw a1, s3, 2
  slliw a0, s3, 2
  add a1, s6, a1
  lw a1, 0(a1)
  add a0, s11, a0
  lw a0, 64(a0)
  bge a1, a0, .Lbb18
.Lbb19:
  slliw a0, s3, 2
  add a0, s6, a0
  lw a0, 0(a0)
  slliw a0, a0, 2
  add a0, s8, a0
  lw s0, 0(a0)
  j .Lbb20
.Lbb18:
  addiw s3, s3, 1
  j .Lbb14
.Lbb21:
  mv a0, s0
  mv a1, s7
  call getNumPos
  slliw a0, a0, 2
  add a0, s6, a0
  lw a0, 0(a0)
  slliw a0, a0, 2
  add a0, s8, a0
  lw s2, 0(a0)
  mv a0, s0
  mv a1, s7
  call getNumPos
  slliw a0, a0, 2
  add a0, s6, a0
  lw a0, 0(a0)
  slliw a0, a0, 2
  add a0, s8, a0
  sw s0, 0(a0)
  mv a0, s0
  mv a1, s7
  call getNumPos
  mv s4, a0
  mv a0, s0
  mv a1, s7
  call getNumPos
  slliw a0, a0, 2
  add a0, s6, a0
  lw a0, 0(a0)
  addiw a0, a0, 1
  slliw a1, s4, 2
  add a1, s6, a1
  sw a0, 0(a1)
  mv s0, s2
.Lbb20:
  mv a0, s0
  mv a1, s7
  call getNumPos
  bne a0, s3, .Lbb21
.Lbb22:
  slliw a2, s3, 2
  slliw a1, s3, 2
  slliw a0, s3, 2
  add a2, s6, a2
  lw a2, 0(a2)
  add a1, s6, a1
  add a0, s6, a0
  slliw a2, a2, 2
  add a2, s8, a2
  sw s0, 0(a2)
  lw a1, 0(a1)
  addiw a1, a1, 1
  sw a1, 0(a0)
  j .Lbb15
.Lbb17:
  bge s0, s11, .Lbb5
.Lbb23:
  bge s1, s0, .Lbb24
.Lbb25:
  slliw a4, s0, 2
  slliw a3, s0, 2
  slliw a2, s0, 2
  slliw a1, s0, 2
  slliw a0, s0, 2
  add a4, s11, a4
  lw a4, 60(a4)
  add a3, s6, a3
  add a2, s6, a2
  add a1, s9, a1
  add a0, s11, a0
  sw a4, 0(a3)
  lw a2, 0(a2)
  lw a1, 128(a1)
  addw a1, a2, a1
  sw a1, 64(a0)
.Lbb24:
  addiw a0, s7, -1
  slliw a1, s0, 2
  add a1, s6, a1
  lw a2, 0(a1)
  slliw a1, s0, 2
  add a1, s11, a1
  lw a3, 64(a1)
  mv a1, s8
  call radixSort
  addiw s0, s0, 1
  j .Lbb17
.Lbb5:
  ld s0, 280(sp)
  ld s1, 272(sp)
  ld s2, 264(sp)
  ld s3, 256(sp)
  ld s4, 248(sp)
  ld s5, 240(sp)
  ld s6, 232(sp)
  ld s7, 224(sp)
  ld s8, 216(sp)
  ld s9, 208(sp)
  ld s11, 200(sp)
  ld ra, 192(sp)
  addi sp, sp, 288
  ret 


main:
.Lbb26:
  addi sp, sp, -48
  sd s0, 40(sp)
  sd s1, 32(sp)
  sd s2, 24(sp)
  sd s3, 16(sp)
  sd s4, 8(sp)
  sd ra, 0(sp)
  la s0, ans
  la s1, a
  mv a0, s1
  call getarray
  mv s4, a0
  li a0, 90
  call _sysy_starttime
  li a0, 9
  li s3, 0
  li a0, 9
  la a1, a
  li a2, 0
  mv a3, s4
  call radixSort
  mv s2, s3
  j .Lbb27
.Lbb28:
  lw a1, 0(s0)
  slliw a3, s2, 2
  addiw a2, s2, 2
  addiw a0, s2, 1
  add a3, s1, a3
  lw a3, 0(a3)
  remw a2, a3, a2
  mulw a2, s2, a2
  addw a1, a1, a2
  addiw a1, a1, 3
  sw a1, 0(s0)
  mv s2, a0
.Lbb27:
  blt s2, s4, .Lbb28
.Lbb29:
  lw a0, 0(s0)
  bge a0, s3, .Lbb30
.Lbb31:
  lw a0, 0(s0)
  sub a0, s3, a0
  sw a0, 0(s0)
.Lbb30:
  li a0, 102
  call _sysy_stoptime
  lw a0, 0(s0)
  call putint
  li a0, 10
  call putch
  li a0, 0
  ld s0, 40(sp)
  ld s1, 32(sp)
  ld s2, 24(sp)
  ld s3, 16(sp)
  ld s4, 8(sp)
  ld ra, 0(sp)
  addi sp, sp, 48
  ret 



.data
base:
  .word 16

.bss
  .align 4
a:
  .space 120000040
ans:
  .space 4
