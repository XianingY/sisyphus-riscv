.global main
radixSort:
.Lbb0:
  addi sp, sp, -336
  sd s0, 328(sp)
  sd s1, 320(sp)
  sd s2, 312(sp)
  sd s3, 304(sp)
  sd s4, 296(sp)
  sd s5, 288(sp)
  sd s6, 280(sp)
  sd s7, 272(sp)
  sd s8, 264(sp)
  sd s9, 256(sp)
  sd s10, 248(sp)
  sd s11, 240(sp)
  sd ra, 232(sp)
  mv t2, a2
  mv t0, a1
  mv t1, a0
  mv s10, sp
  sd s10, 208(sp)
  mv s10, sp
  sd s10, 200(sp)
  mv s10, sp
  sd s10, 192(sp)
  li s1, 0
  ld s11, 192(sp)
  sw s1, 132(s11)
  ld s11, 192(sp)
  sw s1, 136(s11)
  ld s11, 192(sp)
  sw s1, 140(s11)
  ld s11, 192(sp)
  sw s1, 144(s11)
  ld s11, 192(sp)
  sw s1, 148(s11)
  ld s11, 192(sp)
  sw s1, 152(s11)
  ld s11, 192(sp)
  sw s1, 156(s11)
  ld s11, 192(sp)
  sw s1, 160(s11)
  ld s11, 192(sp)
  sw s1, 164(s11)
  ld s11, 192(sp)
  sw s1, 168(s11)
  ld s11, 192(sp)
  sw s1, 172(s11)
  ld s11, 192(sp)
  sw s1, 176(s11)
  ld s11, 192(sp)
  sw s1, 180(s11)
  ld s11, 192(sp)
  sw s1, 184(s11)
  ld s11, 192(sp)
  sw s1, 188(s11)
  ld s11, 200(sp)
  sw s1, 68(s11)
  ld s11, 200(sp)
  sw s1, 72(s11)
  ld s11, 200(sp)
  sw s1, 76(s11)
  ld s11, 200(sp)
  sw s1, 80(s11)
  ld s11, 200(sp)
  sw s1, 84(s11)
  ld s11, 200(sp)
  sw s1, 88(s11)
  ld s11, 200(sp)
  sw s1, 92(s11)
  ld s11, 200(sp)
  sw s1, 96(s11)
  ld s11, 200(sp)
  sw s1, 100(s11)
  ld s11, 200(sp)
  sw s1, 104(s11)
  ld s11, 200(sp)
  sw s1, 108(s11)
  ld s11, 200(sp)
  sw s1, 112(s11)
  ld s11, 200(sp)
  sw s1, 116(s11)
  ld s11, 200(sp)
  sw s1, 120(s11)
  ld s11, 200(sp)
  sw s1, 124(s11)
  ld s11, 208(sp)
  sw s1, 0(s11)
  ld s11, 208(sp)
  sw s1, 4(s11)
  ld s11, 208(sp)
  sw s1, 8(s11)
  ld s11, 208(sp)
  sw s1, 12(s11)
  ld s11, 208(sp)
  sw s1, 16(s11)
  ld s11, 208(sp)
  sw s1, 20(s11)
  ld s11, 208(sp)
  sw s1, 24(s11)
  ld s11, 208(sp)
  sw s1, 28(s11)
  ld s11, 208(sp)
  sw s1, 32(s11)
  ld s11, 208(sp)
  sw s1, 36(s11)
  ld s11, 208(sp)
  sw s1, 40(s11)
  ld s11, 208(sp)
  sw s1, 44(s11)
  ld s11, 208(sp)
  sw s1, 48(s11)
  ld s11, 208(sp)
  sw s1, 52(s11)
  ld s11, 208(sp)
  sw s1, 56(s11)
  ld s11, 208(sp)
  sw s1, 60(s11)
  li a0, -1
  beq t1, a0, .Lbb1
  j .Lbb2
.Lbb3:
  la a0, a
  mv ra, t0
  j .Lbb4
.Lbb2:
  addiw a0, t0, 1
  bge a0, t2, .Lbb1
  j .Lbb3
.Lbb4:
  bge ra, t2, .Lbb5
.Lbb6:
  slliw a1, ra, 2
  add a2, a0, a1
  lw a1, 0(a2)
  mv a3, a1
  mv s9, s1
  j .Lbb7
.Lbb8:
  sraiw a1, a3, 31
  addiw s9, s9, 1
  andi a1, a1, 15
  addw a1, a3, a1
  sraiw a3, a1, 4
.Lbb7:
  blt s9, t1, .Lbb8
.Lbb9:
  lw a2, 0(a2)
  mv a1, a2
  mv s8, s1
  j .Lbb10
.Lbb11:
  sraiw a2, a1, 31
  addiw s8, s8, 1
  andi a2, a2, 15
  addw a1, a1, a2
  sraiw a1, a1, 4
.Lbb10:
  blt s8, t1, .Lbb11
.Lbb12:
  slli a4, a1, 1
  slli a2, a3, 1
  addiw ra, ra, 1
  srli a4, a4, 60
  srli a2, a2, 60
  add a4, a1, a4
  add a5, a3, a2
  andi a2, a4, -16
  andi a4, a5, -16
  subw a1, a1, a2
  subw a3, a3, a4
  slliw a2, a1, 2
  slliw a1, a3, 2
  ld s10, 208(sp)
  add a2, s10, a2
  lw a2, 0(a2)
  add a1, s10, a1
  addiw a2, a2, 1
  sw a2, 0(a1)
  j .Lbb4
.Lbb5:
  ld s10, 208(sp)
  lw s3, 0(s10)
  lw s2, 4(s10)
  lw s0, 8(s10)
  lw t6, 12(s10)
  lw t5, 16(s10)
  lw t4, 20(s10)
  lw t3, 24(s10)
  lw t2, 28(s10)
  lw ra, 32(s10)
  lw a7, 36(s10)
  lw a6, 40(s10)
  lw a5, 44(s10)
  lw a4, 48(s10)
  lw a3, 52(s10)
  lw a2, 56(s10)
  lw a1, 60(s10)
  addw s3, t0, s3
  ld s11, 192(sp)
  sw t0, 128(s11)
  ld s11, 200(sp)
  sw s3, 64(s11)
  ld s10, 200(sp)
  lw s3, 64(s10)
  ld s11, 192(sp)
  sw s3, 132(s11)
  ld s10, 192(sp)
  lw s3, 132(s10)
  addw s2, s3, s2
  ld s11, 200(sp)
  sw s2, 68(s11)
  ld s10, 200(sp)
  lw s2, 68(s10)
  ld s11, 192(sp)
  sw s2, 136(s11)
  ld s10, 192(sp)
  lw s2, 136(s10)
  addw s0, s2, s0
  ld s11, 200(sp)
  sw s0, 72(s11)
  ld s10, 200(sp)
  lw s0, 72(s10)
  ld s11, 192(sp)
  sw s0, 140(s11)
  ld s10, 192(sp)
  lw s0, 140(s10)
  addw t6, s0, t6
  ld s11, 200(sp)
  sw t6, 76(s11)
  ld s10, 200(sp)
  lw t6, 76(s10)
  ld s11, 192(sp)
  sw t6, 144(s11)
  ld s10, 192(sp)
  lw t6, 144(s10)
  addw t5, t6, t5
  ld s11, 200(sp)
  sw t5, 80(s11)
  ld s10, 200(sp)
  lw t5, 80(s10)
  ld s11, 192(sp)
  sw t5, 148(s11)
  ld s10, 192(sp)
  lw t5, 148(s10)
  addw t4, t5, t4
  ld s11, 200(sp)
  sw t4, 84(s11)
  ld s10, 200(sp)
  lw t4, 84(s10)
  ld s11, 192(sp)
  sw t4, 152(s11)
  ld s10, 192(sp)
  lw t4, 152(s10)
  addw t3, t4, t3
  ld s11, 200(sp)
  sw t3, 88(s11)
  ld s10, 200(sp)
  lw t3, 88(s10)
  ld s11, 192(sp)
  sw t3, 156(s11)
  ld s10, 192(sp)
  lw t3, 156(s10)
  addw t2, t3, t2
  ld s11, 200(sp)
  sw t2, 92(s11)
  ld s10, 200(sp)
  lw t2, 92(s10)
  ld s11, 192(sp)
  sw t2, 160(s11)
  ld s10, 192(sp)
  lw t2, 160(s10)
  addw ra, t2, ra
  ld s11, 200(sp)
  sw ra, 96(s11)
  ld s10, 200(sp)
  lw ra, 96(s10)
  ld s11, 192(sp)
  sw ra, 164(s11)
  ld s10, 192(sp)
  lw ra, 164(s10)
  addw a7, ra, a7
  ld s11, 200(sp)
  sw a7, 100(s11)
  ld s10, 200(sp)
  lw a7, 100(s10)
  ld s11, 192(sp)
  sw a7, 168(s11)
  ld s10, 192(sp)
  lw a7, 168(s10)
  addw a6, a7, a6
  ld s11, 200(sp)
  sw a6, 104(s11)
  ld s10, 200(sp)
  lw a6, 104(s10)
  ld s11, 192(sp)
  sw a6, 172(s11)
  ld s10, 192(sp)
  lw a6, 172(s10)
  addw a5, a6, a5
  ld s11, 200(sp)
  sw a5, 108(s11)
  ld s10, 200(sp)
  lw a5, 108(s10)
  ld s11, 192(sp)
  sw a5, 176(s11)
  ld s10, 192(sp)
  lw a5, 176(s10)
  addw a4, a5, a4
  ld s11, 200(sp)
  sw a4, 112(s11)
  ld s10, 200(sp)
  lw a4, 112(s10)
  ld s11, 192(sp)
  sw a4, 180(s11)
  ld s10, 192(sp)
  lw a4, 180(s10)
  addw a3, a4, a3
  ld s11, 200(sp)
  sw a3, 116(s11)
  ld s10, 200(sp)
  lw a3, 116(s10)
  ld s11, 192(sp)
  sw a3, 184(s11)
  ld s10, 192(sp)
  lw a3, 184(s10)
  addw a2, a3, a2
  ld s11, 200(sp)
  sw a2, 120(s11)
  ld s10, 200(sp)
  lw a2, 120(s10)
  ld s11, 192(sp)
  sw a2, 188(s11)
  ld s10, 192(sp)
  lw a2, 188(s10)
  addw a1, a2, a1
  ld s11, 200(sp)
  sw a1, 124(s11)
  mv s7, s1
.Lbb13:
  li s11, 16
  blt s7, s11, .Lbb14
.Lbb15:
  ld s10, 208(sp)
  lw a0, 0(s10)
  addiw s2, t1, -1
  ld s11, 192(sp)
  sw t0, 128(s11)
  addw a0, t0, a0
  ld s11, 200(sp)
  sw a0, 64(s11)
  mv s0, s1
  j .Lbb16
.Lbb14:
  slliw a1, s7, 2
  ld s10, 200(sp)
  add ra, s10, a1
  ld s10, 192(sp)
  add a7, s10, a1
.Lbb17:
  lw a1, 128(a7)
  lw a2, 64(ra)
  bge a1, a2, .Lbb18
.Lbb19:
  lw a1, 128(a7)
  slliw a1, a1, 2
  add a1, a0, a1
  lw a1, 0(a1)
  j .Lbb20
.Lbb18:
  addiw s7, s7, 1
  j .Lbb13
.Lbb20:
  mv a6, a1
  mv s6, s1
  j .Lbb21
.Lbb22:
  sraiw a2, a6, 31
  addiw s6, s6, 1
  andi a2, a2, 15
  addw a2, a6, a2
  sraiw a6, a2, 4
.Lbb21:
  blt s6, t1, .Lbb22
.Lbb23:
  slli a2, a6, 1
  srli a2, a2, 60
  add a2, a6, a2
  andi a2, a2, -16
  subw a2, a6, a2
  beq a2, s7, .Lbb24
.Lbb25:
  mv a5, a1
  mv s5, s1
  j .Lbb26
.Lbb27:
  sraiw a2, a5, 31
  addiw s5, s5, 1
  andi a2, a2, 15
  addw a2, a5, a2
  sraiw a5, a2, 4
.Lbb26:
  blt s5, t1, .Lbb27
.Lbb28:
  slli a2, a5, 1
  srli a2, a2, 60
  add a2, a5, a2
  andi a2, a2, -16
  subw a2, a5, a2
  slliw a2, a2, 2
  ld s10, 192(sp)
  add a2, s10, a2
  lw a2, 128(a2)
  slliw a2, a2, 2
  add a2, a0, a2
  lw a5, 0(a2)
  mv a4, a1
  mv s4, s1
  j .Lbb29
.Lbb30:
  sraiw a2, a4, 31
  addiw s4, s4, 1
  andi a2, a2, 15
  addw a2, a4, a2
  sraiw a4, a2, 4
.Lbb29:
  blt s4, t1, .Lbb30
.Lbb31:
  slli a2, a4, 1
  srli a2, a2, 60
  add a2, a4, a2
  andi a2, a2, -16
  subw a2, a4, a2
  slliw a2, a2, 2
  ld s10, 192(sp)
  add a2, s10, a2
  lw a2, 128(a2)
  slliw a2, a2, 2
  add a2, a0, a2
  sw a1, 0(a2)
  mv a3, a1
  mv s3, s1
  j .Lbb32
.Lbb33:
  sraiw a2, a3, 31
  addiw s3, s3, 1
  andi a2, a2, 15
  addw a2, a3, a2
  sraiw a3, a2, 4
.Lbb32:
  blt s3, t1, .Lbb33
.Lbb34:
  mv s2, s1
  j .Lbb35
.Lbb36:
  sraiw a2, a1, 31
  addiw s2, s2, 1
  andi a2, a2, 15
  addw a1, a1, a2
  sraiw a2, a1, 4
  mv a1, a2
.Lbb35:
  blt s2, t1, .Lbb36
.Lbb37:
  slli a4, a1, 1
  slli a2, a3, 1
  srli a4, a4, 60
  srli a2, a2, 60
  add a4, a1, a4
  add a2, a3, a2
  andi a4, a4, -16
  andi a2, a2, -16
  subw a1, a1, a4
  subw a3, a3, a2
  slliw a2, a1, 2
  slliw a1, a3, 2
  ld s10, 192(sp)
  add a2, s10, a2
  lw a2, 128(a2)
  add a1, s10, a1
  addiw a2, a2, 1
  sw a2, 128(a1)
  mv a1, a5
  j .Lbb20
.Lbb24:
  lw a3, 128(a7)
  mv a2, a3
  slliw a3, a3, 2
  add a3, a0, a3
  addiw a2, a2, 1
  sw a2, 128(a7)
  sw a1, 0(a3)
  j .Lbb17
.Lbb16:
  li s11, 16
  bge s0, s11, .Lbb1
.Lbb38:
  bge s1, s0, .Lbb39
.Lbb40:
  slliw a1, s0, 2
  ld s10, 200(sp)
  add a0, s10, a1
  lw a3, 60(a0)
  ld s10, 208(sp)
  add a2, s10, a1
  lw a2, 0(a2)
  ld s10, 192(sp)
  add a1, s10, a1
  sw a3, 128(a1)
  mv a1, a3
  addw a1, a1, a2
  sw a1, 64(a0)
.Lbb39:
  slliw a0, s0, 2
  ld s10, 192(sp)
  add a1, s10, a0
  lw a1, 128(a1)
  ld s10, 200(sp)
  add a0, s10, a0
  lw a2, 64(a0)
  mv a0, s2
  call radixSort
  addiw s0, s0, 1
  j .Lbb16
.Lbb1:
  ld s0, 328(sp)
  ld s1, 320(sp)
  ld s2, 312(sp)
  ld s3, 304(sp)
  ld s4, 296(sp)
  ld s5, 288(sp)
  ld s6, 280(sp)
  ld s7, 272(sp)
  ld s8, 264(sp)
  ld s9, 256(sp)
  ld s10, 248(sp)
  ld s11, 240(sp)
  ld ra, 232(sp)
  addi sp, sp, 336
  ret 


main:
.Lbb41:
  addi sp, sp, -48
  sd s0, 40(sp)
  sd s1, 32(sp)
  sd s2, 24(sp)
  sd s3, 16(sp)
  sd s4, 8(sp)
  sd ra, 0(sp)
  la s0, a
  mv a0, s0
  call getarray
  mv s4, a0
  li a0, 90
  call _sysy_starttime
  li s3, 0
  li a0, 9
  li a1, 0
  mv a2, s4
  call radixSort
  la s1, ans
  lw a0, 0(s1)
  mv s2, s3
  j .Lbb42
.Lbb43:
  slliw a3, s2, 2
  addiw a2, s2, 2
  addiw a1, s2, 1
  add a3, s0, a3
  lw a3, 0(a3)
  remw a2, a3, a2
  mulw a2, s2, a2
  addw a0, a0, a2
  addiw a0, a0, 3
  mv s2, a1
.Lbb42:
  blt s2, s4, .Lbb43
.Lbb44:
  sw a0, 0(s1)
  bge a0, s3, .Lbb45
.Lbb46:
  sub a0, s3, a0
  sw a0, 0(s1)
.Lbb45:
  li a0, 102
  call _sysy_stoptime
  lw a0, 0(s1)
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

.bss
  .align 4
a:
  .space 120000040
ans:
  .space 4
