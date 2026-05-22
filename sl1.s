.global main
main:
.Lbb0:
  addi sp, sp, -224
  sd s0, 216(sp)
  sd s1, 208(sp)
  sd s2, 200(sp)
  sd s3, 192(sp)
  sd s4, 184(sp)
  sd s5, 176(sp)
  sd s6, 168(sp)
  sd s7, 160(sp)
  sd s8, 152(sp)
  sd s9, 144(sp)
  sd s10, 136(sp)
  sd s11, 128(sp)
  sd ra, 120(sp)
  la s1, y
  la s0, x
  call getint
  mv s9, a0
  call getint
  mv s10, a0
  sd s10, 0(sp)
  li a0, 13
  call _sysy_starttime
  li s2, 0
  li a1, 1
  mv a3, s2
.Lbb1:
  bge a3, s9, .Lbb2
.Lbb3:
  addiw a4, a3, 32
  blt s9, a4, .Lbb4
  j .Lbb5
.Lbb2:
  mv s5, a1
  mv s3, a1
  mv s4, s2
  j .Lbb6
.Lbb4:
  mv a4, s9
.Lbb5:
  mv a2, a3
.Lbb7:
  bge a2, a4, .Lbb8
.Lbb9:
  mv s7, s2
  j .Lbb10
.Lbb8:
  addiw a3, a3, 32
  j .Lbb1
.Lbb10:
  bge s7, s9, .Lbb11
.Lbb12:
  addiw s8, s7, 32
  blt s9, s8, .Lbb13
  j .Lbb14
.Lbb11:
  addiw a2, a2, 1
  j .Lbb7
.Lbb13:
  mv s8, s9
.Lbb14:
  mv a0, s7
.Lbb15:
  bge a0, s8, .Lbb16
.Lbb17:
  mv s6, s2
  j .Lbb18
.Lbb16:
  addiw s7, s7, 32
  j .Lbb10
.Lbb19:
  li s11, 1440000
  mulw t0, a2, s11
  li s11, 2400
  mulw a5, a0, s11
  li s11, 1440000
  mulw t1, a2, s11
  li s11, 2400
  mulw ra, a0, s11
  slliw a7, s6, 2
  slliw a6, s6, 2
  addiw s6, s6, 1
  add t2, s0, t0
  add t0, s1, t1
  add a5, t2, a5
  add ra, t0, ra
  add a5, a5, a7
  add a6, ra, a6
  sw a1, 0(a5)
  sw s2, 0(a6)
.Lbb18:
  blt s6, s9, .Lbb19
.Lbb20:
  addiw a0, a0, 1
  j .Lbb15
.Lbb6:
  addiw a0, s9, -1
  bge s4, a0, .Lbb21
.Lbb22:
  addiw a6, s4, 32
  addiw a0, s9, -1
  blt a0, a6, .Lbb23
  j .Lbb24
.Lbb21:
  li a0, 53
  call _sysy_stoptime
  mv a0, s9
  la a1, x
  call putarray
  sraiw a0, s9, 31
  andi a0, a0, 1
  addw a0, s9, a0
  sraiw a0, a0, 1
  sraiw a1, s9, 31
  andi a1, a1, 1
  addw a1, s9, a1
  sraiw a1, a1, 1
  li s11, 1440000
  mulw a0, a0, s11
  add a0, s0, a0
  li s11, 2400
  mulw a1, a1, s11
  add a1, a0, a1
  mv a0, s9
  call putarray
  li s11, 1440000
  mulw a0, s5, s11
  li s11, 1440000
  subw a0, a0, s11
  add a0, s0, a0
  li s11, 2400
  mulw a1, s3, s11
  li s11, 2400
  subw a1, a1, s11
  add a1, a0, a1
  mv a0, s9
  call putarray
  li a0, 0
  ld s0, 216(sp)
  ld s1, 208(sp)
  ld s2, 200(sp)
  ld s3, 192(sp)
  ld s4, 184(sp)
  ld s5, 176(sp)
  ld s6, 168(sp)
  ld s7, 160(sp)
  ld s8, 152(sp)
  ld s9, 144(sp)
  ld s10, 136(sp)
  ld s11, 128(sp)
  ld ra, 120(sp)
  addi sp, sp, 224
  ret 
.Lbb23:
  addiw a6, s9, -1
.Lbb24:
  mv a5, s4
.Lbb25:
  bge a5, a6, .Lbb26
.Lbb27:
  mv a4, a1
  mv s1, s2
  j .Lbb28
.Lbb26:
  addiw s4, s4, 32
  mv s5, a5
  j .Lbb6
.Lbb28:
  addiw a0, s9, -1
  bge s1, a0, .Lbb29
.Lbb30:
  addiw a3, s1, 32
  addiw a0, s9, -1
  blt a0, a3, .Lbb31
  j .Lbb32
.Lbb29:
  addiw a5, a5, 1
  mv s3, a4
  j .Lbb25
.Lbb31:
  addiw a3, s9, -1
.Lbb32:
  mv a2, s1
.Lbb33:
  bge a2, a3, .Lbb34
.Lbb35:
  mv a0, a1
  j .Lbb36
.Lbb34:
  addiw s1, s1, 32
  mv a4, a2
  j .Lbb28
.Lbb37:
  li s11, 1440000
  mulw s7, a5, s11
  li s11, 2400
  mulw a4, a2, s11
  li s11, 1440000
  mulw s8, a5, s11
  li s11, 2400
  mulw a7, a2, s11
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 40(sp)
  li s11, 2400
  mulw s10, a2, s11
  sd s10, 32(sp)
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 24(sp)
  li s11, 2400
  mulw s10, a2, s11
  sd s10, 16(sp)
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 8(sp)
  li s11, 2400
  mulw t4, a2, s11
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 48(sp)
  li s11, 2400
  mulw t6, a2, s11
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 72(sp)
  li s11, 2400
  mulw s10, a2, s11
  sd s10, 64(sp)
  li s11, 1440000
  mulw s10, a5, s11
  sd s10, 56(sp)
  li s11, 2400
  mulw s6, a2, s11
  slliw t0, a0, 2
  slliw ra, a0, 2
  slliw s5, a0, 2
  slliw s3, a0, 2
  slliw t5, a0, 2
  slliw t3, a0, 2
  slliw t1, a0, 2
  slliw t2, a0, 2
  addiw a0, a0, 1
  li s11, 1440000
  subw s10, s7, s11
  sd s10, 80(sp)
  li s11, 1440000
  addw s8, s8, s11
  ld s11, 40(sp)
  add s10, s0, s11
  sd s10, 40(sp)
  ld s10, 32(sp)
  li s11, 2400
  subw s10, s10, s11
  sd s10, 32(sp)
  ld s11, 24(sp)
  add s10, s0, s11
  sd s10, 24(sp)
  ld s10, 16(sp)
  li s11, 2400
  addw s10, s10, s11
  sd s10, 16(sp)
  ld s11, 8(sp)
  add s10, s0, s11
  sd s10, 8(sp)
  ld s11, 48(sp)
  add s10, s0, s11
  sd s10, 48(sp)
  ld s10, 72(sp)
  li s11, 1440000
  subw s10, s10, s11
  sd s10, 72(sp)
  ld s10, 64(sp)
  li s11, 2400
  subw s7, s10, s11
  ld s11, 56(sp)
  add s10, s0, s11
  sd s10, 88(sp)
  ld s11, 80(sp)
  add s10, s0, s11
  sd s10, 56(sp)
  add s10, s0, s8
  sd s10, 64(sp)
  ld s10, 40(sp)
  ld s11, 32(sp)
  add s10, s10, s11
  sd s10, 32(sp)
  ld s10, 24(sp)
  ld s11, 16(sp)
  add s10, s10, s11
  sd s10, 24(sp)
  ld s10, 8(sp)
  add t4, s10, t4
  ld s10, 48(sp)
  add s10, s10, t6
  sd s10, 16(sp)
  ld s11, 72(sp)
  add s8, s0, s11
  ld s10, 88(sp)
  add s6, s10, s6
  ld s10, 56(sp)
  add s10, s10, a4
  sd s10, 8(sp)
  ld s10, 64(sp)
  add t6, s10, a7
  ld s10, 32(sp)
  add a4, s10, s5
  lw a4, 0(a4)
  ld s10, 24(sp)
  add a7, s10, s3
  lw a7, 0(a7)
  add t4, t4, t5
  lw t4, -4(t4)
  ld s10, 16(sp)
  add t3, s10, t3
  lw t3, 4(t3)
  add t5, s8, s7
  add t2, s6, t2
  ld s10, 8(sp)
  add t0, s10, t0
  lw t0, 0(t0)
  add ra, t6, ra
  lw t6, 0(ra)
  add ra, t5, t1
  lw ra, -4(ra)
  addw t0, t0, t6
  addw a4, t0, a4
  addw a4, a4, a7
  addw a4, a4, t4
  addw a4, a4, t3
  addw a4, a4, ra
  ld s11, 0(sp)
  divw a4, a4, s11
  sw a4, 0(t2)
.Lbb36:
  addiw a4, s9, -1
  blt a0, a4, .Lbb37
.Lbb38:
  addiw a2, a2, 1
  j .Lbb33



.data

.bss
  .align 4
x:
  .space 864000000
y:
  .space 864000000
