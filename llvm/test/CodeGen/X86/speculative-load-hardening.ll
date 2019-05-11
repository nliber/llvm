; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -x86-slh-lfence | FileCheck %s --check-prefix=X64-LFENCE
;
; FIXME: Add support for 32-bit and other EH ABIs.

declare void @leak(i32 %v1, i32 %v2)

declare void @sink(i32)

define i32 @test_trivial_entry_load(i32* %ptr) speculative_load_hardening {
; X64-LABEL: test_trivial_entry_load:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movq %rsp, %rcx
; X64-NEXT:    movq $-1, %rax
; X64-NEXT:    sarq $63, %rcx
; X64-NEXT:    movl (%rdi), %eax
; X64-NEXT:    orl %ecx, %eax
; X64-NEXT:    shlq $47, %rcx
; X64-NEXT:    orq %rcx, %rsp
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_trivial_entry_load:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    movl (%rdi), %eax
; X64-LFENCE-NEXT:    retq
entry:
  %v = load i32, i32* %ptr
  ret i32 %v
}

define void @test_basic_conditions(i32 %a, i32 %b, i32 %c, i32* %ptr1, i32* %ptr2, i32** %ptr3) speculative_load_hardening {
; X64-LABEL: test_basic_conditions:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %r15
; X64-NEXT:    .cfi_def_cfa_offset 16
; X64-NEXT:    pushq %r14
; X64-NEXT:    .cfi_def_cfa_offset 24
; X64-NEXT:    pushq %rbx
; X64-NEXT:    .cfi_def_cfa_offset 32
; X64-NEXT:    .cfi_offset %rbx, -32
; X64-NEXT:    .cfi_offset %r14, -24
; X64-NEXT:    .cfi_offset %r15, -16
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq $-1, %rbx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    testl %edi, %edi
; X64-NEXT:    jne .LBB1_1
; X64-NEXT:  # %bb.2: # %then1
; X64-NEXT:    cmovneq %rbx, %rax
; X64-NEXT:    testl %esi, %esi
; X64-NEXT:    je .LBB1_4
; X64-NEXT:  .LBB1_1:
; X64-NEXT:    cmoveq %rbx, %rax
; X64-NEXT:  .LBB1_8: # %exit
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    .cfi_def_cfa_offset 24
; X64-NEXT:    popq %r14
; X64-NEXT:    .cfi_def_cfa_offset 16
; X64-NEXT:    popq %r15
; X64-NEXT:    .cfi_def_cfa_offset 8
; X64-NEXT:    retq
; X64-NEXT:  .LBB1_4: # %then2
; X64-NEXT:    .cfi_def_cfa_offset 32
; X64-NEXT:    movq %r8, %r14
; X64-NEXT:    cmovneq %rbx, %rax
; X64-NEXT:    testl %edx, %edx
; X64-NEXT:    je .LBB1_6
; X64-NEXT:  # %bb.5: # %else3
; X64-NEXT:    cmoveq %rbx, %rax
; X64-NEXT:    movslq (%r9), %rcx
; X64-NEXT:    orq %rax, %rcx
; X64-NEXT:    leaq (%r14,%rcx,4), %r15
; X64-NEXT:    movl %ecx, (%r14,%rcx,4)
; X64-NEXT:    jmp .LBB1_7
; X64-NEXT:  .LBB1_6: # %then3
; X64-NEXT:    cmovneq %rbx, %rax
; X64-NEXT:    movl (%rcx), %ecx
; X64-NEXT:    addl (%r14), %ecx
; X64-NEXT:    movslq %ecx, %rdi
; X64-NEXT:    orq %rax, %rdi
; X64-NEXT:    movl (%r14,%rdi,4), %esi
; X64-NEXT:    orl %eax, %esi
; X64-NEXT:    movq (%r9), %r15
; X64-NEXT:    orq %rax, %r15
; X64-NEXT:    addl (%r15), %esi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    # kill: def $edi killed $edi killed $rdi
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq leak
; X64-NEXT:  .Lslh_ret_addr0:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr0, %rcx
; X64-NEXT:    cmovneq %rbx, %rax
; X64-NEXT:  .LBB1_7: # %merge
; X64-NEXT:    movslq (%r15), %rcx
; X64-NEXT:    orq %rax, %rcx
; X64-NEXT:    movl $0, (%r14,%rcx,4)
; X64-NEXT:    jmp .LBB1_8
;
; X64-LFENCE-LABEL: test_basic_conditions:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 16
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 24
; X64-LFENCE-NEXT:    pushq %rax
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 32
; X64-LFENCE-NEXT:    .cfi_offset %rbx, -24
; X64-LFENCE-NEXT:    .cfi_offset %r14, -16
; X64-LFENCE-NEXT:    testl %edi, %edi
; X64-LFENCE-NEXT:    jne .LBB1_6
; X64-LFENCE-NEXT:  # %bb.1: # %then1
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    testl %esi, %esi
; X64-LFENCE-NEXT:    jne .LBB1_6
; X64-LFENCE-NEXT:  # %bb.2: # %then2
; X64-LFENCE-NEXT:    movq %r8, %rbx
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    testl %edx, %edx
; X64-LFENCE-NEXT:    je .LBB1_3
; X64-LFENCE-NEXT:  # %bb.4: # %else3
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movslq (%r9), %rax
; X64-LFENCE-NEXT:    leaq (%rbx,%rax,4), %r14
; X64-LFENCE-NEXT:    movl %eax, (%rbx,%rax,4)
; X64-LFENCE-NEXT:    jmp .LBB1_5
; X64-LFENCE-NEXT:  .LBB1_3: # %then3
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movl (%rcx), %eax
; X64-LFENCE-NEXT:    addl (%rbx), %eax
; X64-LFENCE-NEXT:    movslq %eax, %rdi
; X64-LFENCE-NEXT:    movl (%rbx,%rdi,4), %esi
; X64-LFENCE-NEXT:    movq (%r9), %r14
; X64-LFENCE-NEXT:    addl (%r14), %esi
; X64-LFENCE-NEXT:    # kill: def $edi killed $edi killed $rdi
; X64-LFENCE-NEXT:    callq leak
; X64-LFENCE-NEXT:  .LBB1_5: # %merge
; X64-LFENCE-NEXT:    movslq (%r14), %rax
; X64-LFENCE-NEXT:    movl $0, (%rbx,%rax,4)
; X64-LFENCE-NEXT:  .LBB1_6: # %exit
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    addq $8, %rsp
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 24
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 16
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 8
; X64-LFENCE-NEXT:    retq
entry:
  %a.cmp = icmp eq i32 %a, 0
  br i1 %a.cmp, label %then1, label %exit

then1:
  %b.cmp = icmp eq i32 %b, 0
  br i1 %b.cmp, label %then2, label %exit

then2:
  %c.cmp = icmp eq i32 %c, 0
  br i1 %c.cmp, label %then3, label %else3

then3:
  %secret1 = load i32, i32* %ptr1
  %secret2 = load i32, i32* %ptr2
  %secret.sum1 = add i32 %secret1, %secret2
  %ptr2.idx = getelementptr i32, i32* %ptr2, i32 %secret.sum1
  %secret3 = load i32, i32* %ptr2.idx
  %secret4 = load i32*, i32** %ptr3
  %secret5 = load i32, i32* %secret4
  %secret.sum2 = add i32 %secret3, %secret5
  call void @leak(i32 %secret.sum1, i32 %secret.sum2)
  br label %merge

else3:
  %secret6 = load i32*, i32** %ptr3
  %cast = ptrtoint i32* %secret6 to i32
  %ptr2.idx2 = getelementptr i32, i32* %ptr2, i32 %cast
  store i32 %cast, i32* %ptr2.idx2
  br label %merge

merge:
  %phi = phi i32* [ %secret4, %then3 ], [ %ptr2.idx2, %else3 ]
  %secret7 = load i32, i32* %phi
  %ptr2.idx3 = getelementptr i32, i32* %ptr2, i32 %secret7
  store i32 0, i32* %ptr2.idx3
  br label %exit

exit:
  ret void
}

define void @test_basic_loop(i32 %a, i32 %b, i32* %ptr1, i32* %ptr2) nounwind speculative_load_hardening {
; X64-LABEL: test_basic_loop:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %rbp
; X64-NEXT:    pushq %r15
; X64-NEXT:    pushq %r14
; X64-NEXT:    pushq %r12
; X64-NEXT:    pushq %rbx
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq $-1, %r15
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    testl %edi, %edi
; X64-NEXT:    je .LBB2_2
; X64-NEXT:  # %bb.1:
; X64-NEXT:    cmoveq %r15, %rax
; X64-NEXT:    jmp .LBB2_5
; X64-NEXT:  .LBB2_2: # %l.header.preheader
; X64-NEXT:    movq %rcx, %r14
; X64-NEXT:    movq %rdx, %r12
; X64-NEXT:    movl %esi, %ebp
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    xorl %ebx, %ebx
; X64-NEXT:    jmp .LBB2_3
; X64-NEXT:    .p2align 4, 0x90
; X64-NEXT:  .LBB2_6: # in Loop: Header=BB2_3 Depth=1
; X64-NEXT:    cmovgeq %r15, %rax
; X64-NEXT:  .LBB2_3: # %l.header
; X64-NEXT:    # =>This Inner Loop Header: Depth=1
; X64-NEXT:    movslq (%r12), %rcx
; X64-NEXT:    orq %rax, %rcx
; X64-NEXT:    movq %rax, %rdx
; X64-NEXT:    orq %r14, %rdx
; X64-NEXT:    movl (%rdx,%rcx,4), %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr1:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr1, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    incl %ebx
; X64-NEXT:    cmpl %ebp, %ebx
; X64-NEXT:    jl .LBB2_6
; X64-NEXT:  # %bb.4:
; X64-NEXT:    cmovlq %r15, %rax
; X64-NEXT:  .LBB2_5: # %exit
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    popq %r12
; X64-NEXT:    popq %r14
; X64-NEXT:    popq %r15
; X64-NEXT:    popq %rbp
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_basic_loop:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %rbp
; X64-LFENCE-NEXT:    pushq %r15
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    pushq %rax
; X64-LFENCE-NEXT:    testl %edi, %edi
; X64-LFENCE-NEXT:    jne .LBB2_3
; X64-LFENCE-NEXT:  # %bb.1: # %l.header.preheader
; X64-LFENCE-NEXT:    movq %rcx, %r14
; X64-LFENCE-NEXT:    movq %rdx, %r15
; X64-LFENCE-NEXT:    movl %esi, %ebp
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    xorl %ebx, %ebx
; X64-LFENCE-NEXT:    .p2align 4, 0x90
; X64-LFENCE-NEXT:  .LBB2_2: # %l.header
; X64-LFENCE-NEXT:    # =>This Inner Loop Header: Depth=1
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movslq (%r15), %rax
; X64-LFENCE-NEXT:    movl (%r14,%rax,4), %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    incl %ebx
; X64-LFENCE-NEXT:    cmpl %ebp, %ebx
; X64-LFENCE-NEXT:    jl .LBB2_2
; X64-LFENCE-NEXT:  .LBB2_3: # %exit
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    addq $8, %rsp
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    popq %r15
; X64-LFENCE-NEXT:    popq %rbp
; X64-LFENCE-NEXT:    retq
entry:
  %a.cmp = icmp eq i32 %a, 0
  br i1 %a.cmp, label %l.header, label %exit

l.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %l.header ]
  %secret = load i32, i32* %ptr1
  %ptr2.idx = getelementptr i32, i32* %ptr2, i32 %secret
  %leak = load i32, i32* %ptr2.idx
  call void @sink(i32 %leak)
  %i.next = add i32 %i, 1
  %i.cmp = icmp slt i32 %i.next, %b
  br i1 %i.cmp, label %l.header, label %exit

exit:
  ret void
}

define void @test_basic_nested_loop(i32 %a, i32 %b, i32 %c, i32* %ptr1, i32* %ptr2) nounwind speculative_load_hardening {
; X64-LABEL: test_basic_nested_loop:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %rbp
; X64-NEXT:    pushq %r15
; X64-NEXT:    pushq %r14
; X64-NEXT:    pushq %r13
; X64-NEXT:    pushq %r12
; X64-NEXT:    pushq %rbx
; X64-NEXT:    pushq %rax
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq $-1, %rbp
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    testl %edi, %edi
; X64-NEXT:    je .LBB3_2
; X64-NEXT:  # %bb.1:
; X64-NEXT:    cmoveq %rbp, %rax
; X64-NEXT:    jmp .LBB3_10
; X64-NEXT:  .LBB3_2: # %l1.header.preheader
; X64-NEXT:    movq %r8, %r14
; X64-NEXT:    movq %rcx, %rbx
; X64-NEXT:    movl %edx, %r12d
; X64-NEXT:    movl %esi, %r15d
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    xorl %r13d, %r13d
; X64-NEXT:    movl %esi, {{[-0-9]+}}(%r{{[sb]}}p) # 4-byte Spill
; X64-NEXT:    testl %r15d, %r15d
; X64-NEXT:    jg .LBB3_5
; X64-NEXT:    jmp .LBB3_4
; X64-NEXT:    .p2align 4, 0x90
; X64-NEXT:  .LBB3_12:
; X64-NEXT:    cmovgeq %rbp, %rax
; X64-NEXT:    testl %r15d, %r15d
; X64-NEXT:    jle .LBB3_4
; X64-NEXT:  .LBB3_5: # %l2.header.preheader
; X64-NEXT:    cmovleq %rbp, %rax
; X64-NEXT:    xorl %r15d, %r15d
; X64-NEXT:    jmp .LBB3_6
; X64-NEXT:    .p2align 4, 0x90
; X64-NEXT:  .LBB3_11: # in Loop: Header=BB3_6 Depth=1
; X64-NEXT:    cmovgeq %rbp, %rax
; X64-NEXT:  .LBB3_6: # %l2.header
; X64-NEXT:    # =>This Inner Loop Header: Depth=1
; X64-NEXT:    movslq (%rbx), %rcx
; X64-NEXT:    orq %rax, %rcx
; X64-NEXT:    movq %rax, %rdx
; X64-NEXT:    orq %r14, %rdx
; X64-NEXT:    movl (%rdx,%rcx,4), %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr2:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr2, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    incl %r15d
; X64-NEXT:    cmpl %r12d, %r15d
; X64-NEXT:    jl .LBB3_11
; X64-NEXT:  # %bb.7:
; X64-NEXT:    cmovlq %rbp, %rax
; X64-NEXT:    movl {{[-0-9]+}}(%r{{[sb]}}p), %r15d # 4-byte Reload
; X64-NEXT:    jmp .LBB3_8
; X64-NEXT:    .p2align 4, 0x90
; X64-NEXT:  .LBB3_4:
; X64-NEXT:    cmovgq %rbp, %rax
; X64-NEXT:  .LBB3_8: # %l1.latch
; X64-NEXT:    movslq (%rbx), %rcx
; X64-NEXT:    orq %rax, %rcx
; X64-NEXT:    movq %rax, %rdx
; X64-NEXT:    orq %r14, %rdx
; X64-NEXT:    movl (%rdx,%rcx,4), %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr3:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr3, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    incl %r13d
; X64-NEXT:    cmpl %r15d, %r13d
; X64-NEXT:    jl .LBB3_12
; X64-NEXT:  # %bb.9:
; X64-NEXT:    cmovlq %rbp, %rax
; X64-NEXT:  .LBB3_10: # %exit
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    addq $8, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    popq %r12
; X64-NEXT:    popq %r13
; X64-NEXT:    popq %r14
; X64-NEXT:    popq %r15
; X64-NEXT:    popq %rbp
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_basic_nested_loop:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %rbp
; X64-LFENCE-NEXT:    pushq %r15
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    pushq %r13
; X64-LFENCE-NEXT:    pushq %r12
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    pushq %rax
; X64-LFENCE-NEXT:    testl %edi, %edi
; X64-LFENCE-NEXT:    jne .LBB3_6
; X64-LFENCE-NEXT:  # %bb.1: # %l1.header.preheader
; X64-LFENCE-NEXT:    movq %r8, %r14
; X64-LFENCE-NEXT:    movq %rcx, %rbx
; X64-LFENCE-NEXT:    movl %edx, %r13d
; X64-LFENCE-NEXT:    movl %esi, %r15d
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    xorl %r12d, %r12d
; X64-LFENCE-NEXT:    .p2align 4, 0x90
; X64-LFENCE-NEXT:  .LBB3_2: # %l1.header
; X64-LFENCE-NEXT:    # =>This Loop Header: Depth=1
; X64-LFENCE-NEXT:    # Child Loop BB3_4 Depth 2
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    testl %r15d, %r15d
; X64-LFENCE-NEXT:    jle .LBB3_5
; X64-LFENCE-NEXT:  # %bb.3: # %l2.header.preheader
; X64-LFENCE-NEXT:    # in Loop: Header=BB3_2 Depth=1
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    xorl %ebp, %ebp
; X64-LFENCE-NEXT:    .p2align 4, 0x90
; X64-LFENCE-NEXT:  .LBB3_4: # %l2.header
; X64-LFENCE-NEXT:    # Parent Loop BB3_2 Depth=1
; X64-LFENCE-NEXT:    # => This Inner Loop Header: Depth=2
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movslq (%rbx), %rax
; X64-LFENCE-NEXT:    movl (%r14,%rax,4), %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    incl %ebp
; X64-LFENCE-NEXT:    cmpl %r13d, %ebp
; X64-LFENCE-NEXT:    jl .LBB3_4
; X64-LFENCE-NEXT:  .LBB3_5: # %l1.latch
; X64-LFENCE-NEXT:    # in Loop: Header=BB3_2 Depth=1
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movslq (%rbx), %rax
; X64-LFENCE-NEXT:    movl (%r14,%rax,4), %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    incl %r12d
; X64-LFENCE-NEXT:    cmpl %r15d, %r12d
; X64-LFENCE-NEXT:    jl .LBB3_2
; X64-LFENCE-NEXT:  .LBB3_6: # %exit
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    addq $8, %rsp
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    popq %r12
; X64-LFENCE-NEXT:    popq %r13
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    popq %r15
; X64-LFENCE-NEXT:    popq %rbp
; X64-LFENCE-NEXT:    retq
entry:
  %a.cmp = icmp eq i32 %a, 0
  br i1 %a.cmp, label %l1.header, label %exit

l1.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %l1.latch ]
  %b.cmp = icmp sgt i32 %b, 0
  br i1 %b.cmp, label %l2.header, label %l1.latch

l2.header:
  %j = phi i32 [ 0, %l1.header ], [ %j.next, %l2.header ]
  %secret = load i32, i32* %ptr1
  %ptr2.idx = getelementptr i32, i32* %ptr2, i32 %secret
  %leak = load i32, i32* %ptr2.idx
  call void @sink(i32 %leak)
  %j.next = add i32 %j, 1
  %j.cmp = icmp slt i32 %j.next, %c
  br i1 %j.cmp, label %l2.header, label %l1.latch

l1.latch:
  %secret2 = load i32, i32* %ptr1
  %ptr2.idx2 = getelementptr i32, i32* %ptr2, i32 %secret2
  %leak2 = load i32, i32* %ptr2.idx2
  call void @sink(i32 %leak2)
  %i.next = add i32 %i, 1
  %i.cmp = icmp slt i32 %i.next, %b
  br i1 %i.cmp, label %l1.header, label %exit

exit:
  ret void
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

define void @test_basic_eh(i32 %a, i32* %ptr1, i32* %ptr2) speculative_load_hardening personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; X64-LABEL: test_basic_eh:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %rbp
; X64-NEXT:    .cfi_def_cfa_offset 16
; X64-NEXT:    pushq %r15
; X64-NEXT:    .cfi_def_cfa_offset 24
; X64-NEXT:    pushq %r14
; X64-NEXT:    .cfi_def_cfa_offset 32
; X64-NEXT:    pushq %rbx
; X64-NEXT:    .cfi_def_cfa_offset 40
; X64-NEXT:    pushq %rax
; X64-NEXT:    .cfi_def_cfa_offset 48
; X64-NEXT:    .cfi_offset %rbx, -40
; X64-NEXT:    .cfi_offset %r14, -32
; X64-NEXT:    .cfi_offset %r15, -24
; X64-NEXT:    .cfi_offset %rbp, -16
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq $-1, %r15
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpl $41, %edi
; X64-NEXT:    jg .LBB4_1
; X64-NEXT:  # %bb.2: # %thrower
; X64-NEXT:    movq %rdx, %r14
; X64-NEXT:    movq %rsi, %rbx
; X64-NEXT:    cmovgq %r15, %rax
; X64-NEXT:    movslq %edi, %rcx
; X64-NEXT:    movl (%rsi,%rcx,4), %ebp
; X64-NEXT:    orl %eax, %ebp
; X64-NEXT:    movl $4, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq __cxa_allocate_exception
; X64-NEXT:  .Lslh_ret_addr4:
; X64-NEXT:    movq %rsp, %rcx
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rdx
; X64-NEXT:    sarq $63, %rcx
; X64-NEXT:    cmpq $.Lslh_ret_addr4, %rdx
; X64-NEXT:    cmovneq %r15, %rcx
; X64-NEXT:    movl %ebp, (%rax)
; X64-NEXT:  .Ltmp0:
; X64-NEXT:    shlq $47, %rcx
; X64-NEXT:    movq %rax, %rdi
; X64-NEXT:    xorl %esi, %esi
; X64-NEXT:    xorl %edx, %edx
; X64-NEXT:    orq %rcx, %rsp
; X64-NEXT:    callq __cxa_throw
; X64-NEXT:  .Lslh_ret_addr5:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr5, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:  .Ltmp1:
; X64-NEXT:    jmp .LBB4_3
; X64-NEXT:  .LBB4_1:
; X64-NEXT:    cmovleq %r15, %rax
; X64-NEXT:  .LBB4_3: # %exit
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    addq $8, %rsp
; X64-NEXT:    .cfi_def_cfa_offset 40
; X64-NEXT:    popq %rbx
; X64-NEXT:    .cfi_def_cfa_offset 32
; X64-NEXT:    popq %r14
; X64-NEXT:    .cfi_def_cfa_offset 24
; X64-NEXT:    popq %r15
; X64-NEXT:    .cfi_def_cfa_offset 16
; X64-NEXT:    popq %rbp
; X64-NEXT:    .cfi_def_cfa_offset 8
; X64-NEXT:    retq
; X64-NEXT:  .LBB4_4: # %lpad
; X64-NEXT:    .cfi_def_cfa_offset 48
; X64-NEXT:  .Ltmp2:
; X64-NEXT:    movq %rsp, %rcx
; X64-NEXT:    sarq $63, %rcx
; X64-NEXT:    movl (%rax), %eax
; X64-NEXT:    addl (%rbx), %eax
; X64-NEXT:    cltq
; X64-NEXT:    orq %rcx, %rax
; X64-NEXT:    movl (%r14,%rax,4), %edi
; X64-NEXT:    orl %ecx, %edi
; X64-NEXT:    shlq $47, %rcx
; X64-NEXT:    orq %rcx, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr6:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr6, %rcx
; X64-NEXT:    cmovneq %r15, %rax
;
; X64-LFENCE-LABEL: test_basic_eh:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %rbp
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 16
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 24
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 32
; X64-LFENCE-NEXT:    .cfi_offset %rbx, -32
; X64-LFENCE-NEXT:    .cfi_offset %r14, -24
; X64-LFENCE-NEXT:    .cfi_offset %rbp, -16
; X64-LFENCE-NEXT:    cmpl $41, %edi
; X64-LFENCE-NEXT:    jg .LBB4_2
; X64-LFENCE-NEXT:  # %bb.1: # %thrower
; X64-LFENCE-NEXT:    movq %rdx, %r14
; X64-LFENCE-NEXT:    movq %rsi, %rbx
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    movslq %edi, %rax
; X64-LFENCE-NEXT:    movl (%rsi,%rax,4), %ebp
; X64-LFENCE-NEXT:    movl $4, %edi
; X64-LFENCE-NEXT:    callq __cxa_allocate_exception
; X64-LFENCE-NEXT:    movl %ebp, (%rax)
; X64-LFENCE-NEXT:  .Ltmp0:
; X64-LFENCE-NEXT:    movq %rax, %rdi
; X64-LFENCE-NEXT:    xorl %esi, %esi
; X64-LFENCE-NEXT:    xorl %edx, %edx
; X64-LFENCE-NEXT:    callq __cxa_throw
; X64-LFENCE-NEXT:  .Ltmp1:
; X64-LFENCE-NEXT:  .LBB4_2: # %exit
; X64-LFENCE-NEXT:    lfence
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 24
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 16
; X64-LFENCE-NEXT:    popq %rbp
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 8
; X64-LFENCE-NEXT:    retq
; X64-LFENCE-NEXT:  .LBB4_3: # %lpad
; X64-LFENCE-NEXT:    .cfi_def_cfa_offset 32
; X64-LFENCE-NEXT:  .Ltmp2:
; X64-LFENCE-NEXT:    movl (%rax), %eax
; X64-LFENCE-NEXT:    addl (%rbx), %eax
; X64-LFENCE-NEXT:    cltq
; X64-LFENCE-NEXT:    movl (%r14,%rax,4), %edi
; X64-LFENCE-NEXT:    callq sink
entry:
  %a.cmp = icmp slt i32 %a, 42
  br i1 %a.cmp, label %thrower, label %exit

thrower:
  %badidx = getelementptr i32, i32* %ptr1, i32 %a
  %secret1 = load i32, i32* %badidx
  %e.ptr = call i8* @__cxa_allocate_exception(i64 4)
  %e.ptr.cast = bitcast i8* %e.ptr to i32*
  store i32 %secret1, i32* %e.ptr.cast
  invoke void @__cxa_throw(i8* %e.ptr, i8* null, i8* null)
          to label %exit unwind label %lpad

exit:
  ret void

lpad:
  %e = landingpad { i8*, i32 }
          catch i8* null
  %e.catch.ptr = extractvalue { i8*, i32 } %e, 0
  %e.catch.ptr.cast = bitcast i8* %e.catch.ptr to i32*
  %secret1.catch = load i32, i32* %e.catch.ptr.cast
  %secret2 = load i32, i32* %ptr1
  %secret.sum = add i32 %secret1.catch, %secret2
  %ptr2.idx = getelementptr i32, i32* %ptr2, i32 %secret.sum
  %leak = load i32, i32* %ptr2.idx
  call void @sink(i32 %leak)
  unreachable
}

declare void @sink_float(float)
declare void @sink_double(double)

; Test direct and converting loads of floating point values.
define void @test_fp_loads(float* %fptr, double* %dptr, i32* %i32ptr, i64* %i64ptr) nounwind speculative_load_hardening {
; X64-LABEL: test_fp_loads:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %r15
; X64-NEXT:    pushq %r14
; X64-NEXT:    pushq %r13
; X64-NEXT:    pushq %r12
; X64-NEXT:    pushq %rbx
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq %rcx, %r15
; X64-NEXT:    movq %rdx, %r14
; X64-NEXT:    movq %rsi, %rbx
; X64-NEXT:    movq %rdi, %r12
; X64-NEXT:    movq $-1, %r13
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    orq %rax, %r12
; X64-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_float
; X64-NEXT:  .Lslh_ret_addr7:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr7, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    orq %rax, %rbx
; X64-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_double
; X64-NEXT:  .Lslh_ret_addr8:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr8, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-NEXT:    cvtsd2ss %xmm0, %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_float
; X64-NEXT:  .Lslh_ret_addr9:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr9, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-NEXT:    cvtss2sd %xmm0, %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_double
; X64-NEXT:  .Lslh_ret_addr10:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr10, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    orq %rax, %r14
; X64-NEXT:    xorps %xmm0, %xmm0
; X64-NEXT:    cvtsi2ssl (%r14), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_float
; X64-NEXT:  .Lslh_ret_addr11:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr11, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    orq %rax, %r15
; X64-NEXT:    xorps %xmm0, %xmm0
; X64-NEXT:    cvtsi2sdq (%r15), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_double
; X64-NEXT:  .Lslh_ret_addr12:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr12, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    xorps %xmm0, %xmm0
; X64-NEXT:    cvtsi2ssq (%r15), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_float
; X64-NEXT:  .Lslh_ret_addr13:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr13, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    xorps %xmm0, %xmm0
; X64-NEXT:    cvtsi2sdl (%r14), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_double
; X64-NEXT:  .Lslh_ret_addr14:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr14, %rcx
; X64-NEXT:    cmovneq %r13, %rax
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    popq %r12
; X64-NEXT:    popq %r13
; X64-NEXT:    popq %r14
; X64-NEXT:    popq %r15
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_fp_loads:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %r15
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    pushq %r12
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    pushq %rax
; X64-LFENCE-NEXT:    movq %rcx, %r15
; X64-LFENCE-NEXT:    movq %rdx, %r14
; X64-LFENCE-NEXT:    movq %rsi, %rbx
; X64-LFENCE-NEXT:    movq %rdi, %r12
; X64-LFENCE-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-LFENCE-NEXT:    callq sink_float
; X64-LFENCE-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-LFENCE-NEXT:    callq sink_double
; X64-LFENCE-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-LFENCE-NEXT:    cvtsd2ss %xmm0, %xmm0
; X64-LFENCE-NEXT:    callq sink_float
; X64-LFENCE-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-LFENCE-NEXT:    cvtss2sd %xmm0, %xmm0
; X64-LFENCE-NEXT:    callq sink_double
; X64-LFENCE-NEXT:    xorps %xmm0, %xmm0
; X64-LFENCE-NEXT:    cvtsi2ssl (%r14), %xmm0
; X64-LFENCE-NEXT:    callq sink_float
; X64-LFENCE-NEXT:    xorps %xmm0, %xmm0
; X64-LFENCE-NEXT:    cvtsi2sdq (%r15), %xmm0
; X64-LFENCE-NEXT:    callq sink_double
; X64-LFENCE-NEXT:    xorps %xmm0, %xmm0
; X64-LFENCE-NEXT:    cvtsi2ssq (%r15), %xmm0
; X64-LFENCE-NEXT:    callq sink_float
; X64-LFENCE-NEXT:    xorps %xmm0, %xmm0
; X64-LFENCE-NEXT:    cvtsi2sdl (%r14), %xmm0
; X64-LFENCE-NEXT:    callq sink_double
; X64-LFENCE-NEXT:    addq $8, %rsp
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    popq %r12
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    popq %r15
; X64-LFENCE-NEXT:    retq
entry:
  %f1 = load float, float* %fptr
  call void @sink_float(float %f1)
  %d1 = load double, double* %dptr
  call void @sink_double(double %d1)
  %f2.d = load double, double* %dptr
  %f2 = fptrunc double %f2.d to float
  call void @sink_float(float %f2)
  %d2.f = load float, float* %fptr
  %d2 = fpext float %d2.f to double
  call void @sink_double(double %d2)
  %f3.i = load i32, i32* %i32ptr
  %f3 = sitofp i32 %f3.i to float
  call void @sink_float(float %f3)
  %d3.i = load i64, i64* %i64ptr
  %d3 = sitofp i64 %d3.i to double
  call void @sink_double(double %d3)
  %f4.i = load i64, i64* %i64ptr
  %f4 = sitofp i64 %f4.i to float
  call void @sink_float(float %f4)
  %d4.i = load i32, i32* %i32ptr
  %d4 = sitofp i32 %d4.i to double
  call void @sink_double(double %d4)
  ret void
}

declare void @sink_v4f32(<4 x float>)
declare void @sink_v2f64(<2 x double>)
declare void @sink_v16i8(<16 x i8>)
declare void @sink_v8i16(<8 x i16>)
declare void @sink_v4i32(<4 x i32>)
declare void @sink_v2i64(<2 x i64>)

; Test loads of vectors.
define void @test_vec_loads(<4 x float>* %v4f32ptr, <2 x double>* %v2f64ptr, <16 x i8>* %v16i8ptr, <8 x i16>* %v8i16ptr, <4 x i32>* %v4i32ptr, <2 x i64>* %v2i64ptr) nounwind speculative_load_hardening {
; X64-LABEL: test_vec_loads:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %rbp
; X64-NEXT:    pushq %r15
; X64-NEXT:    pushq %r14
; X64-NEXT:    pushq %r13
; X64-NEXT:    pushq %r12
; X64-NEXT:    pushq %rbx
; X64-NEXT:    pushq %rax
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq %r9, %r14
; X64-NEXT:    movq %r8, %r15
; X64-NEXT:    movq %rcx, %r12
; X64-NEXT:    movq %rdx, %r13
; X64-NEXT:    movq %rsi, %rbx
; X64-NEXT:    movq $-1, %rbp
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    orq %rax, %rdi
; X64-NEXT:    movaps (%rdi), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v4f32
; X64-NEXT:  .Lslh_ret_addr15:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr15, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    orq %rax, %rbx
; X64-NEXT:    movaps (%rbx), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v2f64
; X64-NEXT:  .Lslh_ret_addr16:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr16, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    orq %rax, %r13
; X64-NEXT:    movaps (%r13), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v16i8
; X64-NEXT:  .Lslh_ret_addr17:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr17, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    orq %rax, %r12
; X64-NEXT:    movaps (%r12), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v8i16
; X64-NEXT:  .Lslh_ret_addr18:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr18, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    orq %rax, %r15
; X64-NEXT:    movaps (%r15), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v4i32
; X64-NEXT:  .Lslh_ret_addr19:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr19, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    orq %rax, %r14
; X64-NEXT:    movaps (%r14), %xmm0
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink_v2i64
; X64-NEXT:  .Lslh_ret_addr20:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr20, %rcx
; X64-NEXT:    cmovneq %rbp, %rax
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    addq $8, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    popq %r12
; X64-NEXT:    popq %r13
; X64-NEXT:    popq %r14
; X64-NEXT:    popq %r15
; X64-NEXT:    popq %rbp
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_vec_loads:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %r15
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    pushq %r13
; X64-LFENCE-NEXT:    pushq %r12
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    movq %r9, %r14
; X64-LFENCE-NEXT:    movq %r8, %r15
; X64-LFENCE-NEXT:    movq %rcx, %r12
; X64-LFENCE-NEXT:    movq %rdx, %r13
; X64-LFENCE-NEXT:    movq %rsi, %rbx
; X64-LFENCE-NEXT:    movaps (%rdi), %xmm0
; X64-LFENCE-NEXT:    callq sink_v4f32
; X64-LFENCE-NEXT:    movaps (%rbx), %xmm0
; X64-LFENCE-NEXT:    callq sink_v2f64
; X64-LFENCE-NEXT:    movaps (%r13), %xmm0
; X64-LFENCE-NEXT:    callq sink_v16i8
; X64-LFENCE-NEXT:    movaps (%r12), %xmm0
; X64-LFENCE-NEXT:    callq sink_v8i16
; X64-LFENCE-NEXT:    movaps (%r15), %xmm0
; X64-LFENCE-NEXT:    callq sink_v4i32
; X64-LFENCE-NEXT:    movaps (%r14), %xmm0
; X64-LFENCE-NEXT:    callq sink_v2i64
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    popq %r12
; X64-LFENCE-NEXT:    popq %r13
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    popq %r15
; X64-LFENCE-NEXT:    retq
entry:
  %x1 = load <4 x float>, <4 x float>* %v4f32ptr
  call void @sink_v4f32(<4 x float> %x1)
  %x2 = load <2 x double>, <2 x double>* %v2f64ptr
  call void @sink_v2f64(<2 x double> %x2)
  %x3 = load <16 x i8>, <16 x i8>* %v16i8ptr
  call void @sink_v16i8(<16 x i8> %x3)
  %x4 = load <8 x i16>, <8 x i16>* %v8i16ptr
  call void @sink_v8i16(<8 x i16> %x4)
  %x5 = load <4 x i32>, <4 x i32>* %v4i32ptr
  call void @sink_v4i32(<4 x i32> %x5)
  %x6 = load <2 x i64>, <2 x i64>* %v2i64ptr
  call void @sink_v2i64(<2 x i64> %x6)
  ret void
}

define void @test_deferred_hardening(i32* %ptr1, i32* %ptr2, i32 %x) nounwind speculative_load_hardening {
; X64-LABEL: test_deferred_hardening:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pushq %r15
; X64-NEXT:    pushq %r14
; X64-NEXT:    pushq %rbx
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq %rsi, %r14
; X64-NEXT:    movq %rdi, %rbx
; X64-NEXT:    movq $-1, %r15
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    movl (%rdi), %edi
; X64-NEXT:    incl %edi
; X64-NEXT:    imull %edx, %edi
; X64-NEXT:    orl %eax, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr21:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr21, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    movl (%rbx), %ecx
; X64-NEXT:    movl (%r14), %edx
; X64-NEXT:    leal 1(%rcx,%rdx), %edi
; X64-NEXT:    orl %eax, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr22:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr22, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    movl (%rbx), %edi
; X64-NEXT:    shll $7, %edi
; X64-NEXT:    orl %eax, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr23:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr23, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    movswl (%rbx), %edi
; X64-NEXT:    shrl $7, %edi
; X64-NEXT:    notl %edi
; X64-NEXT:    orl $-65536, %edi # imm = 0xFFFF0000
; X64-NEXT:    orl %eax, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr24:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr24, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    movzwl (%rbx), %ecx
; X64-NEXT:    rolw $9, %cx
; X64-NEXT:    movswl %cx, %edi
; X64-NEXT:    negl %edi
; X64-NEXT:    orl %eax, %edi
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    callq sink
; X64-NEXT:  .Lslh_ret_addr25:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq -{{[0-9]+}}(%rsp), %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    cmpq $.Lslh_ret_addr25, %rcx
; X64-NEXT:    cmovneq %r15, %rax
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    popq %rbx
; X64-NEXT:    popq %r14
; X64-NEXT:    popq %r15
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: test_deferred_hardening:
; X64-LFENCE:       # %bb.0: # %entry
; X64-LFENCE-NEXT:    pushq %r14
; X64-LFENCE-NEXT:    pushq %rbx
; X64-LFENCE-NEXT:    pushq %rax
; X64-LFENCE-NEXT:    movq %rsi, %r14
; X64-LFENCE-NEXT:    movq %rdi, %rbx
; X64-LFENCE-NEXT:    movl (%rdi), %edi
; X64-LFENCE-NEXT:    incl %edi
; X64-LFENCE-NEXT:    imull %edx, %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    movl (%rbx), %eax
; X64-LFENCE-NEXT:    movl (%r14), %ecx
; X64-LFENCE-NEXT:    leal 1(%rax,%rcx), %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    movl (%rbx), %edi
; X64-LFENCE-NEXT:    shll $7, %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    movswl (%rbx), %edi
; X64-LFENCE-NEXT:    shrl $7, %edi
; X64-LFENCE-NEXT:    notl %edi
; X64-LFENCE-NEXT:    orl $-65536, %edi # imm = 0xFFFF0000
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    movzwl (%rbx), %eax
; X64-LFENCE-NEXT:    rolw $9, %ax
; X64-LFENCE-NEXT:    movswl %ax, %edi
; X64-LFENCE-NEXT:    negl %edi
; X64-LFENCE-NEXT:    callq sink
; X64-LFENCE-NEXT:    addq $8, %rsp
; X64-LFENCE-NEXT:    popq %rbx
; X64-LFENCE-NEXT:    popq %r14
; X64-LFENCE-NEXT:    retq
entry:
  %a1 = load i32, i32* %ptr1
  %a2 = add i32 %a1, 1
  %a3 = mul i32 %a2, %x
  call void @sink(i32 %a3)
  %b1 = load i32, i32* %ptr1
  %b2 = add i32 %b1, 1
  %b3 = load i32, i32* %ptr2
  %b4 = add i32 %b2, %b3
  call void @sink(i32 %b4)
  %c1 = load i32, i32* %ptr1
  %c2 = shl i32 %c1, 7
  call void @sink(i32 %c2)
  %d1 = load i32, i32* %ptr1
  ; Check trunc and integer ops narrower than i32.
  %d2 = trunc i32 %d1 to i16
  %d3 = ashr i16 %d2, 7
  %d4 = zext i16 %d3 to i32
  %d5 = xor i32 %d4, -1
  call void @sink(i32 %d5)
  %e1 = load i32, i32* %ptr1
  %e2 = trunc i32 %e1 to i16
  %e3 = lshr i16 %e2, 7
  %e4 = shl i16 %e2, 9
  %e5 = or i16 %e3, %e4
  %e6 = sext i16 %e5 to i32
  %e7 = sub i32 0, %e6
  call void @sink(i32 %e7)
  ret void
}

; Make sure we don't crash on idempotent atomic operations which have a
; hardcoded reference to RSP+offset.
define void @idempotent_atomic(i32* %x) speculative_load_hardening {
; X64-LABEL: idempotent_atomic:
; X64:       # %bb.0:
; X64-NEXT:    movq %rsp, %rax
; X64-NEXT:    movq $-1, %rcx
; X64-NEXT:    sarq $63, %rax
; X64-NEXT:    lock orl $0, (%rsp)
; X64-NEXT:    shlq $47, %rax
; X64-NEXT:    orq %rax, %rsp
; X64-NEXT:    retq
;
; X64-LFENCE-LABEL: idempotent_atomic:
; X64-LFENCE:       # %bb.0:
; X64-LFENCE-NEXT:    lock orl $0, (%rsp)
; X64-LFENCE-NEXT:    retq
  %tmp = atomicrmw or i32* %x, i32 0 seq_cst
  ret void
}
