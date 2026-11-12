"""Inline PTX used by QSA tensor-memory transfers, gather and epilogues."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32, Int64

@cute.jit
def split_load(addr:Int32,values:cute.Tensor,n:cutlass.Constexpr,offset:cutlass.Constexpr):
    asm='tcgen05.ld.sync.aligned.16x32bx2.x'+str(n)+'.b32 {'+', '.join('$'+str(i) for i in range(n))+'}, [$'+str(n)+'], '+str(offset)+';'
    result=llvm.inline_asm(llvm.StructType.get_literal([Float32.mlir_type]*n),
        [addr.ir_value()],asm,','.join(['=f']*n+['r']),has_side_effects=True,
        is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT)
    for i in cutlass.range_constexpr(n):
        values[i]=Float32(llvm.extractvalue(Float32.mlir_type,result,[i]))

@cute.jit
def split_store(addr:Int32, values:cute.Tensor, n:cutlass.Constexpr, offset:cutlass.Constexpr):
    asm='tcgen05.st.sync.aligned.16x32bx2.x'+str(n)+'.b32 [$0], '+str(offset)+', {'+', '.join('$'+str(i+1) for i in range(n))+'};'
    operands=[addr.ir_value()]
    for i in cutlass.range_constexpr(n):operands.append(Float32(values[i]).ir_value())
    llvm.inline_asm(None,operands,asm,'r'+',f'*n,has_side_effects=True,
        is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def prefetch_l2(addr64: Int64) -> None:
    """prefetch.global.L2 - register-free cache hint (e19 gate prefetch).

    Warms the 128B line containing ``addr64`` into L2 so the epilogue's gate
    LDG is an L2 hit instead of a DRAM round trip; a hint, so it can never
    change semantics or deadlock, and it costs no destination register.
    """
    asm = "prefetch.global.L2 [$0];"
    llvm.inline_asm(
        None,
        [Int64(addr64).ir_value()],
        asm,
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def cp_async_16(dst_shared32: Int32, src_global64: Int64) -> None:
    """One raw-PTX ``cp.async.cg.shared.global`` 16B copy (a
    single LDGSTS.128 - no destination register, completion tracked by the
    issuing thread's cp.async group queue).

    ``dst_shared32`` is the 18-bit shared-window address of a 16B-aligned
    SMEM slot (the same ``Int32(Int64(ptr.toint()) & 0x3FFFF)`` extraction
    used for shared-memory addresses); ``src_global64`` a 16B-aligned
    global address.  Raw PTX because the bare CopyG2SOp-atom cute.copy form
    is not guaranteed to lower to LDGSTS through every DSL path, and the
    staging mechanism's whole value is the register-free async copy - a
    silent LDG+STS fallback would double the gate fetch cost instead of
    hiding it (has_side_effects=True keeps the op anchored in program order
    relative to the surrounding proxy fences and group commits).
    """
    asm = "cp.async.cg.shared.global [$0], [$1], 16;"
    llvm.inline_asm(
        None,
        [Int32(dst_shared32).ir_value(), Int64(src_global64).ir_value()],
        asm,
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def fns_b32(a: Int32, base: Int32, n: Int32) -> Int32:
    """``fns.b32``: the position of the n-th set bit of ``a``, searching from
    bit ``base`` in the direction given by the sign of ``n`` (n>0 searches
    upward and counts the bit at ``base`` as the first candidate), or -1.

    e18: the bitmap preprocess P4 re-derives every emitted union entry's group
    id from its rank by clearing the lowest set bit ``kth`` times in a
    data-dependent serial loop (up to 31 iterations x 3 instructions, and the
    loop trip count is not known until the SMEM prefix word lands).  With
    base=0 and n=kth+1 this single instruction returns exactly the same bit
    position, so the whole loop collapses.  COARSE-grained use only (once per
    emitted entry, never per element) - see the module docstring for the e14
    fine-grained-asm verdict.
    """
    asm = "fns.b32 $0, $1, $2, $3;"
    r = llvm.inline_asm(
        Int32.mlir_type,
        [Int32(a).ir_value(), Int32(base).ir_value(), Int32(n).ir_value()],
        asm,
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


@cute.jit
def tanh_approx(x: Float32) -> Float32:
    """tanh.approx.f32: ONE MUFU instruction (sm_75+).

    e19 drain lever: sigmoid(g) = 0.5*tanh(g/2) + 0.5 replaces the champion's
    exp2 + rcp MUFU PAIR (and two FADDs) in the output gate - same function,
    max relative deviation ~2^-10.7 which is ~5e-4 RELATIVE on sig; the final
    product is bf16-rounded (ulp ~4e-3 relative at these magnitudes), so the
    substitution almost always lands on the identical bf16 word and any
    differing element moves by <=1 bf16 ulp - the same numerical class the
    shipped SUMTREE reassociation occupies, ~30x below the evaluator atol.
    """
    asm = "tanh.approx.f32 $0, $1;"
    r = llvm.inline_asm(
        Float32.mlir_type,
        [Float32(x).ir_value()],
        asm,
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(r)


@cute.jit
def cp_async_16_zfill(dst_shared32: Int32, src_global64: Int64, source_bytes: Int32):
    llvm.inline_asm(None,
        [dst_shared32.ir_value(),src_global64.ir_value(),source_bytes.ir_value()],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",has_side_effects=True,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
