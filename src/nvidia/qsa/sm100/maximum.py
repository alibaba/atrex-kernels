"""FP32 maximum reductions using SM100 three-input maximum."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32
from cutlass._mlir.dialects import llvm

@cute.jit
def max3(a: Float32, b: Float32, c: Float32):
    return Float32(llvm.inline_asm(Float32.mlir_type,
        [a.ir_value(),b.ir_value(),c.ir_value()],
        "max.f32 $0,$1,$2,$3;", "=f,f,f,f",
        has_side_effects=False,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))

@cute.jit
def max3_nan(a: Float32, b: Float32, c: Float32):
    return Float32(llvm.inline_asm(Float32.mlir_type,
        [a.ir_value(),b.ir_value(),c.ir_value()],
        "max.NaN.f32 $0,$1,$2,$3;", "=f,f,f,f",
        has_side_effects=False,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))

@cute.jit
def max2_nan(a: Float32, b: Float32):
    return Float32(llvm.inline_asm(Float32.mlir_type,
        [a.ir_value(),b.ir_value()],
        "max.NaN.f32 $0,$1,$2;", "=f,f,f",
        has_side_effects=False,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))

@cute.jit
def maximum(values, count: cutlass.Constexpr):
    items=[]
    for i in cutlass.range_constexpr(count):
        items.append(Float32(values[i]))
    for level in cutlass.range_constexpr(4):
        if cutlass.const_expr(len(items)>1):
            reduced=[]
            for i in cutlass.range_constexpr((len(items)+2)//3):
                if cutlass.const_expr(3*i+2<len(items)):
                    reduced.append(max3_nan(items[3*i],items[3*i+1],items[3*i+2]))
                elif cutlass.const_expr(3*i+1<len(items)):
                    reduced.append(max2_nan(items[3*i],items[3*i+1]))
                else:
                    reduced.append(items[3*i])
            items=reduced
    return items[0]
