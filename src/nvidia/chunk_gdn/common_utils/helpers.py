"""CuTeDSL helpers shared by NVIDIA Chunk-GDN kernels."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm


def round_down(a: int, b: int) -> int:
    return (a // b) * b


@cute.jit
def select_tensor_10(t: cute.Tensor) -> cute.Tensor:
    """select_tensor<1,0>: swap first two modes of a 2-D tensor."""
    return cute.make_tensor(
        t.iterator.align(t.iterator.max_alignment),
        cute.make_layout(
            (t.layout.shape[1], t.layout.shape[0]) + t.layout.shape[2:],
            stride=(t.layout.stride[1], t.layout.stride[0]) + t.layout.stride[2:],
        ),
    )


@cute.jit
def smid():
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def tensormap_replace_global_dim_1(
    tensormap_ptr: cute.Pointer,
    new_val: cutlass.Int32,
):
    ptr_i64 = tensormap_ptr.toint().ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, new_val.ir_value()],
        "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], 1, $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class SM80:
    @staticmethod
    @cute.jit
    def convert_c_layout_to_a_layout(c_layout, tiled_mma):
        c_frag_atom_size = cute.size(c_layout, mode=[0])
        a_frag_atom_size = cute.size(tiled_mma.tv_layout_A, mode=[1])
        ratio = a_frag_atom_size // c_frag_atom_size
        if cutlass.const_expr(ratio == 1):
            return c_layout

        divided = cute.logical_divide(c_layout, (None, None, ratio))
        frag_layout = cute.flatten(
            cute.make_layout(
                (divided.shape[0], divided.shape[2][0]),
                stride=(divided.stride[0], divided.stride[2][0]),
            )
        )
        return cute.make_layout(
            (frag_layout.shape, divided.shape[1], divided.shape[2][1]),
            stride=(
                frag_layout.stride,
                divided.stride[1],
                divided.stride[2][1],
            ),
        )

    @staticmethod
    @cute.jit
    def make_acc_into_op(acc: cute.Tensor, tiled_mma, dtype) -> cute.Tensor:
        operand = cute.make_fragment_like(
            SM80.convert_c_layout_to_a_layout(acc.layout, tiled_mma),
            dtype,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand
