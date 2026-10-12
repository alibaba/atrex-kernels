#pragma once

#include <cute/tensor.hpp>

namespace flash {

template<int PrefetchColumns, class Accumulator, class Probability, class Values,
         class SharedValues, class TiledMma, class TiledCopy, class ThreadCopy, class Rescale>
CUTLASS_DEVICE void decode_pv_rescale_overlap(
    Accumulator& accumulator, Probability& probability, Values& values,
    SharedValues const& shared_values, TiledMma tiled_mma,
    TiledCopy tiled_copy, ThreadCopy thread_copy, Rescale rescale) {
    using namespace cute;
    Tensor copy_view = thread_copy.retile_D(values);
    CUTE_STATIC_ASSERT_V(size<2>(probability) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size<2>(values) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size<1>(shared_values) == size<1>(copy_view));
    static_assert(PrefetchColumns >= 0);
    CUTE_STATIC_ASSERT_V(Int<PrefetchColumns>{} <= size<1>(copy_view));
    #pragma unroll
    for (int column = 0; column < PrefetchColumns; ++column) {
        cute::copy(tiled_copy, shared_values(_, column, _0{}), copy_view(_, column, _0{}));
    }
    rescale();
    #pragma hggc mmatiestrictly
    {
        #pragma unroll
        for (int column = PrefetchColumns; column < size<1>(copy_view); ++column) {
            cute::copy(tiled_copy, shared_values(_, column, _0{}), copy_view(_, column, _0{}));
        }
        cute::gemm(tiled_mma, probability(_, _, _0{}), values(_, _, _0{}), accumulator);
    }
}

}
