#if DECODE_Q1_ONLY
#pragma once

#include <cute/tensor.hpp>

#ifndef DECODE_O_SPILL_COLUMNS
#define DECODE_O_SPILL_COLUMNS 0
#endif

#ifndef DECODE_O_SPILL_PHASE
#define DECODE_O_SPILL_PHASE 0
#endif

namespace flash {

template<int SpillColumns, class Accumulator>
CUTLASS_DEVICE void decode_stash_o(Accumulator& accumulator,
                                   volatile float* spill, int thread_idx) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size<0>(accumulator) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(accumulator) == _1{});
    CUTE_STATIC_ASSERT_V(size<2>(accumulator) == Int<16>{});
    for_each(make_int_sequence<SpillColumns>{}, [&](auto column) {
        for_each(make_int_sequence<8>{}, [&](auto value) {
            spill[(column * Int<8>{} + value) * 128 + thread_idx] =
                accumulator(value, _0{}, column + Int<16 - SpillColumns>{});
            accumulator(value, _0{}, column + Int<16 - SpillColumns>{}) = 0.0f;
        });
    });
    __ppu_sched_bound();
}

template<int SpillColumns, bool First, class Accumulator, class Probability,
         class Values, class SharedValues, class TiledMma, class TiledCopy,
         class ThreadCopy, class Softmax, class Scale>
CUTLASS_DEVICE void decode_pv_o_spill(
    Accumulator& accumulator, Probability& probability, Values& values,
    SharedValues const& shared_values, TiledMma tiled_mma,
    TiledCopy tiled_copy, ThreadCopy thread_copy, Softmax& softmax,
    Scale const& scale, volatile float* spill, int thread_idx) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size<0>(accumulator) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(accumulator) == _1{});
    CUTE_STATIC_ASSERT_V(size<2>(accumulator) == Int<16>{});
    CUTE_STATIC_ASSERT_V(size<2>(probability) == _1{});
    CUTE_STATIC_ASSERT_V(size<2>(values) == _1{});
    Tensor copy_view = thread_copy.retile_D(values);
    CUTE_STATIC_ASSERT_V(size<1>(copy_view) == Int<16>{});
    for_each(make_int_sequence<16 - SpillColumns>{}, [&](auto column) {
        Tensor resident = make_tensor(accumulator(_, _, column).data(),
            make_layout(get<0>(accumulator.layout()), get<1>(accumulator.layout()), Layout<_1>{}));
        if constexpr (!First) { softmax.rescale_o(resident, scale); }
    });
    #pragma hggc mmatiestrictly
    {
        for_each(make_int_sequence<16 - SpillColumns>{}, [&](auto column) {
            cute::copy(tiled_copy, shared_values(_, column, _0{}), copy_view(_, column, _0{}));
            cute::gemm(tiled_mma, probability(_, _0{}, _0{}),
                       values(_, column, _0{}), accumulator(_, _0{}, column));
        });
    }
    for_each(make_int_sequence<SpillColumns>{}, [&](auto spilled_column) {
        __ppu_sched_bound();
        auto column = spilled_column + Int<16 - SpillColumns>{};
        Tensor partial = make_tensor<float>(make_layout(
            get<0>(accumulator.layout()), get<1>(accumulator.layout()), Layout<_1>{}));
        if constexpr (First) {
            clear(partial);
        } else {
            for_each(make_int_sequence<8>{}, [&](auto value) {
                partial(value) = spill[(spilled_column * Int<8>{} + value) * 128 + thread_idx];
            });
            softmax.rescale_o(partial, scale);
        }
        cute::copy(tiled_copy, shared_values(_, column, _0{}), copy_view(_, column, _0{}));
        cute::gemm(tiled_mma, probability(_, _0{}, _0{}),
                   values(_, column, _0{}), partial(_, _0{}, _0{}));
        for_each(make_int_sequence<8>{}, [&](auto value) {
            spill[(spilled_column * Int<8>{} + value) * 128 + thread_idx] = partial(value);
        });
        __ppu_sched_bound();
    });
}

template<int SpillColumns, class Accumulator>
CUTLASS_DEVICE void decode_restore_o(Accumulator& accumulator,
                                     volatile float const* spill, int thread_idx) {
    using namespace cute;
    for_each(make_int_sequence<SpillColumns>{}, [&](auto column) {
        for_each(make_int_sequence<8>{}, [&](auto value) {
            accumulator(value, _0{}, column + Int<16 - SpillColumns>{}) =
                spill[(column * Int<8>{} + value) * 128 + thread_idx];
        });
    });
}

}
#else
#endif
