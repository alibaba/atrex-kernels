#if DECODE_Q1_ONLY
#pragma once

#include <cute/tensor.hpp>

namespace flash {

template<int SpillColumns, bool RestoreQuery, bool First, class Query, class Accumulator>
CUTLASS_DEVICE void decode_exchange_qo(Query& query, Accumulator& accumulator,
                                       volatile uint32_t* exchange, int thread_idx) {
    using namespace cute;
    Tensor packed_query = recast<uint32_t>(query);
    CUTE_STATIC_ASSERT_V(size(packed_query) == Int<32>{});
    CUTE_STATIC_ASSERT_V(size<0>(accumulator) == Int<8>{});
    CUTE_STATIC_ASSERT_V(size<1>(accumulator) == _1{});
    CUTE_STATIC_ASSERT_V(size<2>(accumulator) == Int<16>{});
    static_assert(SpillColumns == 4 || SpillColumns == 6);
    static_assert(!RestoreQuery || !First);
    for_each(make_int_sequence<SpillColumns * 8>{}, [&](auto word) {
        auto value = word % Int<8>{};
        auto column = word / Int<8>{} + Int<16 - SpillColumns>{};
        int const offset = word * 128 + thread_idx;
        if constexpr (RestoreQuery) {
            uint32_t saved_query = 0;
            if constexpr (CUTE_STATIC_V(word) < 32) { saved_query = exchange[offset]; }
            exchange[offset] = __float_as_uint(accumulator(value, _0{}, column));
            accumulator(value, _0{}, column) = 0.0f;
            if constexpr (CUTE_STATIC_V(word) < 32) { packed_query(word) = saved_query; }
        } else {
            uint32_t saved_output = 0;
            if constexpr (!First) { saved_output = exchange[offset]; }
            if constexpr (CUTE_STATIC_V(word) < 32) {
                exchange[offset] = packed_query(word);
                packed_query(word) = 0;
            }
            if constexpr (!First) { accumulator(value, _0{}, column) = __uint_as_float(saved_output); }
        }
    });
    __ppu_sched_bound();
}

}
#else
#endif
