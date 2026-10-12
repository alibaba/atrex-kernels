#pragma once

#include <cute/tensor.hpp>

#ifndef DECODE_TRANSPOSE_EIGHT
#define DECODE_TRANSPOSE_EIGHT 0
#endif

#if DECODE_Q1_ONLY
#ifndef DECODE_TRANSPOSE_SLICE
#define DECODE_TRANSPOSE_SLICE 0
#endif

#else
#endif
namespace flash {
#if DECODE_Q1_ONLY

template<class Fragment>
CUTLASS_DEVICE void decode_permute_fragment(Fragment& fragment) {
    using namespace cute;
    Tensor words = recast<uint4>(fragment);
    #pragma unroll
    for (int word = 0; word < size(words); ++word) {
        uint4 bytes = words(word);
        uint32_t first = __byte_perm(bytes.x, bytes.y, 0x0040);
        uint32_t second = __byte_perm(bytes.x, bytes.y, 0x0051);
        uint32_t third = __byte_perm(bytes.x, bytes.y, 0x0062);
        uint32_t fourth = __byte_perm(bytes.x, bytes.y, 0x0073);
        first = __byte_perm(first, bytes.z, 0x0410);
        second = __byte_perm(second, bytes.z, 0x0510);
        third = __byte_perm(third, bytes.z, 0x0610);
        fourth = __byte_perm(fourth, bytes.z, 0x0710);
        bytes.x = __byte_perm(first, bytes.w, 0x4210);
        bytes.y = __byte_perm(second, bytes.w, 0x5210);
        bytes.z = __byte_perm(third, bytes.w, 0x6210);
        bytes.w = __byte_perm(fourth, bytes.w, 0x7210);
        words(word) = bytes;
    }
}
#else
#endif

template<class LoadCopy, class StoreCopy, class Source, class Destination>
CUTLASS_DEVICE void decode_transpose_disjoint(LoadCopy load_copy, StoreCopy store_copy,
                                             Source source, Destination destination) {
    using namespace cute;
#if DECODE_Q1_ONLY
#if DECODE_TRANSPOSE_SLICE == 1
    #pragma unroll
    for (int row = 0; row < size<1>(destination); ++row) {
        Tensor fragment = make_fragment_like(source(_, _, row));
        static_assert(decltype(size(fragment))::value % 16 == 0);
        cute::copy(load_copy, source(_, _, row), fragment);
        decode_permute_fragment(fragment);
        #pragma unroll
        for (int column = 0; column < size<2>(destination); ++column) {
            cute::copy(store_copy, fragment(_, column), destination(_, row, column));
        }
    }
#elif DECODE_TRANSPOSE_SLICE == 2
    #pragma unroll
    for (int row = 0; row < size<1>(destination); ++row) {
        #pragma unroll
        for (int column = 0; column < size<2>(destination); ++column) {
            Tensor fragment = make_fragment_like(source(_, column, row));
            static_assert(decltype(size(fragment))::value % 16 == 0);
            cute::copy(load_copy, source(_, column, row), fragment);
            decode_permute_fragment(fragment);
            cute::copy(store_copy, fragment, destination(_, row, column));
        }
    }
#else
    Tensor fragment = make_fragment_like(source);
    cute::copy(load_copy, source, fragment);
    Tensor words = recast<uint4>(fragment);
    #pragma unroll
    for (int word = 0; word < size(words); ++word) {
        uint4 bytes = words(word);
#if DECODE_TRANSPOSE_EIGHT
        uint32_t low_xy = __byte_perm(bytes.x, bytes.y, 0x5410);
        uint32_t high_xy = __byte_perm(bytes.x, bytes.y, 0x7632);
        uint32_t low_zw = __byte_perm(bytes.z, bytes.w, 0x5410);
        uint32_t high_zw = __byte_perm(bytes.z, bytes.w, 0x7632);
        bytes.x = __byte_perm(low_xy, low_zw, 0x6420);
        bytes.y = __byte_perm(low_xy, low_zw, 0x7531);
        bytes.z = __byte_perm(high_xy, high_zw, 0x6420);
        bytes.w = __byte_perm(high_xy, high_zw, 0x7531);
#else
        uint32_t first = __byte_perm(bytes.x, bytes.y, 0x0040);
        uint32_t second = __byte_perm(bytes.x, bytes.y, 0x0051);
        uint32_t third = __byte_perm(bytes.x, bytes.y, 0x0062);
        uint32_t fourth = __byte_perm(bytes.x, bytes.y, 0x0073);
        first = __byte_perm(first, bytes.z, 0x0410);
        second = __byte_perm(second, bytes.z, 0x0510);
        third = __byte_perm(third, bytes.z, 0x0610);
        fourth = __byte_perm(fourth, bytes.z, 0x0710);
        bytes.x = __byte_perm(first, bytes.w, 0x4210);
        bytes.y = __byte_perm(second, bytes.w, 0x5210);
        bytes.z = __byte_perm(third, bytes.w, 0x6210);
        bytes.w = __byte_perm(fourth, bytes.w, 0x7210);
#endif
        words(word) = bytes;
    }
    #pragma unroll
    for (int column = 0; column < size<2>(destination); ++column) {
        #pragma unroll
        for (int row = 0; row < size<1>(destination); ++row) {
            cute::copy(store_copy, fragment(_, column, row), destination(_, row, column));
        }
    }
#endif
#else
    Tensor fragment = make_fragment_like(source);
    cute::copy(load_copy, source, fragment);
    Tensor words = recast<uint4>(fragment);
    #pragma unroll
    for (int word = 0; word < size(words); ++word) {
        uint4 bytes = words(word);
#if DECODE_TRANSPOSE_EIGHT
        uint32_t low_xy = __byte_perm(bytes.x, bytes.y, 0x5410);
        uint32_t high_xy = __byte_perm(bytes.x, bytes.y, 0x7632);
        uint32_t low_zw = __byte_perm(bytes.z, bytes.w, 0x5410);
        uint32_t high_zw = __byte_perm(bytes.z, bytes.w, 0x7632);
        bytes.x = __byte_perm(low_xy, low_zw, 0x6420);
        bytes.y = __byte_perm(low_xy, low_zw, 0x7531);
        bytes.z = __byte_perm(high_xy, high_zw, 0x6420);
        bytes.w = __byte_perm(high_xy, high_zw, 0x7531);
#else
        uint32_t first = __byte_perm(bytes.x, bytes.y, 0x0040);
        uint32_t second = __byte_perm(bytes.x, bytes.y, 0x0051);
        uint32_t third = __byte_perm(bytes.x, bytes.y, 0x0062);
        uint32_t fourth = __byte_perm(bytes.x, bytes.y, 0x0073);
        first = __byte_perm(first, bytes.z, 0x0410);
        second = __byte_perm(second, bytes.z, 0x0510);
        third = __byte_perm(third, bytes.z, 0x0610);
        fourth = __byte_perm(fourth, bytes.z, 0x0710);
        bytes.x = __byte_perm(first, bytes.w, 0x4210);
        bytes.y = __byte_perm(second, bytes.w, 0x5210);
        bytes.z = __byte_perm(third, bytes.w, 0x6210);
        bytes.w = __byte_perm(fourth, bytes.w, 0x7210);
#endif
        words(word) = bytes;
    }
    #pragma unroll
    for (int column = 0; column < size<2>(destination); ++column) {
        #pragma unroll
        for (int row = 0; row < size<1>(destination); ++row) {
            cute::copy(store_copy, fragment(_, column, row), destination(_, row, column));
        }
    }
#endif
}

}
