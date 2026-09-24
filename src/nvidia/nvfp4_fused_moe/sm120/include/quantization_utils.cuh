#pragma once

#include <optional>
#include <type_traits>
#include "cuda_type_utils.cuh"
#include "moe_common.cuh"

using namespace moe;

// ============================================================================
// QuantizationSFLayout enum (from flashinfer/fp4_layout.cuh)
// ============================================================================

enum class QuantizationSFLayout {
    SWIZZLED_128x4,
    SWIZZLED_8x4,
    LINEAR
};

// ============================================================================
// Type converters for packed vectors
// ============================================================================

namespace quant {

template <class Type>
struct TypeConverter { using PackedType = void; };

template <> struct TypeConverter<half> { using PackedType = half2; };
template <> struct TypeConverter<__nv_bfloat16> { using PackedType = __nv_bfloat162; };
template <> struct TypeConverter<__nv_fp8_e4m3> { using PackedType = __nv_fp8x2_e4m3; };

template <class Type, int NUM_ELTS = 8>
struct PackedVec {
    typename TypeConverter<Type>::PackedType elts[NUM_ELTS / 2];
};

template <int NUM_ELTS>
struct PackedVec<__nv_fp8_e4m3, NUM_ELTS> {
    __nv_fp8x2_e4m3 elts[NUM_ELTS / 2];
};

} // namespace quant

using quant::PackedVec;

// ============================================================================
// FP4/MXFP8 Conversion Functions
// ============================================================================

inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
          "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
    return val;
#else
    return 0;
#endif
}

inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
    return val;
#else
    return 0;
#endif
}

inline __device__ uint64_t fp32_vec_to_e2m1(float2 (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint64_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n .reg .b8 byte1;\n .reg .b8 byte2;\n .reg .b8 byte3;\n"
        ".reg .b8 byte4;\n .reg .b8 byte5;\n .reg .b8 byte6;\n .reg .b8 byte7;\n"
        ".reg .b32 val0;\n .reg .b32 val1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
        "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
        "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
        "mov.b64 %0, {val0, val1};\n"
        "}"
        : "=l"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
          "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
          "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
    return val;
#else
    return 0;
#endif
}

inline __device__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4]) {
    union {
        uint64_t val;
        __nv_fp8x2_e4m3 elts[4];
    } u;
    u.elts[0] = __nv_fp8x2_e4m3(array[0]);
    u.elts[1] = __nv_fp8x2_e4m3(array[1]);
    u.elts[2] = __nv_fp8x2_e4m3(array[2]);
    u.elts[3] = __nv_fp8x2_e4m3(array[3]);
    return u.val;
}

inline __device__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float exp2f_rcp(uint8_t exp) {
    constexpr uint32_t FP32_EXPONENT_BIAS = 127;
    return (exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

// ============================================================================
// cvt_warp_fp16_to_fp4
// ============================================================================

template <class Type, int SF_VEC_SIZE, int CVT_ELTS_PER_THREAD_T, bool UE8M0_SF>
__device__ std::conditional_t<CVT_ELTS_PER_THREAD_T == 16, uint64_t, uint32_t>
cvt_warp_fp16_to_fp4(PackedVec<Type, CVT_ELTS_PER_THREAD_T>& vec, float SFScaleVal,
                     uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(CVT_ELTS_PER_THREAD_T == 8 || CVT_ELTS_PER_THREAD_T == 16);
    using ReturnType = std::conditional_t<CVT_ELTS_PER_THREAD_T == 16, uint64_t, uint32_t>;

    auto localMax = cuda_abs(vec.elts[0]);
    #pragma unroll
    for (int i = 1; i < CVT_ELTS_PER_THREAD_T / 2; i++) {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD_T;
    if constexpr (CVT_NUM_THREADS_PER_SF >= 2) {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    }
    if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
    }
    float vecMax = float(cuda_max(localMax.x, localMax.y));

    uint8_t fp8SFVal;
    float outputScale;
    if constexpr (UE8M0_SF) {
        __nv_fp8_e8m0 tmp;
        vecMax *= reciprocal_approximate_ftz(6.0f);
        tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);
        fp8SFVal = tmp.__x;
        outputScale = vecMax != 0 ? exp2f_rcp(fp8SFVal) : 0.0f;
    } else {
        auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        fp8SFVal = tmp.__x;
        SFValue = static_cast<float>(tmp);
        outputScale = vecMax != 0
                      ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
    }

    if (SFout) {
        *SFout = fp8SFVal;
    }

    float2 fp2Vals[CVT_ELTS_PER_THREAD_T / 2];
    #pragma unroll
    for (int i = 0; i < CVT_ELTS_PER_THREAD_T / 2; i++) {
        if constexpr (std::is_same_v<Type, half>) {
            fp2Vals[i] = __half22float2(vec.elts[i]);
        } else {
            fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        }
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    ReturnType e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
    return e2m1Vec;
#else
    return 0;
#endif
}

// ============================================================================
// Scale factor offset calculation
// ============================================================================

inline __device__ __host__ int64_t get_sf_out_offset_128x4(
    std::optional<int> batchIdx, int mIdx, int kIdx,
    std::optional<int> numRows, int numColVecs) {
    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;
    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4 * innerKStride;
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * innerMStride;
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * outerMStride;
    int32_t numKTiles = (numColVecs + 4 - 1) / 4;
    int32_t mTileIdx = mIdx / (32 * 4);
    int64_t mTileStride = numKTiles * kTileStride;
    int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
    int64_t bTileStride = numMTiles * mTileStride;
    int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                       kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;
    return SFOffset;
}

inline __device__ __host__ int64_t get_sf_out_offset_8x4(
    std::optional<int> batchIdx, int mIdx, int kIdx,
    std::optional<int> numRows, int numCols) {
    const int32_t mTile = 8;
    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;
    int32_t innerMIdx = (mIdx % mTile);
    int64_t mStride = 4 * innerKStride;
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = mTile * mStride;
    int32_t numKTiles = (numCols + 4 - 1) / 4;
    int32_t mTileIdx = mIdx / mTile;
    int64_t mTileStride = numKTiles * kTileStride;
    int32_t numMTiles = (numRows.value_or(0) + 8 - 1) / 8;
    int64_t bTileStride = numMTiles * mTileStride;
    int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                       kTileIdx * kTileStride + innerMIdx * mStride + innerKIdx * innerKStride;
    return SFOffset;
}

template <class SFType, int CVT_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_get_sf_out_offset(
    std::optional<int> batchIdx, int rowIdx, int colVecIdx,
    std::optional<int> numRows, int numColVecs,
    SFType* SFout, QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2 ||
                  CVT_NUM_THREADS_PER_SF == 4);
    if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0) {
        if (layout == QuantizationSFLayout::SWIZZLED_128x4 ||
            layout == QuantizationSFLayout::SWIZZLED_8x4) {
            int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
            int32_t mIdx = rowIdx;
            auto SFOffset = layout == QuantizationSFLayout::SWIZZLED_128x4
                            ? get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs)
                            : get_sf_out_offset_8x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
            return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
        } else if (layout == QuantizationSFLayout::LINEAR) {
            int32_t KTileIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
            int32_t numKTiles = numColVecs;
            int64_t mTileStride = numKTiles;
            int64_t BTileStride = numRows.value_or(0) * mTileStride;
            int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
            return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
        }
    }
#endif
    return nullptr;
}

// ============================================================================
// silu and silu_and_mul
// ============================================================================

__device__ __forceinline__ float silu_f(const float& val) {
    return val / (1.0f + __expf(-val));
}

template <class Type, int CVT_ELTS_PER_THREAD_T>
inline __device__ void silu_and_mul(PackedVec<Type, CVT_ELTS_PER_THREAD_T>& x_vec,
                                    const PackedVec<Type, CVT_ELTS_PER_THREAD_T>& y_vec) {
    float2 x[CVT_ELTS_PER_THREAD_T / 2];
    float2 y[CVT_ELTS_PER_THREAD_T / 2];
    #pragma unroll
    for (int i = 0; i < CVT_ELTS_PER_THREAD_T / 2; i++) {
        if constexpr (std::is_same_v<Type, half>) {
            x[i] = __half22float2(x_vec.elts[i]);
            y[i] = __half22float2(y_vec.elts[i]);
            x[i].x = silu_f(x[i].x) * y[i].x;
            x[i].y = silu_f(x[i].y) * y[i].y;
            x_vec.elts[i] = __float22half2_rn(x[i]);
        } else {
            x[i] = __bfloat1622float2(x_vec.elts[i]);
            y[i] = __bfloat1622float2(y_vec.elts[i]);
            x[i].x = silu_f(x[i].x) * y[i].x;
            x[i].y = silu_f(x[i].y) * y[i].y;
            x_vec.elts[i] = __float22bfloat162_rn(x[i]);
        }
    }
}

// ============================================================================
// Helper functions used by expandInputRows and doActivation
// ============================================================================

__host__ __device__ constexpr int64_t getOffsetWeightSF(
    int64_t expert_id, int64_t gemm_n, int64_t gemm_k,
    TmaConst::FpXBlockScalingType scaling_type) {
    int64_t min_n = TmaConst::MinNDimAlignmentNVFP4;
    int64_t min_k = TmaConst::MinKDimAlignmentNVFP4;
    int64_t block_size = TmaConst::NVFP4BlockScaleVectorSize;
    int64_t padded_n = TmaConst::alignToSfDim(gemm_n, min_n);
    int64_t padded_k = TmaConst::alignToSfDim(gemm_k, min_k);
    return expert_id * padded_n * padded_k / block_size;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(
    int64_t expert_id, int64_t token_offset, int64_t gemm_k,
    TmaConst::FpXBlockScalingType scaling_type) {
    int64_t min_n = TmaConst::MinNDimAlignmentNVFP4;
    int64_t min_k = TmaConst::MinKDimAlignmentNVFP4;
    int64_t block_size = TmaConst::NVFP4BlockScaleVectorSize;
    int64_t padded_sf_start = TmaConst::alignToSfDim(
        token_offset + expert_id * (min_n - 1), min_n);
    int64_t padded_k = TmaConst::alignToSfDim(gemm_k, min_k);
    return padded_sf_start * padded_k / block_size;
}

template <class GemmOutputType, class QuantizedType, class ComputeElem, int VecSize>
__device__ auto quantizePackedFPXValue(
    ComputeElem& post_act_val, float global_scale_val, int64_t num_tokens_before_expert,
    int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
    uint8_t* act_sf_flat, TmaConst::FpXBlockScalingType scaling_type) {
    static constexpr int NumThreadsPerSF = VecSize / CVT_ELTS_PER_THREAD;
    static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> ||
                  std::is_same_v<GemmOutputType, half>);
    PackedVec<GemmOutputType> packed_vec{};
    for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
        packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
        packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
    }

    auto act_sf_expert = act_sf_flat + getOffsetActivationSF(
        expert_id, num_tokens_before_expert, num_cols, scaling_type);

    auto sf_out = cvt_quant_get_sf_out_offset<uint8_t, NumThreadsPerSF>(
        std::nullopt, token_id - num_tokens_before_expert, elem_idx,
        std::nullopt, num_cols / VecSize, act_sf_expert,
        QuantizationSFLayout::SWIZZLED_128x4);

    auto func = (scaling_type == TmaConst::FpXBlockScalingType::NVFP4)
        ? &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, false>
        : &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, true>;

    return func(packed_vec, global_scale_val, sf_out);
}

template <int VecSize, int ElementsPerThread>
__device__ void writeSF(
    int64_t num_tokens_before_expert, int64_t expert_id,
    int64_t source_token_id, int64_t token_id, int64_t elem_idx,
    int64_t num_cols, uint8_t* act_sf_flat,
    uint8_t const* input_sf, bool const swizzled_input_sf = true) {
    static constexpr int NumThreadsPerSF = VecSize / ElementsPerThread;

    auto act_sf_expert = act_sf_flat + getOffsetActivationSF(
        expert_id, num_tokens_before_expert, num_cols,
        (VecSize == TmaConst::NVFP4BlockScaleVectorSize)
            ? TmaConst::FpXBlockScalingType::NVFP4
            : TmaConst::FpXBlockScalingType::MXFPX);

    auto sf_out = cvt_quant_get_sf_out_offset<uint8_t, NumThreadsPerSF>(
        std::nullopt, token_id - num_tokens_before_expert, elem_idx,
        std::nullopt, num_cols / VecSize, act_sf_expert,
        QuantizationSFLayout::SWIZZLED_128x4);
    if (sf_out) {
        if (input_sf) {
            if (swizzled_input_sf) {
                auto sf_in = cvt_quant_get_sf_out_offset<uint8_t, NumThreadsPerSF>(
                    std::nullopt, source_token_id, elem_idx, std::nullopt,
                    num_cols / VecSize, const_cast<uint8_t*>(input_sf),
                    QuantizationSFLayout::SWIZZLED_128x4);
                *sf_out = *sf_in;
            } else {
                auto sf_in = cvt_quant_get_sf_out_offset<uint8_t, NumThreadsPerSF>(
                    std::nullopt, source_token_id, elem_idx, std::nullopt,
                    num_cols / VecSize, const_cast<uint8_t*>(input_sf),
                    QuantizationSFLayout::LINEAR);
                *sf_out = *sf_in;
            }
        } else {
            *sf_out = 0x00;
        }
    }
}
