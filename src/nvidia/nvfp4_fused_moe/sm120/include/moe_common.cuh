#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

extern bool g_enable_pdl;

// ============================================================================
// Replacement for cutlass::Array<T, N>
// ============================================================================

template <typename T, int N>
struct alignas(sizeof(T) * N <= 16 ? (sizeof(T) * N) : 16) Array {
    T data[N];

    __host__ __device__ T& operator[](int i) { return data[i]; }
    __host__ __device__ T const& operator[](int i) const { return data[i]; }

    __host__ __device__ void fill(T val) {
        #pragma unroll
        for (int i = 0; i < N; i++) data[i] = val;
    }

    __host__ __device__ int size() const { return N; }
};

// ============================================================================
// Replacement for cutlass::sizeof_bits<T>
// ============================================================================

template <typename T>
struct sizeof_bits {
    static constexpr int value = sizeof(T) * 8;
};

template <>
struct sizeof_bits<__nv_fp4_e2m1> {
    static constexpr int value = 4;
};

// ============================================================================
// Replacement for cutlass::NumericArrayConverter (via arrayConvert)
// ============================================================================

template <typename To, typename From, int N>
__device__ Array<To, N> arrayConvert(Array<From, N> const& src) {
    Array<To, N> dst;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = static_cast<To>(src[i]);
    }
    return dst;
}

// bf16 -> float specialization
template <int N>
__device__ Array<float, N> arrayConvert(Array<__nv_bfloat16, N> const& src) {
    Array<float, N> dst;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = __bfloat162float(src[i]);
    }
    return dst;
}

// float -> bf16 specialization
template <int N>
__device__ Array<__nv_bfloat16, N> arrayConvertToBf16(Array<float, N> const& src) {
    Array<__nv_bfloat16, N> dst;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = __float2bfloat16(src[i]);
    }
    return dst;
}

// half -> float
template <int N>
__device__ Array<float, N> arrayConvert(Array<half, N> const& src) {
    Array<float, N> dst;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = __half2float(src[i]);
    }
    return dst;
}

// float -> half
template <int N>
__device__ Array<half, N> arrayConvertToHalf(Array<float, N> const& src) {
    Array<half, N> dst;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

// ============================================================================
// TmaWarpSpecializedGroupedGemmInput constants (extracted from CUTLASS)
// ============================================================================

namespace TmaConst {
    using ElementSF = uint8_t;
    static constexpr int NVFP4BlockScaleVectorSize = 16;
    static constexpr int MXFPXBlockScaleVectorSize = 32;
    static constexpr int MinNDimAlignmentNVFP4 = 128;
    static constexpr int MinNDimAlignmentMXFPX = 128;
    static constexpr int MinKDimAlignmentNVFP4 = 64;
    static constexpr int MinKDimAlignmentMXFPX = 128;

    enum class FpXBlockScalingType { MXFPX, NVFP4, NONE };

    __host__ __device__ inline constexpr int alignToSfDim(int dim, int alignment) {
        return ((dim + alignment - 1) / alignment) * alignment;
    }
}

// ============================================================================
// Activation types and params
// ============================================================================

enum class ActivationType : int {
    Gelu = 0, Relu, Silu, Swiglu, Geglu, SwigluBias, Relu2, Identity, InvalidType
};

struct ActivationParams {
    ActivationType activation_type;
    float const* swiglu_alpha = nullptr;
    float const* swiglu_beta = nullptr;
    float const* swiglu_limit = nullptr;
};

// ============================================================================
// ScaleMode enum
// ============================================================================

enum ScaleMode : int {
    NO_SCALE = 0,
    DEFAULT = 1,
};

// ============================================================================
// Activation function implementations (replace CUTLASS epilogue::thread)
// ============================================================================

namespace activations {

template <typename T>
struct SiLu {
    __device__ T operator()(T x) const {
        float xf = static_cast<float>(x);
        float result = xf / (1.0f + __expf(-xf));
        return static_cast<T>(result);
    }
};

template <typename T>
struct GELU {
    __device__ T operator()(T x) const {
        float xf = static_cast<float>(x);
        float result = 0.5f * xf * (1.0f + tanhf(0.7978845608f * (xf + 0.044715f * xf * xf * xf)));
        return static_cast<T>(result);
    }
};

template <typename T>
struct ReLu {
    __device__ T operator()(T x) const {
        float xf = static_cast<float>(x);
        return static_cast<T>(fmaxf(0.f, xf));
    }
};

template <typename T>
struct Relu2 {
    __device__ T operator()(T x) const {
        float xf = static_cast<float>(x);
        float r = fmaxf(0.f, xf);
        return static_cast<T>(r * r);
    }
};

template <typename T>
struct Sigmoid {
    __device__ T operator()(T x) const {
        float xf = static_cast<float>(x);
        return static_cast<T>(1.0f / (1.0f + __expf(-xf)));
    }
};

template <typename T>
struct Identity {
    __device__ T operator()(T x) const { return x; }
};

} // namespace activations

// Activation adaptors
template <template<class> class ActFn>
struct IdentityAdaptor {
    template <typename T>
    __device__ T operator()(T x) const {
        return ActFn<T>{}(x);
    }
};

template <template<class> class ActFn>
struct GLUAdaptor {
    template <typename T>
    __device__ T operator()(T gate, T up) const {
        return static_cast<T>(static_cast<float>(ActFn<T>{}(gate)) * static_cast<float>(up));
    }
};

// ============================================================================
// Utility functions
// ============================================================================

__host__ __device__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int64_t ceilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

inline int getMultiProcessorCount() {
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    return sm_count;
}

template <typename T>
__device__ int findTotalEltsLessThanTarget(T const* sorted_indices, int const arr_length, T const target) {
    int low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (sorted_indices[mid] >= target) {
            high = mid - 1;
        } else {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

template <typename T>
__device__ __host__ inline T* safe_inc_ptr(T* ptr, int64_t inc) {
    constexpr int bits = sizeof_bits<T>::value;
    if constexpr (bits >= 8) {
        return ptr + inc;
    } else {
        static_assert(bits == 4);
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + inc / 2);
    }
}

// ============================================================================
// Thread block sizes
// ============================================================================

static constexpr int EXPAND_THREADS_PER_BLOCK = 256;
static constexpr int ACTIVATION_THREADS_PER_BLOCK = 256;
static constexpr int FINALIZE_THREADS_PER_BLOCK = 256;

// CVT_ELTS_PER_THREAD from quantization.cuh
static constexpr int CVT_ELTS_PER_THREAD = 8;

// ============================================================================
// PDL helpers (optional optimization, guarded by arch)
// ============================================================================

__device__ inline void pdl_wait() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
}

__device__ inline void pdl_launch_dependents() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}
