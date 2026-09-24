#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace moe {

template <typename T>
inline __device__ T ldg(T const* val) { return __ldg(val); }

template <>
inline __device__ __nv_bfloat162 ldg(__nv_bfloat162 const* val) { return __ldg(val); }

template <>
inline __device__ __nv_bfloat16 ldg(__nv_bfloat16 const* val) { return __ldg(val); }

template <typename T>
struct TypeConverter { using Type = half2; };

template <> struct TypeConverter<half2> { using Type = half; };
template <> struct TypeConverter<half> { using Type = half2; };
template <> struct TypeConverter<__nv_bfloat162> { using Type = __nv_bfloat16; };
template <> struct TypeConverter<__nv_bfloat16> { using Type = __nv_bfloat162; };

template <typename T>
inline __device__ T hadd2(T a, T b) { return __hadd2(a, b); }

template <>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    return __float22bfloat162_rn({fa.x + fb.x, fa.y + fb.y});
}

template <typename T>
inline __device__ T add(T a, T b) { return __hadd(a, b); }

template <>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

template <typename T>
inline __device__ T hmul2(T a, T b) { return __hmul2(a, b); }

template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    return __float22bfloat162_rn({fa.x * fb.x, fa.y * fb.y});
}

template <typename T>
inline __device__ T mul(T a, T b) { return __hmul(a, b); }

template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

template <typename T>
inline __device__ T fma(T a, T b, T c, T d) {
    return __hadd(__hmul(a, b), __hmul(c, d));
}

template <>
inline __device__ __nv_bfloat16 fma(__nv_bfloat16 a, __nv_bfloat16 b,
                                     __nv_bfloat16 c, __nv_bfloat16 d) {
    float fa = __bfloat162float(a), fb = __bfloat162float(b);
    float fc = __bfloat162float(c), fd = __bfloat162float(d);
    return __float2bfloat16(fa * fb + fc * fd);
}

// cuda_cast: type conversion specializations
template <typename To, typename Ti>
inline __device__ To cuda_cast(Ti val);

template <> inline __device__ float cuda_cast<float, float>(float val) { return val; }
template <> inline __device__ float cuda_cast<float, half>(half val) { return __half2float(val); }
template <> inline __device__ float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) { return __bfloat162float(val); }
template <> inline __device__ half cuda_cast<half, float>(float val) { return __float2half(val); }
template <> inline __device__ half cuda_cast<half, half>(half val) { return val; }
template <> inline __device__ __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) { return __float2bfloat16(val); }
template <> inline __device__ __nv_bfloat16 cuda_cast<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 val) { return val; }
template <> inline __device__ float2 cuda_cast<float2, half2>(half2 val) { return __half22float2(val); }
template <> inline __device__ float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) { return __bfloat1622float2(val); }
template <> inline __device__ float2 cuda_cast<float2, float2>(float2 val) { return val; }
template <> inline __device__ half2 cuda_cast<half2, float2>(float2 val) { return __float22half2_rn(val); }
template <> inline __device__ __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val) { return __float22bfloat162_rn(val); }
template <> inline __device__ __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat162>(__nv_bfloat162 val) { return val; }
template <> inline __device__ half2 cuda_cast<half2, half2>(half2 val) { return val; }

template <> inline __device__ int8_t cuda_cast<int8_t, float>(float val) {
    union { int8_t int8[4]; int32_t int32; };
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(int32) : "f"(val));
    return int8[0];
}
template <> inline __device__ int8_t cuda_cast<int8_t, half>(half val) {
    return cuda_cast<int8_t>(cuda_cast<float>(val));
}
template <> inline __device__ int8_t cuda_cast<int8_t, __nv_bfloat16>(__nv_bfloat16 val) {
    return cuda_cast<int8_t>(cuda_cast<float>(val));
}

template <> inline __device__ __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, float>(float val) {
    return __nv_fp8_e4m3(val);
}
template <> inline __device__ __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, half>(half val) {
    return __nv_fp8_e4m3(val);
}
template <> inline __device__ __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, __nv_bfloat16>(__nv_bfloat16 val) {
    return __nv_fp8_e4m3(val);
}
template <> inline __device__ float cuda_cast<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
    return float(val);
}

// cuda_abs
template <typename T>
inline __device__ T cuda_abs(T val);

template <> inline __device__ float cuda_abs<float>(float val) { return fabsf(val); }
template <> inline __device__ half cuda_abs<half>(half val) { return __habs(val); }
template <> inline __device__ __nv_bfloat16 cuda_abs<__nv_bfloat16>(__nv_bfloat16 val) {
    return __float2bfloat16(fabsf(__bfloat162float(val)));
}
template <> inline __device__ half2 cuda_abs<half2>(half2 val) { return __habs2(val); }
template <> inline __device__ __nv_bfloat162 cuda_abs<__nv_bfloat162>(__nv_bfloat162 val) {
    float2 f = __bfloat1622float2(val);
    return __float22bfloat162_rn({fabsf(f.x), fabsf(f.y)});
}
template <> inline __device__ float2 cuda_abs<float2>(float2 val) {
    return {fabsf(val.x), fabsf(val.y)};
}

// cuda_max
template <typename T>
inline __device__ T cuda_max(T a, T b);

template <> inline __device__ float cuda_max<float>(float a, float b) { return fmaxf(a, b); }
template <> inline __device__ half cuda_max<half>(half a, half b) { return __hmax(a, b); }
template <> inline __device__ __nv_bfloat16 cuda_max<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}
template <> inline __device__ half2 cuda_max<half2>(half2 a, half2 b) { return __hmax2(a, b); }
template <> inline __device__ __nv_bfloat162 cuda_max<__nv_bfloat162>(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
    return __float22bfloat162_rn({fmaxf(fa.x, fb.x), fmaxf(fa.y, fb.y)});
}

// cuda_min
template <typename T>
inline __device__ T cuda_min(T a, T b);

template <> inline __device__ float cuda_min<float>(float a, float b) { return fminf(a, b); }
template <> inline __device__ half cuda_min<half>(half a, half b) { return __hmin(a, b); }
template <> inline __device__ __nv_bfloat16 cuda_min<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
}

// cuda_clamp
template <typename T>
inline __device__ T cuda_clamp(T val, T minVal, T maxVal) {
    return cuda_min(cuda_max(val, minVal), maxVal);
}

} // namespace moe
