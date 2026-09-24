// ===== Legacy v4 primary shape (E=256, hidden=2048, inter=512) =====
// GEMM2: K=inter=512, N=hidden=2048.
#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n128
#define TASK30_BLOCK_N_VALUE 128
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n64
#define TASK30_BLOCK_N_VALUE 64
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

// ===== Qwen3-MoE primary shape (E=128, hidden=2048, inter=768) =====
// GEMM2: K=inter=768, N=hidden=2048 (same N as legacy). Only PRIMARY_K differs.
#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n128_qwen3
#define TASK30_BLOCK_N_VALUE 128
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 768
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n64_qwen3
#define TASK30_BLOCK_N_VALUE 64
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 768
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

// ===== qwen3_5_flash TP2 shape (E=256, hidden=2048, inter=256) =====
// GEMM2: K=inter=256, N=hidden=2048.
#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n128_tp2
#define TASK30_BLOCK_N_VALUE 128
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 256
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n64_tp2
#define TASK30_BLOCK_N_VALUE 64
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 256
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

// ===== qwen3_6_flash TP2 shape (E=128, hidden=2048, inter=384) =====
// GEMM2: K=inter=384, N=hidden=2048.
#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n128_qwen3_tp2
#define TASK30_BLOCK_N_VALUE 128
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 384
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#define TASK30_IMPL_NAMESPACE atrex_task30_gemm2_small_m_v1_n64_qwen3_tp2
#define TASK30_BLOCK_N_VALUE 64
#define TASK30_PRIMARY_N_VALUE 2048
#define TASK30_PRIMARY_K_VALUE 384
#include "task30_gemm2_small_m_sm120.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

namespace atrex_task30_gemm2_small_m {

namespace v128 = atrex_task30_gemm2_small_m_v1_n128;
namespace v64 = atrex_task30_gemm2_small_m_v1_n64;
namespace v128_qwen3 = atrex_task30_gemm2_small_m_v1_n128_qwen3;
namespace v64_qwen3 = atrex_task30_gemm2_small_m_v1_n64_qwen3;
namespace v128_tp2 = atrex_task30_gemm2_small_m_v1_n128_tp2;
namespace v64_tp2 = atrex_task30_gemm2_small_m_v1_n64_tp2;
namespace v128_qwen3_tp2 = atrex_task30_gemm2_small_m_v1_n128_qwen3_tp2;
namespace v64_qwen3_tp2 = atrex_task30_gemm2_small_m_v1_n64_qwen3_tp2;

// Shape detection: which compiled namespace matches the runtime (N, K)?
enum class ShapeKind { Legacy, LegacyTp2, Qwen3, Qwen3Tp2, Unsupported };
static inline ShapeKind classify_shape(int N, int K) {
    if (K == v128::PRIMARY_K && N == v128::PRIMARY_N) return ShapeKind::Legacy;
    if (K == v128_tp2::PRIMARY_K && N == v128_tp2::PRIMARY_N)
        return ShapeKind::LegacyTp2;
    if (K == v128_qwen3::PRIMARY_K && N == v128_qwen3::PRIMARY_N)
        return ShapeKind::Qwen3;
    if (K == v128_qwen3_tp2::PRIMARY_K && N == v128_qwen3_tp2::PRIMARY_N)
        return ShapeKind::Qwen3Tp2;
    return ShapeKind::Unsupported;
}
static inline const char* shape_error_str() {
    return "task30_small_m supports only the v4 primary shape (K=512 N=2048)"
           ", qwen3_5 TP2 shape (K=256 N=2048), Qwen3-MoE shape"
           " (K=768 N=2048), or qwen3_6 TP2 shape (K=384 N=2048)";
}

template <typename Kernel>
static bool set_dynamic_smem_attr(Kernel kernel, int smem, const char* label) {
    if (smem <= 48 * 1024) {
        return true;
    }
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[task30_small_m] cudaFuncSetAttribute failed for %s: %s\n",
                     label, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return true;
}

static bool ensure_kernel_attrs_prepared() {
    bool ok = true;
    ok = set_dynamic_smem_attr(
             v128::atrex_gemm2_small_m_fused_finalize_kernel, v128::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n128") && ok;
    ok = set_dynamic_smem_attr(
             v64::atrex_gemm2_small_m_fused_finalize_kernel, v64::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n64") && ok;
    ok = set_dynamic_smem_attr(
             v128::atrex_gemm2_m1_row_fused_finalize_kernel, v128::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n128") && ok;
    ok = set_dynamic_smem_attr(
             v64::atrex_gemm2_m1_row_fused_finalize_kernel, v64::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n64") && ok;
    // Qwen3-MoE shape (PRIMARY_K=768) instances.
    ok = set_dynamic_smem_attr(
             v128_qwen3::atrex_gemm2_small_m_fused_finalize_kernel,
             v128_qwen3::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n128_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3::atrex_gemm2_small_m_fused_finalize_kernel,
             v64_qwen3::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n64_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v128_qwen3::atrex_gemm2_m1_row_fused_finalize_kernel,
             v128_qwen3::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n128_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3::atrex_gemm2_m1_row_fused_finalize_kernel,
             v64_qwen3::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n64_qwen3") && ok;
    // TP2 shape instances change PRIMARY_K for scale-factor strides.
    ok = set_dynamic_smem_attr(
             v128_tp2::atrex_gemm2_small_m_fused_finalize_kernel,
             v128_tp2::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n128_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_tp2::atrex_gemm2_small_m_fused_finalize_kernel,
             v64_tp2::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n64_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v128_tp2::atrex_gemm2_m1_row_fused_finalize_kernel,
             v128_tp2::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n128_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_tp2::atrex_gemm2_m1_row_fused_finalize_kernel,
             v64_tp2::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n64_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v128_qwen3_tp2::atrex_gemm2_small_m_fused_finalize_kernel,
             v128_qwen3_tp2::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n128_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3_tp2::atrex_gemm2_small_m_fused_finalize_kernel,
             v64_qwen3_tp2::SMEM_BYTES,
             "gemm2_small_m_fused_finalize_kernel_n64_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v128_qwen3_tp2::atrex_gemm2_m1_row_fused_finalize_kernel,
             v128_qwen3_tp2::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n128_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3_tp2::atrex_gemm2_m1_row_fused_finalize_kernel,
             v64_qwen3_tp2::SMEM_BYTES,
             "gemm2_m1_row_fused_finalize_kernel_n64_qwen3_tp2") && ok;
    return ok;
}

int64_t workspace_bytes(int num_experts, int N, int K,
                        int64_t expanded_num_tokens) {
    (void)num_experts;
    (void)N;
    (void)K;
    (void)expanded_num_tokens;
    return 256;
}

bool encode_tma_a_2d(CUtensorMap* desc,
                     void const* a_fp4,
                     int64_t expanded_num_tokens,
                     int K_half,
                     uint32_t row_bytes,
                     uint32_t block_m,
                     const char* tag) {
    uint64_t globalDim[2];
    uint64_t globalStride[1];
    uint32_t boxDim[2];
    uint32_t elemStride[2];
    globalDim[0] = static_cast<uint64_t>(K_half);
    globalDim[1] = static_cast<uint64_t>(expanded_num_tokens);
    globalStride[0] = static_cast<uint64_t>(K_half);
    boxDim[0] = row_bytes;
    boxDim[1] = block_m;
    elemStride[0] = 1;
    elemStride[1] = 1;

    CUresult res = cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<void*>(a_fp4), globalDim, globalStride, boxDim,
        elemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr,
                     "[task30_small_m] %s failed: %d "
                     "(A=%p expanded=%lld K_half=%d row_bytes=%u block_m=%u)\n",
                     tag, static_cast<int>(res), a_fp4,
                     static_cast<long long>(expanded_num_tokens), K_half,
                     row_bytes, block_m);
        return false;
    }
    return true;
}

bool encode_tma_b_3d(CUtensorMap* desc,
                     void const* b_fp4,
                     int num_experts,
                     int N,
                     int K_half,
                     uint32_t row_bytes,
                     uint32_t block_n,
                     const char* tag) {
    uint64_t globalDim[3];
    uint64_t globalStride[2];
    uint32_t boxDim[3];
    uint32_t elemStride[3];
    globalDim[0] = static_cast<uint64_t>(K_half);
    globalDim[1] = static_cast<uint64_t>(N);
    globalDim[2] = static_cast<uint64_t>(num_experts);
    globalStride[0] = static_cast<uint64_t>(K_half);
    globalStride[1] = static_cast<uint64_t>(N) *
                      static_cast<uint64_t>(K_half);
    boxDim[0] = row_bytes;
    boxDim[1] = block_n;
    boxDim[2] = 1;
    elemStride[0] = 1;
    elemStride[1] = 1;
    elemStride[2] = 1;

    CUresult res = cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        const_cast<void*>(b_fp4), globalDim, globalStride, boxDim,
        elemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr,
                     "[task30_small_m] %s failed: %d "
                     "(B=%p E=%d N=%d K_half=%d row_bytes=%u block_n=%u)\n",
                     tag, static_cast<int>(res), b_fp4, num_experts, N,
                     K_half, row_bytes, block_n);
        return false;
    }
    return true;
}

using Gemm2SmallMKernel = void (*)(
    CUtensorMap,
    CUtensorMap,
    const uint8_t*,
    const uint8_t*,
    const float*,
    __nv_bfloat16*,
    const int64_t*,
    const int*,
    const float*,
    int,
    int,
    int,
    int);

using Gemm2M1RowKernel = void (*)(
    CUtensorMap,
    CUtensorMap,
    const uint8_t*,
    const uint8_t*,
    const float*,
    __nv_bfloat16*,
    const int64_t*,
    const int*,
    const float*,
    int,
    int,
    int,
    int,
    int,
    int);

void forward_impl(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    int block_n,
    int primary_n,
    int primary_k,
    int threads,
    int smem,
    Gemm2SmallMKernel kernel,
    const char* label) {
    (void)workspace;
    if (expanded_num_tokens <= 0 || M <= 0) {
        return;
    }
    if (N != primary_n || K != primary_k) {
        std::fprintf(stderr,
                     "[%s] kernel-compiled-for (N=%d K=%d) does not match runtime (N=%d K=%d), E=%d\n",
                     label, primary_n, primary_k, N, K, num_experts);
        return;
    }

    CUtensorMap h_tma_a;
    CUtensorMap h_tma_b;
    int K_half = K / 2;
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,
                         v128::ROW_BYTES, v128::BLOCK_M,
                         "TMA A encode")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v128::ROW_BYTES, block_n,
                         "TMA B 3D encode")) {
        return;
    }
    if (!ensure_kernel_attrs_prepared()) {
        return;
    }

    const int n_tiles = (N + block_n - 1) / block_n;
    const int grid = num_experts * n_tiles;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = threads;
    config.dynamicSmemBytes = smem;
    config.stream = stream;

    cudaLaunchKernelEx(&config, kernel,
                       h_tma_a, h_tma_b,
                       reinterpret_cast<const uint8_t*>(sf_a),
                       reinterpret_cast<const uint8_t*>(sf_b),
                       alpha,
                       reinterpret_cast<__nv_bfloat16*>(final_output_bf16),
                       expert_first_token_offset,
                       permuted_row_to_unpermuted_row,
                       sorted_scales,
                       num_experts, N, K, M);
}

void forward_m1_row_impl(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    int block_n,
    int primary_n,
    int primary_k,
    int threads,
    int smem,
    Gemm2M1RowKernel kernel,
    const char* label,
    int split_k) {
    (void)workspace;
    if (expanded_num_tokens <= 0 || M <= 0 || split_k <= 0) {
        return;
    }
    if (M != 1) {
        std::fprintf(stderr, "[%s] M=1 row-grid path got M=%d\n", label, M);
        return;
    }
    if (N != primary_n || K != primary_k) {
        std::fprintf(stderr,
                     "[%s] kernel-compiled-for (N=%d K=%d) does not match runtime (N=%d K=%d), E=%d\n",
                     label, primary_n, primary_k, N, K, num_experts);
        return;
    }

    CUtensorMap h_tma_a;
    CUtensorMap h_tma_b;
    int K_half = K / 2;
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,
                         v128::ROW_BYTES, v128::BLOCK_M,
                         "TMA A encode m1")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v128::ROW_BYTES, block_n,
                         "TMA B 3D encode m1")) {
        return;
    }
    if (!ensure_kernel_attrs_prepared()) {
        return;
    }

    const int n_tiles = (N + block_n - 1) / block_n;
    const int grid =
        static_cast<int>(expanded_num_tokens) * n_tiles * split_k;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = threads;
    config.dynamicSmemBytes = smem;
    config.stream = stream;

    cudaLaunchKernelEx(&config, kernel,
                       h_tma_a, h_tma_b,
                       reinterpret_cast<const uint8_t*>(sf_a),
                       reinterpret_cast<const uint8_t*>(sf_b),
                       alpha,
                       reinterpret_cast<__nv_bfloat16*>(final_output_bf16),
                       expert_first_token_offset,
                       permuted_row_to_unpermuted_row,
                       sorted_scales,
                       num_experts, N, K, M,
                       static_cast<int>(expanded_num_tokens),
                       split_k);
}

// NS is one of {v128, v64, v128_qwen3, v64_qwen3}. Macros avoid the
// namespace-as-template-parameter restriction.
#define TASK30_FORWARD_M1_CALL(NS, LABEL) do {                                 \
    forward_m1_row_impl(                                                       \
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,                    \
        expert_first_token_offset, permuted_row_to_unpermuted_row,             \
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,              \
        workspace, stream, NS::BLOCK_N, NS::PRIMARY_N, NS::PRIMARY_K,          \
        NS::THREADS, NS::SMEM_BYTES,                                           \
        NS::atrex_gemm2_m1_row_fused_finalize_kernel,                                \
        LABEL, split_k);                                                       \
} while (0)

#define TASK30_FORWARD_SMALL_M_CALL(NS, LABEL) do {                            \
    forward_impl(                                                              \
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,                    \
        expert_first_token_offset, permuted_row_to_unpermuted_row,             \
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,              \
        workspace, stream, NS::BLOCK_N, NS::PRIMARY_N, NS::PRIMARY_K,          \
        NS::THREADS, NS::SMEM_BYTES,                                           \
        NS::atrex_gemm2_small_m_fused_finalize_kernel,                               \
        LABEL);                                                                \
} while (0)

void forward_m1_n128(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    int split_k) {
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK30_FORWARD_M1_CALL(v128, "task30_m1_row_n128"); return;
        case ShapeKind::LegacyTp2:
            TASK30_FORWARD_M1_CALL(v128_tp2, "task30_m1_row_n128_tp2"); return;
        case ShapeKind::Qwen3:
            TASK30_FORWARD_M1_CALL(v128_qwen3, "task30_m1_row_n128_qwen3"); return;
        case ShapeKind::Qwen3Tp2:
            TASK30_FORWARD_M1_CALL(v128_qwen3_tp2,
                "task30_m1_row_n128_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task30_m1_row_n128] %s; got N=%d K=%d E=%d\n",
                         shape_error_str(), N, K, num_experts);
            return;
    }
}

void forward_m1_n64(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    int split_k) {
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK30_FORWARD_M1_CALL(v64, "task30_m1_row_n64"); return;
        case ShapeKind::LegacyTp2:
            TASK30_FORWARD_M1_CALL(v64_tp2, "task30_m1_row_n64_tp2"); return;
        case ShapeKind::Qwen3:
            TASK30_FORWARD_M1_CALL(v64_qwen3, "task30_m1_row_n64_qwen3"); return;
        case ShapeKind::Qwen3Tp2:
            TASK30_FORWARD_M1_CALL(v64_qwen3_tp2,
                "task30_m1_row_n64_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task30_m1_row_n64] %s; got N=%d K=%d E=%d\n",
                         shape_error_str(), N, K, num_experts);
            return;
    }
}

void forward_n128(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    if (M == 1) {
        forward_m1_n128(
            a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            sorted_scales, num_experts, N, K, M, expanded_num_tokens,
            workspace, stream, 1);
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK30_FORWARD_SMALL_M_CALL(v128, "task30_small_m_n128"); return;
        case ShapeKind::LegacyTp2:
            TASK30_FORWARD_SMALL_M_CALL(v128_tp2,
                "task30_small_m_n128_tp2");
            return;
        case ShapeKind::Qwen3:
            TASK30_FORWARD_SMALL_M_CALL(v128_qwen3,
                "task30_small_m_n128_qwen3");
            return;
        case ShapeKind::Qwen3Tp2:
            TASK30_FORWARD_SMALL_M_CALL(v128_qwen3_tp2,
                "task30_small_m_n128_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task30_small_m_n128] %s; got N=%d K=%d E=%d\n",
                         shape_error_str(), N, K, num_experts);
            return;
    }
}

void forward_n64(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    if (M == 1) {
        forward_m1_n64(
            a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            sorted_scales, num_experts, N, K, M, expanded_num_tokens,
            workspace, stream, 1);
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK30_FORWARD_SMALL_M_CALL(v64, "task30_small_m_n64"); return;
        case ShapeKind::LegacyTp2:
            TASK30_FORWARD_SMALL_M_CALL(v64_tp2,
                "task30_small_m_n64_tp2");
            return;
        case ShapeKind::Qwen3:
            TASK30_FORWARD_SMALL_M_CALL(v64_qwen3,
                "task30_small_m_n64_qwen3");
            return;
        case ShapeKind::Qwen3Tp2:
            TASK30_FORWARD_SMALL_M_CALL(v64_qwen3_tp2,
                "task30_small_m_n64_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task30_small_m_n64] %s; got N=%d K=%d E=%d\n",
                         shape_error_str(), N, K, num_experts);
            return;
    }
}
#undef TASK30_FORWARD_M1_CALL
#undef TASK30_FORWARD_SMALL_M_CALL

void forward_auto(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    if (M == 1) {
        forward_m1_n64(
            a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            sorted_scales, num_experts, N, K, M, expanded_num_tokens,
            workspace, stream, 2);
        return;
    }
    forward_n128(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        workspace, stream);
}

}  // namespace atrex_task30_gemm2_small_m

extern "C" char const* atrex_task30_gemm2_small_m_variant() {
    return "task30_small_m_v6_m1_rowgrid_n64_split2_else_grouped_m16_n128_allrows";
}

extern "C" int atrex_task30_gemm2_small_m_prepare() {
    return atrex_task30_gemm2_small_m::ensure_kernel_attrs_prepared() ? 0 : 1;
}

extern "C" int64_t atrex_task30_gemm2_small_m_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    return atrex_task30_gemm2_small_m::workspace_bytes(
        num_experts, N, K, expanded_num_tokens);
}

#define TASK30_FWD_ARGS \
    void const* a_fp4, \
    void const* b_fp4, \
    void const* sf_a, \
    void const* sf_b, \
    float const* alpha, \
    void* final_output_bf16, \
    int64_t const* expert_first_token_offset, \
    int const* permuted_row_to_unpermuted_row, \
    float const* sorted_scales, \
    int num_experts, \
    int N, \
    int K, \
    int M, \
    int64_t expanded_num_tokens, \
    void* workspace, \
    cudaStream_t stream

extern "C" void atrex_task30_gemm2_small_m_forward(TASK30_FWD_ARGS) {
    atrex_task30_gemm2_small_m::forward_auto(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        workspace, stream);
}

extern "C" void atrex_task30_gemm2_small_m_forward_n64(TASK30_FWD_ARGS) {
    atrex_task30_gemm2_small_m::forward_n64(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        workspace, stream);
}

extern "C" void atrex_task30_gemm2_small_m_forward_n128(TASK30_FWD_ARGS) {
    atrex_task30_gemm2_small_m::forward_n128(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        workspace, stream);
}

#undef TASK30_FWD_ARGS
