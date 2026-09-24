// ===== Legacy v4 primary shape (E=256, hidden=2048, inter=512) =====
// GEMM1: K=hidden=2048, N=2*inter=1024.
#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n128
#define TASK29_BLOCK_N_VALUE 128
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n64
#define TASK29_BLOCK_N_VALUE 64
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

// ===== Qwen3-MoE primary shape (E=128, hidden=2048, inter=768) =====
// GEMM1: K=hidden=2048 (same as legacy), N=2*inter=1536.
// Only PRIMARY_N differs from the legacy instance; INTRA_SPLIT logic relies
// on PRIMARY_K=2048 which is unchanged.
#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n128_qwen3
#define TASK29_BLOCK_N_VALUE 128
#define TASK29_PRIMARY_N_VALUE 1536
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n64_qwen3
#define TASK29_BLOCK_N_VALUE 64
#define TASK29_PRIMARY_N_VALUE 1536
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

// ===== qwen3_5_flash TP2 shape (E=256, hidden=2048, inter=256) =====
// GEMM1: K=hidden=2048, N=2*inter=512.
#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n128_tp2
#define TASK29_BLOCK_N_VALUE 128
#define TASK29_PRIMARY_N_VALUE 512
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n64_tp2
#define TASK29_BLOCK_N_VALUE 64
#define TASK29_PRIMARY_N_VALUE 512
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

// ===== qwen3_6_flash TP2 shape (E=128, hidden=2048, inter=384) =====
// GEMM1: K=hidden=2048, N=2*inter=768.
#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n128_qwen3_tp2
#define TASK29_BLOCK_N_VALUE 128
#define TASK29_PRIMARY_N_VALUE 768
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1_n64_qwen3_tp2
#define TASK29_BLOCK_N_VALUE 64
#define TASK29_PRIMARY_N_VALUE 768
#define TASK29_PRIMARY_K_VALUE 2048
#include "task29_gemm1_small_m_sm120.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

namespace atrex_task29_gemm1_small_m {

namespace v1 = atrex_task29_gemm1_small_m_v1_n128;
namespace v64 = atrex_task29_gemm1_small_m_v1_n64;
namespace v1_qwen3 = atrex_task29_gemm1_small_m_v1_n128_qwen3;
namespace v64_qwen3 = atrex_task29_gemm1_small_m_v1_n64_qwen3;
namespace v1_tp2 = atrex_task29_gemm1_small_m_v1_n128_tp2;
namespace v64_tp2 = atrex_task29_gemm1_small_m_v1_n64_tp2;
namespace v1_qwen3_tp2 = atrex_task29_gemm1_small_m_v1_n128_qwen3_tp2;
namespace v64_qwen3_tp2 = atrex_task29_gemm1_small_m_v1_n64_qwen3_tp2;

// Shape detection: which compiled namespace matches the runtime (N, K)?
enum class ShapeKind { Legacy, LegacyTp2, Qwen3, Qwen3Tp2, Unsupported };
static inline ShapeKind classify_shape(int N, int K) {
    if (K == v1::PRIMARY_K && N == v1::PRIMARY_N) return ShapeKind::Legacy;
    if (K == v1_tp2::PRIMARY_K && N == v1_tp2::PRIMARY_N)
        return ShapeKind::LegacyTp2;
    if (K == v1_qwen3::PRIMARY_K && N == v1_qwen3::PRIMARY_N)
        return ShapeKind::Qwen3;
    if (K == v1_qwen3_tp2::PRIMARY_K && N == v1_qwen3_tp2::PRIMARY_N)
        return ShapeKind::Qwen3Tp2;
    return ShapeKind::Unsupported;
}
static inline const char* shape_error_str() {
    return "task29_small_m supports only the v4 primary shape (K=2048 N=1024)"
           ", qwen3_5 TP2 shape (K=2048 N=512), Qwen3-MoE shape"
           " (K=2048 N=1536), or qwen3_6 TP2 shape (K=2048 N=768)";
}

static uint64_t g_fused_reduce_epoch = 1;

static inline int64_t align256(int64_t x) {
    return (x + 255) & ~255LL;
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
                     "[task29_small_m] cudaFuncSetAttribute failed for %s: %s\n",
                     label, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return true;
}

static bool ensure_kernel_attrs_prepared() {
    bool ok = true;
    ok = set_dynamic_smem_attr(
             v1::atrex_gemm1_m1_splitk_kernel, v1::SMEM_BYTES,
             "atrex_gemm1_m1_splitk_kernel") && ok;
    ok = set_dynamic_smem_attr(
             v1::atrex_gemm1_m1_splitk_fused_reduce_kernel, v1::SMEM_BYTES,
             "atrex_gemm1_m1_splitk_fused_reduce_kernel") && ok;
    ok = set_dynamic_smem_attr(
             v1::atrex_gemm1_grouped_m16_kernel, v1::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n128") && ok;
    ok = set_dynamic_smem_attr(
             v1::atrex_gemm1_grouped_m16_fused_act_kernel, v1::SMEM_BYTES,
             "gemm1_grouped_m16_fused_act_kernel_n128") && ok;
    ok = set_dynamic_smem_attr(
             v64::atrex_gemm1_grouped_m16_kernel, v64::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n64") && ok;
    ok = set_dynamic_smem_attr(
             v1::atrex_gemm1_m1_intra_split4_kernel, v1::INTRA_SMEM_BYTES,
             "atrex_gemm1_m1_intra_split4_kernel") && ok;
    // Qwen3-MoE shape (PRIMARY_N=1536) instances. SMEM_BYTES is unchanged
    // (smem only depends on BLOCK_M/N/K which match), but we still call
    // cudaFuncSetAttribute per kernel-function pointer.
    ok = set_dynamic_smem_attr(
             v1_qwen3::atrex_gemm1_m1_splitk_kernel, v1_qwen3::SMEM_BYTES,
             "gemm1_m1_splitk_kernel_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3::atrex_gemm1_m1_splitk_fused_reduce_kernel,
             v1_qwen3::SMEM_BYTES,
             "gemm1_m1_splitk_fused_reduce_kernel_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3::atrex_gemm1_grouped_m16_kernel, v1_qwen3::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n128_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3::atrex_gemm1_grouped_m16_fused_act_kernel,
             v1_qwen3::SMEM_BYTES,
             "gemm1_grouped_m16_fused_act_kernel_n128_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3::atrex_gemm1_grouped_m16_kernel, v64_qwen3::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n64_qwen3") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3::atrex_gemm1_m1_intra_split4_kernel,
             v1_qwen3::INTRA_SMEM_BYTES,
             "gemm1_m1_intra_split4_kernel_qwen3") && ok;
    // TP2 shape instances use the same tile sizes and smem layout, but
    // PRIMARY_N changes scale-factor strides.
    ok = set_dynamic_smem_attr(
             v1_tp2::atrex_gemm1_m1_splitk_kernel, v1_tp2::SMEM_BYTES,
             "gemm1_m1_splitk_kernel_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_tp2::atrex_gemm1_m1_splitk_fused_reduce_kernel,
             v1_tp2::SMEM_BYTES,
             "gemm1_m1_splitk_fused_reduce_kernel_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_tp2::atrex_gemm1_grouped_m16_kernel, v1_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n128_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_tp2::atrex_gemm1_grouped_m16_fused_act_kernel,
             v1_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_fused_act_kernel_n128_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_tp2::atrex_gemm1_grouped_m16_kernel, v64_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n64_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_tp2::atrex_gemm1_m1_intra_split4_kernel,
             v1_tp2::INTRA_SMEM_BYTES,
             "gemm1_m1_intra_split4_kernel_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3_tp2::atrex_gemm1_m1_splitk_kernel,
             v1_qwen3_tp2::SMEM_BYTES,
             "gemm1_m1_splitk_kernel_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3_tp2::atrex_gemm1_m1_splitk_fused_reduce_kernel,
             v1_qwen3_tp2::SMEM_BYTES,
             "gemm1_m1_splitk_fused_reduce_kernel_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3_tp2::atrex_gemm1_grouped_m16_kernel,
             v1_qwen3_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n128_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3_tp2::atrex_gemm1_grouped_m16_fused_act_kernel,
             v1_qwen3_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_fused_act_kernel_n128_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v64_qwen3_tp2::atrex_gemm1_grouped_m16_kernel,
             v64_qwen3_tp2::SMEM_BYTES,
             "gemm1_grouped_m16_kernel_n64_qwen3_tp2") && ok;
    ok = set_dynamic_smem_attr(
             v1_qwen3_tp2::atrex_gemm1_m1_intra_split4_kernel,
             v1_qwen3_tp2::INTRA_SMEM_BYTES,
             "gemm1_m1_intra_split4_kernel_qwen3_tp2") && ok;
    return ok;
}

int64_t cuda_workspace_bytes(int N, int64_t expanded_num_tokens, int split_k) {
    if (split_k <= 1) {
        return 256;
    }
    int64_t partial_bytes =
        static_cast<int64_t>(split_k) * expanded_num_tokens * N *
        static_cast<int64_t>(sizeof(float));
    return align256(partial_bytes);
}

int64_t cuda_fused_workspace_bytes(int N, int64_t expanded_num_tokens,
                                   int split_k) {
    if (split_k <= 1) {
        return 256;
    }
    int n_tiles = (N + v1::BLOCK_N - 1) / v1::BLOCK_N;
    int64_t partial_bytes =
        static_cast<int64_t>(split_k) * expanded_num_tokens * N *
        static_cast<int64_t>(sizeof(float));
    int64_t state_bytes =
        expanded_num_tokens * n_tiles *
        static_cast<int64_t>(sizeof(unsigned long long));
    return align256(partial_bytes) + align256(state_bytes);
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
                     "[task29_small_m] %s failed: %d "
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
                     "[task29_small_m] %s failed: %d "
                     "(B=%p E=%d N=%d K_half=%d row_bytes=%u block_n=%u)\n",
                     tag, static_cast<int>(res), b_fp4, num_experts, N,
                     K_half, row_bytes, block_n);
        return false;
    }
    return true;
}

// Dispatch macro: NS is one of `v1` or `v1_qwen3`. Picks the kernel and
// per-shape compile-time constants (BLOCK_N, PRIMARY_N/K, SMEM_BYTES, THREADS).
// We use a macro rather than a function template because NS is a namespace
// alias, which cannot be passed as a template type parameter.
#define TASK29_FORWARD_V1_BODY(NS) do {                                       \
    CUtensorMap h_tma_a;                                                       \
    CUtensorMap h_tma_b;                                                       \
    int K_half = K / 2;                                                        \
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,         \
                         NS::ROW_BYTES, NS::BLOCK_M,                           \
                         "TMA A encode")) {                                    \
        return;                                                                \
    }                                                                          \
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,              \
                         NS::ROW_BYTES, NS::BLOCK_N,                           \
                         "TMA B 3D encode")) {                                 \
        return;                                                                \
    }                                                                          \
    if (!ensure_kernel_attrs_prepared()) {                                      \
        return;                                                                \
    }                                                                          \
    int smem = NS::SMEM_BYTES;                                                 \
    int n_tiles = (N + NS::BLOCK_N - 1) / NS::BLOCK_N;                         \
    int grid = static_cast<int>(expanded_num_tokens) * n_tiles * split_k;      \
    float* partials =                                                          \
        split_k > 1 ? reinterpret_cast<float*>(workspace) : nullptr;           \
    cudaLaunchConfig_t config = {};                                            \
    config.gridDim = grid;                                                     \
    config.blockDim = NS::THREADS;                                             \
    config.dynamicSmemBytes = smem;                                            \
    config.stream = stream;                                                    \
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_m1_splitk_kernel,                    \
                       h_tma_a, h_tma_b,                                       \
                       reinterpret_cast<const uint8_t*>(sf_a),                 \
                       reinterpret_cast<const uint8_t*>(sf_b),                 \
                       alpha,                                                  \
                       reinterpret_cast<__nv_bfloat16*>(output_bf16),          \
                       partials,                                               \
                       expert_first_token_offset,                              \
                       num_experts, N, K,                                      \
                       static_cast<int>(expanded_num_tokens),                  \
                       split_k);                                               \
    if (split_k > 1) {                                                         \
        int total = static_cast<int>(expanded_num_tokens) * N;                 \
        int block = 256;                                                       \
        int reduce_grid = (total + block - 1) / block;                         \
        NS::atrex_reduce_splitk_kernel<<<reduce_grid, block, 0, stream>>>(           \
            partials, alpha,                                                   \
            reinterpret_cast<__nv_bfloat16*>(output_bf16),                     \
            expert_first_token_offset, num_experts, N,                         \
            static_cast<int>(expanded_num_tokens), split_k);                   \
    }                                                                          \
} while (0)

void forward_v1(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k,
    void* workspace,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0 || split_k <= 0) {
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:  TASK29_FORWARD_V1_BODY(v1); return;
        case ShapeKind::LegacyTp2: TASK29_FORWARD_V1_BODY(v1_tp2); return;
        case ShapeKind::Qwen3:   TASK29_FORWARD_V1_BODY(v1_qwen3); return;
        case ShapeKind::Qwen3Tp2: TASK29_FORWARD_V1_BODY(v1_qwen3_tp2); return;
        default:
            std::fprintf(stderr, "[task29_small_m] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef TASK29_FORWARD_V1_BODY

#define TASK29_FORWARD_FUSED_REDUCE_BODY(NS) do {                              \
    CUtensorMap h_tma_a;                                                       \
    CUtensorMap h_tma_b;                                                       \
    int K_half = K / 2;                                                        \
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,         \
                         NS::ROW_BYTES, NS::BLOCK_M,                           \
                         "TMA A encode fused")) {                              \
        return;                                                                \
    }                                                                          \
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,              \
                         NS::ROW_BYTES, NS::BLOCK_N,                           \
                         "TMA B 3D encode fused")) {                           \
        return;                                                                \
    }                                                                          \
    if (!ensure_kernel_attrs_prepared()) {                                      \
        return;                                                                \
    }                                                                          \
    int smem = NS::SMEM_BYTES;                                                 \
    int n_tiles = (N + NS::BLOCK_N - 1) / NS::BLOCK_N;                         \
    int grid = static_cast<int>(expanded_num_tokens) * n_tiles * split_k;      \
    uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);                       \
    float* partials = split_k > 1 ? reinterpret_cast<float*>(ws) : nullptr;    \
    int64_t partial_bytes =                                                    \
        static_cast<int64_t>(split_k) * expanded_num_tokens * N *              \
        static_cast<int64_t>(sizeof(float));                                   \
    unsigned long long* done_states = split_k > 1                              \
        ? reinterpret_cast<unsigned long long*>(ws + align256(partial_bytes))  \
        : nullptr;                                                             \
    int64_t done_state_bytes =                                                 \
        static_cast<int64_t>(expanded_num_tokens) * n_tiles *                  \
        static_cast<int64_t>(sizeof(unsigned long long));                      \
    if (split_k > 1) {                                                         \
        cudaMemsetAsync(done_states, 0, done_state_bytes, stream);             \
    }                                                                          \
    uint64_t epoch = g_fused_reduce_epoch++;                                   \
    unsigned long long epoch_state =                                           \
        (0x9e3779b97f4a7c15ULL + (epoch << 8)) & ~0xffULL;                     \
    cudaLaunchConfig_t config = {};                                            \
    config.gridDim = grid;                                                     \
    config.blockDim = NS::THREADS;                                             \
    config.dynamicSmemBytes = smem;                                            \
    config.stream = stream;                                                    \
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_m1_splitk_fused_reduce_kernel,       \
                       h_tma_a, h_tma_b,                                       \
                       reinterpret_cast<const uint8_t*>(sf_a),                 \
                       reinterpret_cast<const uint8_t*>(sf_b),                 \
                       alpha,                                                  \
                       reinterpret_cast<__nv_bfloat16*>(output_bf16),          \
                       partials, done_states, epoch_state,                     \
                       expert_first_token_offset,                              \
                       num_experts, N, K,                                      \
                       static_cast<int>(expanded_num_tokens),                  \
                       split_k);                                               \
} while (0)

void forward_fused_reduce(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k,
    void* workspace,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0 || split_k <= 0) {
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy: TASK29_FORWARD_FUSED_REDUCE_BODY(v1); return;
        case ShapeKind::LegacyTp2:
            TASK29_FORWARD_FUSED_REDUCE_BODY(v1_tp2); return;
        case ShapeKind::Qwen3:  TASK29_FORWARD_FUSED_REDUCE_BODY(v1_qwen3); return;
        case ShapeKind::Qwen3Tp2:
            TASK29_FORWARD_FUSED_REDUCE_BODY(v1_qwen3_tp2); return;
        default:
            std::fprintf(stderr, "[task29_small_m_fused] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef TASK29_FORWARD_FUSED_REDUCE_BODY

using GroupedM16Kernel = void (*)(
    CUtensorMap,
    CUtensorMap,
    const uint8_t*,
    const uint8_t*,
    const float*,
    __nv_bfloat16*,
    const int64_t*,
    int,
    int,
    int);

void forward_grouped_m16_impl(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int block_n,
    int primary_n,
    int primary_k,
    int threads,
    int smem,
    GroupedM16Kernel kernel,
    const char* label,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0) {
        return;
    }
    if (K != primary_k || N != primary_n) {
        std::fprintf(stderr,
                     "[%s] kernel-compiled-for (N=%d K=%d) does not match runtime (N=%d K=%d)\n",
                     label, primary_n, primary_k, N, K);
        return;
    }

    CUtensorMap h_tma_a;
    CUtensorMap h_tma_b;
    int K_half = K / 2;
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,
                         v1::ROW_BYTES, v1::BLOCK_M,
                         "TMA A encode grouped_m16")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v1::ROW_BYTES, block_n,
                         "TMA B 3D encode grouped_m16")) {
        return;
    }
    if (!ensure_kernel_attrs_prepared()) {
        return;
    }

    int n_tiles = (N + block_n - 1) / block_n;
    int grid = num_experts * n_tiles;

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
                       reinterpret_cast<__nv_bfloat16*>(output_bf16),
                       expert_first_token_offset,
                       num_experts, N, K);
}

// Wrapper macro: NS is one of {v1, v64, v1_qwen3, v64_qwen3}. Picks per-shape
// constants and the grouped_m16 kernel, then calls forward_grouped_m16_impl.
#define TASK29_FORWARD_GROUPED_M16_CALL(NS, LABEL) do {                        \
    forward_grouped_m16_impl(                                                  \
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,                          \
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,     \
        NS::BLOCK_N, NS::PRIMARY_N, NS::PRIMARY_K, NS::THREADS,                \
        NS::SMEM_BYTES, NS::atrex_gemm1_grouped_m16_kernel, LABEL, stream);          \
} while (0)

void forward_grouped_m16_n128(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream) {
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v1, "task29_small_m_grouped_m16_n128");
            return;
        case ShapeKind::LegacyTp2:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v1_tp2, "task29_small_m_grouped_m16_n128_tp2");
            return;
        case ShapeKind::Qwen3:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v1_qwen3, "task29_small_m_grouped_m16_n128_qwen3");
            return;
        case ShapeKind::Qwen3Tp2:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v1_qwen3_tp2,
                "task29_small_m_grouped_m16_n128_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task29_small_m_grouped_m16_n128] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}

#define TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(NS) do {                     \
    CUtensorMap h_tma_a;                                                       \
    CUtensorMap h_tma_b;                                                       \
    int K_half = K / 2;                                                        \
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,         \
                         NS::ROW_BYTES, NS::BLOCK_M,                           \
                         "TMA A encode grouped_m16_fused_act")) {              \
        return;                                                                \
    }                                                                          \
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,              \
                         NS::ROW_BYTES, NS::BLOCK_N,                           \
                         "TMA B 3D encode grouped_m16_fused_act")) {           \
        return;                                                                \
    }                                                                          \
    if (!ensure_kernel_attrs_prepared()) {                                      \
        return;                                                                \
    }                                                                          \
    int smem = NS::SMEM_BYTES;                                                 \
    int n_tiles = (N + NS::BLOCK_N - 1) / NS::BLOCK_N;                         \
    int grid = num_experts * n_tiles;                                          \
    cudaLaunchConfig_t config = {};                                            \
    config.gridDim = grid;                                                     \
    config.blockDim = NS::THREADS;                                             \
    config.dynamicSmemBytes = smem;                                            \
    config.stream = stream;                                                    \
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_grouped_m16_fused_act_kernel,        \
                       h_tma_a, h_tma_b,                                       \
                       reinterpret_cast<const uint8_t*>(sf_a),                 \
                       reinterpret_cast<const uint8_t*>(sf_b),                 \
                       alpha,                                                  \
                       reinterpret_cast<uint8_t*>(output_fp4),                 \
                       fc2_act_global_scale,                                   \
                       reinterpret_cast<uint8_t*>(fc2_act_sf),                 \
                       expert_first_token_offset,                              \
                       num_experts, N, K);                                     \
} while (0)

void forward_grouped_m16_fused_act_n128(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0) {
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(v1); return;
        case ShapeKind::LegacyTp2:
            TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(v1_tp2); return;
        case ShapeKind::Qwen3:
            TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(v1_qwen3); return;
        case ShapeKind::Qwen3Tp2:
            TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(v1_qwen3_tp2); return;
        default:
            std::fprintf(stderr,
                         "[task29_small_m_grouped_m16_fused_act_n128] %s;"
                         " got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY

void forward_grouped_m16_n64(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream) {
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v64, "task29_small_m_grouped_m16_n64");
            return;
        case ShapeKind::LegacyTp2:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v64_tp2, "task29_small_m_grouped_m16_n64_tp2");
            return;
        case ShapeKind::Qwen3:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v64_qwen3, "task29_small_m_grouped_m16_n64_qwen3");
            return;
        case ShapeKind::Qwen3Tp2:
            TASK29_FORWARD_GROUPED_M16_CALL(
                v64_qwen3_tp2,
                "task29_small_m_grouped_m16_n64_qwen3_tp2");
            return;
        default:
            std::fprintf(stderr,
                         "[task29_small_m_grouped_m16_n64] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}

void forward_grouped_m16(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 256) {
        forward_grouped_m16_n64(
            a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
            expert_first_token_offset, num_experts, N, K,
            expanded_num_tokens, stream);
    } else {
        forward_grouped_m16_n128(
            a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
            expert_first_token_offset, num_experts, N, K,
            expanded_num_tokens, stream);
    }
}

#define TASK29_FORWARD_INTRA4_BODY(NS) do {                                    \
    CUtensorMap h_tma_a;                                                       \
    CUtensorMap h_tma_b;                                                       \
    int K_half = K / 2;                                                        \
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,         \
                         NS::ROW_BYTES, NS::BLOCK_M,                           \
                         "TMA A encode intra4")) {                             \
        return;                                                                \
    }                                                                          \
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,              \
                         NS::ROW_BYTES, NS::BLOCK_N,                           \
                         "TMA B 3D encode intra4")) {                          \
        return;                                                                \
    }                                                                          \
    if (!ensure_kernel_attrs_prepared()) {                                      \
        return;                                                                \
    }                                                                          \
    int smem = NS::INTRA_SMEM_BYTES;                                           \
    int n_tiles = (N + NS::BLOCK_N - 1) / NS::BLOCK_N;                         \
    int grid = static_cast<int>(expanded_num_tokens) * n_tiles;                \
    cudaLaunchConfig_t config = {};                                            \
    config.gridDim = grid;                                                     \
    config.blockDim = NS::INTRA_THREADS;                                       \
    config.dynamicSmemBytes = smem;                                            \
    config.stream = stream;                                                    \
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_m1_intra_split4_kernel,              \
                       h_tma_a, h_tma_b,                                       \
                       reinterpret_cast<const uint8_t*>(sf_a),                 \
                       reinterpret_cast<const uint8_t*>(sf_b),                 \
                       alpha,                                                  \
                       reinterpret_cast<__nv_bfloat16*>(output_bf16),          \
                       expert_first_token_offset,                              \
                       num_experts, N, K,                                      \
                       static_cast<int>(expanded_num_tokens));                 \
} while (0)

void forward_intra4(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0) {
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::Legacy: TASK29_FORWARD_INTRA4_BODY(v1); return;
        case ShapeKind::LegacyTp2: TASK29_FORWARD_INTRA4_BODY(v1_tp2); return;
        case ShapeKind::Qwen3:  TASK29_FORWARD_INTRA4_BODY(v1_qwen3); return;
        case ShapeKind::Qwen3Tp2:
            TASK29_FORWARD_INTRA4_BODY(v1_qwen3_tp2); return;
        default:
            std::fprintf(stderr, "[task29_small_m_intra4] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef TASK29_FORWARD_INTRA4_BODY

}  // namespace atrex_task29_gemm1_small_m

extern "C" char const* atrex_task29_gemm1_small_m_variant() {
    return "task29_small_m_v5_grouped_m16_allrows_auto_n64_le_m32_n128_ge_m64";
}

extern "C" int atrex_task29_gemm1_small_m_prepare() {
    return atrex_task29_gemm1_small_m::ensure_kernel_attrs_prepared() ? 0 : 1;
}

extern "C" int64_t atrex_task29_gemm1_small_m_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k) {
    (void)num_experts;
    (void)K;
    return atrex_task29_gemm1_small_m::cuda_workspace_bytes(
        N, expanded_num_tokens, split_k);
}

extern "C" void atrex_task29_gemm1_small_m_forward(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k,
    void* workspace,
    cudaStream_t stream) {
    atrex_task29_gemm1_small_m::forward_v1(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        split_k, workspace, stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_fused_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k) {
    (void)num_experts;
    (void)K;
    return atrex_task29_gemm1_small_m::cuda_fused_workspace_bytes(
        N, expanded_num_tokens, split_k);
}

extern "C" void atrex_task29_gemm1_small_m_forward_fused(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k,
    void* workspace,
    cudaStream_t stream) {
    atrex_task29_gemm1_small_m::forward_fused_reduce(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        split_k, workspace, stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_grouped_m16_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    (void)num_experts;
    (void)N;
    (void)K;
    (void)expanded_num_tokens;
    return 256;
}

extern "C" void atrex_task29_gemm1_small_m_forward_grouped_m16(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_task29_gemm1_small_m::forward_grouped_m16(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_grouped_m16_n64_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    return atrex_task29_gemm1_small_m_grouped_m16_workspace_bytes(
        num_experts, N, K, expanded_num_tokens);
}

extern "C" void atrex_task29_gemm1_small_m_forward_grouped_m16_n64(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_task29_gemm1_small_m::forward_grouped_m16_n64(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_grouped_m16_n128_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    return atrex_task29_gemm1_small_m_grouped_m16_workspace_bytes(
        num_experts, N, K, expanded_num_tokens);
}

extern "C" void atrex_task29_gemm1_small_m_forward_grouped_m16_n128(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_task29_gemm1_small_m::forward_grouped_m16_n128(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_grouped_m16_fused_act_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    return atrex_task29_gemm1_small_m_grouped_m16_workspace_bytes(
        num_experts, N, K, expanded_num_tokens);
}

extern "C" void atrex_task29_gemm1_small_m_forward_grouped_m16_fused_act(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_task29_gemm1_small_m::forward_grouped_m16_fused_act_n128(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_fp4,
        fc2_act_global_scale, fc2_act_sf,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        stream);
}

extern "C" int64_t atrex_task29_gemm1_small_m_intra4_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    (void)num_experts;
    (void)N;
    (void)K;
    (void)expanded_num_tokens;
    return 256;
}

extern "C" void atrex_task29_gemm1_small_m_forward_intra4(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_task29_gemm1_small_m::forward_intra4(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        stream);
}
