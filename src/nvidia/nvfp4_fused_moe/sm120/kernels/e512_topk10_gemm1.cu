#include <atomic>

// ============================================================================
// e512_topk10 GEMM1 (up/gate projection) small-M kernels for SM120.
//
// Additive port of the e512_topk10 variant only (E=512, topk=10,
// hidden/K=2560, N=2*inter=640) from revision 90e4bf46. The topk=8 shapes
// stay in task29_gemm1_small_m_cuda.cu and are not touched here.
//
// The implementation namespace is atrex_-prefixed so every launched __global__
// demangles to a name starting with "atrex_" (kernel-integration contract).
// extern "C" entry points use the atrex_e512t10_ prefix to stay distinct from the
// dev atrex_task29_* symbols inside the same JIT module.
//
// FlashInfer materializes BF16 before SwiGLU and once more before the dynamic
// scale factor / FP4 requantization. Both boundaries are matched in the fused
// epilogue via TASK29_ROUND_*_BF16_BOUNDARY while keeping a single launch.
// ============================================================================
#undef TASK29_ROUND_BF16_BOUNDARY
#define TASK29_ROUND_BF16_BOUNDARY 1
#undef TASK29_ROUND_POSTACT_BF16_BOUNDARY
#define TASK29_ROUND_POSTACT_BF16_BOUNDARY 1
#define TASK29_IMPL_NAMESPACE atrex_e512t10_task29_gemm1_small_m_v1_n128_e512t10
#define TASK29_BLOCK_N_VALUE 128
#define TASK29_PRIMARY_N_VALUE 640
#define TASK29_PRIMARY_K_VALUE 2560
#include "e512_topk10_gemm1.cuh"
#undef TASK29_PRIMARY_K_VALUE
#undef TASK29_PRIMARY_N_VALUE
#undef TASK29_BLOCK_N_VALUE
#undef TASK29_IMPL_NAMESPACE
#undef TASK29_ROUND_POSTACT_BF16_BOUNDARY
#undef TASK29_ROUND_BF16_BOUNDARY

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

namespace atrex_e512t10_task29_gemm1_small_m {

namespace v1_e512t10 = atrex_e512t10_task29_gemm1_small_m_v1_n128_e512t10;

// Shape detection: only the e512_topk10 instance is compiled into this file.
enum class ShapeKind {
    E512Topk10,
    Unsupported
};
static inline ShapeKind classify_shape(int N, int K) {
    if (K == v1_e512t10::PRIMARY_K && N == v1_e512t10::PRIMARY_N)
        return ShapeKind::E512Topk10;
    return ShapeKind::Unsupported;
}
static inline const char* shape_error_str() {
    return "e512_topk10 gemm1 small-M supports only the e512_topk10 shape"
           " (K=2560 N=640)";
}

static std::atomic<uint64_t> g_fused_reduce_epoch{1};

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
                     "[e512_topk10_gemm1] cudaFuncSetAttribute failed for %s: %s\n",
                     label, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return true;
}

// Only the two kernels the e512_topk10 dispatch actually launches need their
// dynamic shared-memory ceiling raised: the M==1 fused split-K reduce kernel
// and the 1<M grouped-m16 fused-activation kernel.
static bool ensure_kernel_attrs_prepared() {
    bool ok = true;
    ok = set_dynamic_smem_attr(
             v1_e512t10::atrex_gemm1_m1_splitk_fused_reduce_kernel,
             v1_e512t10::SMEM_BYTES,
             "gemm1_m1_splitk_fused_reduce_kernel_e512t10") && ok;
    ok = set_dynamic_smem_attr(
             v1_e512t10::atrex_gemm1_grouped_m16_fused_act_kernel,
             v1_e512t10::SMEM_BYTES,
             "gemm1_grouped_m16_fused_act_kernel_n128_e512t10") && ok;
    return ok;
}

int64_t cuda_fused_workspace_bytes(int N, int64_t expanded_num_tokens,
                                   int split_k) {
    if (split_k <= 1) {
        return 256;
    }
    int n_tiles = (N + v1_e512t10::BLOCK_N - 1) / v1_e512t10::BLOCK_N;
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
                     "[e512_topk10_gemm1] %s failed: %d "
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
                     "[e512_topk10_gemm1] %s failed: %d "
                     "(B=%p E=%d N=%d K_half=%d row_bytes=%u block_n=%u)\n",
                     tag, static_cast<int>(res), b_fp4, num_experts, N,
                     K_half, row_bytes, block_n);
        return false;
    }
    return true;
}

// M==1 fused split-K reduce: a single launch computes the partial products and
// reduces them in-kernel, resetting the completion state so a CUDA-graph replay
// reuses the same workspace without a separate memset node.
#define QF_TASK29_FORWARD_FUSED_REDUCE_BODY(NS) do {                           \
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
    uint64_t epoch = g_fused_reduce_epoch.fetch_add(                           \
        1, std::memory_order_relaxed);                                          \
    unsigned long long epoch_state =                                           \
        (0x9e3779b97f4a7c15ULL + (epoch << 8)) & ~0xffULL;                     \
    cudaLaunchConfig_t config = {};                                            \
    config.gridDim = grid;                                                     \
    config.blockDim = NS::THREADS;                                             \
    config.dynamicSmemBytes = smem;                                            \
    config.stream = stream;                                                    \
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_m1_splitk_fused_reduce_kernel,  \
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
        case ShapeKind::E512Topk10:
            QF_TASK29_FORWARD_FUSED_REDUCE_BODY(v1_e512t10); return;
        default:
            std::fprintf(stderr,
                         "[e512_topk10_gemm1_fused] %s; got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef QF_TASK29_FORWARD_FUSED_REDUCE_BODY

// 1<M grouped-m16 with fused SwiGLU + FP4 requantization. The compact shared
// scale-factor staging path is selected inside the kernel when
// use_shared_sf_staging is set and the shape is e512_topk10.
#define QF_TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(NS) do {                  \
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
    cudaLaunchKernelEx(&config, NS::atrex_gemm1_grouped_m16_fused_act_kernel,   \
                       h_tma_a, h_tma_b,                                       \
                       reinterpret_cast<const uint8_t*>(sf_a),                 \
                       reinterpret_cast<const uint8_t*>(sf_b),                 \
                       alpha,                                                  \
                       reinterpret_cast<uint8_t*>(output_fp4),                 \
                       fc2_act_global_scale,                                   \
                       reinterpret_cast<uint8_t*>(fc2_act_sf),                 \
                       expert_first_token_offset,                              \
                       num_experts, N, K, use_shared_sf_staging);              \
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
    bool use_shared_sf_staging,
    cudaStream_t stream) {
    if (expanded_num_tokens <= 0) {
        return;
    }
    switch (classify_shape(N, K)) {
        case ShapeKind::E512Topk10:
            QF_TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY(v1_e512t10); return;
        default:
            std::fprintf(stderr,
                         "[e512_topk10_gemm1_grouped_m16_fused_act] %s;"
                         " got N=%d K=%d\n",
                         shape_error_str(), N, K);
            return;
    }
}
#undef QF_TASK29_FORWARD_GROUPED_M16_FUSED_ACT_BODY

}  // namespace atrex_e512t10_task29_gemm1_small_m

// ============================================================================
// extern "C" entry points (atrex_e512t10_ prefix). Signatures must match the
// declarations in e512_topk10_pybind.cu exactly.
// ============================================================================

extern "C" char const* atrex_e512t10_task29_variant() {
    return "e512_topk10_task29_gemm1_small_m_v1_n128_bf16_pre_post_boundaries";
}

extern "C" int atrex_e512t10_task29_prepare() {
    return atrex_e512t10_task29_gemm1_small_m::ensure_kernel_attrs_prepared() ? 0 : 1;
}

extern "C" int64_t atrex_e512t10_task29_fused_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k) {
    (void)num_experts;
    (void)K;
    return atrex_e512t10_task29_gemm1_small_m::cuda_fused_workspace_bytes(
        N, expanded_num_tokens, split_k);
}

extern "C" void atrex_e512t10_task29_forward_fused(
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
    atrex_e512t10_task29_gemm1_small_m::forward_fused_reduce(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_bf16,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        split_k, workspace, stream);
}

extern "C" int64_t atrex_e512t10_task29_grouped_m16_fused_act_workspace_bytes(
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

extern "C" void atrex_e512t10_task29_forward_grouped_m16_fused_act(
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
    bool use_shared_sf_staging,
    void* workspace,
    cudaStream_t stream) {
    (void)workspace;
    atrex_e512t10_task29_gemm1_small_m::forward_grouped_m16_fused_act_n128(
        a_fp4, b_fp4, sf_a, sf_b, alpha, output_fp4,
        fc2_act_global_scale, fc2_act_sf,
        expert_first_token_offset, num_experts, N, K, expanded_num_tokens,
        use_shared_sf_staging, stream);
}
