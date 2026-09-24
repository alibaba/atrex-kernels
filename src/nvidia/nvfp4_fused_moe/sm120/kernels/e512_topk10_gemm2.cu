// ============================================================================
// e512_topk10 GEMM2 (down projection) small-M kernels for SM120.
//
// Additive port of the e512_topk10 variant only (E=512, topk=10,
// N=hidden=2560, K=inter=320; K has a single 64-wide tail atom) from revision
// 90e4bf46. The topk=8 shapes stay in task30_gemm2_small_m_cuda.cu.
//
// e512_topk10 uses the in-kernel fixed-order FP32 finalize
// (FIXED_ORDER_FINALIZE=true, FIXED_FINALIZE_TOPK=10 baked into the header):
// each expert row writes its BF16 partial to expert_rows_bf16 and the last
// completing block per (row, n_tile) reduces the ten contributions in a fixed
// order guarded by completion_counters, so no separate finalize_moe_routing or
// atomic BF16 accumulate pass is needed.
//
// atrex_-prefixed implementation namespace => launched __global__ names start
// with "atrex_"; extern "C" uses atrex_e512t10_ to stay distinct from dev's
// atrex_task30_* symbols in the same JIT module.
// ============================================================================
#define TASK30_IMPL_NAMESPACE atrex_e512t10_task30_gemm2_small_m_v1_n128_e512t10
#define TASK30_BLOCK_N_VALUE 128
#define TASK30_PRIMARY_N_VALUE 2560
#define TASK30_PRIMARY_K_VALUE 320
#include "e512_topk10_gemm2.cuh"
#undef TASK30_PRIMARY_K_VALUE
#undef TASK30_PRIMARY_N_VALUE
#undef TASK30_BLOCK_N_VALUE
#undef TASK30_IMPL_NAMESPACE

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

namespace atrex_e512t10_task30_gemm2_small_m {

namespace v128_e512t10 = atrex_e512t10_task30_gemm2_small_m_v1_n128_e512t10;

// Shape detection: only the e512_topk10 instance is compiled into this file.
enum class ShapeKind {
    E512Topk10,
    Unsupported
};
static inline ShapeKind classify_shape(int N, int K) {
    if (K == v128_e512t10::PRIMARY_K && N == v128_e512t10::PRIMARY_N)
        return ShapeKind::E512Topk10;
    return ShapeKind::Unsupported;
}
static inline const char* shape_error_str() {
    return "e512_topk10 gemm2 small-M supports only the e512_topk10 shape"
           " (K=320 N=2560)";
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
                     "[e512_topk10_gemm2] cudaFuncSetAttribute failed for %s: %s\n",
                     label, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return true;
}

// Only the fixed-order (<true>) kernels are launched by e512_topk10.
static bool ensure_kernel_attrs_prepared() {
    bool ok = true;
    ok = set_dynamic_smem_attr(
             v128_e512t10::atrex_gemm2_small_m_kernel<true>,
             v128_e512t10::SMEM_BYTES,
             "gemm2_small_m_fixed_finalize_kernel_n128_e512t10") && ok;
    ok = set_dynamic_smem_attr(
             v128_e512t10::atrex_gemm2_m1_row_kernel<true>,
             v128_e512t10::SMEM_BYTES,
             "gemm2_m1_row_fixed_finalize_kernel_n128_e512t10") && ok;
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
                     "[e512_topk10_gemm2] %s failed: %d "
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
                     "[e512_topk10_gemm2] %s failed: %d "
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
    int,
    bool,
    __nv_bfloat16*,
    const float*,
    int*,
    bool,
    const int*);

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
    int,
    __nv_bfloat16*,
    const float*,
    int*,
    bool,
    const int*);

// 1<M small-M grouped path. ROW_BYTES/BLOCK_M are shape-independent tile
// constants, so they are taken from the only compiled instance.
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
    bool use_shared_sf_staging,
    void* workspace,
    cudaStream_t stream,
    void* expert_rows_bf16,
    float const* topk_weights,
    int* completion_counters,
    bool output_preinitialized,
    int const* topk_ids,
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
                         v128_e512t10::ROW_BYTES, v128_e512t10::BLOCK_M,
                         "TMA A encode")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v128_e512t10::ROW_BYTES, block_n,
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
                       num_experts, N, K, M, use_shared_sf_staging,
                       reinterpret_cast<__nv_bfloat16*>(expert_rows_bf16),
                       topk_weights, completion_counters,
                       output_preinitialized, topk_ids);
}

// M==1 row-grid path: one CTA owns the full K range for an expert row, which
// preserves the BF16 expert-row boundary the generic GEMM2 oracle uses before
// the fixed top-k reduction.
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
    void* expert_rows_bf16,
    float const* topk_weights,
    int* completion_counters,
    bool output_preinitialized,
    int const* topk_ids,
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
                         v128_e512t10::ROW_BYTES, v128_e512t10::BLOCK_M,
                         "TMA A encode m1")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v128_e512t10::ROW_BYTES, block_n,
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
                       split_k,
                       reinterpret_cast<__nv_bfloat16*>(expert_rows_bf16),
                       topk_weights, completion_counters,
                       output_preinitialized, topk_ids);
}

void forward_fixed_e512t10(
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
    bool use_shared_sf_staging,
    void* workspace,
    void* expert_rows_bf16,
    float const* topk_weights,
    int* completion_counters,
    bool output_preinitialized,
    int const* topk_ids,
    cudaStream_t stream) {
    if (classify_shape(N, K) != ShapeKind::E512Topk10 ||
        num_experts != 512 || expanded_num_tokens != 10LL * M ||
        expert_rows_bf16 == nullptr || topk_weights == nullptr ||
        completion_counters == nullptr || topk_ids == nullptr) {
        std::fprintf(
            stderr,
            "[e512_topk10_gemm2_fixed] requires E=512 topk=10 N=2560 "
            "K=320 and non-null fixed-order buffers; got E=%d M=%d "
            "expanded=%lld N=%d K=%d\n",
            num_experts, M, static_cast<long long>(expanded_num_tokens),
            N, K);
        return;
    }

    if (M == 1) {
        forward_m1_row_impl(
            a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            sorted_scales, num_experts, N, K, M, expanded_num_tokens,
            workspace, stream,
            expert_rows_bf16, topk_weights, completion_counters,
            output_preinitialized, topk_ids,
            v128_e512t10::BLOCK_N,
            v128_e512t10::PRIMARY_N,
            v128_e512t10::PRIMARY_K,
            v128_e512t10::THREADS,
            v128_e512t10::SMEM_BYTES,
            v128_e512t10::atrex_gemm2_m1_row_kernel<true>,
            "e512_topk10_gemm2_m1_row_fixed_n128", 1);
        return;
    }

    forward_impl(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        use_shared_sf_staging, workspace, stream,
        expert_rows_bf16, topk_weights, completion_counters,
        output_preinitialized, topk_ids,
        v128_e512t10::BLOCK_N,
        v128_e512t10::PRIMARY_N,
        v128_e512t10::PRIMARY_K,
        v128_e512t10::THREADS,
        v128_e512t10::SMEM_BYTES,
        v128_e512t10::atrex_gemm2_small_m_kernel<true>,
        "e512_topk10_gemm2_small_m_fixed_n128");
}

}  // namespace atrex_e512t10_task30_gemm2_small_m

// ============================================================================
// extern "C" entry points (atrex_e512t10_ prefix). Signatures must match the
// declarations in e512_topk10_pybind.cu exactly.
// ============================================================================

extern "C" char const* atrex_e512t10_task30_variant() {
    return "e512_topk10_task30_gemm2_small_m_in_kernel_fixed_fp32_finalize";
}

extern "C" int atrex_e512t10_task30_prepare() {
    return atrex_e512t10_task30_gemm2_small_m::ensure_kernel_attrs_prepared() ? 0 : 1;
}

extern "C" int64_t atrex_e512t10_task30_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    return atrex_e512t10_task30_gemm2_small_m::workspace_bytes(
        num_experts, N, K, expanded_num_tokens);
}

extern "C" void atrex_e512t10_task30_forward_fixed_e512t10(
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
    bool use_shared_sf_staging,
    void* workspace,
    void* expert_rows_bf16,
    float const* topk_weights,
    int* completion_counters,
    bool output_preinitialized,
    int const* topk_ids,
    cudaStream_t stream) {
    atrex_e512t10_task30_gemm2_small_m::forward_fixed_e512t10(
        a_fp4, b_fp4, sf_a, sf_b, alpha, final_output_bf16,
        expert_first_token_offset, permuted_row_to_unpermuted_row,
        sorted_scales, num_experts, N, K, M, expanded_num_tokens,
        use_shared_sf_staging, workspace, expert_rows_bf16, topk_weights,
        completion_counters, output_preinitialized, topk_ids, stream);
}
