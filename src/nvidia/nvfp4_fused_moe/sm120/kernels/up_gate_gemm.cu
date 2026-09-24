#include "gemm_sm120_v20.cuh"
#include "gemm_sm120_v20_fused_act.cuh"

#include <cstdio>

namespace {

bool check_cu_result(CUresult status, const char* op) {
    if (status == CUDA_SUCCESS) {
        return true;
    }
    const char* name = nullptr;
    const char* str = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &str);
    std::printf("[hybrid_v3_up_gate] %s failed: %d (%s: %s)\n",
                op, static_cast<int>(status),
                name ? name : "unknown", str ? str : "unknown");
    return false;
}

bool check_cuda_result(cudaError_t status, const char* op) {
    if (status == cudaSuccess) {
        return true;
    }
    std::printf("[hybrid_v3_up_gate] %s failed: %s\n",
                op, cudaGetErrorString(status));
    return false;
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

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<void*>(a_fp4),
        globalDim,
        globalStride,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (!check_cu_result(status, tag)) {
        std::printf("[hybrid_v3_up_gate] A 2D TMA args: A=%p expanded=%lld K_half=%d "
                    "row_bytes=%u block_m=%u\n",
                    a_fp4, static_cast<long long>(expanded_num_tokens), K_half,
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
    globalStride[1] = static_cast<uint64_t>(N) * static_cast<uint64_t>(K_half);
    boxDim[0] = row_bytes;
    boxDim[1] = block_n;
    boxDim[2] = 1;
    elemStride[0] = 1;
    elemStride[1] = 1;
    elemStride[2] = 1;

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        const_cast<void*>(b_fp4),
        globalDim,
        globalStride,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (!check_cu_result(status, tag)) {
        std::printf("[hybrid_v3_up_gate] B 3D TMA args: B=%p E=%d N=%d K_half=%d "
                    "row_bytes=%u block_n=%u stride1=%llu\n",
                    b_fp4, num_experts, N, K_half, row_bytes, block_n,
                    static_cast<unsigned long long>(globalStride[1]));
        return false;
    }
    return true;
}

} // namespace

// ============================================================================
// extern "C" launcher — v20 fused act (TMA+ldmatrix body + SwiGLU+FP4 epilogue)
//
// workspace layout:
//   [0 .. 4)                              total_tiles (int)
//   [4 .. 4 + max_tiles*12)               tile_info (int[max_tiles][3])
// ============================================================================

extern "C" void gemm_forward_v20_fused_act(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    void* act_sf_flat,
    float const* fc2_act_global_scale,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N_gemm,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream)
{
    namespace v20fa = atrex_gemm_v20_fused_act;

    int K_half = K / 2;
    int* d_total_tiles = reinterpret_cast<int*>(workspace);
    int* d_tile_info   = d_total_tiles + 1;

    int upper_bound_tiles = ((int)(expanded_num_tokens / v20fa::BLOCK_M) + num_experts) *
                            ((N_gemm + v20fa::BLOCK_N - 1) / v20fa::BLOCK_N);

    CUtensorMap h_tma_a;
    CUtensorMap h_tma_b;
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,
                         v20fa::ROW_BYTES, v20fa::BLOCK_M,
                         "cuTensorMapEncodeTiled(A fused_act)")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N_gemm, K_half,
                         v20fa::ROW_BYTES, v20fa::BLOCK_N,
                         "cuTensorMapEncodeTiled(B fused_act 3D)")) {
        return;
    }

    // Tile info kernel (reuse v20's)
    {
        int tile_threads = num_experts < 1024 ? num_experts : 1024;
        int tile_smem = (num_experts * 2 + 1) * (int)sizeof(int);

        cudaLaunchConfig_t config = {};
        config.gridDim = 1;
        config.blockDim = tile_threads;
        config.dynamicSmemBytes = tile_smem;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = g_enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaError_t status = cudaLaunchKernelEx(&config, atrex_gemm_v20::atrex_compute_tile_info_v20_kernel,
            expert_first_token_offset,
            d_tile_info,
            d_total_tiles,
            num_experts, N_gemm);
        if (!check_cuda_result(status, "atrex_compute_tile_info_v20_kernel(fused_act) launch")) {
            return;
        }
    }

    if (upper_bound_tiles == 0) return;

    int smem = v20fa::SMEM_BYTES;
    if (smem > 48 * 1024) {
        cudaError_t status = cudaFuncSetAttribute(v20fa::atrex_grouped_gemm_v20_fused_act_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (!check_cuda_result(status, "atrex_grouped_gemm_v20_fused_act_kernel set smem")) {
            return;
        }
    }

    {
        cudaLaunchConfig_t config = {};
        config.gridDim = upper_bound_tiles;
        config.blockDim = v20fa::THREADS;
        config.dynamicSmemBytes = smem;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = g_enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaError_t status = cudaLaunchKernelEx(&config, v20fa::atrex_grouped_gemm_v20_fused_act_kernel,
            h_tma_a,
            h_tma_b,
            reinterpret_cast<const uint8_t*>(sf_a),
            reinterpret_cast<const uint8_t*>(sf_b),
            alpha,
            reinterpret_cast<uint8_t*>(output_fp4),
            reinterpret_cast<uint8_t*>(act_sf_flat),
            fc2_act_global_scale,
            expert_first_token_offset,
            d_tile_info,
            d_total_tiles,
            num_experts, N_gemm, K);
        if (!check_cuda_result(status, "atrex_grouped_gemm_v20_fused_act_kernel launch")) {
            return;
        }
    }
}

// ============================================================================
// v20 launcher — v19 + ldmatrix.x2 for B loads
// ============================================================================

extern "C" void gemm_forward_v20(
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
    cudaStream_t stream)
{
    namespace v20 = atrex_gemm_v20;

    int K_half = K / 2;
    int* d_total_tiles = reinterpret_cast<int*>(workspace);
    int* d_tile_info   = d_total_tiles + 1;

    int upper_bound_tiles = ((int)(expanded_num_tokens / v20::BLOCK_M) + num_experts) *
                            ((N + v20::BLOCK_N - 1) / v20::BLOCK_N);

    CUtensorMap h_tma_a;
    CUtensorMap h_tma_b;
    if (!encode_tma_a_2d(&h_tma_a, a_fp4, expanded_num_tokens, K_half,
                         v20::ROW_BYTES, v20::BLOCK_M,
                         "cuTensorMapEncodeTiled(A v20)")) {
        return;
    }
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half,
                         v20::ROW_BYTES, v20::BLOCK_N,
                         "cuTensorMapEncodeTiled(B v20 3D)")) {
        return;
    }

    {
        int tile_threads = num_experts < 1024 ? num_experts : 1024;
        int tile_smem = (num_experts * 2 + 1) * (int)sizeof(int);

        cudaLaunchConfig_t config = {};
        config.gridDim = 1;
        config.blockDim = tile_threads;
        config.dynamicSmemBytes = tile_smem;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = g_enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaError_t status = cudaLaunchKernelEx(&config, v20::atrex_compute_tile_info_v20_kernel,
            expert_first_token_offset,
            d_tile_info,
            d_total_tiles,
            num_experts, N);
        if (!check_cuda_result(status, "atrex_compute_tile_info_v20_kernel launch")) {
            return;
        }
    }

    if (upper_bound_tiles == 0) return;

    int smem = v20::SMEM_BYTES;
    if (smem > 48 * 1024) {
        cudaError_t status = cudaFuncSetAttribute(v20::atrex_grouped_gemm_nvfp4_v20_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (!check_cuda_result(status, "atrex_grouped_gemm_nvfp4_v20_kernel set smem")) {
            return;
        }
    }

    {
        cudaLaunchConfig_t config = {};
        config.gridDim = upper_bound_tiles;
        config.blockDim = v20::THREADS;
        config.dynamicSmemBytes = smem;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = g_enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaError_t status = cudaLaunchKernelEx(&config, v20::atrex_grouped_gemm_nvfp4_v20_kernel,
            h_tma_a,
            h_tma_b,
            reinterpret_cast<const uint8_t*>(sf_a),
            reinterpret_cast<const uint8_t*>(sf_b),
            alpha,
            reinterpret_cast<__nv_bfloat16*>(output_bf16),
            expert_first_token_offset,
            d_tile_info,
            d_total_tiles,
            num_experts, N, K);
        if (!check_cuda_result(status, "atrex_grouped_gemm_nvfp4_v20_kernel launch")) {
            return;
        }
    }
}
