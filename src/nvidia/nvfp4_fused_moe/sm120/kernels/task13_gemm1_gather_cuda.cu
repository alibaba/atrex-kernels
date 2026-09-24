#include "task13_gemm1_gather_fused_act_sm120_v1.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

extern bool g_enable_pdl;

namespace atrex_task13_gemm1_gather {

namespace v1 = atrex_task13_gemm1_gather_v1;

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
                     "[task13_cuda_v1] cudaFuncSetAttribute failed for %s: %s\n",
                     label, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return true;
}

static bool ensure_kernel_attrs_prepared() {
    bool ok = true;
    ok = set_dynamic_smem_attr(
             v1::atrex_grouped_gemm_nvfp4_kernel<false, true>, v1::SMEM_BYTES,
             "grouped_gemm_nvfp4_kernel_nvfp4_input") && ok;
    ok = set_dynamic_smem_attr(
             v1::atrex_grouped_gemm_nvfp4_kernel<false, false>, v1::SMEM_BYTES,
             "grouped_gemm_nvfp4_kernel_bf16_input") && ok;
    return ok;
}

int64_t cuda_workspace_bytes(int num_experts, int N, int64_t expanded_num_tokens) {
    (void)N;
    (void)expanded_num_tokens;
    int64_t bytes = static_cast<int64_t>(num_experts + 2) * sizeof(int);
    return align256(bytes);
}

static bool encode_tma_b_3d(
    CUtensorMap* desc,
    void const* b_fp4,
    int num_experts,
    int N,
    int K_half) {
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
    boxDim[0] = v1::ROW_BYTES;
    boxDim[1] = v1::BLOCK_N;
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
                     "[task13_cuda_v1] TMA B 3D encode failed: %d "
                     "(B=%p E=%d N=%d K_half=%d)\n",
                     static_cast<int>(res), b_fp4, num_experts, N, K_half);
        return false;
    }
    return true;
}

void forward_gather_fused_act_v1(
    void const* hidden_states,
    void const* input_sf,
    void const* b_fp4,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    float const* fc1_act_global_scale,
    int const* permuted_source_rows,
    int64_t const* expert_first_token_offset,
    int num_tokens,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    bool input_is_nvfp4) {
    int K_half = K / 2;
    int* d_total_tiles = reinterpret_cast<int*>(workspace);
    int* d_tile_prefix_sums = d_total_tiles + 1;

    int n_tiles = (N + v1::BLOCK_N - 1) / v1::BLOCK_N;
    int full_upper_bound_tiles =
        (((int)expanded_num_tokens + v1::BLOCK_M - 1) / v1::BLOCK_M +
         num_experts) *
        n_tiles;
    int upper_bound_tiles = full_upper_bound_tiles;

    CUtensorMap h_tma_b;
    if (!encode_tma_b_3d(&h_tma_b, b_fp4, num_experts, N, K_half)) {
        return;
    }

    if (upper_bound_tiles == 0) {
        return;
    }

    if (!ensure_kernel_attrs_prepared()) {
        return;
    }

    int tile_threads = num_experts < 1024 ? num_experts : 1024;
    int tile_smem = num_experts * static_cast<int>(sizeof(int));

    cudaLaunchConfig_t tile_config = {};
    tile_config.gridDim = 1;
    tile_config.blockDim = tile_threads;
    tile_config.dynamicSmemBytes = tile_smem;
    tile_config.stream = stream;
    cudaLaunchAttribute tile_attrs[1];
    tile_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    tile_attrs[0].val.programmaticStreamSerializationAllowed = g_enable_pdl;
    tile_config.numAttrs = 1;
    tile_config.attrs = tile_attrs;

    cudaLaunchKernelEx(&tile_config, v1::atrex_compute_tile_info_kernel,
                       expert_first_token_offset, d_tile_prefix_sums,
                       d_total_tiles, num_experts, N);

    int smem = v1::SMEM_BYTES;

    {
        cudaLaunchConfig_t config = {};
        config.gridDim = upper_bound_tiles;
        config.blockDim = v1::THREADS;
        config.dynamicSmemBytes = smem;
        config.stream = stream;

        if (input_is_nvfp4 && input_sf != nullptr) {
            cudaLaunchKernelEx(&config, v1::atrex_grouped_gemm_nvfp4_kernel<false, true>,
                               hidden_states,
                               reinterpret_cast<const uint8_t*>(input_sf),
                               h_tma_b,
                               reinterpret_cast<const uint8_t*>(sf_b), alpha,
                               reinterpret_cast<uint8_t*>(output_fp4),
                               fc2_act_global_scale,
                               reinterpret_cast<uint8_t*>(fc2_act_sf),
                               fc1_act_global_scale,
                               permuted_source_rows,
                               expert_first_token_offset, d_tile_prefix_sums,
                               d_total_tiles, num_experts, num_tokens, N, K, 0);
        } else {
            cudaLaunchKernelEx(&config, v1::atrex_grouped_gemm_nvfp4_kernel<false, false>,
                               hidden_states, nullptr, h_tma_b,
                               reinterpret_cast<const uint8_t*>(sf_b), alpha,
                               reinterpret_cast<uint8_t*>(output_fp4),
                               fc2_act_global_scale,
                               reinterpret_cast<uint8_t*>(fc2_act_sf),
                               fc1_act_global_scale,
                               permuted_source_rows,
                               expert_first_token_offset, d_tile_prefix_sums,
                               d_total_tiles, num_experts, num_tokens, N, K, 0);
        }
    }
}

}  // namespace atrex_task13_gemm1_gather

extern "C" char const* task13_gemm1_gather_fused_act_variant() {
    return "task13_gather_bf16_or_nvfp4_fused_act_b_tma_l2_256B_tile_80x256x128_stage2";
}

extern "C" int task13_gemm1_gather_fused_act_prepare() {
    return atrex_task13_gemm1_gather::ensure_kernel_attrs_prepared() ? 0 : 1;
}

extern "C" int64_t task13_gemm1_gather_fused_act_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens) {
    (void)K;
    return atrex_task13_gemm1_gather::cuda_workspace_bytes(
        num_experts, N, expanded_num_tokens);
}

extern "C" int64_t task13_gather_fc2_act_sf_bytes(
    int num_experts,
    int inter_size,
    int64_t expanded_num_tokens) {
    int64_t padded_expanded_sf = atrex_task13_gemm1_gather_v1::sf::align_to(
        static_cast<int>(expanded_num_tokens +
                         static_cast<int64_t>(num_experts) *
                             (atrex_task13_gemm1_gather_v1::sf::MIN_N - 1)),
        atrex_task13_gemm1_gather_v1::sf::MIN_N);
    int64_t padded_inter = atrex_task13_gemm1_gather_v1::sf::align_to(
        inter_size, atrex_task13_gemm1_gather_v1::sf::MIN_K);
    return atrex_task13_gemm1_gather::align256(
        padded_expanded_sf * padded_inter /
        atrex_task13_gemm1_gather_v1::sf::NVFP4_BLOCK);
}

extern "C" void task13_gemm1_gather_fused_act_forward(
    void const* hidden_states,
    void const* input_sf,
    void const* b_fp4,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    float const* fc1_act_global_scale,
    int const* permuted_source_rows,
    int64_t const* expert_first_token_offset,
    int num_tokens,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    bool input_is_nvfp4) {
    atrex_task13_gemm1_gather::forward_gather_fused_act_v1(
        hidden_states, input_sf, b_fp4, sf_b, alpha, output_fp4,
        fc2_act_global_scale, fc2_act_sf,
        fc1_act_global_scale, permuted_source_rows,
        expert_first_token_offset, num_tokens, num_experts, N, K,
        expanded_num_tokens, workspace, stream, input_is_nvfp4);
}
