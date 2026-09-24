#include "moe_common.cuh"
#include <cub/cub.cuh>

// ============================================================================
// atrex_routingSortCompactSmallMKernel
// ============================================================================

template <int kNumExperts, int kMaxTokens, int kTopK>
__global__ void atrex_routingSortCompactSmallMKernel(
    int const* __restrict__ token_selected_experts,
    int64_t* __restrict__ expert_first_token_offset,
    int* __restrict__ permuted_token_selected_experts,
    int* __restrict__ permuted_row_to_unpermuted_row,
    int* __restrict__ unpermuted_row_to_permuted_row,
    int num_tokens,
    int num_experts_per_token,
    int start_expert_id,
    uint16_t* __restrict__ output_to_zero,
    int64_t output_numel,
    int* __restrict__ completion_counters_to_zero,
    int64_t completion_counter_numel,
    float const* __restrict__ a1_global_scale,
    float const* __restrict__ w1_global_scale,
    float* __restrict__ gemm1_alpha,
    float const* __restrict__ a2_global_scale,
    float const* __restrict__ w2_global_scale,
    float* __restrict__ gemm2_alpha)
{
    using BlockScan = cub::BlockScan<int, kNumExperts>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int selected_experts[kMaxTokens * kTopK];

    pdl_wait();

    int const expert = threadIdx.x;
    int const expanded = num_tokens * num_experts_per_token;
    if (expert < expanded) {
        selected_experts[expert] =
            token_selected_experts[expert] - start_expert_id;
    }

    // For task30-owned compact outputs, fold the BF16 memset into this
    // already-captured routing node. A zero output_numel preserves
    // accumulation into a buffer produced on another stream.
    for (int64_t idx = expert; idx < output_numel; idx += kNumExperts) {
        output_to_zero[idx] = 0;
    }
    for (int64_t idx = expert; idx < completion_counter_numel;
         idx += kNumExperts) {
        completion_counters_to_zero[idx] = 0;
    }
    // These per-expert scales are static-sized metadata. Producing them in
    // routing removes four elementwise CUDA Graph nodes per decode layer.
    if (gemm1_alpha != nullptr) {
        gemm1_alpha[expert] =
            1.0f / (a1_global_scale[expert] * w1_global_scale[expert]);
        gemm2_alpha[expert] =
            1.0f / (a2_global_scale[expert] * w2_global_scale[expert]);
    }
    __syncthreads();

    // torch.topk produces unique expert IDs within each token. A 32-bit mask
    // therefore represents all rows owned by this expert for compact M.
    uint32_t token_mask = 0;
    for (int token = 0; token < num_tokens; ++token) {
        int const token_base = token * num_experts_per_token;
        for (int k_rank = 0; k_rank < num_experts_per_token; ++k_rank) {
            if (selected_experts[token_base + k_rank] == expert) {
                token_mask |= 1u << token;
                break;
            }
        }
    }

    int const count = __popc(token_mask);
    int expert_start;
    BlockScan(temp_storage).ExclusiveSum(count, expert_start);

    expert_first_token_offset[expert] = expert_start;
    if (expert == kNumExperts - 1) {
        expert_first_token_offset[kNumExperts] = expert_start + count;
    }

    // Stable scatter in token order within each expert. The public
    // unpermuted-row convention is topk-major: k_rank * M + token.
    int local_row = 0;
    uint32_t remaining = token_mask;
    while (remaining != 0) {
        int const source_token = __ffs(static_cast<int>(remaining)) - 1;
        remaining &= remaining - 1;
        int const token_base = source_token * num_experts_per_token;
        int source_k_rank = 0;
        for (; source_k_rank < num_experts_per_token; ++source_k_rank) {
            if (selected_experts[token_base + source_k_rank] == expert) {
                break;
            }
        }
        int const permuted_row = expert_start + local_row++;
        int const unpermuted_row = source_k_rank * num_tokens + source_token;
        permuted_token_selected_experts[permuted_row] = expert;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }

    __syncthreads();
    pdl_launch_dependents();
}

// ============================================================================
// atrex_blockExpertPrefixSumKernel
// ============================================================================

template <int kNumTokensPerBlock, bool kFuseAux>
__global__ void atrex_blockExpertPrefixSumKernel(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_token,
    int const start_expert_id,
    uint16_t* __restrict__ output_to_zero,
    int64_t output_numel,
    int* __restrict__ completion_counters_to_zero,
    int64_t completion_counter_numel,
    float const* __restrict__ a1_global_scale,
    float const* __restrict__ w1_global_scale,
    float* __restrict__ gemm1_alpha,
    float const* __restrict__ a2_global_scale,
    float const* __restrict__ w2_global_scale,
    float* __restrict__ gemm2_alpha)
{
    using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

    pdl_wait();

    // Reuse the generic routing grid for fixed per-layer work without
    // changing its expert-parallel CTA topology. Each output element is
    // owned by exactly one routing thread across the full 2-D grid.
    if constexpr (kFuseAux) {
        if (output_numel > 0) {
            int64_t const linear_thread =
                (static_cast<int64_t>(block_id) * gridDim.x +
                 target_expert_id) * blockDim.x + threadIdx.x;
            int64_t const total_threads =
                static_cast<int64_t>(gridDim.x) * gridDim.y * blockDim.x;
            for (int64_t idx = linear_thread; idx < output_numel;
                 idx += total_threads) {
                output_to_zero[idx] = 0;
            }
        }
        if (completion_counter_numel > 0) {
            int64_t const linear_thread =
                (static_cast<int64_t>(block_id) * gridDim.x +
                 target_expert_id) * blockDim.x + threadIdx.x;
            int64_t const total_threads =
                static_cast<int64_t>(gridDim.x) * gridDim.y * blockDim.x;
            for (int64_t idx = linear_thread;
                 idx < completion_counter_numel; idx += total_threads) {
                completion_counters_to_zero[idx] = 0;
            }
        }

        // One thread in the first token block owns the two alpha values for
        // its expert. Routing remains fully parallel across experts.
        if (gemm1_alpha != nullptr && block_id == 0 && threadIdx.x == 0) {
            gemm1_alpha[target_expert_id] =
                1.0f / (a1_global_scale[target_expert_id] *
                        w1_global_scale[target_expert_id]);
            gemm2_alpha[target_expert_id] =
                1.0f / (a2_global_scale[target_expert_id] *
                        w2_global_scale[target_expert_id]);
        }
    }

    int expanded_token_id = -1;
    if (token_id < num_tokens) {
        for (int i = 0; i < num_experts_per_token; i++) {
            int const expert_id =
                token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
            if (expert_id == target_expert_id) {
                expanded_token_id = i * num_tokens + token_id;
                break;
            }
        }
    }

    int const has_matched = expanded_token_id >= 0 ? 1 : 0;
    int index;
    BlockScan(temp_storage).ExclusiveSum(has_matched, index);

    if (has_matched) {
        blocked_row_to_unpermuted_row[target_expert_id * num_tokens +
                                      block_id * kNumTokensPerBlock + index] = expanded_token_id;
    }
    if (threadIdx.x == kNumTokensPerBlock - 1) {
        blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] =
            index + has_matched;
    }

    pdl_launch_dependents();
}

// ============================================================================
// atrex_globalExpertPrefixSumLargeKernel
// ============================================================================

template <int kNumThreadsPerBlock>
__global__ void atrex_globalExpertPrefixSumLargeKernel(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int64_t const num_experts_per_node,
    int64_t const num_blocks_per_seq,
    int64_t const num_elem_per_thread)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int offset = threadIdx.x * num_elem_per_thread;
    int cnt = 0;

    pdl_wait();

    for (int i = 0; i < num_elem_per_thread; i++) {
        if (offset + i < num_experts_per_node * num_blocks_per_seq) {
            cnt += blocked_expert_counts[offset + i];
        }
    }

    int cumsum;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    for (int i = 0; i < num_elem_per_thread; i++) {
        if (offset + i < num_experts_per_node * num_blocks_per_seq) {
            blocked_expert_counts_cumsum[offset + i] = cumsum;
            if ((offset + i) % num_blocks_per_seq == 0) {
                expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
            }
            cumsum += blocked_expert_counts[offset + i];
            if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
                expert_first_token_offset[num_experts_per_node] = cumsum;
            }
        }
    }

    pdl_launch_dependents();
}

// ============================================================================
// atrex_globalExpertPrefixSumKernel (small variant)
// ============================================================================

template <int kNumThreadsPerBlock>
__global__ void atrex_globalExpertPrefixSumKernel(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int64_t const num_experts_per_node,
    int64_t const num_blocks_per_seq)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    pdl_wait();

    int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq
                        ? blocked_expert_counts[threadIdx.x]
                        : 0;
    int cumsum;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    if (threadIdx.x < num_experts_per_node * num_blocks_per_seq) {
        blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
        if (threadIdx.x % num_blocks_per_seq == 0) {
            expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
        }
        if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1) {
            expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
        }
    }

    pdl_launch_dependents();
}

// ============================================================================
// atrex_mergeExpertPrefixSumKernel
// ============================================================================

__global__ void atrex_mergeExpertPrefixSumKernel(
    int const* blocked_expert_counts,
    int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int const num_tokens)
{
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * blockDim.x + threadIdx.x;

    pdl_wait();

    int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    int const offset =
        blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
    if (threadIdx.x < cnt) {
        int const unpermuted_row =
            blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
        int const permuted_row = offset + threadIdx.x;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        permuted_token_selected_experts[permuted_row] = target_expert_id;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }

    pdl_launch_dependents();
}

__global__ void atrex_mergeExpertPrefixSumWithScalesKernel(
    int const* blocked_expert_counts,
    int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row,
    float const* topk_weights,
    int* permuted_source_rows,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    float* permuted_scales,
    int const num_tokens,
    int const topk)
{
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * blockDim.x + threadIdx.x;

    pdl_wait();

    int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    int const offset =
        blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
    if (threadIdx.x < cnt) {
        int const unpermuted_row =
            blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
        int const permuted_row = offset + threadIdx.x;
        int const source_row = unpermuted_row % num_tokens;
        int const source_k_rank = unpermuted_row / num_tokens;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        permuted_source_rows[permuted_row] = source_row;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        if (permuted_scales) {
            permuted_scales[permuted_row] =
                topk_weights[source_row * topk + source_k_rank];
        }
    }

    pdl_launch_dependents();
}

__global__ void atrex_directRouteCountWithScalesKernel(
    int const* __restrict__ token_selected_experts,
    int* __restrict__ expert_counts,
    int num_tokens,
    int topk,
    int num_experts_per_node,
    int start_expert_id)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * topk;
    if (linear >= total) {
        return;
    }

    int expert = token_selected_experts[linear] - start_expert_id;
    if (expert >= 0 && expert < num_experts_per_node) {
        atomicAdd(&expert_counts[expert], 1);
    }
}

__global__ void atrex_directRoutePrefixWithScalesKernel(
    int const* __restrict__ expert_counts,
    int* __restrict__ expert_cursors,
    int64_t* __restrict__ expert_first_token_offset,
    int num_experts_per_node)
{
    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert = threadIdx.x;
    int count = expert < num_experts_per_node ? expert_counts[expert] : 0;
    int prefix;
    BlockScan(temp_storage).ExclusiveSum(count, prefix);

    if (expert < num_experts_per_node) {
        expert_first_token_offset[expert] = prefix;
        expert_cursors[expert] = prefix;
    }
    if (expert == num_experts_per_node - 1) {
        expert_first_token_offset[num_experts_per_node] = prefix + count;
    }
}

__global__ void atrex_directRouteScatterWithScalesKernel(
    int const* __restrict__ token_selected_experts,
    float const* __restrict__ topk_weights,
    int* __restrict__ expert_cursors,
    int* __restrict__ permuted_source_rows,
    int* __restrict__ permuted_row_to_unpermuted_row,
    int* __restrict__ unpermuted_row_to_permuted_row,
    float* __restrict__ permuted_scales,
    int num_tokens,
    int topk,
    int num_experts_per_node,
    int start_expert_id)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * topk;
    if (linear >= total) {
        return;
    }

    int expert = token_selected_experts[linear] - start_expert_id;
    if (expert < 0 || expert >= num_experts_per_node) {
        return;
    }

    int source_row = linear / topk;
    int permuted_row = atomicAdd(&expert_cursors[expert], 1);

    permuted_source_rows[permuted_row] = source_row;
    if (num_tokens >= 2048) {
        permuted_row_to_unpermuted_row[permuted_row] = source_row;
    } else {
        int source_k_rank = linear - source_row * topk;
        int unpermuted_row = source_k_rank * num_tokens + source_row;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }
    if (permuted_scales) {
        permuted_scales[permuted_row] = topk_weights[linear];
    }
}

// ============================================================================
// Host helper
// ============================================================================

static int64_t computeNumTokensPerBlock(int64_t num_tokens, int64_t num_experts_per_node) {
    for (int64_t n = 32; n <= 1024; n *= 2) {
        int64_t num_blocks = ceilDiv(num_tokens, n);
        if (num_blocks * num_experts_per_node <= n) return n;
    }
    return 1024;
}

// ============================================================================
// Extern C launchers
// ============================================================================

extern "C" void routing_sort(
    int const* token_selected_experts,  // [M, topk]
    int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row,
    int64_t* expert_first_token_offset, // [E+1]
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t num_tokens,
    int64_t num_experts_per_node,
    int64_t num_experts_per_token,
    int start_expert_id,
    uint16_t* output_to_zero,
    int64_t output_numel,
    int* completion_counters_to_zero,
    int64_t completion_counter_numel,
    float const* a1_global_scale,
    float const* w1_global_scale,
    float* gemm1_alpha,
    float const* a2_global_scale,
    float const* w2_global_scale,
    float* gemm2_alpha,
    cudaStream_t stream)
{
    bool enable_pdl = g_enable_pdl;

    bool const compact_requested =
        num_tokens == 1 || output_numel > 0 ||
        completion_counter_numel > 0 || gemm1_alpha != nullptr;
    if (compact_requested && num_tokens >= 1 && num_tokens <= 16 &&
        num_experts_per_node == 512 && num_experts_per_token == 10) {
        cudaLaunchConfig_t config = {};
        config.gridDim = 1;
        config.blockDim = 512;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(
            &config, atrex_routingSortCompactSmallMKernel<512, 16, 10>,
            token_selected_experts,
            expert_first_token_offset, permuted_token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            static_cast<int>(num_tokens),
            static_cast<int>(num_experts_per_token), start_expert_id,
            output_to_zero, output_numel,
            completion_counters_to_zero, completion_counter_numel,
            a1_global_scale, w1_global_scale,
            gemm1_alpha, a2_global_scale, w2_global_scale, gemm2_alpha);
        return;
    }

    int64_t num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
    int64_t num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);

    // Step 1: blockExpertPrefixSum
    {
        dim3 blocks(num_experts_per_node, num_blocks_per_seq);
        dim3 threads(num_tokens_per_block);

        cudaLaunchConfig_t config = {};
        config.gridDim = blocks;
        config.blockDim = threads;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        bool const fuse_aux = output_numel > 0 ||
                              completion_counter_numel > 0 ||
                              gemm1_alpha != nullptr;
#define SELECT_BLOCK_PREFIX(tokens)                                      \
        (fuse_aux ? atrex_blockExpertPrefixSumKernel<tokens, true>             \
                  : atrex_blockExpertPrefixSumKernel<tokens, false>)
        auto func = SELECT_BLOCK_PREFIX(1024);
        if (num_tokens_per_block <= 32) func = SELECT_BLOCK_PREFIX(32);
        else if (num_tokens_per_block <= 64) func = SELECT_BLOCK_PREFIX(64);
        else if (num_tokens_per_block <= 128) func = SELECT_BLOCK_PREFIX(128);
        else if (num_tokens_per_block <= 256) func = SELECT_BLOCK_PREFIX(256);
        else if (num_tokens_per_block <= 512) func = SELECT_BLOCK_PREFIX(512);
#undef SELECT_BLOCK_PREFIX

        cudaLaunchKernelEx(&config, func,
            token_selected_experts, blocked_expert_counts,
            blocked_row_to_unpermuted_row, num_tokens,
            num_experts_per_token, start_expert_id,
            output_to_zero, output_numel,
            completion_counters_to_zero, completion_counter_numel,
            a1_global_scale, w1_global_scale, gemm1_alpha,
            a2_global_scale, w2_global_scale, gemm2_alpha);
    }

    // Step 2: globalExpertPrefixSum
    {
        int64_t num_elements = num_experts_per_node * num_blocks_per_seq;

        cudaLaunchConfig_t config = {};
        config.gridDim = 1;
        config.blockDim = 1024;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        if (num_elements <= 1024) {
            auto func = atrex_globalExpertPrefixSumKernel<1024>;
            if (num_elements <= 32) { func = atrex_globalExpertPrefixSumKernel<32>; config.blockDim = 32; }
            else if (num_elements <= 64) { func = atrex_globalExpertPrefixSumKernel<64>; config.blockDim = 64; }
            else if (num_elements <= 128) { func = atrex_globalExpertPrefixSumKernel<128>; config.blockDim = 128; }
            else if (num_elements <= 256) { func = atrex_globalExpertPrefixSumKernel<256>; config.blockDim = 256; }
            else if (num_elements <= 512) { func = atrex_globalExpertPrefixSumKernel<512>; config.blockDim = 512; }
            cudaLaunchKernelEx(&config, func,
                blocked_expert_counts, blocked_expert_counts_cumsum,
                expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
        } else {
            int64_t num_elem_per_thread = ceilDiv(num_elements, (int64_t)1024);
            cudaLaunchKernelEx(&config, atrex_globalExpertPrefixSumLargeKernel<1024>,
                blocked_expert_counts, blocked_expert_counts_cumsum,
                expert_first_token_offset, num_experts_per_node, num_blocks_per_seq,
                num_elem_per_thread);
        }
    }

    // Step 3: mergeExpertPrefixSum
    {
        dim3 blocks(num_experts_per_node, num_blocks_per_seq);
        dim3 threads(num_tokens_per_block);

        cudaLaunchConfig_t config = {};
        config.gridDim = blocks;
        config.blockDim = threads;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, atrex_mergeExpertPrefixSumKernel,
            blocked_expert_counts, blocked_expert_counts_cumsum,
            blocked_row_to_unpermuted_row,
            permuted_token_selected_experts, permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row, (int)num_tokens);
    }
}

extern "C" void routing_sort_with_scales(
    int const* token_selected_experts,  // [M, topk]
    float const* topk_weights,          // [M, topk]
    int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row,
    int64_t* expert_first_token_offset, // [E+1]
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    float* permuted_scales,
    int64_t num_tokens,
    int64_t num_experts_per_node,
    int64_t num_experts_per_token,
    int start_expert_id,
    cudaStream_t stream)
{
    int total = static_cast<int>(num_tokens * num_experts_per_token);
    int threads = 1024;
    int blocks = ceilDiv(total, threads);

    cudaMemsetAsync(blocked_expert_counts, 0,
                    num_experts_per_node * sizeof(int), stream);

    atrex_directRouteCountWithScalesKernel<<<blocks, threads, 0, stream>>>(
        token_selected_experts, blocked_expert_counts,
        static_cast<int>(num_tokens),
        static_cast<int>(num_experts_per_token),
        static_cast<int>(num_experts_per_node),
        start_expert_id);

    atrex_directRoutePrefixWithScalesKernel<<<1, 256, 0, stream>>>(
        blocked_expert_counts, blocked_expert_counts_cumsum,
        expert_first_token_offset,
        static_cast<int>(num_experts_per_node));

    atrex_directRouteScatterWithScalesKernel<<<blocks, threads, 0, stream>>>(
        token_selected_experts, topk_weights, blocked_expert_counts_cumsum,
        permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, permuted_scales,
        static_cast<int>(num_tokens),
        static_cast<int>(num_experts_per_token),
        static_cast<int>(num_experts_per_node),
        start_expert_id);
}
