#include "moe_common.cuh"

// ============================================================================
// finalizeMoeRoutingKernel — multi-row version
//
// Each block processes ROWS_PER_BLOCK output rows.
// With M=6000 and ROWS_PER_BLOCK=8: 750 blocks → 1 wave at occ=1 or 2 waves at occ=2.
// ============================================================================

// When Accumulate=true, the kernel treats the existing contents of
// reduced_unpermuted_output as a pre-initialized base (e.g. shared-experts
// output written by a prior op on this stream — Python side has done the
// cross-stream sync via shared_event before launch) and adds the routed-
// experts topk sum on top. Each (row, vec8) cell is written by exactly one
// thread, so a plain load-add-store is race-free; no atomics needed. When
// Accumulate=false the behavior is identical to the original overwrite
// kernel.
template <int TOPK, int ROWS_PER_BLOCK, bool Accumulate>
__global__ __launch_bounds__(256)
void atrex_finalizeMoeRoutingKernel_opt(
    __nv_bfloat16 const* __restrict__ expanded_permuted_rows,
    __nv_bfloat16* __restrict__ reduced_unpermuted_output,
    float const* __restrict__ scales,
    int const* __restrict__ unpermuted_row_to_permuted_row,
    int const* __restrict__ token_selected_experts,
    int const padded_cols,
    int const unpadded_cols,
    int const num_rows,
    int const num_experts_per_node,
    int const start_expert_id)
{
    int const base_row = blockIdx.x * ROWS_PER_BLOCK;
    int const M = num_rows;
    int const tid = threadIdx.x;

    constexpr int ELEM_SIZE = 8;
    int const num_elems = unpadded_cols / ELEM_SIZE;
    int const padded_elems = padded_cols / ELEM_SIZE;

    using Vec8 = Array<__nv_bfloat16, 8>;

    pdl_wait();

    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int original_row = base_row + r;
        if (original_row >= M) break;

        __nv_bfloat16 const* row_ptrs[TOPK];
        float row_scales[TOPK];
        int num_valid_k = 0;

        // Match routing_sort's expert-range guard before reading unperm_map:
        // vLLM dummy-run topk_ids == -1 slots are skipped by routing_sort.
        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            int topk_idx = original_row * TOPK + k;
            int selected_expert =
                token_selected_experts[topk_idx] - start_expert_id;
            if (selected_expert < 0 ||
                selected_expert >= num_experts_per_node) continue;

            int expanded_orig = original_row + k * M;
            int permuted_row = unpermuted_row_to_permuted_row[expanded_orig];
            if (permuted_row < 0 || permuted_row >= M * TOPK) continue;
            row_ptrs[num_valid_k] = expanded_permuted_rows +
                                    (int64_t)permuted_row * padded_cols;
            row_scales[num_valid_k] = scales[topk_idx];
            num_valid_k++;
        }

        __nv_bfloat16* out_row = reduced_unpermuted_output +
                                  (int64_t)original_row * unpadded_cols;

        for (int ei = tid; ei < num_elems; ei += 256) {
            float acc[8];
            if constexpr (Accumulate) {
                Vec8 prev = reinterpret_cast<Vec8 const*>(out_row)[ei];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    acc[i] = __bfloat162float(prev[i]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[i] = 0.f;
            }

            #pragma unroll 4
            for (int k = 0; k < num_valid_k; k++) {
                Vec8 v = reinterpret_cast<Vec8 const*>(row_ptrs[k])[ei];
                float s = row_scales[k];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    acc[i] += s * __bfloat162float(v[i]);
            }

            Vec8 out;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                out[i] = __float2bfloat16(acc[i]);
            reinterpret_cast<Vec8*>(out_row)[ei] = out;
        }
    }

    pdl_launch_dependents();
}

// ============================================================================
// Extern C launcher
// ============================================================================

// Declared in fused_moe_forward.cu. Reads the same thread-local flag that
// fused_moe_set_output_preinitialized() writes. We branch on it here so the
// hybrid_v1/v2/v3/v4/v5 small-M (M < FUSED_FINALIZE_M_THRESHOLD) paths
// inherit the same preinit-accumulate semantics that the M >= 2048 fused-
// finalize path already has via cutlass red.add — without changing any
// caller's signature.
extern "C" int fused_moe_get_output_preinitialized();

extern "C" void finalize_moe_routing(
    void const* expanded_permuted_rows,
    void* reduced_unpermuted_output,
    void const* bias,
    float const* scales,
    int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts,
    int64_t num_rows,
    int64_t padded_cols,
    int64_t unpadded_cols,
    int64_t experts_per_token,
    int num_experts_per_node,
    int start_expert_id,
    cudaStream_t stream)
{
    bool enable_pdl = g_enable_pdl;

    constexpr int ROWS_PER_BLOCK = 4;
    int blocks = ((int)num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = 256;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    bool accumulate = fused_moe_get_output_preinitialized() != 0;

    if (accumulate) {
        cudaLaunchKernelEx(&config,
            atrex_finalizeMoeRoutingKernel_opt<8, ROWS_PER_BLOCK, true>,
            reinterpret_cast<__nv_bfloat16 const*>(expanded_permuted_rows),
            reinterpret_cast<__nv_bfloat16*>(reduced_unpermuted_output),
            scales,
            unpermuted_row_to_permuted_row,
            token_selected_experts,
            (int)padded_cols, (int)unpadded_cols, (int)num_rows,
            num_experts_per_node, start_expert_id);
    } else {
        cudaLaunchKernelEx(&config,
            atrex_finalizeMoeRoutingKernel_opt<8, ROWS_PER_BLOCK, false>,
            reinterpret_cast<__nv_bfloat16 const*>(expanded_permuted_rows),
            reinterpret_cast<__nv_bfloat16*>(reduced_unpermuted_output),
            scales,
            unpermuted_row_to_permuted_row,
            token_selected_experts,
            (int)padded_cols, (int)unpadded_cols, (int)num_rows,
            num_experts_per_node, start_expert_id);
    }
}
