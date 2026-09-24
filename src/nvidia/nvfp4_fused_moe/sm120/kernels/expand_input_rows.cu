#include "moe_common.cuh"
#include "quantization_utils.cuh"

// ============================================================================
// Optimized expandInputRows — 128 threads, 16 bf16 per thread
//
// Key optimizations vs original:
//   1. Each thread processes 16 elements = 1 SF vector → no warp shuffle
//   2. Precomputed per-row SF base offset → eliminates get_sf_out_offset_128x4
//   3. Fused bf16→float conversion with absmax → no redundant bf162↔float roundtrips
// ============================================================================

__global__ __launch_bounds__(128, 8)
void atrex_expandInputRowsKernel_opt(
    __nv_bfloat16 const* __restrict__ unpermuted_input,
    uint8_t* __restrict__ permuted_output,
    float const* __restrict__ unpermuted_scales,
    float* __restrict__ permuted_scales,
    int const* __restrict__ permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const hidden_size,
    int const k,
    float const* __restrict__ fc1_act_global_scale,
    bool const use_per_expert_act_scale,
    int64_t const* __restrict__ expert_first_token_offset,
    uint8_t* __restrict__ fc1_act_sf_flat,
    int const num_experts_per_node,
    int const* __restrict__ token_expert_ids,
    bool const skip_sf_padding)
{
    pdl_wait();

    constexpr int SF_VEC = TmaConst::NVFP4BlockScaleVectorSize;  // 16
    constexpr int64_t MIN_N = TmaConst::MinNDimAlignmentNVFP4;   // 128

    int const tid = threadIdx.x;
    int const num_sf_vecs = hidden_size / SF_VEC;

    // SF layout constants (loop-invariant)
    int const sf_padded_K = TmaConst::alignToSfDim(hidden_size,
                                (int)TmaConst::MinKDimAlignmentNVFP4);
    int const sf_k_vecs = sf_padded_K / SF_VEC;
    int const sf_numKTiles = (sf_k_vecs + 3) / 4;
    int const sf_mTileStride = sf_numKTiles * 512;

    int const num_valid_tokens =
        (int)expert_first_token_offset[num_experts_per_node];

    for (int prow = blockIdx.x; prow < num_valid_tokens; prow += gridDim.x) {
        int const uprow = permuted_row_to_unpermuted_row[prow];
        int const source_row = uprow % num_tokens;
        int const source_k_rank = uprow / num_tokens;

        int const expert = token_expert_ids[prow];
        int const scale_idx = use_per_expert_act_scale ? expert : 0;
        float const global_scale = fc1_act_global_scale
                                   ? fc1_act_global_scale[scale_idx] : 1.0f;
        int64_t const tok_before = expert_first_token_offset[expert];

        // SF expert base (same as getOffsetActivationSF)
        int64_t const psf = TmaConst::alignToSfDim(
            (int)(tok_before + expert * (MIN_N - 1)), (int)MIN_N);
        uint8_t* sf_base = fc1_act_sf_flat + psf * sf_k_vecs;

        // Precomputed per-row SF offset within SWIZZLED_128x4 layout
        int const lr = (int)(prow - tok_before);
        int const sf_row_base = (lr / 128) * sf_mTileStride +
                                (lr % 32) * 16 +
                                ((lr % 128) / 32) * 4;

        __nv_bfloat16 const* src =
            unpermuted_input + (int64_t)source_row * hidden_size;
        uint64_t* dst = reinterpret_cast<uint64_t*>(permuted_output) +
                        (int64_t)prow * num_sf_vecs;

        for (int kv = tid; kv < num_sf_vecs; kv += 128) {
            // Load 16 bf16 as two 128-bit loads → 8 bf162 pairs
            __nv_bfloat162 p[8];
            *reinterpret_cast<uint4*>(&p[0]) =
                *reinterpret_cast<uint4 const*>(src + kv * SF_VEC);
            *reinterpret_cast<uint4*>(&p[4]) =
                *reinterpret_cast<uint4 const*>(src + kv * SF_VEC + 8);

            // Fused bf16→float conversion + absmax (no bf162 intermediary)
            float2 f2[8];
            float fmax = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                f2[i] = __bfloat1622float2(p[i]);
                fmax = fmaxf(fmax, fmaxf(fabsf(f2[i].x), fabsf(f2[i].y)));
            }

            // FP8 e4m3 scale factor
            float sv = global_scale *
                       (fmax * reciprocal_approximate_ftz(6.0f));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sv);
            uint8_t sf_val = sf8.__x;
            sv = static_cast<float>(sf8);
            float oscale = fmax != 0.f
                ? reciprocal_approximate_ftz(
                    sv * reciprocal_approximate_ftz(global_scale))
                : 0.f;

            // Write SF with precomputed offset
            sf_base[sf_row_base + (kv / 4) * 512 + (kv % 4)] = sf_val;

            // Scale and convert to FP4
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                f2[i].x *= oscale;
                f2[i].y *= oscale;
            }
            dst[kv] = fp32_vec_to_e2m1(f2);
        }

        // K-dim SF padding (for padded_hidden > hidden)
        for (int kv = num_sf_vecs + tid; kv < sf_k_vecs; kv += 128)
            sf_base[sf_row_base + (kv / 4) * 512 + (kv % 4)] = 0;

        // Copy permuted scale
        if (tid == 0 && permuted_scales) {
            int64_t idx = (int64_t)source_row * k + source_k_rank;
            permuted_scales[prow] = unpermuted_scales
                                    ? unpermuted_scales[idx] : 1.0f;
        }
    }

    pdl_launch_dependents();

    if (skip_sf_padding) {
        return;
    }

    // N-dim SF padding (expert token count → align to 128)
    int const npad = (int)MIN_N * num_experts_per_node;
    for (int pt = blockIdx.x; pt < npad; pt += gridDim.x) {
        int e = pt / (int)MIN_N;
        int64_t ts = expert_first_token_offset[e];
        int64_t te = expert_first_token_offset[e + 1];
        int toks = (int)(te - ts);
        int ptoks = TmaConst::alignToSfDim(toks, (int)MIN_N);
        int pcount = ptoks - toks;
        int pidx = pt % (int)MIN_N;

        if (pidx < pcount) {
            int64_t psf = TmaConst::alignToSfDim(
                (int)(ts + e * (int64_t)(MIN_N - 1)), (int)MIN_N);
            uint8_t* sb = fc1_act_sf_flat + psf * sf_k_vecs;

            int pr = toks + pidx;
            int srb = (pr / 128) * sf_mTileStride +
                      (pr % 32) * 16 +
                      ((pr % 128) / 32) * 4;

            for (int kv = tid; kv < sf_k_vecs; kv += 128)
                sb[srb + (kv / 4) * 512 + (kv % 4)] = 0;
        }
    }
}

// ============================================================================
// NVFP4 input path — permute pre-quantized FP4 data + reformat flat SF
// ============================================================================

__global__ __launch_bounds__(128, 8)
void atrex_expandInputRowsKernel_nvfp4(
    uint8_t const* __restrict__ unpermuted_fp4,        // [M, K/2] packed FP4
    uint8_t const* __restrict__ unpermuted_input_sf,   // [M, K/16] flat FP8 e4m3
    uint8_t* __restrict__ permuted_output,             // [M*topk, K/2] packed FP4
    float const* __restrict__ unpermuted_scales,
    float* __restrict__ permuted_scales,
    int const* __restrict__ permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const hidden_size,
    int const k,
    int64_t const* __restrict__ expert_first_token_offset,
    uint8_t* __restrict__ fc1_act_sf_flat,
    int const num_experts_per_node,
    int const* __restrict__ token_expert_ids,
    bool const skip_sf_padding)
{
    pdl_wait();

    constexpr int SF_VEC = TmaConst::NVFP4BlockScaleVectorSize;  // 16
    constexpr int64_t MIN_N = TmaConst::MinNDimAlignmentNVFP4;   // 128

    int const tid = threadIdx.x;
    int const num_sf_vecs = hidden_size / SF_VEC;

    int const sf_padded_K = TmaConst::alignToSfDim(hidden_size,
                                (int)TmaConst::MinKDimAlignmentNVFP4);
    int const sf_k_vecs = sf_padded_K / SF_VEC;
    int const sf_numKTiles = (sf_k_vecs + 3) / 4;
    int const sf_mTileStride = sf_numKTiles * 512;

    int const num_valid_tokens =
        (int)expert_first_token_offset[num_experts_per_node];

    for (int prow = blockIdx.x; prow < num_valid_tokens; prow += gridDim.x) {
        int const uprow = permuted_row_to_unpermuted_row[prow];
        int const source_row = uprow % num_tokens;
        int const source_k_rank = uprow / num_tokens;

        int const expert = token_expert_ids[prow];
        int64_t const tok_before = expert_first_token_offset[expert];

        int64_t const psf = TmaConst::alignToSfDim(
            (int)(tok_before + expert * (MIN_N - 1)), (int)MIN_N);
        uint8_t* sf_base = fc1_act_sf_flat + psf * sf_k_vecs;

        int const lr = (int)(prow - tok_before);
        int const sf_row_base = (lr / 128) * sf_mTileStride +
                                (lr % 32) * 16 +
                                ((lr % 128) / 32) * 4;

        uint64_t const* fp4_src = reinterpret_cast<uint64_t const*>(
            unpermuted_fp4) + (int64_t)source_row * num_sf_vecs;
        uint64_t* fp4_dst = reinterpret_cast<uint64_t*>(
            permuted_output) + (int64_t)prow * num_sf_vecs;
        uint8_t const* sf_src = unpermuted_input_sf +
            (int64_t)source_row * num_sf_vecs;

        for (int kv = tid; kv < num_sf_vecs; kv += 128) {
            fp4_dst[kv] = fp4_src[kv];
            sf_base[sf_row_base + (kv / 4) * 512 + (kv % 4)] = sf_src[kv];
        }

        for (int kv = num_sf_vecs + tid; kv < sf_k_vecs; kv += 128)
            sf_base[sf_row_base + (kv / 4) * 512 + (kv % 4)] = 0;

        if (tid == 0 && permuted_scales) {
            int64_t idx = (int64_t)source_row * k + source_k_rank;
            permuted_scales[prow] = unpermuted_scales
                                    ? unpermuted_scales[idx] : 1.0f;
        }
    }

    pdl_launch_dependents();

    if (skip_sf_padding) {
        return;
    }

    int const npad = (int)MIN_N * num_experts_per_node;
    for (int pt = blockIdx.x; pt < npad; pt += gridDim.x) {
        int e = pt / (int)MIN_N;
        int64_t ts = expert_first_token_offset[e];
        int64_t te = expert_first_token_offset[e + 1];
        int toks = (int)(te - ts);
        int ptoks = TmaConst::alignToSfDim(toks, (int)MIN_N);
        int pcount = ptoks - toks;
        int pidx = pt % (int)MIN_N;

        if (pidx < pcount) {
            int64_t psf = TmaConst::alignToSfDim(
                (int)(ts + e * (int64_t)(MIN_N - 1)), (int)MIN_N);
            uint8_t* sb = fc1_act_sf_flat + psf * sf_k_vecs;

            int pr = toks + pidx;
            int srb = (pr / 128) * sf_mTileStride +
                      (pr % 32) * 16 +
                      ((pr % 128) / 32) * 4;

            for (int kv = tid; kv < sf_k_vecs; kv += 128)
                sb[srb + (kv / 4) * 512 + (kv % 4)] = 0;
        }
    }
}

// ============================================================================
// Extern C launcher
// ============================================================================

extern "C" void expand_input_rows(
    void const* unpermuted_input,       // [M, K] bf16
    void* permuted_output,              // [M*topk, K/2] packed FP4
    float const* unpermuted_scales,     // [M, topk] or nullptr
    float* permuted_scales,             // [M*topk] or nullptr
    int const* permuted_row_to_unpermuted_row,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t k,
    float const* fc1_act_global_scale,
    bool use_per_expert_act_scale,
    int64_t const* expert_first_token_offset,
    uint8_t* fc1_act_sf_flat,
    uint8_t const* input_sf,
    bool swizzled_input_sf,
    int64_t num_experts_per_node,
    int const* token_expert_ids,
    bool skip_sf_padding,
    cudaStream_t stream)
{
    bool enable_pdl = g_enable_pdl;

    constexpr int64_t min_num_tokens_alignment = TmaConst::MinNDimAlignmentNVFP4;
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    int sm_count = getMultiProcessorCount();
    int64_t useful_rows = num_tokens * k;
    int64_t launch_work = skip_sf_padding
        ? useful_rows
        : std::max(useful_rows, num_padding_tokens);
    int64_t blocks = std::min((int64_t)(sm_count * 16),
                              std::max<int64_t>(launch_work, 1));

    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = 128;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    if (input_sf != nullptr) {
        cudaLaunchKernelEx(&config, atrex_expandInputRowsKernel_nvfp4,
            reinterpret_cast<uint8_t const*>(unpermuted_input),
            input_sf,
            reinterpret_cast<uint8_t*>(permuted_output),
            unpermuted_scales, permuted_scales,
            permuted_row_to_unpermuted_row,
            (int)num_tokens, (int)hidden_size, (int)k,
            expert_first_token_offset,
            fc1_act_sf_flat,
            (int)num_experts_per_node,
            token_expert_ids,
            skip_sf_padding);
    } else {
        cudaLaunchKernelEx(&config, atrex_expandInputRowsKernel_opt,
            reinterpret_cast<__nv_bfloat16 const*>(unpermuted_input),
            reinterpret_cast<uint8_t*>(permuted_output),
            unpermuted_scales, permuted_scales,
            permuted_row_to_unpermuted_row,
            (int)num_tokens, (int)hidden_size, (int)k,
            fc1_act_global_scale, use_per_expert_act_scale,
            expert_first_token_offset,
            fc1_act_sf_flat,
            (int)num_experts_per_node,
            token_expert_ids,
            skip_sf_padding);
    }
}
