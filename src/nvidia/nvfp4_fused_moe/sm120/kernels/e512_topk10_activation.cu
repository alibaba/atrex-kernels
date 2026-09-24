#include "moe_common.cuh"
#include "quantization_utils.cuh"

// ============================================================================
// e512_topk10 activation — SwiGLU + FP4 requantization with BF16 boundaries
//
// Additive companion to up_gate_activation.cu. Differs from the dev topk=8
// kernel in two e512_topk10-specific ways:
//   1. RoundPostActBF16: materialize SiLU(gate)*up to BF16 before absmax/FP4
//      requantization, matching the FlashInfer boundary the e512_topk10 path
//      reproduces (dev keeps everything in float).
//   2. skip_sf_padding: when the fused SF staging buffer is used, the N-dim SF
//      padding is already zeroed upstream, so this kernel returns right after
//      the main loop instead of re-padding.
//
// Symbols are renamed with an atrex_e512t10_ prefix so they coexist with the dev
// atrex_doActivationKernel_opt / do_activation in the same JIT module without
// ODR or duplicate-symbol collisions, while still satisfying the skill §4 gate
// that every launched __global__ name starts with "atrex_".
// ============================================================================

// launch_bounds(128, 8) lets the launcher pick 96 threads for the
// inter_size=768 shape (elems_per_row=96), while keeping the SM occupancy ceiling
// (128*8 = 1024 threads/SM) identical to the legacy (64*16). For inter_size=512
// the launcher still picks 64 threads.
template <bool Interleaved, bool RoundPostActBF16 = false>
__global__ __launch_bounds__(128, 8)
void atrex_e512t10_doActivationKernel_opt(
    uint8_t* __restrict__ output,
    __nv_bfloat16 const* __restrict__ gemm_result,
    float const* __restrict__ fp8_quant,
    int64_t const* __restrict__ expert_first_token_offset,
    int const num_experts_per_node,
    int64_t const inter_size,
    float const* __restrict__ fc2_act_global_scale,
    bool const use_per_expert_act_scale,
    uint8_t* __restrict__ fc2_act_sf_flat,
    int const* __restrict__ token_expert_ids,
    bool const skip_sf_padding)
{
    pdl_wait();

    constexpr int SF_VEC = TmaConst::NVFP4BlockScaleVectorSize;  // 16
    constexpr int ELTS_PER_THREAD = 8;
    constexpr int NUM_THREADS_PER_SF = SF_VEC / ELTS_PER_THREAD;  // 2
    constexpr int64_t MIN_N = TmaConst::MinNDimAlignmentNVFP4;

    int const tid = threadIdx.x;
    int const num_elems = (int)inter_size / ELTS_PER_THREAD;
    int const gated_off_bf16 = (int)inter_size;

    // SF layout constants
    int const sf_padded_K = TmaConst::alignToSfDim(
        (int)inter_size, (int)TmaConst::MinKDimAlignmentNVFP4);
    int const sf_k_vecs = sf_padded_K / SF_VEC;
    int const sf_numKTiles = (sf_k_vecs + 3) / 4;
    int const sf_mTileStride = sf_numKTiles * 512;

    int const num_valid_tokens =
        (int)expert_first_token_offset[num_experts_per_node];

    for (int token = blockIdx.x; token < num_valid_tokens;
         token += gridDim.x)
    {
        int const expert = token_expert_ids[token];
        int const scale_idx = use_per_expert_act_scale ? expert : 0;
        float const quant_scale = fp8_quant ? fp8_quant[scale_idx] : 1.f;
        float const global_scale = fc2_act_global_scale
                                   ? fc2_act_global_scale[scale_idx] : 1.0f;
        int64_t const tok_before = expert_first_token_offset[expert];

        // SF expert base + per-row offset
        int64_t const psf = TmaConst::alignToSfDim(
            (int)(tok_before + expert * (MIN_N - 1)), (int)MIN_N);
        uint8_t* sf_base = fc2_act_sf_flat + psf * sf_k_vecs;

        int const lr = (int)(token - tok_before);
        int const sf_row_base = (lr / 128) * sf_mTileStride +
                                (lr % 32) * 16 +
                                ((lr % 128) / 32) * 4;

        // Source/dest pointers
        __nv_bfloat16 const* row_base =
            gemm_result + (int64_t)token * inter_size * 2;
        using Vec8 = Array<__nv_bfloat16, 8>;
        Vec8 const* up_ptr = reinterpret_cast<Vec8 const*>(row_base);
        Vec8 const* gate_ptr = reinterpret_cast<Vec8 const*>(
            row_base + gated_off_bf16);
        uint32_t* out_ptr = reinterpret_cast<uint32_t*>(
            output + (int64_t)token * inter_size / 2);

        for (int ei = tid; ei < num_elems; ei += blockDim.x) {
            Vec8 gate_raw, up_raw;
            if constexpr (Interleaved) {
                // gate/up alternate in groups of 8: [first0:8, second0:8, ...]
                up_raw   = up_ptr[2 * ei];
                gate_raw = up_ptr[2 * ei + 1];
            } else {
                gate_raw = gate_ptr[ei];
                up_raw   = up_ptr[ei];
            }

            // SiLU(gate) * up in float, track absmax directly
            float vals[8];
            float fmax = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float g = __bfloat162float(gate_raw[i]);
                float u = __bfloat162float(up_raw[i]);
                float sg = g / (1.0f + __expf(-g));
                vals[i] = sg * u * quant_scale;
                if constexpr (RoundPostActBF16) {
                    vals[i] = __bfloat162float(__float2bfloat16_rn(vals[i]));
                }
                fmax = fmaxf(fmax, fabsf(vals[i]));
            }

            // Shuffle absmax across 2 threads sharing one SF vector
            fmax = fmaxf(fmax, __shfl_xor_sync(0xFFFFFFFF, fmax, 1));

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

            // Write SF (even threads only)
            if ((tid & 1) == 0) {
                int kIdx = ei / NUM_THREADS_PER_SF;
                sf_base[sf_row_base + (kIdx / 4) * 512 + (kIdx % 4)] =
                    sf_val;
            }

            // Scale and convert to FP4
            #pragma unroll
            for (int i = 0; i < 8; i++)
                vals[i] *= oscale;
            out_ptr[ei] = fp32_vec_to_e2m1(vals);
        }

        // K-dim SF padding (inter_size=512 aligned to 64 → no padding needed,
        // but handle general case)
        int const num_sf_data = (int)inter_size / SF_VEC;
        for (int kv = num_sf_data + tid; kv < sf_k_vecs; kv += blockDim.x) {
            if ((tid & 1) == 0)
                sf_base[sf_row_base + (kv / 4) * 512 + (kv % 4)] = 0;
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
            uint8_t* sb = fc2_act_sf_flat + psf * sf_k_vecs;

            int pr = toks + pidx;
            int srb = (pr / 128) * sf_mTileStride +
                      (pr % 32) * 16 +
                      ((pr % 128) / 32) * 4;

            for (int kv = tid; kv < sf_k_vecs; kv += blockDim.x) {
                if ((tid & 1) == 0)
                    sb[srb + (kv / 4) * 512 + (kv % 4)] = 0;
            }
        }
    }
}

// ============================================================================
// Extern C launcher
// ============================================================================

extern "C" void atrex_e512t10_do_activation(
    void* output,
    void const* gemm_result,
    float const* fp8_quant,
    void const* bias,
    bool bias_is_broadcast,
    int64_t const* expert_first_token_offset,
    int num_experts_per_node,
    int64_t inter_size,
    int64_t expanded_num_tokens,
    float const* fc2_act_global_scale,
    bool use_per_expert_act_scale,
    uint8_t* fc2_act_sf_flat,
    int const* token_expert_ids,
    cudaStream_t stream,
    bool interleaved,
    bool skip_sf_padding,
    bool round_post_act_bf16)
{
    bool enable_pdl = g_enable_pdl;

    constexpr int64_t min_num_tokens_alignment = TmaConst::MinNDimAlignmentNVFP4;
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    int64_t elems_per_row = inter_size / CVT_ELTS_PER_THREAD;
    int64_t threads = ((elems_per_row + 31) / 32) * 32;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;

    int sm_count = getMultiProcessorCount();
    int64_t target_threads_per_sm = 2048;
    int64_t blocks_per_sm = target_threads_per_sm / threads;
    int64_t launch_work = skip_sf_padding
        ? expanded_num_tokens
        : std::max(expanded_num_tokens, num_padding_tokens);
    int64_t blocks = std::min((int64_t)(sm_count * blocks_per_sm),
                              std::max<int64_t>(launch_work, 1));

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

    auto launch = [&](auto kernel_fn) {
        cudaLaunchKernelEx(&config, kernel_fn,
            reinterpret_cast<uint8_t*>(output),
            reinterpret_cast<__nv_bfloat16 const*>(gemm_result),
            fp8_quant,
            expert_first_token_offset,
            num_experts_per_node,
            inter_size,
            fc2_act_global_scale,
            use_per_expert_act_scale,
            fc2_act_sf_flat,
            token_expert_ids,
            skip_sf_padding);
    };
    if (interleaved) {
        if (round_post_act_bf16)
            launch(atrex_e512t10_doActivationKernel_opt<true, true>);
        else
            launch(atrex_e512t10_doActivationKernel_opt<true, false>);
    } else {
        if (round_post_act_bf16)
            launch(atrex_e512t10_doActivationKernel_opt<false, true>);
        else
            launch(atrex_e512t10_doActivationKernel_opt<false, false>);
    }
}
