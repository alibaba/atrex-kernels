#define device_kernel atrex_ppu_decode_kernel
#define native_decode atrex_ppu_decode
#include <cstdio>
#include "flash_fwd_launch_template.h"
#include "flash_fwd_combine_launch_template.h"

template<bool Causal, bool SmallM, bool Aiu, int Compact = 0>
void launch_decode(Flash_fwd_params &params, hggcStream_t stream) {
    run_flash_fwd<89, 256, 256, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t,
        Causal, false, false, true, true, Aiu, DECODE_AIU_ROWS, false, false,
        true, true, false, Compact != 0, SmallM, Compact != 3>(params, stream);
}
#include "decode_resources.h"

extern "C" int native_decode(void **pointers, int64_t *metadata, float scale,
                              int splits, int variant, void *stream_ptr) {
    Flash_fwd_params params{};
    params.q_ptr = pointers[0];
    params.k_ptr = pointers[1];
    params.v_ptr = pointers[2];
    params.o_ptr = pointers[3];
    params.cu_seqlens_q = static_cast<int*>(pointers[4]);
    params.seqused_k = static_cast<int*>(pointers[5]);
    params.page_table = static_cast<int*>(pointers[6]);
    params.q_descale_ptr = static_cast<float*>(pointers[7]);
    params.k_descale_ptr = static_cast<float*>(pointers[8]);
    params.v_descale_ptr = static_cast<float*>(pointers[9]);
    params.softmax_lse_ptr = pointers[10];
    params.oaccum_ptr = pointers[11];
    params.softmax_lseaccum_ptr = pointers[12];
    params.tile_count_semaphore = static_cast<int*>(pointers[13]);
    params.b = metadata[0];
    params.b_k = params.b;
    params.total_q = metadata[1];
    params.seqlen_q = metadata[2];
    params.seqlen_k = metadata[3];
    params.num_pages = metadata[4];
    params.page_size = metadata[5];
    params.page_table_batch_stride = metadata[6];
    params.q_row_stride = metadata[7];
    params.q_head_stride = metadata[8];
    params.k_batch_stride = metadata[9];
    params.k_row_stride = metadata[10];
    params.k_head_stride = metadata[11];
    params.v_batch_stride = metadata[12];
    params.v_row_stride = metadata[13];
    params.v_head_stride = metadata[14];
    params.v_dim_stride = 1;
    params.o_row_stride = metadata[15];
    params.o_head_stride = metadata[16];
    params.q_descale_batch_stride = metadata[17];
    params.q_descale_head_stride = metadata[18];
    params.k_descale_batch_stride = metadata[19];
    params.k_descale_head_stride = metadata[20];
    params.v_descale_batch_stride = metadata[21];
    params.v_descale_head_stride = metadata[22];
    params.num_sm = metadata[23];
    params.h = metadata[24];
    params.h_k = metadata[25];
    params.d = params.d_rounded = params.dv = params.dv_rounded = 256;
    params.seqlen_q_rounded = cutlass::round_up(params.seqlen_q, 128);
    params.seqlen_k_rounded = cutlass::round_up(params.seqlen_k, 128);
    params.oaccum_row_stride = 256;
    params.oaccum_head_stride = params.total_q * 256;
    params.oaccum_split_stride = params.h * params.total_q * 256;
    params.lseaccum_head_stride = params.total_q;
    params.lseaccum_split_stride = params.h * params.total_q;
    params.scale_softmax = scale;
    params.num_splits = splits == 0 ? 32 : splits;
    params.num_splits_dynamic_ptr = splits == 0 ? static_cast<int*>(pointers[14]) : nullptr;
    params.arch = 89;
    params.pack_gqa = true;
    params.is_e4m3 = true;
    params.is_varlen_q = params.total_q != params.b * params.seqlen_q;
    params.decode_scheduler_block_m = variant == 4 && params.is_varlen_q ? 32 : 0;
    params.is_causal = params.seqlen_q != 1;
#if DECODE_REQUEST_ORDER
#if DECODE_Q1_ONLY
    if (splits == 0 && (params.is_causal || DECODE_REQUEST_ORDER >= 2) &&
        params.total_q / params.b >= 0.5f * params.seqlen_q) {
        params.decode_request_order = params.tile_count_semaphore + 32;
    }
#else
    if (splits == 0 && params.is_causal &&
        params.total_q / params.b >= 0.5f * params.seqlen_q) {
        params.decode_request_order = params.tile_count_semaphore + 32;
    }
#endif
#endif
    params.window_size_left = params.seqlen_k - 1;
    params.window_size_right = params.is_causal ? 0 : params.seqlen_q - 1;
    auto stream = reinterpret_cast<hggcStream_t>(stream_ptr);
    auto status = splits == 0 ? hggcSuccess : hggcMemsetAsync(params.tile_count_semaphore, 0, sizeof(int), stream);
    if (status != hggcSuccess) return static_cast<int>(status);
#if DECODE_Q1_ONLY
    if (params.is_causal || variant != 5) return static_cast<int>(hggcErrorInvalidValue);
    launch_decode<false, false, true, 2>(params, stream);
#else
    if (!params.is_causal || variant != 4) return static_cast<int>(hggcErrorInvalidValue);
    launch_decode<true, true, true>(params, stream);
#endif
    run_flash_fwd_combine<89, 8, 256, 5, true, true, cutlass::bfloat16_t, float>(params, stream, false);
    return static_cast<int>(hggcGetLastError());
}
