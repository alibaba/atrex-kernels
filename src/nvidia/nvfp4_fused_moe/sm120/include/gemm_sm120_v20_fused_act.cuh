#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "moe_common.cuh"
#include "quantization_utils.cuh"

namespace atrex_gemm_v20_fused_act {

// Tile config identical to v20
static constexpr int BLOCK_M = 128;
static constexpr int BLOCK_N = 128;
static constexpr int BLOCK_K = 128;

static constexpr int ATOM_M = 16;
static constexpr int ATOM_N = 8;
static constexpr int ATOM_K = 64;
static constexpr int K_BLOCKS = BLOCK_K / ATOM_K;   // 2

static constexpr int ATOMS_M = BLOCK_M / ATOM_M;    // 8
static constexpr int ATOMS_N = BLOCK_N / ATOM_N;    // 16

static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 4;
static constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 8
static constexpr int THREADS = NUM_WARPS * 32;      // 256

static constexpr int ATOMS_M_PER_WARP = ATOMS_M / WARPS_M; // 4
static constexpr int ATOMS_N_PER_WARP = ATOMS_N / WARPS_N; // 4

static constexpr int ROW_BYTES = BLOCK_K / 2;        // 64

static constexpr int SA_BYTES    = BLOCK_M * ROW_BYTES;      // 8192
static constexpr int SB_BYTES    = BLOCK_N * ROW_BYTES;      // 8192
static constexpr int SF_A_STAGE  = K_BLOCKS * 512;            // 1024
static constexpr int SF_B_STAGE  = K_BLOCKS * 512;            // 1024
static constexpr int STAGE_BYTES = SA_BYTES + SB_BYTES + SF_A_STAGE + SF_B_STAGE; // 18432
static constexpr int NUM_STAGES  = 2;
static constexpr int MBAR_OFFSET = STAGE_BYTES * NUM_STAGES;  // 36864
static constexpr int SMEM_BYTES  = MBAR_OFFSET + NUM_STAGES * 8; // 36880

__device__ __forceinline__ int sw64(int addr) {
    return addr ^ ((addr >> 3) & 0x30);
}

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :: "r"(smem_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :: "r"(smem_addr), "r"(tx_bytes));
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t ready;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P;\n"
        "}\n"
        : "=r"(ready)
        : "r"(smem_addr), "r"(phase));
    return ready != 0;
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase) {
    while (!mbarrier_try_wait_parity(mbar, phase)) {}
}

__device__ __forceinline__ void tma_copy_2d(
    void const* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord0, int32_t coord1)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(desc),
           "r"(coord0), "r"(coord1),
           "r"(mbar_addr));
}

__device__ __forceinline__ void tma_copy_3d(
    void const* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord0, int32_t coord1, int32_t coord2)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(smem_addr), "l"(desc),
           "r"(coord0), "r"(coord1), "r"(coord2),
           "r"(mbar_addr));
}

__device__ __forceinline__ void prefetch_tma_descriptor(void const* desc)
{
    asm volatile("prefetch.tensormap [%0];\n"
                 :: "l"(reinterpret_cast<uint64_t>(desc))
                 : "memory");
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%0,  %1,  %2,  %3},"
        "{%10},"
        "{%11, %12},"
        "{%13},"
        "{%14, %15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
    );
#endif
}

__device__ __forceinline__ float fused_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================================
// v20 fused act kernel: TMA+ldmatrix MMA body + SwiGLU+FP4 epilogue
// ============================================================================

__global__ __launch_bounds__(THREADS, 2)
void atrex_grouped_gemm_v20_fused_act_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float*   __restrict__ alpha,
    uint8_t*       __restrict__ output_fp4,
    uint8_t*       __restrict__ act_sf_flat,
    const float*   __restrict__ fc2_act_global_scale,
    const int64_t* __restrict__ expert_first_token_offset,
    const int*     __restrict__ tile_info,
    const int*     __restrict__ d_total_tiles,
    int num_experts, int N_gemm, int K)
{
    pdl_wait();
    const int total_tiles = *d_total_tiles;
    const int tile_id = blockIdx.x;
    if (tile_id >= total_tiles) { pdl_launch_dependents(); return; }

    const int expert_id = tile_info[tile_id * 3 + 0];
    const int tile_m    = tile_info[tile_id * 3 + 1];
    const int tile_n    = tile_info[tile_id * 3 + 2];

    const int64_t expert_off = expert_first_token_offset[expert_id];
    const int expert_tokens  = (int)(expert_first_token_offset[expert_id + 1] - expert_off);
    const int m_start = tile_m * BLOCK_M;
    const int n_start = tile_n * BLOCK_N;
    const int a_row_abs = (int)(expert_off + m_start);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;
    const int g       = lane_id & 3;
    const int l       = lane_id >> 2;

    float acc[ATOMS_M_PER_WARP][ATOMS_N_PER_WARP][4];
    #pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++)
        #pragma unroll
        for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
            acc[mi][ni][0] = 0.f;
            acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f;
            acc[mi][ni][3] = 0.f;
        }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);

    const int num_k_tiles = K / BLOCK_K;

    // SF addressing (same as v20)
    constexpr int64_t sf_bsz = TmaConst::NVFP4BlockScaleVectorSize;
    constexpr int64_t sf_min_n = TmaConst::MinNDimAlignmentNVFP4;
    constexpr int64_t sf_min_k = TmaConst::MinKDimAlignmentNVFP4;
    const int sf_padded_K = TmaConst::alignToSfDim(K, (int)sf_min_k);
    const int sf_num_k_vecs = sf_padded_K / sf_bsz;
    const int sf_numKTiles = (sf_num_k_vecs + 3) / 4;
    const int64_t sf_mTileStride = (int64_t)sf_numKTiles * 512;

    const int64_t sf_a_padded_start = TmaConst::alignToSfDim(
        (int)(expert_off + expert_id * (sf_min_n - 1)), (int)sf_min_n);
    const uint8_t* sf_a_tile_ptr = sf_a_base +
        sf_a_padded_start * sf_padded_K / sf_bsz + tile_m * sf_mTileStride;

    const int sf_padded_N = TmaConst::alignToSfDim(N_gemm, (int)sf_min_n);
    const uint8_t* sf_b_tile_ptr = sf_b_base +
        (int64_t)expert_id * sf_padded_N * sf_padded_K / sf_bsz +
        tile_n * sf_mTileStride;

    // SF per-thread assignment (threads 0-127)
    int sf_thread_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64) {
        sf_thread_off = threadIdx.x * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_off;
    } else if (threadIdx.x < 128) {
        sf_thread_off = (threadIdx.x - 64) * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_off;
    }

    // Initialize 2 mbarriers
    if (threadIdx.x == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

    // A ldmatrix per-lane constants (same as v20)
    const int ldsm_a_m_off = ((lane_id >> 3) & 1) * 8;
    const int ldsm_a_k_off = (lane_id >> 4) * 16;
    const int ldsm_a_row   = lane_id & 7;

    // B ldmatrix per-lane constants (same as v20)
    const int ldsm_b_row   = lane_id & 7;
    const int ldsm_b_k_off = ((lane_id >> 3) & 1) * 16;

    // B col base per ni
    int b_col_base[ATOMS_N_PER_WARP];
    #pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        b_col_base[ni] = warp_n * ATOMS_N_PER_WARP * ATOM_N + ni * ATOM_N;
    }

    // B SF read offsets
    int b_sf_base[ATOMS_N_PER_WARP];
    #pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int col_tile = b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    // A SF read offsets
    int a_sf_base[ATOMS_M_PER_WARP];
    #pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
        int sf_row = warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M + ((lane_id & 1) * 8 + l);
        a_sf_base[mi] = (sf_row & 31) * 16 + (sf_row >> 5) * 4;
    }

    // ---- Prologue: issue TMA + SF loads for stage 0 ----
    if (threadIdx.x == 0) {
        mbarrier_arrive_expect_tx(&mbar[0], SA_BYTES + SB_BYTES);
        tma_copy_2d(&tma_a_desc, &mbar[0], smem, 0, a_row_abs);
        tma_copy_3d(&tma_b_desc, &mbar[0], smem + SA_BYTES, 0, n_start, expert_id);
    }
    {
        uint8_t* sSF_A = smem + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        if (threadIdx.x < 64)
            cp_async_cg_16(sSF_A + sf_thread_off, sf_thread_src);
        else if (threadIdx.x < 128)
            cp_async_cg_16(sSF_B + sf_thread_off, sf_thread_src);
    }
    cp_async_commit();

    // ---- Main K-loop with 2-stage TMA pipeline (v20 body) ----
    int phase = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int buf = kt & 1;
        uint8_t* sA_curr = smem + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - buf;
            const int next_kt = kt + 1;
            const int next_k_coord = next_kt * ROW_BYTES;
            uint8_t* next_stage = smem + next_buf * STAGE_BYTES;

            if (threadIdx.x == 0) {
                mbarrier_arrive_expect_tx(&mbar[next_buf], SA_BYTES + SB_BYTES);
                tma_copy_2d(&tma_a_desc, &mbar[next_buf], next_stage, next_k_coord, a_row_abs);
                tma_copy_3d(&tma_b_desc, &mbar[next_buf], next_stage + SA_BYTES,
                            next_k_coord, n_start, expert_id);
            }

            uint8_t* next_SF_A = next_stage + SA_BYTES + SB_BYTES;
            uint8_t* next_SF_B = next_SF_A + SF_A_STAGE;
            if (threadIdx.x < 64)
                cp_async_cg_16(next_SF_A + sf_thread_off,
                    sf_thread_src + (int64_t)next_kt * SF_A_STAGE);
            else if (threadIdx.x < 128)
                cp_async_cg_16(next_SF_B + sf_thread_off,
                    sf_thread_src + (int64_t)next_kt * SF_B_STAGE);

            cp_async_commit();
        }

        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            const int kb_off = kb * (ATOM_K / 2);
            const int kb_sf  = kb * 512;

            // B loads: ldmatrix.x2 (v20)
            uint32_t b_r[ATOMS_N_PER_WARP][2];
            uint32_t sfb_r[ATOMS_N_PER_WARP];
            #pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                int b_row = b_col_base[ni] + ldsm_b_row;
                uint32_t ldsm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    sB_curr + sw64(b_row * ROW_BYTES + kb_off + ldsm_b_k_off)));

                asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_r[ni][0]), "=r"(b_r[ni][1])
                    : "r"(ldsm_b_addr));

                sfb_r[ni] = *reinterpret_cast<const uint32_t*>(sSF_B_curr + kb_sf + b_sf_base[ni]);
            }

            // A loads: ldmatrix.x4 (v20)
            #pragma unroll
            for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
                int atom_base = warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M;
                int phys_row = atom_base + ldsm_a_m_off + ldsm_a_row;
                uint32_t ldsm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off + ldsm_a_k_off)));

                uint32_t a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(ldsm_addr));

                uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                    sSF_A_curr + kb_sf + a_sf_base[mi]);

                #pragma unroll
                for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                    mma_nvfp4_m16n8k64(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3,
                        b_r[ni][0], b_r[ni][1],
                        sfa_v, sfb_r[ni]);
                }
            }
        }

        if (buf == 1) phase ^= 1;
        __syncthreads();
    }

    // ================================================================
    // Fused SwiGLU + FP4 Quantization Epilogue (from fused_act)
    // ================================================================
    float alpha_v = alpha ? alpha[expert_id] : 1.0f;
    float global_scale = fc2_act_global_scale ? fc2_act_global_scale[expert_id] : 1.0f;

    int inter_size = N_gemm / 2;
    int half_inter = inter_size / 2;

    constexpr int64_t OUT_MIN_N = TmaConst::MinNDimAlignmentNVFP4;
    int out_sf_padded_K = TmaConst::alignToSfDim(inter_size,
        (int)TmaConst::MinKDimAlignmentNVFP4);
    int out_sf_k_vecs = out_sf_padded_K / TmaConst::NVFP4BlockScaleVectorSize;
    int out_sf_numKTiles = (out_sf_k_vecs + 3) / 4;
    int out_sf_mTileStride = out_sf_numKTiles * 512;

    int64_t psf = TmaConst::alignToSfDim(
        (int)(expert_off + expert_id * (OUT_MIN_N - 1)), (int)OUT_MIN_N);
    uint8_t* sf_expert = act_sf_flat + psf * out_sf_k_vecs;

    int out_col_base = tile_n * (BLOCK_N / 2) + warp_n * (ATOMS_N_PER_WARP * ATOM_N / 2);
    int sf_kIdx = out_col_base / (int)TmaConst::NVFP4BlockScaleVectorSize;

    #pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
        int m_base = warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M;

        // SwiGLU: pair interleaved gate/up accumulators
        // Interleaved layout: ni=0 (gate0), ni=1 (up0), ni=2 (gate1), ni=3 (up1)
        // Standalone activation applies silu to second-half (up), multiply by first-half (gate)
        // So: result = silu(up) * gate = silu(ni=1) * ni=0, silu(ni=3) * ni=2
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            float ga0 = acc[mi][0][v] * alpha_v;
            float up0 = acc[mi][1][v] * alpha_v;
            acc[mi][0][v] = fused_silu(up0) * ga0;

            float ga1 = acc[mi][2][v] * alpha_v;
            float up1 = acc[mi][3][v] * alpha_v;
            acc[mi][2][v] = fused_silu(up1) * ga1;
        }

        #pragma unroll
        for (int rs = 0; rs < 2; rs++) {
            int r_local = m_start + m_base + (rs == 0 ? l : l + 8);
            if (r_local >= expert_tokens) continue;
            int r_global = (int)expert_off + r_local;

            float v0 = acc[mi][0][rs * 2];
            float v1 = acc[mi][0][rs * 2 + 1];
            float v2 = acc[mi][2][rs * 2];
            float v3 = acc[mi][2][rs * 2 + 1];

            float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                             fmaxf(fabsf(v2), fabsf(v3)));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));

            float sv = global_scale * (mx * reciprocal_approximate_ftz(6.0f));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sv);
            uint8_t sf_byte = sf8.__x;
            sv = static_cast<float>(sf8);
            float oscale = mx != 0.f
                ? reciprocal_approximate_ftz(
                      sv * reciprocal_approximate_ftz(global_scale))
                : 0.f;

            v0 *= oscale; v1 *= oscale; v2 *= oscale; v3 *= oscale;

            uint32_t packed;
            asm volatile(
                "{\n"
                ".reg .b8 b0, b1, b2, b3;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
                "mov.b32 %0, {b0, b1, b0, b0};\n"
                "}\n"
                : "=r"(packed)
                : "f"(v0), "f"(v1), "f"(v2), "f"(v3));

            int byte_base = r_global * half_inter + out_col_base / 2;
            output_fp4[byte_base + g]     = (uint8_t)(packed & 0xFF);
            output_fp4[byte_base + 4 + g] = (uint8_t)((packed >> 8) & 0xFF);

            if (g == 0) {
                int sf_row_base = (r_local / 128) * out_sf_mTileStride +
                                  (r_local % 32) * 16 +
                                  ((r_local % 128) / 32) * 4;
                sf_expert[sf_row_base + (sf_kIdx / 4) * 512 +
                          (sf_kIdx % 4)] = sf_byte;
            }
        }
    }

    pdl_launch_dependents();
}

} // namespace atrex_gemm_v20_fused_act
