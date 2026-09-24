#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace atrex_task13_gemm1_gather_v1 {

// Pure CUDA SM120 NVFP4 GEMM1 kernel. This file intentionally has no
// CUTLASS/CUTE dependencies.

namespace sf {
static constexpr int NVFP4_BLOCK = 16;
static constexpr int MIN_N = 128;
static constexpr int MIN_K = 64;

__host__ __device__ inline int align_to(int dim, int alignment) {
    return ((dim + alignment - 1) / alignment) * alignment;
}
}  // namespace sf

__device__ inline void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif
}

__device__ inline void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

static constexpr int BLOCK_M = 80;
static constexpr int BLOCK_N = 256;
static constexpr int BLOCK_K = 128;

static constexpr int ATOM_M = 16;
static constexpr int ATOM_N = 8;
static constexpr int ATOM_K = 64;
static constexpr int K_BLOCKS = BLOCK_K / ATOM_K;

static constexpr int ATOMS_M = BLOCK_M / ATOM_M;
static constexpr int ATOMS_N = BLOCK_N / ATOM_N;

static constexpr int WARPS_M = 1;
static constexpr int WARPS_N = 8;
static constexpr int NUM_WARPS = WARPS_M * WARPS_N;
static constexpr int THREADS = NUM_WARPS * 32;

static constexpr int ATOMS_M_PER_WARP = ATOMS_M / WARPS_M;
static constexpr int ATOMS_N_PER_WARP = ATOMS_N / WARPS_N;

static constexpr int ROW_BYTES = BLOCK_K / 2;
static constexpr int SA_BYTES = BLOCK_M * ROW_BYTES;
static constexpr int SB_BYTES = BLOCK_N * ROW_BYTES;
static constexpr int SF_A_STAGE = K_BLOCKS * 512;
static constexpr int SF_B_128_STAGE = K_BLOCKS * 512;
static constexpr int SF_B_STAGE = SF_B_128_STAGE * (BLOCK_N / 128);
static constexpr int STAGE_BYTES = SA_BYTES + SB_BYTES + SF_A_STAGE + SF_B_STAGE;
static constexpr int NUM_STAGES = 2;
static constexpr int MBAR_OFFSET = STAGE_BYTES * NUM_STAGES;
static constexpr int SOURCE_ROWS_OFFSET = MBAR_OFFSET + NUM_STAGES * 8;
static constexpr int SOURCE_ROWS_BYTES = BLOCK_M * static_cast<int>(sizeof(int));
static constexpr int SMEM_KLOOP_BYTES = SOURCE_ROWS_OFFSET + SOURCE_ROWS_BYTES;
static constexpr int SMEM_EPIL_BYTES =
    BLOCK_M * (BLOCK_N + 8) * static_cast<int>(sizeof(__nv_bfloat16));
static constexpr int SMEM_BYTES =
    (SMEM_KLOOP_BYTES > SMEM_EPIL_BYTES ? SMEM_KLOOP_BYTES : SMEM_EPIL_BYTES);
static constexpr float FP4_MAX_INV = 1.0f / 6.0f;

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

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar,
                                                          uint32_t tx_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :: "r"(smem_addr), "r"(tx_bytes));
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* mbar,
                                                         uint32_t phase) {
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

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar,
                                                     uint32_t phase) {
    while (!mbarrier_try_wait_parity(mbar, phase)) {}
}

__device__ __forceinline__ void tma_copy_2d(
    void const* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord0, int32_t coord1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(desc), "r"(coord0), "r"(coord1), "r"(mbar_addr));
}

__device__ __forceinline__ void tma_copy_3d(
    void const* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord0, int32_t coord1, int32_t coord2) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(smem_addr), "l"(desc), "r"(coord0), "r"(coord1),
           "r"(coord2), "r"(mbar_addr));
}

__device__ __forceinline__ void prefetch_tma_descriptor(void const* desc) {
    asm volatile("prefetch.tensormap [%0];\n"
                 :: "l"(reinterpret_cast<uint64_t>(desc))
                 : "memory");
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
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
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
#endif
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float exp2_approx_ftz(float a) {
    float b;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float fused_silu(float x) {
    constexpr float neg_log2e = -1.4426950408889634f;
    return x * reciprocal_approximate_ftz(1.0f + exp2_approx_ftz(x * neg_log2e));
}

__device__ __forceinline__ uint64_t fp32_vec_to_e2m1_16(float2 (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint64_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n .reg .b8 byte1;\n .reg .b8 byte2;\n .reg .b8 byte3;\n"
        ".reg .b8 byte4;\n .reg .b8 byte5;\n .reg .b8 byte6;\n .reg .b8 byte7;\n"
        ".reg .b32 val0;\n .reg .b32 val1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
        "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
        "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
        "mov.b64 %0, {val0, val1};\n"
        "}\n"
        : "=l"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
          "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
          "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
    return val;
#else
    return 0;
#endif
}

__device__ __forceinline__ void quantize_gather_a_stage(
    uint8_t* __restrict__ stage,
    const __nv_bfloat16* __restrict__ hidden_states,
    const int* __restrict__ permuted_source_rows,
    int64_t expert_off,
    int m_start,
    int expert_tokens,
    int K,
    int kt,
    float global_scale) {
    constexpr int SF_VEC = sf::NVFP4_BLOCK;
    constexpr int SF_VECS_PER_TILE = BLOCK_K / SF_VEC;
    uint8_t* sA = stage;
    uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;

    for (int linear = threadIdx.x; linear < BLOCK_M * SF_VECS_PER_TILE;
         linear += THREADS) {
        int row = linear / SF_VECS_PER_TILE;
        int kv = linear - row * SF_VECS_PER_TILE;
        int r_local = m_start + row;
        int sA_off = sw64(row * ROW_BYTES + kv * (SF_VEC / 2));
        int sf_off = (kv / 4) * 512 + (row & 31) * 16 +
                     (row >> 5) * 4 + (kv & 3);

        if (r_local >= expert_tokens) {
            *reinterpret_cast<uint64_t*>(sA + sA_off) = 0;
            sSF_A[sf_off] = 0;
            continue;
        }

        int sorted_row = static_cast<int>(expert_off) + r_local;
        int source_row = __ldg(&permuted_source_rows[sorted_row]);
        int k_base = kt * BLOCK_K + kv * SF_VEC;

        if (k_base + SF_VEC > K) {
            *reinterpret_cast<uint64_t*>(sA + sA_off) = 0;
            sSF_A[sf_off] = 0;
            continue;
        }

        const __nv_bfloat16* src =
            hidden_states + static_cast<int64_t>(source_row) * K + k_base;
        __nv_bfloat162 p[8];
        *reinterpret_cast<uint4*>(&p[0]) =
            *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(&p[4]) =
            *reinterpret_cast<const uint4*>(src + 8);

        float2 f2[8];
        float fmax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            f2[i] = __bfloat1622float2(p[i]);
            fmax = fmaxf(fmax, fmaxf(fabsf(f2[i].x), fabsf(f2[i].y)));
        }

        float sv = global_scale * (fmax * FP4_MAX_INV);
        __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sv);
        uint8_t sf_val = sf8.__x;
        sv = static_cast<float>(sf8);
        float oscale = fmax != 0.f
            ? reciprocal_approximate_ftz(
                sv * reciprocal_approximate_ftz(global_scale))
            : 0.f;

#pragma unroll
        for (int i = 0; i < 8; i++) {
            f2[i].x *= oscale;
            f2[i].y *= oscale;
        }

        *reinterpret_cast<uint64_t*>(sA + sA_off) = fp32_vec_to_e2m1_16(f2);
        sSF_A[sf_off] = sf_val;
    }
}

__device__ __forceinline__ void load_prequant_gather_a_stage(
    uint8_t* __restrict__ stage,
    const uint8_t* __restrict__ hidden_fp4,
    const uint8_t* __restrict__ input_sf,
    const int* __restrict__ source_rows,
    int K,
    int kt) {
    constexpr int SF_VEC = sf::NVFP4_BLOCK;
    constexpr int SF_VECS_PER_TILE = BLOCK_K / SF_VEC;
    constexpr int SF_VEC_PAIRS_PER_TILE = SF_VECS_PER_TILE / 2;
    uint8_t* sA = stage;
    uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
    const int src_row_bytes = K / 2;
    const int src_sf_vecs = K / SF_VEC;

    for (int linear = threadIdx.x; linear < BLOCK_M * SF_VEC_PAIRS_PER_TILE;
         linear += THREADS) {
        int row = linear / SF_VEC_PAIRS_PER_TILE;
        int kv_pair = linear - row * SF_VEC_PAIRS_PER_TILE;
        int kv = kv_pair * 2;
        int sA_off = sw64(row * ROW_BYTES + kv * (SF_VEC / 2));
        int sf_off = (kv / 4) * 512 + (row & 31) * 16 +
                     (row >> 5) * 4 + (kv & 3);
        int k_vec = kt * SF_VECS_PER_TILE + kv;
        int source_row = source_rows[row];

        if (source_row < 0 || k_vec >= src_sf_vecs) {
            *reinterpret_cast<uint64_t*>(sA + sA_off) = 0;
            *reinterpret_cast<uint64_t*>(sA + sA_off + 8) = 0;
            *reinterpret_cast<uint16_t*>(sSF_A + sf_off) = 0;
            continue;
        }

        const uint8_t* fp4_src = hidden_fp4 +
            static_cast<int64_t>(source_row) * src_row_bytes +
            k_vec * (SF_VEC / 2);
        const uint8_t* sf_src = input_sf +
            static_cast<int64_t>(source_row) * src_sf_vecs + k_vec;
        if (k_vec + 1 < src_sf_vecs) {
            cp_async_cg_16(sA + sA_off, fp4_src);
            *reinterpret_cast<uint16_t*>(sSF_A + sf_off) =
                *reinterpret_cast<const uint16_t*>(sf_src);
        } else {
            *reinterpret_cast<uint64_t*>(sA + sA_off) =
                *reinterpret_cast<const uint64_t*>(fp4_src);
            *reinterpret_cast<uint64_t*>(sA + sA_off + 8) = 0;
            sSF_A[sf_off] = __ldg(sf_src);
            sSF_A[sf_off + 1] = 0;
        }
    }
}

__device__ __forceinline__ int upper_bound_prefix(
    const int* __restrict__ prefix,
    int count,
    int value) {
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = __ldg(&prefix[mid]);
        if (v <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void atrex_compute_tile_info_kernel(
    const int64_t* __restrict__ expert_first_token_offset,
    int* __restrict__ tile_prefix_sums,
    int* __restrict__ d_total_tiles,
    int num_experts,
    int N) {
    pdl_wait();
    extern __shared__ int smem_tile[];
    int* s_tiles_per_expert = smem_tile;

    int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        int tokens = static_cast<int>(
            expert_first_token_offset[e + 1] - expert_first_token_offset[e]);
        int m_tiles = (tokens + BLOCK_M - 1) / BLOCK_M;
        s_tiles_per_expert[e] = m_tiles * n_tiles;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int running = 0;
        tile_prefix_sums[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            running += s_tiles_per_expert[e];
            tile_prefix_sums[e + 1] = running;
        }
        *d_total_tiles = running;
    }
    pdl_launch_dependents();
}

template <bool FixedTask22, bool InputNvfp4 = false>
__global__ __launch_bounds__(THREADS, 2)
void atrex_grouped_gemm_nvfp4_kernel(
    const void* __restrict__ hidden_states,
    const uint8_t* __restrict__ input_sf,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    uint8_t* __restrict__ output_fp4,
    const float* __restrict__ fc2_act_global_scale,
    uint8_t* __restrict__ act_sf_flat,
    const float* __restrict__ fc1_act_global_scale,
    const int* __restrict__ permuted_source_rows,
    const int64_t* __restrict__ expert_first_token_offset,
    int* __restrict__ tile_prefix_sums,
    int* __restrict__ d_total_tiles,
    int num_experts,
    int num_tokens,
    int N,
    int K,
    int tile_m_offset) {
    const int n_tiles = FixedTask22 ? 8 : (N + BLOCK_N - 1) / BLOCK_N;
    const int tile_id = blockIdx.x;
    if constexpr (!FixedTask22) {
        const int total_tiles = __ldg(d_total_tiles);
        if (tile_id >= total_tiles) {
            return;
        }
    }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);
    int expert_id;
    int tile_m;
    int tile_n;
    if constexpr (FixedTask22) {
        if (tile_id == 0 && threadIdx.x == 0) {
            int running = 0;
            tile_prefix_sums[0] = 0;
            for (int e = 0; e < num_experts; e++) {
                int64_t expert_start = __ldg(&expert_first_token_offset[e]);
                int tokens = static_cast<int>(
                    __ldg(&expert_first_token_offset[e + 1]) - expert_start);
                int m_tiles = (tokens + BLOCK_M - 1) / BLOCK_M;
                int overflow_tiles = m_tiles > 2 ? (m_tiles - 2) * n_tiles : 0;
                running += overflow_tiles;
                tile_prefix_sums[e + 1] = running;
            }
            *d_total_tiles = running;
        }
        int local_tile = tile_id & 15;
        expert_id = tile_id >> 4;
        tile_m = local_tile >> 3;
        tile_n = local_tile & 7;
        if (threadIdx.x == 0) {
            mbarrier_init(&mbar[0], 1);
            mbarrier_init(&mbar[1], 1);
            prefetch_tma_descriptor(&tma_b_desc);
        }
        __syncthreads();
    } else {
        int* tile_meta = reinterpret_cast<int*>(smem);
        if (threadIdx.x == 0) {
            expert_id =
                upper_bound_prefix(tile_prefix_sums, num_experts + 1, tile_id) - 1;
            int local_tile = tile_id - __ldg(&tile_prefix_sums[expert_id]);
            int64_t expert_start = __ldg(&expert_first_token_offset[expert_id]);
            int tokens = static_cast<int>(
                __ldg(&expert_first_token_offset[expert_id + 1]) - expert_start);
            int m_tiles = (tokens + BLOCK_M - 1) / BLOCK_M;
            int local_tile_n = local_tile / m_tiles;
            int local_tile_m = local_tile - local_tile_n * m_tiles;
            tile_m = local_tile_m + tile_m_offset;
            tile_n = local_tile_n;
            tile_meta[0] = expert_id;
            tile_meta[1] = tile_m;
            tile_meta[2] = tile_n;
            mbarrier_init(&mbar[0], 1);
            mbarrier_init(&mbar[1], 1);
            prefetch_tma_descriptor(&tma_b_desc);
        }
        __syncthreads();
        expert_id = tile_meta[0];
        tile_m = tile_meta[1];
        tile_n = tile_meta[2];
    }

    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    int expert_tokens =
        static_cast<int>(__ldg(&expert_first_token_offset[expert_id + 1]) -
                         expert_off);
    const int m_start = tile_m * BLOCK_M;
    const int n_start = tile_n * BLOCK_N;

    int* source_rows = nullptr;
    if constexpr (InputNvfp4) {
        source_rows = reinterpret_cast<int*>(smem + SOURCE_ROWS_OFFSET);
        for (int row = threadIdx.x; row < BLOCK_M; row += THREADS) {
            int r_local = m_start + row;
            int source_row = -1;
            if (r_local < expert_tokens) {
                int sorted_row = static_cast<int>(expert_off) + r_local;
                source_row = __ldg(&permuted_source_rows[sorted_row]);
            }
            source_rows[row] = source_row;
        }
        __syncthreads();
    }

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    float acc[ATOMS_M_PER_WARP][ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
#pragma unroll
        for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
            acc[mi][ni][0] = 0.f;
            acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f;
            acc[mi][ni][3] = 0.f;
        }
    }

    const int num_k_tiles = FixedTask22 ? 16 : (K + BLOCK_K - 1) / BLOCK_K;

    const int sf_padded_K = FixedTask22 ? 2048 : sf::align_to(K, sf::MIN_K);
    const int sf_num_k_vecs = sf_padded_K / sf::NVFP4_BLOCK;
    const int sf_num_k_tiles = (sf_num_k_vecs + 3) / 4;
    const int64_t sf_m_tile_stride = static_cast<int64_t>(sf_num_k_tiles) * 512;

    const int sf_padded_N = FixedTask22 ? 1024 : sf::align_to(N, sf::MIN_N);
    constexpr int SF_B_N_CHUNKS = BLOCK_N / 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * sf_padded_N * sf_padded_K /
            sf::NVFP4_BLOCK +
        tile_n * SF_B_N_CHUNKS * sf_m_tile_stride;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64 * SF_B_N_CHUNKS) {
        int sf_chunk = threadIdx.x / 64;
        int sf_lane = threadIdx.x - sf_chunk * 64;
        sf_thread_dst_off = sf_chunk * SF_B_128_STAGE + sf_lane * 16;
        sf_thread_src = sf_b_tile_ptr + sf_chunk * sf_m_tile_stride +
                        sf_lane * 16;
    }

    float fc1_global_scale = fc1_act_global_scale
        ? __ldg(&fc1_act_global_scale[expert_id]) : 1.0f;

    const int ldsm_a_m_off = ((lane_id >> 3) & 1) * 8;
    const int ldsm_a_k_off = (lane_id >> 4) * 16;
    const int ldsm_a_row = lane_id & 7;
    const int ldsm_b_row = lane_id & 7;
    const int ldsm_b_k_off = ((lane_id >> 3) & 1) * 16;

    int b_col_base[ATOMS_N_PER_WARP];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        b_col_base[ni] =
            warp_n * ATOMS_N_PER_WARP * ATOM_N + ni * ATOM_N;
    }

    int b_sf_base[ATOMS_N_PER_WARP];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int col_tile = b_col_base[ni] + l;
        int col_chunk = col_tile / 128;
        int local_col = col_tile - col_chunk * 128;
        b_sf_base[ni] = col_chunk * SF_B_128_STAGE +
                        (local_col & 31) * 16 + (local_col >> 5) * 4;
    }

    int a_sf_base[ATOMS_M_PER_WARP];
#pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
        int sf_row = warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M +
                     ((lane_id & 1) * 8 + l);
        a_sf_base[mi] = (sf_row & 31) * 16 + (sf_row >> 5) * 4;
    }

    if (threadIdx.x == 0) {
        mbarrier_arrive_expect_tx(&mbar[0], SB_BYTES);
        tma_copy_3d(&tma_b_desc, &mbar[0], smem + SA_BYTES,
                    0, n_start, expert_id);
    }
    if constexpr (InputNvfp4) {
        load_prequant_gather_a_stage(
            smem, reinterpret_cast<const uint8_t*>(hidden_states), input_sf,
            source_rows, K, 0);
    } else {
        quantize_gather_a_stage(
            smem, reinterpret_cast<const __nv_bfloat16*>(hidden_states),
            permuted_source_rows, expert_off, m_start,
            expert_tokens, K, 0, fc1_global_scale);
    }
    {
        uint8_t* sSF_A = smem + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        if (threadIdx.x < 64 * SF_B_N_CHUNKS) {
            cp_async_cg_16(sSF_B + sf_thread_dst_off, sf_thread_src);
        }
    }
    cp_async_commit();

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
                mbarrier_arrive_expect_tx(&mbar[next_buf], SB_BYTES);
                tma_copy_3d(&tma_b_desc, &mbar[next_buf],
                            next_stage + SA_BYTES,
                            next_k_coord, n_start, expert_id);
            }
            if constexpr (InputNvfp4) {
                load_prequant_gather_a_stage(
                    next_stage, reinterpret_cast<const uint8_t*>(hidden_states),
                    input_sf, source_rows, K, next_kt);
            } else {
                quantize_gather_a_stage(
                    next_stage,
                    reinterpret_cast<const __nv_bfloat16*>(hidden_states),
                    permuted_source_rows, expert_off, m_start,
                    expert_tokens, K, next_kt, fc1_global_scale);
            }

            uint8_t* next_SF_A = next_stage + SA_BYTES + SB_BYTES;
            uint8_t* next_SF_B = next_SF_A + SF_A_STAGE;
            if (threadIdx.x < 64 * SF_B_N_CHUNKS) {
                cp_async_cg_16(next_SF_B + sf_thread_dst_off,
                               sf_thread_src + static_cast<int64_t>(next_kt) *
                                                   SF_B_128_STAGE);
            }
            cp_async_commit();
        }

#pragma unroll
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            const int kb_off = kb * (ATOM_K / 2);
            const int kb_sf_a = kb * 512;
            const int kb_sf_b = kb * 512;

            uint32_t b_r[ATOMS_N_PER_WARP][2];
            uint32_t sfb_r[ATOMS_N_PER_WARP];
#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                int b_row = b_col_base[ni] + ldsm_b_row;
                uint32_t ldsm_b_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        sB_curr + sw64(b_row * ROW_BYTES + kb_off +
                                       ldsm_b_k_off)));
                asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_r[ni][0]), "=r"(b_r[ni][1])
                    : "r"(ldsm_b_addr));

                sfb_r[ni] = *reinterpret_cast<const uint32_t*>(
                    sSF_B_curr + kb_sf_b + b_sf_base[ni]);
            }

#pragma unroll
            for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
                int atom_base =
                    warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M;
                int phys_row = atom_base + ldsm_a_m_off + ldsm_a_row;
                uint32_t ldsm_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        sA_curr +
                        sw64(phys_row * ROW_BYTES + kb_off + ldsm_a_k_off)));

                uint32_t a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(ldsm_addr));

                uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                    sSF_A_curr + kb_sf_a + a_sf_base[mi]);

#pragma unroll
                for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                    mma_nvfp4_m16n8k64(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                        sfa_v, sfb_r[ni]);
                }
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
    float global_scale = fc2_act_global_scale
        ? __ldg(&fc2_act_global_scale[expert_id]) : 1.0f;
    float inv_global_scale = reciprocal_approximate_ftz(global_scale);

    const int inter_size = FixedTask22 ? 512 : N / 2;
    const int half_inter_bytes = inter_size / 2;
    const int out_sf_padded_K = FixedTask22 ? 512 : sf::align_to(inter_size, sf::MIN_K);
    const int out_sf_k_vecs = out_sf_padded_K / sf::NVFP4_BLOCK;
    const int out_sf_num_k_tiles = (out_sf_k_vecs + 3) / 4;
    const int out_sf_m_tile_stride = out_sf_num_k_tiles * 512;
    int64_t psf = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    uint8_t* sf_expert = act_sf_flat + psf * out_sf_k_vecs;

    int out_col_base = tile_n * (BLOCK_N / 2) +
                       warp_n * (ATOMS_N_PER_WARP * ATOM_N / 2);
    int out_byte_col = out_col_base / 2;
    int sf_k_idx = out_col_base / sf::NVFP4_BLOCK;

#pragma unroll
    for (int mi = 0; mi < ATOMS_M_PER_WARP; mi++) {
        int m_base = warp_m * ATOMS_M_PER_WARP * ATOM_M + mi * ATOM_M;

#pragma unroll
        for (int rs = 0; rs < 2; rs++) {
            int row_in_tile = m_base + l + rs * 8;
            int r_local = m_start + row_in_tile;
            if (r_local >= expert_tokens) {
                continue;
            }
            if constexpr (!FixedTask22) {
                if (out_col_base + 15 >= inter_size) {
                    continue;
                }
            }
            int r_global = static_cast<int>(expert_off) + r_local;

            int ai = rs * 2;
            float first0 = acc[mi][0][ai] * alpha_v;
            float second0 = acc[mi][1][ai] * alpha_v;
            float v0 = fused_silu(second0) * first0;

            float first1 = acc[mi][0][ai + 1] * alpha_v;
            float second1 = acc[mi][1][ai + 1] * alpha_v;
            float v1 = fused_silu(second1) * first1;

            float first2 = acc[mi][2][ai] * alpha_v;
            float second2 = acc[mi][3][ai] * alpha_v;
            float v2 = fused_silu(second2) * first2;

            float first3 = acc[mi][2][ai + 1] * alpha_v;
            float second3 = acc[mi][3][ai + 1] * alpha_v;
            float v3 = fused_silu(second3) * first3;

            float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)),
                             fmaxf(fabsf(v2), fabsf(v3)));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));

            float sv = global_scale * (mx * FP4_MAX_INV);
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sv);
            uint8_t sf_byte = sf8.__x;
            sv = static_cast<float>(sf8);
            float oscale = mx != 0.f
                ? reciprocal_approximate_ftz(sv * inv_global_scale)
                : 0.f;

            v0 *= oscale;
            v1 *= oscale;
            v2 *= oscale;
            v3 *= oscale;

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

            int byte_base =
                r_global * half_inter_bytes + out_byte_col;
            output_fp4[byte_base + g] = static_cast<uint8_t>(packed & 0xff);
            output_fp4[byte_base + 4 + g] =
                static_cast<uint8_t>((packed >> 8) & 0xff);

            if (g == 0) {
                int sf_row_base =
                    (r_local / sf::MIN_N) * out_sf_m_tile_stride +
                    (r_local & 31) * 16 +
                    ((r_local & (sf::MIN_N - 1)) >> 5) * 4;
                sf_expert[sf_row_base + (sf_k_idx / 4) * 512 +
                          (sf_k_idx & 3)] = sf_byte;
            }
        }
    }

}

}  // namespace atrex_task13_gemm1_gather_v1
