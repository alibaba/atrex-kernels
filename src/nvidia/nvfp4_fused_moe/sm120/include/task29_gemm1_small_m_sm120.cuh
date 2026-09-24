#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TASK29_IMPL_NAMESPACE
#define TASK29_IMPL_NAMESPACE atrex_task29_gemm1_small_m_v1
#endif

#ifndef TASK29_BLOCK_N_VALUE
#define TASK29_BLOCK_N_VALUE 128
#endif

// PRIMARY_N / PRIMARY_K are the matrix N / K dims this kernel instance is
// compiled for. They are NOT tile dims (those are BLOCK_N / BLOCK_K) — they
// only show up in SF (block-scale) address arithmetic:
//   - PRIMARY_K controls the A-side per-M-row SF stride.
//   - PRIMARY_N * PRIMARY_K is the per-expert SF stride for the B (weight)
//     tensor, so it MUST match how the upstream `w1_sf` quantization packs
//     experts (PRIMARY_N * PRIMARY_K / 16 bytes per expert).
// Defaults match the legacy v4 primary shape (hidden=2048, inter=512 →
// GEMM1 K=hidden=2048, N=2*inter=1024).
#ifndef TASK29_PRIMARY_N_VALUE
#define TASK29_PRIMARY_N_VALUE 1024
#endif
#ifndef TASK29_PRIMARY_K_VALUE
#define TASK29_PRIMARY_K_VALUE 2048
#endif

namespace TASK29_IMPL_NAMESPACE {

namespace sf {
static constexpr int NVFP4_BLOCK = 16;
static constexpr int MIN_N = 128;

__host__ __device__ inline int align_to(int dim, int alignment) {
    return ((dim + alignment - 1) / alignment) * alignment;
}
}  // namespace sf

static constexpr int BLOCK_M = 16;
static constexpr int BLOCK_N = TASK29_BLOCK_N_VALUE;
static constexpr int BLOCK_K = 128;

static constexpr int ATOM_M = 16;
static constexpr int ATOM_N = 8;
static constexpr int ATOM_K = 64;
static constexpr int K_BLOCKS = BLOCK_K / ATOM_K;

static constexpr int ATOMS_M = BLOCK_M / ATOM_M;
static constexpr int ATOMS_N = BLOCK_N / ATOM_N;

static constexpr int WARPS_M = 1;
static constexpr int WARPS_N = 4;
static constexpr int NUM_WARPS = WARPS_M * WARPS_N;
static constexpr int THREADS = NUM_WARPS * 32;

static constexpr int ATOMS_M_PER_WARP = ATOMS_M / WARPS_M;
static constexpr int ATOMS_N_PER_WARP = ATOMS_N / WARPS_N;

static constexpr int ROW_BYTES = BLOCK_K / 2;
static constexpr int SA_BYTES = BLOCK_M * ROW_BYTES;
static constexpr int SB_BYTES = BLOCK_N * ROW_BYTES;

// The input scale-factor buffer is padded/swizzled in 128-row groups by the
// custom_moe setup path. For M=1 we still copy the full 128-row scale tile so
// the SM120 scale-vector addressing matches task24 exactly.
static constexpr int SF_A_STAGE = K_BLOCKS * 512;
static constexpr int SF_B_STAGE = K_BLOCKS * 512;
static constexpr int STAGE_BYTES = SA_BYTES + SB_BYTES + SF_A_STAGE + SF_B_STAGE;
static constexpr int NUM_STAGES = 2;
static constexpr int MBAR_OFFSET = STAGE_BYTES * NUM_STAGES;
static constexpr int SMEM_BYTES = MBAR_OFFSET + NUM_STAGES * 8;

static constexpr int PRIMARY_K = TASK29_PRIMARY_K_VALUE;
static constexpr int PRIMARY_N = TASK29_PRIMARY_N_VALUE;
static constexpr int SF_M_TILE_STRIDE = (PRIMARY_K / BLOCK_K) * SF_A_STAGE;
static constexpr float FP4_MAX_INV = 1.0f / 6.0f;

static constexpr int INTRA_SPLIT = 4;
static constexpr int INTRA_WARPS = WARPS_N * INTRA_SPLIT;
static constexpr int INTRA_THREADS = INTRA_WARPS * 32;
static constexpr int INTRA_GROUP_BYTES = STAGE_BYTES * NUM_STAGES;
static constexpr int INTRA_MBAR_OFFSET = INTRA_GROUP_BYTES * INTRA_SPLIT;
static constexpr int INTRA_PARTIAL_OFFSET =
    INTRA_MBAR_OFFSET + INTRA_SPLIT * NUM_STAGES * 8;
static constexpr int INTRA_PARTIAL_BYTES =
    INTRA_SPLIT * BLOCK_N * static_cast<int>(sizeof(float));
static constexpr int INTRA_SMEM_BYTES =
    INTRA_PARTIAL_OFFSET + INTRA_PARTIAL_BYTES;

__device__ __forceinline__ int sw64(int addr) {
    return addr ^ ((addr >> 3) & 0x30);
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fused_silu(float x) {
    return x * reciprocal_approximate_ftz(1.0f + __expf(-x));
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

__device__ __forceinline__ int find_expert_for_row(
    const int64_t* __restrict__ expert_first_token_offset,
    int row_idx,
    int num_experts) {
    int lo = 0;
    int hi = num_experts;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        int64_t off = __ldg(&expert_first_token_offset[mid]);
        if (off <= row_idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ __launch_bounds__(THREADS, 4)
void atrex_gemm1_m1_splitk_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partials,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int expanded_num_tokens,
    int split_k) {
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int split_id = blockIdx.x % split_k;
    const int tile_n = (blockIdx.x / split_k) % n_tiles;
    const int row_idx = blockIdx.x / (split_k * n_tiles);
    if (row_idx >= expanded_num_tokens) {
        return;
    }

    const int expert_id = find_expert_for_row(
        expert_first_token_offset, row_idx, num_experts);
    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    const int64_t expert_next = __ldg(&expert_first_token_offset[expert_id + 1]);
    if (row_idx < expert_off || row_idx >= expert_next) {
        return;
    }

    const int total_k_tiles = K / BLOCK_K;
    const int split_tiles = (total_k_tiles + split_k - 1) / split_k;
    const int kt_begin = split_id * split_tiles;
    int kt_end = kt_begin + split_tiles;
    if (kt_end > total_k_tiles) {
        kt_end = total_k_tiles;
    }
    if (kt_begin >= kt_end) {
        return;
    }

    const int n_start = tile_n * BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int warp_n = warp_id;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    float acc[ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        acc[ni][0] = 0.f;
        acc[ni][1] = 0.f;
        acc[ni][2] = 0.f;
        acc[ni][3] = 0.f;
    }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);

    const int64_t sf_a_padded_start = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    const uint8_t* sf_a_tile_ptr =
        sf_a_base + sf_a_padded_start * PRIMARY_K / sf::NVFP4_BLOCK;

    const int sf_b_super_tile = n_start / 128;
    const int sf_b_row_start = sf_b_super_tile * 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * PRIMARY_N * PRIMARY_K /
            sf::NVFP4_BLOCK +
        sf_b_super_tile * SF_M_TILE_STRIDE;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64) {
        sf_thread_dst_off = threadIdx.x * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_dst_off;
    } else {
        int t = threadIdx.x - 64;
        sf_thread_dst_off = t * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_dst_off;
    }

    if (threadIdx.x == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

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
        int col_tile = n_start - sf_b_row_start + b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    const int sf_row = ((lane_id & 1) * 8 + l);
    const int a_sf_base = (sf_row & 31) * 16 + (sf_row >> 5) * 4;

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = smem + buf * STAGE_BYTES;
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord, row_idx);
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        if (threadIdx.x < 64) {
            cp_async_cg_16(sSF_A + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        } else {
            cp_async_cg_16(sSF_B + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        }
        cp_async_commit();
    };

    issue_stage(kt_begin, 0);

    int phase = 0;
    for (int kt = kt_begin; kt < kt_end; kt++) {
        const int local_kt = kt - kt_begin;
        const int buf = local_kt & 1;
        uint8_t* sA_curr = smem + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < kt_end) {
            issue_stage(kt + 1, 1 - buf);
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

            const int phys_row = ldsm_a_m_off + ldsm_a_row;
            uint32_t ldsm_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off +
                                   ldsm_a_k_off)));

            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ldsm_addr));

            uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                sSF_A_curr + kb_sf_a + a_sf_base);

#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                mma_nvfp4_m16n8k64(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                    sfa_v, sfb_r[ni]);
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    const float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
    const int row_base = row_idx * N + n_start;

#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        if (l == 0 && n_start + col + 1 < N) {
            int out_idx = row_base + col;
            if (split_k == 1) {
                output[out_idx] = __float2bfloat16(acc[ni][0] * alpha_v);
                output[out_idx + 1] = __float2bfloat16(acc[ni][1] * alpha_v);
            } else {
                int64_t pbase =
                    (static_cast<int64_t>(split_id) * expanded_num_tokens +
                     row_idx) * N + n_start + col;
                partials[pbase] = acc[ni][0];
                partials[pbase + 1] = acc[ni][1];
            }
        }
    }
}

__global__ __launch_bounds__(THREADS, 4)
void atrex_gemm1_grouped_m16_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int K) {
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int expert_id = blockIdx.x / n_tiles;
    const int tile_n = blockIdx.x - expert_id * n_tiles;
    if (expert_id >= num_experts) {
        return;
    }

    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    const int64_t expert_next = __ldg(&expert_first_token_offset[expert_id + 1]);
    int total_rows = static_cast<int>(expert_next - expert_off);
    if (total_rows <= 0) {
        return;
    }

    const int n_start = tile_n * BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int warp_n = warp_id;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    for (int chunk_row = 0; chunk_row < total_rows; chunk_row += BLOCK_M) {
    int valid_rows = total_rows - chunk_row;
    if (valid_rows > BLOCK_M) {
        valid_rows = BLOCK_M;
    }
    const int64_t row_start = expert_off + chunk_row;
    const int sf_chunk_tile = chunk_row / sf::MIN_N;
    const int sf_chunk_row = chunk_row - sf_chunk_tile * sf::MIN_N;

    float acc[ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        acc[ni][0] = 0.f;
        acc[ni][1] = 0.f;
        acc[ni][2] = 0.f;
        acc[ni][3] = 0.f;
    }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);

    const int64_t sf_a_padded_start = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    const uint8_t* sf_a_tile_ptr =
        sf_a_base + sf_a_padded_start * PRIMARY_K / sf::NVFP4_BLOCK +
        sf_chunk_tile * SF_M_TILE_STRIDE;

    const int sf_b_super_tile = n_start / 128;
    const int sf_b_row_start = sf_b_super_tile * 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * PRIMARY_N * PRIMARY_K /
            sf::NVFP4_BLOCK +
        sf_b_super_tile * SF_M_TILE_STRIDE;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64) {
        sf_thread_dst_off = threadIdx.x * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_dst_off;
    } else {
        int t = threadIdx.x - 64;
        sf_thread_dst_off = t * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_dst_off;
    }

    if (threadIdx.x == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

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
        int col_tile = n_start - sf_b_row_start + b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    const int sf_row = sf_chunk_row + ((lane_id & 1) * 8 + l);
    const int a_sf_base = (sf_row & 31) * 16 + (sf_row >> 5) * 4;

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = smem + buf * STAGE_BYTES;
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord,
                        static_cast<int>(row_start));
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        if (threadIdx.x < 64) {
            cp_async_cg_16(sSF_A + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        } else {
            cp_async_cg_16(sSF_B + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        }
        cp_async_commit();
    };

    issue_stage(0, 0);

    int phase = 0;
    const int total_k_tiles = K / BLOCK_K;
    for (int kt = 0; kt < total_k_tiles; kt++) {
        const int buf = kt & 1;
        uint8_t* sA_curr = smem + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < total_k_tiles) {
            issue_stage(kt + 1, 1 - buf);
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

            const int phys_row = ldsm_a_m_off + ldsm_a_row;
            uint32_t ldsm_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off +
                                   ldsm_a_k_off)));

            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ldsm_addr));

            uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                sSF_A_curr + kb_sf_a + a_sf_base);

#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                mma_nvfp4_m16n8k64(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                    sfa_v, sfb_r[ni]);
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    const float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        if (n_start + col + 1 < N) {
            int row0 = l;
            int row1 = row0 + 8;
            if (row0 < valid_rows) {
                int64_t out_idx =
                    (row_start + row0) * static_cast<int64_t>(N) +
                    n_start + col;
                output[out_idx] = __float2bfloat16(acc[ni][0] * alpha_v);
                output[out_idx + 1] =
                    __float2bfloat16(acc[ni][1] * alpha_v);
            }
            if (row1 < valid_rows) {
                int64_t out_idx =
                    (row_start + row1) * static_cast<int64_t>(N) +
                    n_start + col;
                output[out_idx] = __float2bfloat16(acc[ni][2] * alpha_v);
                output[out_idx + 1] =
                    __float2bfloat16(acc[ni][3] * alpha_v);
            }
        }
    }
    }
}

__global__ void atrex_gemm1_grouped_m16_fused_act_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    uint8_t* __restrict__ output_fp4,
    const float* __restrict__ fc2_act_global_scale,
    uint8_t* __restrict__ act_sf_flat,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int K) {
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int expert_id = blockIdx.x / n_tiles;
    const int tile_n = blockIdx.x - expert_id * n_tiles;
    if (expert_id >= num_experts) {
        return;
    }

    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    const int64_t expert_next = __ldg(&expert_first_token_offset[expert_id + 1]);
    int total_rows = static_cast<int>(expert_next - expert_off);
    if (total_rows <= 0) {
        return;
    }

    const int n_start = tile_n * BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int warp_n = warp_id;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    for (int chunk_row = 0; chunk_row < total_rows; chunk_row += BLOCK_M) {
    int valid_rows = total_rows - chunk_row;
    if (valid_rows > BLOCK_M) {
        valid_rows = BLOCK_M;
    }
    const int64_t row_start = expert_off + chunk_row;
    const int sf_chunk_tile = chunk_row / sf::MIN_N;
    const int sf_chunk_row = chunk_row - sf_chunk_tile * sf::MIN_N;

    float acc[ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        acc[ni][0] = 0.f;
        acc[ni][1] = 0.f;
        acc[ni][2] = 0.f;
        acc[ni][3] = 0.f;
    }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);

    const int64_t sf_a_padded_start = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    const uint8_t* sf_a_tile_ptr =
        sf_a_base + sf_a_padded_start * PRIMARY_K / sf::NVFP4_BLOCK +
        sf_chunk_tile * SF_M_TILE_STRIDE;

    const int sf_b_super_tile = n_start / 128;
    const int sf_b_row_start = sf_b_super_tile * 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * PRIMARY_N * PRIMARY_K /
            sf::NVFP4_BLOCK +
        sf_b_super_tile * SF_M_TILE_STRIDE;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64) {
        sf_thread_dst_off = threadIdx.x * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_dst_off;
    } else {
        int t = threadIdx.x - 64;
        sf_thread_dst_off = t * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_dst_off;
    }

    if (threadIdx.x == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

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
        int col_tile = n_start - sf_b_row_start + b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    const int sf_row = sf_chunk_row + ((lane_id & 1) * 8 + l);
    const int a_sf_base = (sf_row & 31) * 16 + (sf_row >> 5) * 4;

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = smem + buf * STAGE_BYTES;
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord,
                        static_cast<int>(row_start));
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        if (threadIdx.x < 64) {
            cp_async_cg_16(sSF_A + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        } else {
            cp_async_cg_16(sSF_B + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        }
        cp_async_commit();
    };

    issue_stage(0, 0);

    int phase = 0;
    const int total_k_tiles = K / BLOCK_K;
    for (int kt = 0; kt < total_k_tiles; kt++) {
        const int buf = kt & 1;
        uint8_t* sA_curr = smem + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < total_k_tiles) {
            issue_stage(kt + 1, 1 - buf);
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

            const int phys_row = ldsm_a_m_off + ldsm_a_row;
            uint32_t ldsm_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off +
                                   ldsm_a_k_off)));

            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ldsm_addr));

            uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                sSF_A_curr + kb_sf_a + a_sf_base);

#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                mma_nvfp4_m16n8k64(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                    sfa_v, sfb_r[ni]);
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    const float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
    const float global_scale = fc2_act_global_scale
        ? __ldg(&fc2_act_global_scale[expert_id]) : 1.0f;
    const float inv_global_scale = reciprocal_approximate_ftz(global_scale);

    const int inter_size = N / 2;
    const int half_inter_bytes = inter_size / 2;
    const int out_sf_k_vecs = inter_size / sf::NVFP4_BLOCK;
    const int out_sf_m_tile_stride = ((out_sf_k_vecs + 3) / 4) * 512;
    int64_t psf = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)),
        sf::MIN_N);
    uint8_t* sf_expert = act_sf_flat + psf * out_sf_k_vecs;

    const int out_col_base =
        n_start / 2 + warp_n * (ATOMS_N_PER_WARP * ATOM_N / 2);
    const int out_byte_col = out_col_base / 2;
    const int sf_k_idx = out_col_base / sf::NVFP4_BLOCK;

    auto fused_pair = [&](int gate_ni, int up_ni, int acc_idx) -> float {
        float gate = acc[gate_ni][acc_idx] * alpha_v;
        float up = acc[up_ni][acc_idx] * alpha_v;
        return fused_silu(up) * gate;
    };

#pragma unroll
    for (int rs = 0; rs < 2; rs++) {
        int row = l + rs * 8;
        if (row >= valid_rows) {
            continue;
        }

        int ai = rs * 2;
        float v0 = fused_pair(0, 1, ai);
        float v1 = fused_pair(0, 1, ai + 1);
        float v2 = 0.f;
        float v3 = 0.f;
        if constexpr (ATOMS_N_PER_WARP >= 4) {
            v2 = fused_pair(2, 3, ai);
            v3 = fused_pair(2, 3, ai + 1);
        }

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

        int64_t byte_base =
            (row_start + row) * static_cast<int64_t>(half_inter_bytes) +
            out_byte_col;
        output_fp4[byte_base + g] = static_cast<uint8_t>(packed & 0xff);
        if constexpr (ATOMS_N_PER_WARP >= 4) {
            output_fp4[byte_base + 4 + g] =
                static_cast<uint8_t>((packed >> 8) & 0xff);
        }

        if (g == 0) {
            int sf_row_base =
                ((chunk_row + row) / 128) * out_sf_m_tile_stride +
                ((chunk_row + row) % 32) * 16 +
                (((chunk_row + row) % 128) / 32) * 4;
            sf_expert[sf_row_base + (sf_k_idx / 4) * 512 +
                      (sf_k_idx % 4)] = sf_byte;
        }
    }
    }
}

__global__ void atrex_reduce_splitk_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int expanded_num_tokens,
    int split_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = expanded_num_tokens * N;
    if (idx >= total) {
        return;
    }
    int row = idx / N;
    int expert_id = find_expert_for_row(
        expert_first_token_offset, row, num_experts);
    float sum = 0.f;
    int64_t stride = static_cast<int64_t>(expanded_num_tokens) * N;
    for (int s = 0; s < split_k; s++) {
        sum += partials[static_cast<int64_t>(s) * stride + idx];
    }
    float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
    output[idx] = __float2bfloat16(sum * alpha_v);
}

__device__ __forceinline__ void ensure_fused_reduce_epoch(
    unsigned long long* __restrict__ state,
    unsigned long long epoch_state) {
    constexpr unsigned long long COUNT_MASK = 0xffULL;
    constexpr unsigned long long EPOCH_MASK = ~COUNT_MASK;
    while (true) {
        unsigned long long old = atomicAdd(state, 0ULL);
        if ((old & EPOCH_MASK) == epoch_state) {
            return;
        }
        if (atomicCAS(state, old, epoch_state) == old) {
            return;
        }
    }
}

__global__ __launch_bounds__(THREADS, 4)
void atrex_gemm1_m1_splitk_fused_reduce_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partials,
    unsigned long long* __restrict__ done_states,
    unsigned long long epoch_state,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int expanded_num_tokens,
    int split_k) {
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int split_id = blockIdx.x % split_k;
    const int tile_n = (blockIdx.x / split_k) % n_tiles;
    const int row_idx = blockIdx.x / (split_k * n_tiles);
    if (row_idx >= expanded_num_tokens) {
        return;
    }

    const int expert_id = find_expert_for_row(
        expert_first_token_offset, row_idx, num_experts);
    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    const int64_t expert_next = __ldg(&expert_first_token_offset[expert_id + 1]);
    if (row_idx < expert_off || row_idx >= expert_next) {
        return;
    }

    const int total_k_tiles = K / BLOCK_K;
    const int split_tiles = (total_k_tiles + split_k - 1) / split_k;
    const int kt_begin = split_id * split_tiles;
    int kt_end = kt_begin + split_tiles;
    if (kt_end > total_k_tiles) {
        kt_end = total_k_tiles;
    }
    if (kt_begin >= kt_end) {
        return;
    }

    const int n_start = tile_n * BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int warp_n = warp_id;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    if (split_k > 1 && threadIdx.x == 0) {
        int state_idx = row_idx * n_tiles + tile_n;
        ensure_fused_reduce_epoch(&done_states[state_idx], epoch_state);
    }

    float acc[ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        acc[ni][0] = 0.f;
        acc[ni][1] = 0.f;
        acc[ni][2] = 0.f;
        acc[ni][3] = 0.f;
    }

    extern __shared__ uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFFSET);

    const int64_t sf_a_padded_start = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    const uint8_t* sf_a_tile_ptr =
        sf_a_base + sf_a_padded_start * PRIMARY_K / sf::NVFP4_BLOCK;

    const int sf_b_super_tile = n_start / 128;
    const int sf_b_row_start = sf_b_super_tile * 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * PRIMARY_N * PRIMARY_K /
            sf::NVFP4_BLOCK +
        sf_b_super_tile * SF_M_TILE_STRIDE;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (threadIdx.x < 64) {
        sf_thread_dst_off = threadIdx.x * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_dst_off;
    } else {
        int t = threadIdx.x - 64;
        sf_thread_dst_off = t * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_dst_off;
    }

    if (threadIdx.x == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

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
        int col_tile = n_start - sf_b_row_start + b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    const int sf_row = ((lane_id & 1) * 8 + l);
    const int a_sf_base = (sf_row & 31) * 16 + (sf_row >> 5) * 4;

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = smem + buf * STAGE_BYTES;
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord, row_idx);
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        if (threadIdx.x < 64) {
            cp_async_cg_16(sSF_A + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        } else {
            cp_async_cg_16(sSF_B + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        }
        cp_async_commit();
    };

    issue_stage(kt_begin, 0);

    int phase = 0;
    for (int kt = kt_begin; kt < kt_end; kt++) {
        const int local_kt = kt - kt_begin;
        const int buf = local_kt & 1;
        uint8_t* sA_curr = smem + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < kt_end) {
            issue_stage(kt + 1, 1 - buf);
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

            const int phys_row = ldsm_a_m_off + ldsm_a_row;
            uint32_t ldsm_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off +
                                   ldsm_a_k_off)));

            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ldsm_addr));

            uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                sSF_A_curr + kb_sf_a + a_sf_base);

#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                mma_nvfp4_m16n8k64(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                    sfa_v, sfb_r[ni]);
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    const float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
    const int row_base = row_idx * N + n_start;

    if (split_k == 1) {
#pragma unroll
        for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
            int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
            int col = n_atom * ATOM_N + g * 2;
            if (l == 0 && n_start + col + 1 < N) {
                int out_idx = row_base + col;
                output[out_idx] = __float2bfloat16(acc[ni][0] * alpha_v);
                output[out_idx + 1] = __float2bfloat16(acc[ni][1] * alpha_v);
            }
        }
        return;
    }

#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        if (l == 0 && n_start + col + 1 < N) {
            int64_t pbase =
                (static_cast<int64_t>(split_id) * expanded_num_tokens +
                 row_idx) * N + n_start + col;
            partials[pbase] = acc[ni][0];
            partials[pbase + 1] = acc[ni][1];
        }
    }

    __syncthreads();
    __threadfence();
    __syncthreads();

    int state_idx = row_idx * n_tiles + tile_n;
    int is_last_split = 0;
    if (threadIdx.x == 0) {
        unsigned long long old_count =
            atomicAdd(&done_states[state_idx], 1ULL) & 0xffULL;
        is_last_split =
            (old_count + 1ULL == static_cast<unsigned long long>(split_k));
    }
    if (!__syncthreads_or(is_last_split)) {
        return;
    }

    if (threadIdx.x < BLOCK_N && n_start + threadIdx.x < N) {
        int idx = row_base + threadIdx.x;
        int64_t stride = static_cast<int64_t>(expanded_num_tokens) * N;
        float sum = 0.f;
        for (int s = 0; s < split_k; s++) {
            sum += partials[static_cast<int64_t>(s) * stride + idx];
        }
        output[idx] = __float2bfloat16(sum * alpha_v);
    }
}

__global__ __launch_bounds__(INTRA_THREADS, 1)
void atrex_gemm1_m1_intra_split4_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int expanded_num_tokens) {
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int tile_n = blockIdx.x % n_tiles;
    const int row_idx = blockIdx.x / n_tiles;
    if (row_idx >= expanded_num_tokens) {
        return;
    }

    const int expert_id = find_expert_for_row(
        expert_first_token_offset, row_idx, num_experts);
    const int64_t expert_off = __ldg(&expert_first_token_offset[expert_id]);
    const int64_t expert_next = __ldg(&expert_first_token_offset[expert_id + 1]);
    if (row_idx < expert_off || row_idx >= expert_next) {
        return;
    }

    const int n_start = tile_n * BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int split_id = warp_id / WARPS_N;
    const int warp_n = warp_id - split_id * WARPS_N;
    const int local_tid = threadIdx.x - split_id * THREADS;
    const int g = lane_id & 3;
    const int l = lane_id >> 2;

    const int total_k_tiles = K / BLOCK_K;
    constexpr int split_tiles = (PRIMARY_K / BLOCK_K) / INTRA_SPLIT;
    const int kt_begin = split_id * split_tiles;
    const int kt_end = kt_begin + split_tiles;
    if (kt_end > total_k_tiles) {
        return;
    }

    float acc[ATOMS_N_PER_WARP][4];
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        acc[ni][0] = 0.f;
        acc[ni][1] = 0.f;
        acc[ni][2] = 0.f;
        acc[ni][3] = 0.f;
    }

    extern __shared__ uint8_t smem[];
    uint8_t* group_base = smem + split_id * INTRA_GROUP_BYTES;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        smem + INTRA_MBAR_OFFSET + split_id * NUM_STAGES * 8);

    const int64_t sf_a_padded_start = sf::align_to(
        static_cast<int>(expert_off + expert_id * (sf::MIN_N - 1)), sf::MIN_N);
    const uint8_t* sf_a_tile_ptr =
        sf_a_base + sf_a_padded_start * PRIMARY_K / sf::NVFP4_BLOCK;

    const int sf_b_super_tile = n_start / 128;
    const int sf_b_row_start = sf_b_super_tile * 128;
    const uint8_t* sf_b_tile_ptr =
        sf_b_base +
        static_cast<int64_t>(expert_id) * PRIMARY_N * PRIMARY_K /
            sf::NVFP4_BLOCK +
        sf_b_super_tile * SF_M_TILE_STRIDE;

    int sf_thread_dst_off = -1;
    const uint8_t* sf_thread_src = nullptr;
    if (local_tid < 64) {
        sf_thread_dst_off = local_tid * 16;
        sf_thread_src = sf_a_tile_ptr + sf_thread_dst_off;
    } else {
        int t = local_tid - 64;
        sf_thread_dst_off = t * 16;
        sf_thread_src = sf_b_tile_ptr + sf_thread_dst_off;
    }

    if (local_tid == 0) {
        mbarrier_init(&mbar[0], 1);
        mbarrier_init(&mbar[1], 1);
        prefetch_tma_descriptor(&tma_a_desc);
        prefetch_tma_descriptor(&tma_b_desc);
    }
    __syncthreads();

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
        int col_tile = n_start - sf_b_row_start + b_col_base[ni] + l;
        b_sf_base[ni] = (col_tile & 31) * 16 + (col_tile >> 5) * 4;
    }

    const int sf_row = ((lane_id & 1) * 8 + l);
    const int a_sf_base = (sf_row & 31) * 16 + (sf_row >> 5) * 4;

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = group_base + buf * STAGE_BYTES;
        if (local_tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord, row_idx);
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        if (local_tid < 64) {
            cp_async_cg_16(sSF_A + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        } else {
            cp_async_cg_16(sSF_B + sf_thread_dst_off,
                           sf_thread_src + sf_k_off);
        }
        cp_async_commit();
    };

    issue_stage(kt_begin, 0);

    int phase = 0;
    for (int kt = kt_begin; kt < kt_end; kt++) {
        const int local_kt = kt - kt_begin;
        const int buf = local_kt & 1;
        uint8_t* sA_curr = group_base + buf * STAGE_BYTES;
        uint8_t* sB_curr = sA_curr + SA_BYTES;
        uint8_t* sSF_A_curr = sB_curr + SB_BYTES;
        uint8_t* sSF_B_curr = sSF_A_curr + SF_A_STAGE;

        mbarrier_wait_parity(&mbar[buf], phase);
        cp_async_wait_group<0>();
        __syncthreads();

        if (kt + 1 < kt_end) {
            issue_stage(kt + 1, 1 - buf);
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

            const int phys_row = ldsm_a_m_off + ldsm_a_row;
            uint32_t ldsm_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(
                    sA_curr + sw64(phys_row * ROW_BYTES + kb_off +
                                   ldsm_a_k_off)));

            uint32_t a0, a1, a2, a3;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ldsm_addr));

            uint32_t sfa_v = *reinterpret_cast<const uint32_t*>(
                sSF_A_curr + kb_sf_a + a_sf_base);

#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                mma_nvfp4_m16n8k64(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    a0, a1, a2, a3, b_r[ni][0], b_r[ni][1],
                    sfa_v, sfb_r[ni]);
            }
        }

        if (buf == 1) {
            phase ^= 1;
        }
        __syncthreads();
    }

    float* partial_s = reinterpret_cast<float*>(smem + INTRA_PARTIAL_OFFSET);
#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        if (l == 0 && col + 1 < BLOCK_N) {
            int p = split_id * BLOCK_N + col;
            partial_s[p] = acc[ni][0];
            partial_s[p + 1] = acc[ni][1];
        }
    }
    __syncthreads();

    if (threadIdx.x < BLOCK_N && n_start + threadIdx.x < N) {
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < INTRA_SPLIT; s++) {
            sum += partial_s[s * BLOCK_N + threadIdx.x];
        }
        float alpha_v = alpha ? __ldg(&alpha[expert_id]) : 1.0f;
        int out_idx = row_idx * N + n_start + threadIdx.x;
        output[out_idx] = __float2bfloat16(sum * alpha_v);
    }
}

}  // namespace TASK29_IMPL_NAMESPACE
