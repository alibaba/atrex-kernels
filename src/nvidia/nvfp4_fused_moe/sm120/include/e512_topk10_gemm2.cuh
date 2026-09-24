#include <cuda.h>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TASK30_IMPL_NAMESPACE
#define TASK30_IMPL_NAMESPACE task30_gemm2_small_m_v1
#endif

#ifndef TASK30_BLOCK_N_VALUE
#define TASK30_BLOCK_N_VALUE 128
#endif

// PRIMARY_N / PRIMARY_K are the matrix N / K dims this kernel instance is
// compiled for. They are NOT tile dims (those are BLOCK_N / BLOCK_K) — they
// only show up in SF (block-scale) address arithmetic:
//   - PRIMARY_K controls the A-side per-M-row SF stride.
//   - PRIMARY_N * PRIMARY_K is the per-expert SF stride for the B (weight)
//     tensor, so it MUST match how the upstream `w2_sf` quantization packs
//     experts (PRIMARY_N * PRIMARY_K / 16 bytes per expert).
// Defaults match the legacy v4 primary shape (hidden=2048, inter=512 →
// GEMM2 K=inter=512, N=hidden=2048).
#ifndef TASK30_PRIMARY_N_VALUE
#define TASK30_PRIMARY_N_VALUE 2048
#endif
#ifndef TASK30_PRIMARY_K_VALUE
#define TASK30_PRIMARY_K_VALUE 512
#endif

namespace TASK30_IMPL_NAMESPACE {

namespace sf {
static constexpr int NVFP4_BLOCK = 16;
static constexpr int MIN_N = 128;

__host__ __device__ inline int align_to(int dim, int alignment) {
    return ((dim + alignment - 1) / alignment) * alignment;
}
}  // namespace sf

static constexpr int BLOCK_M = 16;
static constexpr int BLOCK_N = TASK30_BLOCK_N_VALUE;
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

// The shared scale-factor stage keeps the full 128-row swizzled shape. The
// M=1 row-grid path loads only its valid four-byte fragment and zero-fills the
// unused rows directly in shared memory.
static constexpr int SF_A_STAGE = K_BLOCKS * 512;
static constexpr int SF_B_STAGE = K_BLOCKS * 512;
static constexpr int STAGE_BYTES = SA_BYTES + SB_BYTES + SF_A_STAGE + SF_B_STAGE;
static constexpr int NUM_STAGES = 2;
static constexpr int MBAR_OFFSET = STAGE_BYTES * NUM_STAGES;
static constexpr int SMEM_BYTES = MBAR_OFFSET + NUM_STAGES * 8;

// GEMM2: [expanded, K=inter] x [E, N=hidden, K=inter]^T. The legacy epilogue
// scatters routed contributions with BF16 atomics. The fixed-order e512_topk10
// epilogue instead materializes one BF16 row per top-k slot and lets the last
// producer CTA reduce those slots in rank order with FP32 accumulation.
// PRIMARY_K/N are compile-time constants used only for SF address arithmetic;
// see comment above for the vLLM-side packing constraint.
static constexpr int PRIMARY_K = TASK30_PRIMARY_K_VALUE;
static constexpr int PRIMARY_N = TASK30_PRIMARY_N_VALUE;
// A swizzled SF block covers one 64-wide MMA atom.  Express the stride in
// those atoms instead of 128-wide K tiles so shapes with a single 64-wide
// tail (for example K=320) do not round the per-row scale stride down.
static constexpr int SF_M_TILE_STRIDE = (PRIMARY_K / ATOM_K) * 512;

__device__ __forceinline__ int sw64(int addr) {
    return addr ^ ((addr >> 3) & 0x30);
}

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void stage_sf_a_rows_16(
    uint8_t* smem_base,
    const uint8_t* gmem_base,
    int source_row_begin,
    int dest_row_begin,
    int valid_rows,
    int sf_lane,
    bool atom_valid) {
    int const atom = sf_lane >> 5;
    int const row_mod = sf_lane & 31;
    uint8_t* smem_segment =
        smem_base + atom * 512 + row_mod * 16;
    *reinterpret_cast<uint4*>(smem_segment) = make_uint4(0, 0, 0, 0);
    if (!atom_valid) {
        return;
    }

#pragma unroll
    for (int group = 0; group < 4; ++group) {
        int const dest_row = row_mod + group * 32;
        if (dest_row >= dest_row_begin &&
            dest_row < dest_row_begin + valid_rows) {
            int const source_row =
                source_row_begin + dest_row - dest_row_begin;
            int const source_offset =
                atom * 512 + (source_row & 31) * 16 +
                (source_row >> 5) * 4;
            *reinterpret_cast<uint32_t*>(smem_segment + group * 4) =
                *reinterpret_cast<const uint32_t*>(
                    gmem_base + source_offset);
        }
    }
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

static constexpr int FIXED_FINALIZE_TOPK = 10;

// Number of non-sentinel (>= 0) top-k slots for a token. Fixed-order finalize
// uses this as its completion threshold so that partial top-k (some slots set
// to -1 and dropped by routing_sort) still fires the reduction once every
// valid expert row has arrived. Dense top-k yields FIXED_FINALIZE_TOPK, which
// preserves the original behavior exactly.
__device__ __forceinline__ int e512t10_valid_slot_count(
    const int* __restrict__ topk_ids, int token, int topk) {
    int count = 0;
    for (int k = 0; k < topk; ++k) {
        if (__ldg(&topk_ids[token * topk + k]) >= 0) {
            ++count;
        }
    }
    return count;
}

template <int TOPK>
__device__ __forceinline__ void fixed_order_finalize_pair(
    const __nv_bfloat16* __restrict__ expert_rows,
    __nv_bfloat16* __restrict__ final_output,
    const float* __restrict__ topk_weights,
    const int* __restrict__ topk_ids,
    int token,
    int M,
    int N,
    int col,
    bool output_preinitialized) {
    float out0 = 0.0f;
    float out1 = 0.0f;
    if (output_preinitialized) {
        const __nv_bfloat16* previous =
            final_output + static_cast<int64_t>(token) * N + col;
        out0 = __bfloat162float(previous[0]);
        out1 = __bfloat162float(previous[1]);
    }

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        // Skip sentinel slots (topk_ids < 0): routing_sort drops them, so their
        // expert_rows entry is never written. Summing only valid slots keeps
        // partial top-k correct and matches the count-agnostic reference.
        if (__ldg(&topk_ids[token * TOPK + k]) < 0) {
            continue;
        }
        const int expanded_id = k * M + token;
        const __nv_bfloat16* contribution =
            expert_rows + static_cast<int64_t>(expanded_id) * N + col;
        const float weight = __ldg(&topk_weights[token * TOPK + k]);
        out0 += weight * __bfloat162float(contribution[0]);
        out1 += weight * __bfloat162float(contribution[1]);
    }

    __nv_bfloat16* output =
        final_output + static_cast<int64_t>(token) * N + col;
    *reinterpret_cast<__nv_bfloat162*>(output) =
        __floats2bfloat162_rn(out0, out1);
}

template <bool FIXED_ORDER_FINALIZE>
__global__ __launch_bounds__(THREADS, 4)
void atrex_gemm2_m1_row_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ final_output,
    const int64_t* __restrict__ expert_first_token_offset,
    const int* __restrict__ permuted_row_to_unpermuted_row,
    const float* __restrict__ sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int expanded_num_tokens,
    int split_k,
    __nv_bfloat16* __restrict__ expert_rows,
    const float* __restrict__ topk_weights,
    int* __restrict__ completion_counters,
    bool output_preinitialized,
    const int* __restrict__ topk_ids) {
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
    if (row_idx < expert_off || row_idx >= expert_next || row_idx != expert_off) {
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

    const int total_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    const int split_tiles = (total_k_tiles + split_k - 1) / split_k;
    const int kt_begin = split_id * split_tiles;
    int kt_end = kt_begin + split_tiles;
    if (kt_end > total_k_tiles) {
        kt_end = total_k_tiles;
    }
    if (kt_begin >= kt_end) {
        return;
    }

    auto issue_stage = [&](int kt, int buf) {
        const int k_coord = kt * ROW_BYTES;
        uint8_t* stage = smem + buf * STAGE_BYTES;
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[buf], SA_BYTES + SB_BYTES);
            tma_copy_2d(&tma_a_desc, &mbar[buf], stage, k_coord,
                        static_cast<int>(expert_off));
            tma_copy_3d(&tma_b_desc, &mbar[buf], stage + SA_BYTES,
                        k_coord, n_start, expert_id);
        }
        uint8_t* sSF_A = stage + SA_BYTES + SB_BYTES;
        uint8_t* sSF_B = sSF_A + SF_A_STAGE;
        const int64_t sf_k_off = static_cast<int64_t>(kt) * SF_A_STAGE;
        const int sf_atom = sf_thread_dst_off / 512;
        const bool sf_atom_valid =
            kt * BLOCK_K + sf_atom * ATOM_K < K;
        uint8_t* sf_dst = threadIdx.x < 64
            ? sSF_A + sf_thread_dst_off
            : sSF_B + sf_thread_dst_off;
        if (threadIdx.x < 64) {
            *reinterpret_cast<uint4*>(sf_dst) = make_uint4(0, 0, 0, 0);
            if (sf_atom_valid && (threadIdx.x & 31) == 0) {
                *reinterpret_cast<uint32_t*>(sf_dst) =
                    *reinterpret_cast<const uint32_t*>(
                        sf_thread_src + sf_k_off);
            }
        } else if (sf_atom_valid) {
            cp_async_cg_16(sf_dst, sf_thread_src + sf_k_off);
        } else {
            *reinterpret_cast<uint4*>(sf_dst) = make_uint4(0, 0, 0, 0);
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
            if (kt * BLOCK_K + kb * ATOM_K >= K) {
                continue;
            }
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
    const int expanded_id = __ldg(&permuted_row_to_unpermuted_row[row_idx]);
    const int original_token = expanded_id % M;
    const float output_scale = FIXED_ORDER_FINALIZE
        ? alpha_v
        : alpha_v * __ldg(&sorted_scales[row_idx]);

#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        int out_col = n_start + col;
        if (l == 0 && out_col + 1 < N) {
            __nv_bfloat162 value =
                __floats2bfloat162_rn(acc[ni][0] * output_scale,
                                       acc[ni][1] * output_scale);
            if constexpr (FIXED_ORDER_FINALIZE) {
                __nv_bfloat16* out =
                    expert_rows + static_cast<int64_t>(expanded_id) * N +
                    out_col;
                *reinterpret_cast<__nv_bfloat162*>(out) = value;
            } else {
                __nv_bfloat16* out =
                    final_output + static_cast<int64_t>(original_token) * N +
                    out_col;
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(out), value);
            }
        }
    }

    if constexpr (FIXED_ORDER_FINALIZE) {
        // Every producer thread publishes its columns before one thread makes
        // this top-k slot visible through the completion counter.
        if (l == 0) {
            __threadfence();
        }
        __syncthreads();
        int* is_last = reinterpret_cast<int*>(smem);
        if (threadIdx.x == 0) {
            cuda::atomic_ref<int, cuda::thread_scope_device> counter(
                completion_counters[original_token * n_tiles + tile_n]);
            const int old =
                counter.fetch_add(1, cuda::memory_order_acq_rel);
            *is_last = old == e512t10_valid_slot_count(
                topk_ids, original_token, FIXED_FINALIZE_TOPK) - 1;
        }
        __syncthreads();

        if (*is_last && l == 0) {
#pragma unroll
            for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
                const int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
                const int out_col = n_start + n_atom * ATOM_N + g * 2;
                if (out_col + 1 < N) {
                    fixed_order_finalize_pair<FIXED_FINALIZE_TOPK>(
                        expert_rows, final_output, topk_weights, topk_ids,
                        original_token, M, N, out_col,
                        output_preinitialized);
                }
            }
        }
    }
}

template <bool FIXED_ORDER_FINALIZE>
__global__ __launch_bounds__(THREADS, 4)
void atrex_gemm2_small_m_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    const uint8_t* __restrict__ sf_a_base,
    const uint8_t* __restrict__ sf_b_base,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ final_output,
    const int64_t* __restrict__ expert_first_token_offset,
    const int* __restrict__ permuted_row_to_unpermuted_row,
    const float* __restrict__ sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    bool use_shared_sf_staging,
    __nv_bfloat16* __restrict__ expert_rows,
    const float* __restrict__ topk_weights,
    int* __restrict__ completion_counters,
    bool output_preinitialized,
    const int* __restrict__ topk_ids) {
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
        const int sf_atom = sf_thread_dst_off / 512;
        const bool sf_atom_valid =
            kt * BLOCK_K + sf_atom * ATOM_K < K;
        uint8_t* sf_dst = threadIdx.x < 64
            ? sSF_A + sf_thread_dst_off
            : sSF_B + sf_thread_dst_off;
        if (threadIdx.x < 64 && use_shared_sf_staging &&
            PRIMARY_K == 320 && PRIMARY_N == 2560) {
            stage_sf_a_rows_16(
                sSF_A, sf_a_tile_ptr + sf_k_off,
                sf_chunk_row, sf_chunk_row, valid_rows, threadIdx.x,
                sf_atom_valid);
        } else if (sf_atom_valid) {
            cp_async_cg_16(sf_dst, sf_thread_src + sf_k_off);
        } else {
            *reinterpret_cast<uint4*>(sf_dst) = make_uint4(0, 0, 0, 0);
        }
        cp_async_commit();
    };

    issue_stage(0, 0);

    int phase = 0;
    const int total_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
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
            if (kt * BLOCK_K + kb * ATOM_K >= K) {
                continue;
            }
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
    const int row0 = l;
    const int row1 = row0 + 8;
    int expanded_id0 = 0;
    int expanded_id1 = 0;
    int original_token0 = 0;
    int original_token1 = 0;
    float output_scale0 = alpha_v;
    float output_scale1 = alpha_v;
    const bool valid0 = row0 < valid_rows;
    const bool valid1 = row1 < valid_rows;
    if (valid0) {
        const int64_t permuted_row = row_start + row0;
        expanded_id0 =
            __ldg(&permuted_row_to_unpermuted_row[permuted_row]);
        original_token0 = expanded_id0 % M;
        if constexpr (!FIXED_ORDER_FINALIZE) {
            output_scale0 *= __ldg(&sorted_scales[permuted_row]);
        }
    }
    if (valid1) {
        const int64_t permuted_row = row_start + row1;
        expanded_id1 =
            __ldg(&permuted_row_to_unpermuted_row[permuted_row]);
        original_token1 = expanded_id1 % M;
        if constexpr (!FIXED_ORDER_FINALIZE) {
            output_scale1 *= __ldg(&sorted_scales[permuted_row]);
        }
    }

#pragma unroll
    for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
        int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
        int col = n_atom * ATOM_N + g * 2;
        int out_col = n_start + col;
        if (out_col + 1 < N) {
            if (valid0) {
                __nv_bfloat16* out = FIXED_ORDER_FINALIZE
                    ? expert_rows + static_cast<int64_t>(expanded_id0) * N +
                        out_col
                    : final_output +
                        static_cast<int64_t>(original_token0) * N + out_col;
                __nv_bfloat162 value =
                    __floats2bfloat162_rn(acc[ni][0] * output_scale0,
                                           acc[ni][1] * output_scale0);
                if constexpr (FIXED_ORDER_FINALIZE) {
                    *reinterpret_cast<__nv_bfloat162*>(out) = value;
                } else {
                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(out), value);
                }
            }
            if (valid1) {
                __nv_bfloat16* out = FIXED_ORDER_FINALIZE
                    ? expert_rows + static_cast<int64_t>(expanded_id1) * N +
                        out_col
                    : final_output +
                        static_cast<int64_t>(original_token1) * N + out_col;
                __nv_bfloat162 value =
                    __floats2bfloat162_rn(acc[ni][2] * output_scale1,
                                           acc[ni][3] * output_scale1);
                if constexpr (FIXED_ORDER_FINALIZE) {
                    *reinterpret_cast<__nv_bfloat162*>(out) = value;
                } else {
                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(out), value);
                }
            }
        }
    }

    if constexpr (FIXED_ORDER_FINALIZE) {
        if (valid0 || valid1) {
            __threadfence();
        }
        __syncthreads();
        int* is_last = reinterpret_cast<int*>(smem);
        if (threadIdx.x < valid_rows) {
            is_last[threadIdx.x] = 0;
        }
        __syncthreads();

        // One lane owns each row's arrival. Top-k expert IDs are unique per
        // token, so the CTAs contributing to a (token, N-tile) are exactly that
        // token's non-sentinel slots; a fully sentinel token gets none and its
        // output row stays at the value routing_sort cleared it to.
        if (warp_id == 0 && g == 0) {
            if (valid0) {
                cuda::atomic_ref<int, cuda::thread_scope_device> counter(
                    completion_counters[
                        original_token0 * n_tiles + tile_n]);
                const int old =
                    counter.fetch_add(1, cuda::memory_order_acq_rel);
                is_last[row0] = old == e512t10_valid_slot_count(
                    topk_ids, original_token0, FIXED_FINALIZE_TOPK) - 1;
            }
            if (valid1) {
                cuda::atomic_ref<int, cuda::thread_scope_device> counter(
                    completion_counters[
                        original_token1 * n_tiles + tile_n]);
                const int old =
                    counter.fetch_add(1, cuda::memory_order_acq_rel);
                is_last[row1] = old == e512t10_valid_slot_count(
                    topk_ids, original_token1, FIXED_FINALIZE_TOPK) - 1;
            }
        }
        __syncthreads();

#pragma unroll
        for (int ni = 0; ni < ATOMS_N_PER_WARP; ni++) {
            const int n_atom = warp_n * ATOMS_N_PER_WARP + ni;
            const int out_col = n_start + n_atom * ATOM_N + g * 2;
            if (out_col + 1 < N) {
                if (valid0 && is_last[row0]) {
                    fixed_order_finalize_pair<FIXED_FINALIZE_TOPK>(
                        expert_rows, final_output, topk_weights, topk_ids,
                        original_token0, M, N, out_col,
                        output_preinitialized);
                }
                if (valid1 && is_last[row1]) {
                    fixed_order_finalize_pair<FIXED_FINALIZE_TOPK>(
                        expert_rows, final_output, topk_weights, topk_ids,
                        original_token1, M, N, out_col,
                        output_preinitialized);
                }
            }
        }
    }
    }
}

}  // namespace TASK30_IMPL_NAMESPACE
