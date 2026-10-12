/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch,
  class DispatchPolicy_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class GmemTiledCopyScaleA_,
  class GmemTiledCopyScaleB_>
struct CollectiveMmaScale
{
  //
  // Type Aliases
  //
  using DispatchPolicy = DispatchPolicy_;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using GmemTiledCopyScaleA = GmemTiledCopyScaleA_;
  using GmemTiledCopyScaleB = GmemTiledCopyScaleB_;
  using ArchTag = Arch;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    Shape<int,int,int> scale_shape;
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    uint32_t *ptr_scale_A;
    StrideA dScaleA;
    uint32_t *ptr_scale_B;
    StrideB dScaleB;
  };

  // Device side kernel params
  using Params = Arguments;
  // store params to const scale tensor in mainloop
  // avoid modify kernel file for fp4
  Params params_;

  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  // Tensor mA_scale_mkl_;

  //
  // Methods
  //

  template <class ProblemShape>
  CUTLASS_DEVICE
  CollectiveMmaScale(Params params, ProblemShape problem_shape_MNK, int batch_idx = 0) {
    params_ = params;

    auto M = get<0>(problem_shape_MNK);
    auto N = get<1>(problem_shape_MNK);
    auto K = get<2>(problem_shape_MNK);

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementA, false, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);

    // update desc.ptr to current batch, use y_offset as batch will cause smaller bit wide
    gmem_tiled_copy_A.desc_.gmem_ptr += batch_idx * get<2>(params.dA) * sizeof(ElementA);
    gmem_tiled_copy_B.desc_.gmem_ptr += batch_idx * get<2>(params.dB) * sizeof(ElementB);
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    using X = Underscore;
    auto blk_shape_scale = make_shape(size<0>(TileShape{}) / 64, size<1>(TileShape{}) / 64, size<2>(TileShape{}) / 16 * 64 / 4);
    auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);

    using ThrLayoutVMNK = typename TiledMma::ThrLayoutVMNK;
    constexpr int warp_on_m = size<1>(ThrLayoutVMNK{});
    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

    auto M = get<0>(params_.scale_shape);
    auto N = get<1>(params_.scale_shape);
    auto K = get<2>(params_.scale_shape);

    // problem.k has divided 2 since regard input as int8, scale.k is K/16 but not K/32
    // /4 because regard meta as uint32
    int rectify_M = (M + size<0>(TileShape{}) - 1) / size<0>(TileShape{}) * size<0>(TileShape{});
    int rectify_N = (N + size<1>(TileShape{}) - 1) / size<1>(TileShape{}) * size<1>(TileShape{});
    int rectify_K = (cute::ceil_div(K, 16) + 1) / 2 * 2 * 64;

    int K_scale = cute::ceil_div(K, 16*4);
    Tensor mA_scale_mkl = make_tensor(make_gmem_ptr(params_.ptr_scale_A), make_shape(rectify_M, K_scale), make_stride(K_scale, 1));
    Tensor gA_scale = local_tile(mA_scale_mkl, make_shape(size<0>(TileShape{}), size<1>(TileShape{}), size<2>(TileShape{})/16/4), take<0,3>(blk_coord_mnkl), Step< _1,X,_1>{});

    Tensor mB_scale_nkl = make_tensor(make_gmem_ptr(params_.ptr_scale_B), make_shape(rectify_N, K_scale), make_stride(K_scale, 1));
    Tensor gB_scale = local_tile(mB_scale_nkl, make_shape(size<0>(TileShape{}), size<1>(TileShape{}), size<2>(TileShape{})/16/4), take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});

    GmemTiledCopyScaleA gmem_tiled_copy_A_scale;
    // GmemTiledCopyScaleA's thr layout's size is warp_on_m*32
    auto gmem_thr_copy_A_scale = gmem_tiled_copy_A_scale.get_slice(warp_m_idx * 32 + lane_idx);
    Tensor tAgAScale = gmem_thr_copy_A_scale.partition_S(gA_scale);

    GmemTiledCopyScaleB gmem_tiled_copy_B_scale;
    auto gmem_thr_copy_B_scale = gmem_tiled_copy_B_scale.get_slice(warp_n_idx * 32 + lane_idx);
    Tensor tBgBScale = gmem_thr_copy_B_scale.partition_S(gB_scale);

    // used to get vreg size
    uint32_t *tmp_ptr = nullptr;
    constexpr int meta_block_m = size<0>(TileShape{}) / 64;
    constexpr int meta_block_n = size<1>(TileShape{}) / 64;
    // /4 because regard meta as uint32
    constexpr int meta_block_k = size<2>(TileShape{}) / 16 * 64 / 4;
    Tensor sAScale = make_tensor(tmp_ptr, Shape<Int<meta_block_m>, Int<meta_block_k>>{});
    Tensor tAsAScale = gmem_thr_copy_A_scale.partition_D(sAScale);
    // vreg for one stage
    using ScaleARegType = decltype(make_fragment_like(tAsAScale));
    ScaleARegType tArAScale[4];

    Tensor sBScale = make_tensor(tmp_ptr, Shape<Int<meta_block_n>, Int<meta_block_k>>{});
    Tensor tBsBScale = gmem_thr_copy_B_scale.partition_D(sBScale);
    using ScaleBRegType = decltype(make_fragment_like(tBsBScale));
    ScaleBRegType tBrBScale[4];

    auto copy_scale_g2r = [&] (auto& dst, auto& src, int k_pipe, int trans=0) {
      int warp_idx = trans ? warp_n_idx : warp_m_idx;
      auto dst_ptr = dst.data();
      using Element = typename ScaleARegType::element_type;
      auto src_ptr = src.data() + k_pipe * 2;
      auto act_thread_id = warp_idx * 32 + lane_idx;
      auto row_idx = (act_thread_id % 4) * 64 + (act_thread_id%32)/4;
      constexpr int K_BLOCK_NUM = 2;
      for_each(make_int_sequence<K_BLOCK_NUM>{}, [&] (auto k_block) {
        uint32_t tmp0 = *(src_ptr + (row_idx + warp_idx*16)*K_scale + k_block);
        uint32_t tmp1 = *(src_ptr + (row_idx + 8 + warp_idx*16)*K_scale + k_block);
        uint16_t upper16b = tmp0 & 0xffff;
        uint16_t lower16b = tmp1 & 0xffff;
        *(dst_ptr + k_block*2) = upper16b | lower16b << 16;     // little end ?
        *(dst_ptr + k_block*2 + 1) = static_cast<uint16_t>((tmp0 >> 16) & 0xffff) | static_cast<uint16_t>((tmp1 >> 16) & 0xffff) << 16;
      });
    };

    //A scale
    copy_scale_g2r(tArAScale[0], gA_scale, 0);
    copy_scale_g2r(tArAScale[1], gA_scale, 1);
    copy_scale_g2r(tArAScale[2], gA_scale, 2);
    copy_scale_g2r(tArAScale[3], gA_scale, 3);

    //B scale
    copy_scale_g2r(tBrBScale[0], gB_scale, 0, 1);
    copy_scale_g2r(tBrBScale[1], gB_scale, 1, 1);
    copy_scale_g2r(tBrBScale[2], gB_scale, 2, 1);
    copy_scale_g2r(tBrBScale[3], gB_scale, 3, 1);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
        warp_idx
      );
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    // tsm ld swzl needn't distinguish params inner warp
    // use original smem_tiled_copy to avoid use specific tailed_layout_tv like below, use warp_idx*32 to get scaler layout of current warp
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_tsm_ld_tensor<SmemLayoutA>());                 // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    smem_tiled_copy_B.smem_base_ = storage.smem_b.data();
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_tsm_ld_tensor<SmemLayoutB>());                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    int mainloop_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      int meta_stage_idx = mainloop_idx % 4;
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
          cp_async_fence();
          if (k_tile_count > 0) { ++k_tile_iter; }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k_block
        if (meta_stage_idx == 0) {
          cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tArAScale[0](_,_,k_block), tCrB(_,_,k_block), tBrBScale[0](_,_,k_block), src_accum);
        } else if (meta_stage_idx == 1) {
          cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tArAScale[1](_,_,k_block), tCrB(_,_,k_block), tBrBScale[1](_,_,k_block), src_accum);
        } else if (meta_stage_idx == 2) {
          cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tArAScale[2](_,_,k_block), tCrB(_,_,k_block), tBrBScale[2](_,_,k_block), src_accum);
        } else if (meta_stage_idx == 3) {
          cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tArAScale[3](_,_,k_block), tCrB(_,_,k_block), tBrBScale[3](_,_,k_block), src_accum);
        }
        // cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });

      if (meta_stage_idx == 0) {
        copy_scale_g2r(tArAScale[0], gA_scale, mainloop_idx + 4);
        copy_scale_g2r(tBrBScale[0], gB_scale, mainloop_idx + 4, 1);
      } else if (meta_stage_idx == 1) {
        copy_scale_g2r(tArAScale[1], gA_scale, mainloop_idx + 4);
        copy_scale_g2r(tBrBScale[1], gB_scale, mainloop_idx + 4, 1);
      } else if (meta_stage_idx == 2) {
        copy_scale_g2r(tArAScale[2], gA_scale, mainloop_idx + 4);
        copy_scale_g2r(tBrBScale[2], gB_scale, mainloop_idx + 4, 1);
      } else if (meta_stage_idx == 3) {
        copy_scale_g2r(tArAScale[3], gA_scale, mainloop_idx + 4);
        copy_scale_g2r(tBrBScale[3], gB_scale, mainloop_idx + 4, 1);
      }
      ++mainloop_idx;
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
