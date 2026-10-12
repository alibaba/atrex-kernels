/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/fast_math.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE static float divide_float(float a, float b) {
  #ifdef __HGGC_ARCH__
    return __fdividef(a, b);
  #else
    return (a / b);
  #endif
}

struct WorkTileInfo {
  int32_t M_idx = 0;
  int32_t N_idx = 0;
  int32_t L_idx = 0;
  bool is_valid_tile = false;

  CUTLASS_HOST_DEVICE
  bool
  is_valid() const {
    return is_valid_tile;
  }

  CUTLASS_HOST_DEVICE
  static WorkTileInfo
  invalid_work_tile() {
    return {-1, -1, -1, false};
  }

  CUTLASS_HOST_DEVICE
  bool
  is_final_split(uint32_t k_tiles_per_output_tile) const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  int32_t
  reduction_subtile_idx() const {
    return -1;
  }
};

// Users are not supposed to use this class directly.
// This is a CRTP base class for the actual tile schedulers.
template<typename Subclass, typename Params_, bool AdaptAlong>
struct TileSchedulerPPU0015 {
  int32_t current_work_linear_idx_;
  int32_t total_grid_size_;

  using Params = Params_;
  Params scheduler_params;
  struct SharedStorage {
    union {
      int dispatch_info[16];
      WorkTileInfo work_tile_info;
    };
  };
  SharedStorage* const scheduler_smem_;

  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  struct Arguments {
    int tb_per_cu = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    [[maybe_unused]] KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace=nullptr,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1,
    [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    bool alongN = true;
    if (AdaptAlong && (cute::shape<0>(problem_shape_mnkl) <= cute::shape<1>(problem_shape_mnkl))) {
      alongN = false;
    }
    RasterOrderOptions raster_order = arguments.raster_order;
    if (raster_order == RasterOrderOptions::Heuristic) {
      raster_order = alongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;
    }
    int swizzle_step = get_swizzle_step(tile_shape, arguments.tb_per_cu, alongN);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.tb_per_cu,
      raster_order,
      workspace,
      swizzle_step
    );

    return params;
  }

  CUTLASS_DEVICE explicit TileSchedulerPPU0015(Params const& params_, SharedStorage* const scheduler_smem):
  scheduler_params(params_), scheduler_smem_(scheduler_smem) {
    current_work_linear_idx_ = blockIdx.x;
    total_grid_size_ = gridDim.x;
  }

  // Reloaded interface that receives WorkTileInfo to deduce next work.
  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    return cute::make_tuple(fetch_next_work(), true);
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    return args.tb_per_cu >= 1;
  }

  // Given the inputs, computes the total number of output blocks over which this problem will compute.
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape, ClusterShape cluster_shape) {
    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(cta_shape)));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape),
      cta_m, cta_n
    );
  }

  // Given the inputs, computes the total number of output blocks over which this problem will compute.
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl,
                          TileShape tile_shape_mnk,
                          AtomThrShape atom_thr_shape_mnk,
                          ClusterShape cluster_shape_mnk) {
    auto [tiles_m, tiles_n, tiles_l] = product_each(ceil_div(select<0,1,3>(problem_shape_mnkl), take<0,2>(tile_shape_mnk)));
    auto cta_m = round_nearest(tiles_m * size<0>(atom_thr_shape_mnk), size<0>(cluster_shape_mnk));
    auto cta_n = round_nearest(tiles_n * size<1>(atom_thr_shape_mnk), size<1>(cluster_shape_mnk));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape_mnk),
      cta_m, cta_n
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    [[maybe_unused]] Params const& params,
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments,
    bool truncate_by_problem_size=true) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.tb_per_cu,
      arguments.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static dim3
  get_grid_shape(
      Params const& params,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo hw_info) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk);
    Arguments args{};
    args.raster_order = params.raster_order_ == RasterOrder::AlongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape_mnk),
      hw_info,
      args.tb_per_cu,
      args.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  static cutlass::Status
  initialize_workspace(void* workspace, size_t workspace_size) {
    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  template<class BlockShape>
  CUTLASS_HOST_DEVICE
  static int
  get_swizzle_step(BlockShape cta_shape, int tb_per_cu, bool alongN) {
    int tb_per_ce = 4 * tb_per_cu;
    int x_step = tb_per_ce;
    int y_step = 1;
    // size of one cu
    float ce_m = x_step * cute::shape<0>(cta_shape);
    float ce_n = y_step * cute::shape<1>(cta_shape);
    // ce_shape_ratio is better when closer to 1
    float ce_shape_ratio = ce_m > ce_n ? divide_float(ce_m, ce_n) : divide_float(ce_n, ce_m);
    for (int x = x_step; x > 0; --x) {
      if (tb_per_ce % x) {
        continue;
      }
      int y = tb_per_ce / x;
      float cur_ce_m = x * cute::shape<0>(cta_shape);
      float cur_ce_n = y * cute::shape<1>(cta_shape);
      float cur_shape_ratio = cur_ce_m > cur_ce_n ? divide_float(cur_ce_m, cur_ce_n) : divide_float(cur_ce_n, cur_ce_m);
      if (cur_shape_ratio < ce_shape_ratio) {
        x_step = x;
        y_step = y;
        ce_shape_ratio = cur_shape_ratio;
      }
    }
    if (alongN) {
      return x_step;
    } else {
      return y_step;
    }
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(int32_t linear_idx) const {
    if (linear_idx >= scheduler_params.blocks_per_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    int32_t work_idx_l, remainder;
    scheduler_params.divmod_batch_(work_idx_l, remainder, linear_idx);

    auto [work_idx_m, work_idx_n] = scheduler_params.get_work_idx_m_and_n(remainder);
#if 0
    if (threadIdx.x == 0) {
      int32_t ce_idx = cutlass::arch::CuId() / 4;
      int32_t cu_idx = cutlass::arch::CuId() & 0x3;
      printf("linear_idx=%d, blockIdx.x=%d, work_idx_l=%d, work_idx_m=%d, work_idx_n=%d, ce_idx=%d, cu_idx=%d\n", linear_idx, blockIdx.x, work_idx_l, work_idx_m, work_idx_n, ce_idx, cu_idx);
    }
#endif
    return {work_idx_m, work_idx_n, static_cast<int32_t>(work_idx_l), true};
  }
};

// CRTP class for normal static/dynamic tile scheduler
template<typename Subclass, typename Params>
struct TileSchedulerPPU0015Common: public TileSchedulerPPU0015<Subclass, Params, true> {
  using TileSchedulerPPU0015<Subclass, Params, true>::current_work_linear_idx_;
  using TileSchedulerPPU0015<Subclass, Params, true>::total_grid_size_;
  using TileSchedulerPPU0015<Subclass, Params, true>::scheduler_params;
  using TileSchedulerPPU0015<Subclass, Params, true>::scheduler_smem_;
  using typename TileSchedulerPPU0015<Subclass, Params, true>::SharedStorage;

  int preallocate_end_idx_;

  CUTLASS_DEVICE explicit TileSchedulerPPU0015Common(Params const& params_, SharedStorage* const scheduler_smem):
  TileSchedulerPPU0015<Subclass, Params, true>(params_, scheduler_smem) {
    preallocate_end_idx_ = scheduler_params.blocks_per_problem_;
    if (scheduler_params.cu39_begin_number_ > 0) {
      if (blockIdx.x < 36 * scheduler_params.tb_per_cu_) {
        current_work_linear_idx_ = blockIdx.x;
        total_grid_size_ = 36 * scheduler_params.tb_per_cu_;
        preallocate_end_idx_ = scheduler_params.cu39_begin_number_;
      } else {
        int idx_3cu = blockIdx.x - 36 * scheduler_params.tb_per_cu_;
        current_work_linear_idx_ = scheduler_params.cu39_begin_number_ + idx_3cu;
        total_grid_size_ = 3 * scheduler_params.tb_per_cu_;
      }
    }
  }
};

struct StaticPersistentTileSchedulerPPU0015:
public TileSchedulerPPU0015Common<StaticPersistentTileSchedulerPPU0015, StaticTileSchedulerPPU0015Params>
{
  static constexpr bool IsDynamicPersistent = false;
  using TileSchedulerPPU0015Common<StaticPersistentTileSchedulerPPU0015, StaticTileSchedulerPPU0015Params>::preallocate_end_idx_;
  // using Params = StaticTileSchedulerPPU0015Params;

  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerPPU0015(Params const& params_, SharedStorage* const scheduler_smem):
  TileSchedulerPPU0015Common<StaticPersistentTileSchedulerPPU0015, StaticTileSchedulerPPU0015Params>(params_, scheduler_smem) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() {
    int old_idx = current_work_linear_idx_;
    current_work_linear_idx_ += total_grid_size_;
    return get_current_work_for_linear_idx(old_idx);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    if (current_work_linear_idx_ >= preallocate_end_idx_) {
      return WorkTileInfo::invalid_work_tile();
    }
    return get_current_work();
  }

  static size_t get_workspace_size() {
    return 0;
  }
};

struct DynamicPersistentTileSchedulerPPU0015:
public TileSchedulerPPU0015Common<DynamicPersistentTileSchedulerPPU0015, DynamicTileSchedulerPPU0015Params>
{
  static constexpr bool IsDynamicPersistent = true;
  using TileSchedulerPPU0015Common<DynamicPersistentTileSchedulerPPU0015, DynamicTileSchedulerPPU0015Params>::preallocate_end_idx_;

  int* block_counter_;
  int* global_dispatch_blocks_; // 0: init, dispatch_blocks(>0): pre-allocate
  int* dynamic_counter_;
  int dynamic_start_idx_;
  int dispatch_blocks_;

  CUTLASS_DEVICE explicit DynamicPersistentTileSchedulerPPU0015(Params const& params_, SharedStorage* const scheduler_smem):
  TileSchedulerPPU0015Common<DynamicPersistentTileSchedulerPPU0015, DynamicTileSchedulerPPU0015Params>(params_, scheduler_smem) {
    block_counter_ = scheduler_params.workspace_;
    global_dispatch_blocks_ = scheduler_params.workspace_ + 32;
    dynamic_counter_ = scheduler_params.workspace_ + 64;
    dispatch_blocks_ = 0;
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() {
    // wave0: pre-allocate gridDim.x blocks
    if (threadIdx.x == 0) {
      atomicAdd(block_counter_, 1);
      scheduler_smem_->work_tile_info = get_current_work_for_linear_idx(current_work_linear_idx_);
    }
    __syncthreads();
    current_work_linear_idx_ += total_grid_size_;
    return scheduler_smem_->work_tile_info;
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    // wave1 start: determine dispatched blocks in wave0 and set boundary between pre-allocate and dynamic
    if (dispatch_blocks_ == 0) {
      if (threadIdx.x == 0) {
        int dispatch_blocks = *block_counter_;
        int rt_val = atomicCAS(global_dispatch_blocks_, 0, dispatch_blocks);
        if (rt_val != 0) {
          dispatch_blocks = rt_val;
        }
        if (scheduler_params.cu39_begin_number_ > 0) {
          if (dispatch_blocks < 36 * scheduler_params.tb_per_cu_) {
            total_grid_size_ = dispatch_blocks;
            dynamic_start_idx_ = min(gridDim.x + dispatch_blocks * (scheduler_params.wave_high_ - 1),
                                    scheduler_params.blocks_per_problem_ - 3 * scheduler_params.tb_per_cu_ - (gridDim.x - dispatch_blocks) * (scheduler_params.wave_low_ - 1));
            preallocate_end_idx_ = dynamic_start_idx_;
          } else if (dispatch_blocks >= 36 * scheduler_params.tb_per_cu_) {
            int wave_blocks = dispatch_blocks - 36 * scheduler_params.tb_per_cu_;
            dynamic_start_idx_ = min(scheduler_params.cu39_begin_number_ + 3 * scheduler_params.tb_per_cu_ + wave_blocks * (scheduler_params.wave_high_ - 1),
                                    scheduler_params.blocks_per_problem_ - (gridDim.x - dispatch_blocks) * (scheduler_params.wave_low_ - 1));
            if (blockIdx.x >= 36 * scheduler_params.tb_per_cu_) {
              total_grid_size_ = wave_blocks;
              preallocate_end_idx_ = dynamic_start_idx_;
            }
          }
        } else {
          total_grid_size_ = dispatch_blocks;
          dynamic_start_idx_ = min(gridDim.x + dispatch_blocks * (scheduler_params.wave_high_ - 1),
                                  scheduler_params.blocks_per_problem_ - (gridDim.x - dispatch_blocks) * (scheduler_params.wave_low_ - 1));
          preallocate_end_idx_ = dynamic_start_idx_;
        }
        if (blockIdx.x >= dispatch_blocks) {
          current_work_linear_idx_ = dynamic_start_idx_;
        }
        scheduler_smem_->dispatch_info[0] = dispatch_blocks;
        scheduler_smem_->dispatch_info[1] = current_work_linear_idx_;
        scheduler_smem_->dispatch_info[2] = total_grid_size_;
        scheduler_smem_->dispatch_info[3] = preallocate_end_idx_;
        scheduler_smem_->dispatch_info[4] = dynamic_start_idx_;
      }
      __syncthreads();
      dispatch_blocks_ = scheduler_smem_->dispatch_info[0];
      current_work_linear_idx_ = scheduler_smem_->dispatch_info[1];
      total_grid_size_ = scheduler_smem_->dispatch_info[2];
      preallocate_end_idx_ = scheduler_smem_->dispatch_info[3];
      dynamic_start_idx_ = scheduler_smem_->dispatch_info[4];
    }
#if PRINT_DISPATCH
    if (threadIdx.x == 0) {
      printf("blockidx: %d, dispatch_blocks: %d, 39cu_begin_number: %d, dynamic_start_idx: %d\n", blockIdx.x, dispatch_blocks_, scheduler_params.cu39_begin_number_, dynamic_start_idx_);
    }
#endif
    // wave1 and the following waves
    if (current_work_linear_idx_ < preallocate_end_idx_) { // pre-allocate for dispatch_blocks_
      int old_idx = current_work_linear_idx_;
      current_work_linear_idx_ += total_grid_size_;
      return get_current_work_for_linear_idx(old_idx);
    } else if (dynamic_start_idx_ == scheduler_params.blocks_per_problem_) { // pre-allocate all
      return WorkTileInfo::invalid_work_tile();
    } else {
      if (threadIdx.x == 0) {
        int idx = atomicAdd(dynamic_counter_, 1);
        idx += dynamic_start_idx_;
        if (scheduler_params.cu39_begin_number_ > dynamic_start_idx_ && idx >= scheduler_params.cu39_begin_number_) {
          idx += 3 * scheduler_params.tb_per_cu_;
        }
        scheduler_smem_->work_tile_info = get_current_work_for_linear_idx(idx);
      }
      __syncthreads();
      return scheduler_smem_->work_tile_info;
    }
  }

  static size_t get_workspace_size() {
    return 1024;
  }
};

// static scheduler used by batch-gemm, with TB round-up
struct StaticPersistentTileSchedulerPadPPU0015:
public TileSchedulerPPU0015<StaticPersistentTileSchedulerPadPPU0015, StaticTileSchedulerPadPPU0015Params, false>
{
  static constexpr bool IsDynamicPersistent = false;
  using TileSchedulerPPU0015<StaticPersistentTileSchedulerPadPPU0015, StaticTileSchedulerPadPPU0015Params, false>::current_work_linear_idx_;
  using TileSchedulerPPU0015<StaticPersistentTileSchedulerPadPPU0015, StaticTileSchedulerPadPPU0015Params, false>::total_grid_size_;
  using typename TileSchedulerPPU0015<StaticPersistentTileSchedulerPadPPU0015, StaticTileSchedulerPadPPU0015Params, false>::SharedStorage;

  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerPadPPU0015(Params const& params_, SharedStorage* const scheduler_smem):
  TileSchedulerPPU0015<StaticPersistentTileSchedulerPadPPU0015, StaticTileSchedulerPadPPU0015Params, false>(params_, scheduler_smem) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() {
    int old_idx = current_work_linear_idx_;
    current_work_linear_idx_ += total_grid_size_;
    return get_current_work_for_linear_idx(old_idx);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    return get_current_work();
  }

  static size_t get_workspace_size() {
    return 0;
  }
};

} // namespace cutlass::gemm::kernel::detail
