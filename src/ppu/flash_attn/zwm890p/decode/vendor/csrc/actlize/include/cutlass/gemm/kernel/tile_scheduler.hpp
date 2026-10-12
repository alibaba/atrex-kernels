/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/gemm/kernel/ppu0015_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_group.hpp"
namespace cutlass::gemm {

//
// Tags for specifying tile schedulers
//

struct PersistentScheduler { };
struct StreamKScheduler { };
struct GroupScheduler { }; // Only used for Grouped GEMMs

struct SplitKSerialScheduler { };
struct PersistentSchedulerPPU0015 { };
struct DynamicPersistentSchedulerPPU0015 { };

} // namespace cutlass::gemm

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

template <
  class TileSchedulerTag,
  class ArchTag,
  class TileShape,
  class ClusterShape
  , class ProblemShapeType = void
>
struct TileSchedulerSelector {
  static_assert(cutlass::detail::dependent_false<ArchTag>,
      "Could not select a tile scheduler for given parameters.");
};

// Default (void) for PPU maps to PersistentTileSchedulerPPU
template <
  class ArchTag,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
    void,
    ArchTag,
    TileShape,
    ClusterShape
  > {
  using Scheduler = typename TileSchedulerSelector<
      PersistentScheduler,
      ArchTag,
      TileShape,
      ClusterShape
  >::Scheduler;
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  StreamKScheduler,
  Arch,
  TileShape,
  ClusterShape
  > {
  using Scheduler = PersistentTileSchedulerPPUStreamK<TileShape, ClusterShape>;
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
  , class GroupProblemShape
>
struct TileSchedulerSelector<
  GroupScheduler,
  Arch,
  TileShape,
  ClusterShape
  , GroupProblemShape
  > {
  using Scheduler = PersistentTileSchedulerPPUGroup<GroupProblemShape>;
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  cutlass::gemm::PersistentSchedulerPPU0015,
  Arch,
  TileShape,
  ClusterShape
  > {
#ifdef USE_PAD_BLOCK
  // would evaluate for batch-gemm later on
  using Scheduler = StaticPersistentTileSchedulerPadPPU0015;
#else
  using Scheduler = StaticPersistentTileSchedulerPPU0015;
#endif
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  cutlass::gemm::DynamicPersistentSchedulerPPU0015,
  Arch,
  TileShape,
  ClusterShape
  > {
  using Scheduler = DynamicPersistentTileSchedulerPPU0015;
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  PersistentScheduler,
  Arch,
  TileShape,
  ClusterShape
  > {
  using Scheduler = PersistentTileSchedulerPPU;
};

template <
  typename Arch,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  cutlass::gemm::SplitKSerialScheduler,
  Arch,
  TileShape,
  ClusterShape
  > {
 using Scheduler = PersistentTileSchedulerPPU;
};

} // namespace cutlass::gemm::kernel::detail
