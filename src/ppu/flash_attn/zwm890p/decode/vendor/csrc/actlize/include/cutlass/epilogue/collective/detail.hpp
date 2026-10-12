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

#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/util/type_traits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Stride>
constexpr bool
is_m_major() {
  return cutlass::gemm::detail::is_major<0,Stride>();
}

template <class Stride>
constexpr bool
is_n_major() {
  return cutlass::gemm::detail::is_major<1,Stride>();
}

template <class Stride>
constexpr bool
is_im2col() {
  return cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNWC>>
      || cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNHWC>>
      || cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNDHWC>>;
}

template<class Schedule>
struct ppu_is_ptr_array_tma : cute::false_type {};

template<>
struct ppu_is_ptr_array_tma<PtrArrayTmaWarpSpecializedCooperative> : cute::true_type {};

template<>
struct ppu_is_ptr_array_tma<PtrArrayTmaWarpSpecializedPingpong> : cute::true_type {};

template<>
struct ppu_is_ptr_array_tma<PtrArrayTmaWarpSpecialized> : cute::true_type {};

template<class Schedule>
static constexpr bool ppu_is_ptr_array_tma_v = ppu_is_ptr_array_tma<Schedule>::value;

template<class Schedule>
struct ppu_is_ptr_array_tma_cooperative : cute::false_type {};

template<>
struct ppu_is_ptr_array_tma_cooperative<PtrArrayTmaWarpSpecializedCooperative> : cute::true_type {};

template<class Schedule>
static constexpr bool ppu_is_ptr_array_tma_cooperative_v = ppu_is_ptr_array_tma_cooperative<Schedule>::value;

template<class Schedule>
struct ppu_is_ptr_array_tma_pingpong : cute::false_type {};

template<>
struct ppu_is_ptr_array_tma_pingpong<PtrArrayTmaWarpSpecializedPingpong> : cute::true_type {};

template<class Schedule>
static constexpr bool ppu_is_ptr_array_tma_pingpong_v = ppu_is_ptr_array_tma_pingpong<Schedule>::value;

template<class DispatchPolicy>
struct ppu_is_ptr_array_tma_dispatch_policy : cute::false_type {};

template<
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups
>
struct ppu_is_ptr_array_tma_dispatch_policy<
    PPUPtrArrayTmaWarpSpecialized<StagesC,
                                   StagesD,
                                   FragmentSize,
                                   ReuseSmemC,
                                   DelayTmaStore,
                                   NumEpilogueWarpGroups>>
    : cute::true_type {};

template<class DispatchPolicy>
static constexpr bool ppu_is_ptr_array_tma_dispatch_policy_v = ppu_is_ptr_array_tma_dispatch_policy<DispatchPolicy>::value;

using cutlass::atomic_maximum;

template <class T>
static constexpr int elements_per_access_v = cutlass::sizeof_bits<uint32_t>::value / cutlass::sizeof_bits<T>::value;

template <class EpilogueSchedule>
static constexpr bool ppu_is_cooperative_v =
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueSchedule> ||
  ppu_is_ptr_array_tma_cooperative_v<EpilogueSchedule>;

template <class EpilogueSchedule>
static constexpr bool ppu_is_warp_specialized_v =
  (!ppu_is_ptr_array_tma_cooperative_v<EpilogueSchedule> && ppu_is_ptr_array_tma_v<EpilogueSchedule>) ||
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecialized, EpilogueSchedule>;

template <class GmemLayoutTag>
static constexpr bool is_im2col_mode =
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNWC> ||
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNHWC> ||
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNDHWC>;

template <class T>
struct EmptyStorage {
  CUTLASS_HOST_DEVICE
  T* data() { return nullptr; }
};

template<class EpilogueSchedule, class Stride>
CUTLASS_HOST_DEVICE
auto get_epilogue_stride(Stride stride){
  if constexpr (cute::is_base_of_v<cutlass::gemm::EpilogueTransposed, EpilogueSchedule>||
                cute::is_base_of_v<cutlass::epilogue::PtrArrayNoSmemWarpSpecializedTransposed, EpilogueSchedule>) {
    return cute::make_stride(cute::get<1>(stride), cute::get<0>(stride), cute::get<2>(stride));
  }
  else {
    return stride;
  }
}

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithBias {
  static constexpr bool value = false;
  using type = typename ThreadEpilogueOp::ElementCompute;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithBias <ThreadEpilogueOp, cute::void_t<typename ThreadEpilogueOp::ElementBias>> {
  static constexpr bool value = true;
  using type = typename ThreadEpilogueOp::ElementBias;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithPerChannelScaling {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithPerChannelScaling <ThreadEpilogueOp, cute::enable_if_t<ThreadEpilogueOp::IsPerChannelScalingSupported>> {
  static constexpr bool value = true;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithActivation {
  static constexpr bool value = false;
  using type = void;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithActivation <ThreadEpilogueOp, cute::enable_if_t<ThreadEpilogueOp::IsEltActSupported>> {
  static constexpr bool value = true;
  using type = typename ThreadEpilogueOp::ActivationFn;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithElementwiseArguments : cute::false_type {};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithElementwiseArguments<
        ThreadEpilogueOp,
        cute::void_t<typename ThreadEpilogueOp::ElementwiseOp::Arguments>> : cute::true_type {};

// SFINAE helpers for detecting beta/beta_ptr/beta_ptr_array in EVT arguments.
template <class Arguments, class = void>
struct has_beta {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta<Arguments, cute::void_t<decltype(Arguments{}.thread.beta)>> {
  static constexpr bool value = true;
};

template <class Arguments, class = void>
struct has_beta_ptr {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta_ptr<Arguments, cute::void_t<decltype(Arguments{}.thread.beta_ptr)>> {
  static constexpr bool value = true;
};

template <class Arguments, class = void>
struct has_beta_ptr_array {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta_ptr_array<Arguments, cute::void_t<decltype(Arguments{}.thread.beta_ptr_array)>> {
  static constexpr bool value = true;
};

} // namespace detail
} // namespace collective
} // namespace epilogue
} // namespace cutlass
