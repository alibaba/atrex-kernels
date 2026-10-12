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

#include <cute/algorithm/copy.hpp>

namespace cute{

template <
  bool SplitAIU = true,
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA,
  class CopyPolicyB,
  class SrcEngineB, class SrcLayoutB,
  class DstEngineB, class DstLayoutB
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>     && dst_a,
  CopyPolicyB                  const& copy_policy_b,
  Tensor<SrcEngineB, SrcLayoutB> const& src_b,
  Tensor<DstEngineB, DstLayoutB>     && dst_b,
  int &warp_idx
) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if (warp_idx == 0) {
    copy(copy_policy_a, src_a, dst_a);
    copy(copy_policy_b, src_b, dst_b);
  }
#else
  if constexpr (SplitAIU) {
    if (warp_idx == 0) {
      copy(copy_policy_a, src_a, dst_a);
    } else if (warp_idx == 1) {
      copy(copy_policy_b, src_b, dst_b);
    }

  } else {
    if (warp_idx == 0) {
      copy(copy_policy_a, src_a, dst_a);
      copy(copy_policy_b, src_b, dst_b);
    }
  }
#endif
}

template <
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>      & dst_a,
  int &warp_idx
) {
  // if (warp_idx == 0) {
  //   copy(copy_policy_a, src_a, dst_a);
  // }

  // used by fa, will only split on last dim now
  static constexpr int rank  = DstLayoutA::rank;
  if (warp_idx == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<rank-1>(src_a) / 2; ++i) {
      copy(copy_policy_a, src_a(_,_,i), dst_a(_,_,i));
    }
  }
  else if (warp_idx == 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = size<rank-1>(src_a) / 2; i < size<rank-1>(src_a); ++i) {
      copy(copy_policy_a, src_a(_,_,i), dst_a(_,_,i));
    }
  }
}

template <
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>     && dst_a,
  int &warp_idx
) {
  if (warp_idx == 0) {
    copy(copy_policy_a, src_a, dst_a);
  }
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_ldcs(Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;

  CUTE_UNROLL
  for (int i = 0; i < size(dst); ++i) {
    if constexpr (is_gmem<SrcEngine>::value && is_rmem<DstEngine>::value &&
              sizeof(SrcType) == sizeof(DstType)) {
      if constexpr (sizeof_bits_v<SrcType> == 128) {
        *reinterpret_cast<uint4 *>(&dst(i)) = __ldcs(reinterpret_cast<const uint4*>(&src(i)));
      } else if constexpr (sizeof_bits_v<SrcType> == 64) {
        *reinterpret_cast<uint2 *>(&dst(i)) = __ldcs(reinterpret_cast<const uint2*>(&src(i)));
      } else if constexpr (sizeof_bits_v<SrcType> == 32) {
        *reinterpret_cast<uint *>(&dst(i)) = __ldcs(reinterpret_cast<const uint*>(&src(i)));
      } else if constexpr (sizeof_bits_v<SrcType> == 16) {
        *reinterpret_cast<uint16_t *>(&dst(i)) = __ldcs(reinterpret_cast<const uint16_t*>(&src(i)));
      } else if constexpr (sizeof_bits_v<SrcType> == 8) {
        *reinterpret_cast<uint8_t *>(&dst(i)) = __ldcs(reinterpret_cast<const uint8_t*>(&src(i)));
      } else {
        dst(i) = static_cast<DstType>(static_cast<SrcType>(src(i)));
      }
    } else {
      dst(i) = static_cast<DstType>(static_cast<SrcType>(src(i)));
    }
  }
}

// Specializaton for Atom AutoVectorizingCopyAssumedAlignment
template <int MaxVecBits, class CopyInternalType,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_g2r_ldcs(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>, CopyInternalType> const&,
     Tensor<SrcEngine, SrcLayout>                                            const& src,
     Tensor<DstEngine, DstLayout>                                                 & dst)
{
  constexpr int common_elem = CUTE_STATIC_V(max_common_vector(src, dst));
  constexpr int align_bits  = CUTE_STATIC_V(gcd(max_alignment(src), max_alignment(dst), Int<MaxVecBits>{}));
  static_assert(is_integral<decltype(Int<common_elem>{} * sizeof_bits_v<typename SrcEngine::value_type>)>::value, "Error: Attempting a subbit copy!");
  constexpr int vec_bits    = gcd(common_elem * sizeof_bits_v<typename SrcEngine::value_type>, align_bits);

  if constexpr (common_elem > 1 && ((vec_bits % 8) == 0)) {
    // If more than one element vectorizes to 8bits or more, then recast and copy
    using VecType = uint_bit_t<vec_bits>;
    // Preserve volatility
    using SrcVecType = conditional_t<is_volatile_v<typename SrcEngine::element_type>, VecType const volatile, VecType const>;
    using DstVecType = conditional_t<is_volatile_v<typename DstEngine::element_type>, VecType       volatile, VecType      >;

    // Recast
    Tensor src_v = recast<SrcVecType>(src);
    Tensor dst_v = recast<DstVecType>(dst);

    return copy_ldcs(src_v, dst_v);
  } else {
    return copy_ldcs(src, dst);
  }
}

template <class CopyPolicy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_g2r_ldcs(CopyPolicy          const& copy_policy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_g2r_ldcs(copy_policy, src, dst);
}


} // namespace cute
