/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
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
 */

/*! \file
    \brief Block Wise Scale configs specific for PPU MMA.

    Struct ScaleConfig is copied from the official v3.9.x branch, since the definition is not present in CUTLASS v3.6.
    If in the corresponding ppu branch of v3.9, we should directly include the source file below:

    #include  "include/cutlass/detail/blockwise_scale_layout.hpp"
*/

#pragma once

#include "cutlass/layout/matrix.h"
#include "cute/int_tuple.hpp"

namespace cutlass::detail{

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, bool K_majorSFA = false, bool K_majorSFB = false, bool K_padded = false>
struct PPUBlockwiseScaleConfig {

  using ShapeSFA = Shape<Shape<Int<SFVecSizeM>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;
  using ShapeSFB = Shape<Shape<Int<SFVecSizeN>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;

  using StrideSFA = conditional_t<K_majorSFA == false,
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>,
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using StrideSFB = conditional_t<K_majorSFB == false,
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>,
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutSFA = Layout<ShapeSFA, StrideSFA>;
  using LayoutSFB = Layout<ShapeSFB, StrideSFB>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSFA{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFA(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K] = cta_shape_mnk;
      if constexpr (K_majorSFA == false) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }


  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSFB{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFB(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (K_majorSFB == false) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;
      if constexpr (K_majorSFA == false) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(M, SFVecSizeM)));
      }
      else if constexpr (K_padded == false) {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
      else {
        // k_blocks round up to 4 because of device requirements. same to shape and SFB
        return make_stride(make_stride(_0{}, round_up(cute::ceil_div(K, SFVecSizeK), 4)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto mk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, cute::ceil_div(M, SFVecSizeM)),
                 make_shape(Int<SFVecSizeK>{}, K_padded? round_up(cute::ceil_div(K, SFVecSizeK), 4):cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFB(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;

      if constexpr (K_majorSFB == false) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(N, SFVecSizeN)));
      }
      else if constexpr (K_padded == false) {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
      else {
        return make_stride(make_stride(_0{}, round_up(cute::ceil_div(K, SFVecSizeK), 4)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto nk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, cute::ceil_div(N, SFVecSizeN)),
                 make_shape(Int<SFVecSizeK>{}, K_padded? round_up(cute::ceil_div(K, SFVecSizeK), 4):cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(nk_layout), L), append(stride(nk_layout), size(filter_zeros(nk_layout))));
  }
};

template<class MmaTileShape_MNK, bool kmajor_sfa = false, bool kmajor_sfb = false, bool k_padded = false>
constexpr auto ppu_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return PPUBlockwiseScaleConfig<
      size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{}),
      kmajor_sfa, kmajor_sfb, k_padded>{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
