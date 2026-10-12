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

#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>

namespace cute {

struct PPU_AIU_LOAD_BASE {};

// Inspects a tiled copy and whether its copy engine is AIU or not
template<typename CopyOperation, typename CopyInternalType, typename LayoutCopy_TV, typename ShapeTiler_MN>
constexpr bool is_aiu_copy_engine(cute::TiledCopy<cute::Copy_Atom<CopyOperation, CopyInternalType>, LayoutCopy_TV, ShapeTiler_MN> tiled_copy) {
  if constexpr (cute::is_void_v<CopyOperation>) {
    return false;
  }
  else {
    if constexpr (cute::is_base_of_v<cute::PPU_AIU_LOAD_BASE, CopyOperation>) {
      return true;
    }
  }
  return false;
}

struct AiuDesc {
  const uint8_t* gmem_ptr;  /*this ptr is invalid when transfer mix tensor to aiu*/
  int dim_h;
  int dim_w;
  int cube_h = 0; /* only used in ppu1.0*/
  int cube_w = 0; /* only used in ppu1.0*/
  int offset_w = 0; /* only used in ppu1.0*/
  int stride_w = 0; /* only used in ppu1.5*/

  template<typename Element, bool Trans, int Block_MN, int Block_K, typename Stride>
  CUTE_HOST_DEVICE
  void init(const uint8_t* gmem_ptr_, int MN, int K, Stride stride) {

    gmem_ptr = gmem_ptr_;

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    static constexpr int BlockContSize = (Trans ? Block_MN : Block_K) * sizeof_bits<Element>::value / 8;
    static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
    static constexpr int AiuContElemSize = AiuContByteSize / sizeof_bits<Element>::value * 8;

    if constexpr (!Trans) {
      dim_h = MN;
      dim_w = get<0>(stride);
      cube_h = Block_MN;
      cube_w = AiuContElemSize;
      offset_w = 0;
    } else {
      dim_h = K;
      dim_w = get<1>(stride);
      cube_h = Block_K;
      cube_w = AiuContElemSize;
      offset_w = 0;
    }
    if constexpr (sizeof_bits<Element>::value == 4) {
      dim_w /= 2;
      cube_w /= 2;
    }
#else
    if constexpr (!Trans) {
      dim_h = MN;
      dim_w = K;
      stride_w = get<0>(stride);
    } else {
      dim_h = K;
      dim_w = MN;
      stride_w = get<1>(stride);
    }
#endif
  }

  // for 1.5fa usage, transfer no_cont_dim/cont_dim/ld directly
  CUTE_HOST_DEVICE
  void init(const uint8_t* gmem_ptr_, int h, int w, int stride) {
    gmem_ptr = gmem_ptr_;
    dim_h = h;
    dim_w = w;
    stride_w = stride; /* only used in ppu1.5*/
  }


#if DEBUG_PRINT
  CUTE_HOST_DEVICE void
  print(void *smem_ptr, const void* gmem_ptr, int coord_w, int coord_h, int coord_n) {
    if (cute::thread0()) {
      printf("    AIU_LOAD, gmem_ptr = %p, smem_ptr = %p, dim_[h,w] = [%d,%d], cube_[h,w] = [%d,%d], coord_[h,w,n] = [%d,%d,%d], offset_w = %d\n",
              gmem_ptr, smem_ptr, dim_h, dim_w, cube_h, cube_w, coord_h, coord_w, coord_n, offset_w);
    }
  }
#endif
};

}
