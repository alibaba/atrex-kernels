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

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>

#include "cute/arch/copy_aiu_base.hpp"

#define DEBUG_PRINT 0
#if DEBUG_PRINT
#include <cute/util/debug.hpp>
#endif


namespace cute
{

template <class NumBitsPerTMA, typename Element, bool Trans, bool Swzl = true, typename Enable = void>
struct PPU0010_AIU_LOAD : PPU_AIU_LOAD_BASE {};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 16>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
    // desc.print(smem_ptr, gmem_ptr, coord_w, coord_h, coord_n);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr (Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=false,b16");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 16>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=true,b16");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 32>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b32 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 4),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b32 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 4),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=false,b32");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 32>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b32 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * sizeof(float)),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b32 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * sizeof(float)),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=true,b32");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 8>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=false,b8");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 8>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=true,b8");
#endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Swzl>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, Swzl, cute::enable_if_t<sizeof_bits<Element>::value == 4>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
    coord_w /= 2;
    // desc.print(smem_ptr, gmem_ptr, coord_w, coord_h, coord_n);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr(Swzl) {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    } else {
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b8 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
          "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
          "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
          "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
      );
    }
#else
    CUTE_INVALID_CONTROL_PATH("Support for AIU_LOAD has not been enabled for Trans=false,b4");
#endif
  }
};

template<typename Element, int CUBE_H, int CUBE_W, bool SWAP>
CUTE_HOST_DEVICE static void
ppu_tsm_ld_swzl_sim(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int stage) {
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * stage;

  int slice_idx = coord_w / (32 / sizeof(Element));

  // no trans can always copy 32b each time
  uint32_t *slice_base = reinterpret_cast<uint32_t*>(stage_base);
  slice_base += CUBE_H * 8 * slice_idx;
  uint32_t *frag = reinterpret_cast<uint32_t*>(frag_ptr);

  int slice_start_vec = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;

  int lane_idx = threadIdx.x % 32;
  // each warp is arrange in 8x4 for each vreg
  int lane_row_idx = lane_idx / 4 + coord_h;
  int lane_col_idx = lane_idx % 4;
  // four vreg
  CUTE_NO_UNROLL
  for (int v = 0; v < 4; v++) {
    // cur lane's cur vreg is on which row of original tsm layout
    int vreg_row_idx;
    if (!SWAP) {
      // vreg 1/3 is on lower 8 rows
      vreg_row_idx = (v % 2) * 8 + lane_row_idx;
    } else {
      // vreg 2/3 is on lower 8 rows
      vreg_row_idx = (v / 2) * 8 + lane_row_idx;
    }

    // cache line index, four rows in one cache line
    int vreg_line_idx = vreg_row_idx / 4;
    int vreg_vec_idx;
    if (!SWAP) {
      // vreg 2/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v / 2);
    } else {
      // vreg 1/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v % 2);
    }
    // swizzle between two vector in odd cache line
    int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
    // swizzle for each slice
    int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;

    // int32 offset of cur reg
    int lane_32b_off = vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
    frag[v] = slice_base[lane_32b_off];
  }

}

template <typename Element, bool Trans>
struct PPU0010_TSM_LD_SWZL_IMPL;

template <typename Element>
struct PPU0010_TSM_LD_SWZL_IMPL<Element, false> {
  CUTE_HOST_DEVICE void operator()(int *vreg, Element *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=false");
#endif
  }
};

template <typename Element>
struct PPU0010_TSM_LD_SWZL_IMPL<Element, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, Element *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for b8 and Trans=true");
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<half_t, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, half_t *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<bfloat16_t, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, bfloat16_t *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<float, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, float *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n16.x1.swzl.trans.shared.b32 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};


template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum = 1>
struct PPU0010_TSM_LD_SWZL;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, int InstNum>
struct PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, InstNum> {

  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
  int *vreg = reinterpret_cast<int *>(frag_ptr);
  int channel_bytes_offset = coord_w * sizeof(Element);
  PPU0010_TSM_LD_SWZL_IMPL<Element, false>()(vreg, stage_base, coord_h, CUBE_H, channel_bytes_offset);
#else
  CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=false");
  // ppu_tsm_ld_swzl_sim<Element, CUBE_H, CUBE_W, false>(frag_ptr, smem_base, coord_w, coord_h, stage);
#endif
  }
};

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, int InstNum>
struct PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, true, InstNum> {

  CUTE_HOST_DEVICE static void
  // similar to aiu_load with trans, first coord is coord on K, which is coord_h when trans
  copy(void *frag_ptr, void *smem_base, int coord_h, int coord_w, int cube_in_stage = 0, int stage = 0)
  {
    // if (cute::thread0()) {
    //   print("smem_base = 0x%x, CUBE_W = %d, CUBE_H = %d, coord_w = %d, coord_h = %d, cube_in_stage = %d, stage = %d\n",
    //         smem_base, CUBE_W, CUBE_H, coord_w, coord_h, cube_in_stage, stage);
    // }

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
  int *vreg = reinterpret_cast<int *>(frag_ptr);
  int channel_bytes_offset = coord_w * sizeof(Element);
  PPU0010_TSM_LD_SWZL_IMPL<Element, true>()(vreg, stage_base, coord_h, CUBE_H, channel_bytes_offset);
#else
  CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }

};
#undef DEBUG_PRINT
} // namespace cute
