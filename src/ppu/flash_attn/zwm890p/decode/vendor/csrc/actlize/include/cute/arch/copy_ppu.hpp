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

#include <cute/config.hpp>
#include <cute/arch/copy.hpp>

// ppu1.x always supports ldmatrix path, macro only for possible upper-layer call
#define CUTE_ARCH_LDSM_PPU_ENABLED 1
#define CUTE_ARCH_LDSM_PPU_ACTIVATED 1

// ppu1.x always supports async cp path, macro only for possible upper-layer call
#define CUTE_ARCH_CP_ASYNC_PPU_ENABLED 1

namespace cute
{

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// ldmatrix operations (from original ldmatrix ops)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#endif
  }
};

struct PPU_U32x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#endif
  }
};

struct PPU_U32x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#endif
  }
};

struct PPU_U16x2_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#endif
  }
};

struct PPU_U16x4_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#endif
  }
};

struct PPU_U16x8_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#endif
  }
};

//
// Legacy LDSM interfaces that aren't very useful
//

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm(uint128_t const* const smem_ptr,
          T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    PPU_U32x1_LDSM_N::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    PPU_U32x2_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    PPU_U32x4_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm_trans(uint128_t const* const smem_ptr,
                T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    PPU_U16x2_LDSM_T::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    PPU_U16x4_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    PPU_U16x8_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// cp.async operations (from original cp.async ops)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Copy via cp.async with caching at all levels
template <class TS, class TD = TS>
struct PPU_CP_ASYNC_CACHEALWAYS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "ppu.cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "ppu.cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("ppu.cp.async.ca.shared.global.LLC::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)));
  }
};

/// Copy via cp.async with caching at global level
template <class TS, class TD = TS>
struct PPU_CP_ASYNC_CACHEGLOBAL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "ppu.cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "ppu.cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)));
  }
};

/// Copy via cp.async with caching at all levels
template <class TS, class TD = TS>
struct PPU_CP_ASYNC_CACHEALWAYS_ZFILL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "ppu.cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "ppu.cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst,
       bool      pred)
  {
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    int src_size = pred ? sizeof(TS) : 0;
    asm volatile("ppu.cp.async.ca.shared.global.LLC::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)),
           "r"(src_size));
  }
};

/// Copy via cp.async with caching at global level
template <class TS, class TD = TS>
struct PPU_CP_ASYNC_CACHEGLOBAL_ZFILL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "ppu.cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
#ifdef __HGGCCC__
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "ppu.cp.async sizeof(TS) is not supported");
#else
  static_assert(sizeof(TS) == 16, "ppu.cp.async sizeof(TS) is not supported");
#endif

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst,
       bool      pred)
  {
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    int src_size = pred ? sizeof(TS) : 0;
    asm volatile("ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)),
           "r"(src_size));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
CUTE_HOST_DEVICE
void
cp_async_fence()
{
  asm volatile("ppu.cp.async.commit_group;\n" ::);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Blocks until all but N previous cp.async.commit_group operations have committed.
template <int N>
CUTE_HOST_DEVICE
void
cp_async_wait()
{
  if constexpr (N == 0) {
    asm volatile("ppu.cp.async.wait_all;\n" ::);
  } else {
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N));
  }
}

template <int N>
CUTE_HOST_DEVICE
void
cp_async_wait(Int<N>)
{
  return cp_async_wait<N>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
