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

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
#include <hggc_fp16.h>

// PPU supports DP4A/DP2A vector core

namespace cute {

template <class D, class A = D, class B = A, class C = D>
struct PPU_UniversalFMA
{
  using DRegisters = D[1];
  using ARegisters = A[1];
  using BRegisters = B[1];
  using CRegisters = C[1];

  CUTE_HOST_DEVICE static constexpr void
  fma(D      & d,
      A const& a,
      B const& b,
      C const& c)
  {
    d = (C)a * (C)b + c;
  }
};

struct PPU_2x1x1_F16F16F16F16
{
  using DRegisters = uint32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = cutlass::half_t[1];
  using CRegisters = uint32_t[1];
  CUTE_HOST_DEVICE static void
  fma(uint32_t             & d0,
      uint32_t        const& a0,
      cutlass::half_t const& b0,
      uint32_t        const& c0)
  {
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 100))
    __half2 const & A = reinterpret_cast<__half2 const &>(a0);
    __half2 const & B = __half2half2(reinterpret_cast<__half const &>(b0));
    __half2 const & C = reinterpret_cast<__half2 const &>(c0);
    __half2       & D = reinterpret_cast<__half2       &>(d0);
    D = __hfma2(A, B, C);
#endif
  }
};

struct PPU_1x2x1_F16F16F16F16
{
  using DRegisters = uint32_t[1];
  using ARegisters = cutlass::half_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[1];
  CUTE_HOST_DEVICE static void
  fma(uint32_t             & d0,
      cutlass::half_t const& a0,
      uint32_t        const& b0,
      uint32_t        const& c0)
  {
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 100))
    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a0));
    __half2 const & B = reinterpret_cast<__half2 const &>(b0);
    __half2 const & C = reinterpret_cast<__half2 const &>(c0);
    __half2       & D = reinterpret_cast<__half2       &>(d0);
    D = __hfma2(A, B, C);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Integer dot-product operations (DP4A/DP2A)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_DP4A
{
  using DRegisters = int32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = int32_t[1];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(int32_t& d, uint32_t const& a, uint32_t const& b, int32_t const& c)
  {
    asm volatile("ppu.dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
  }
};


} // namespace cute
