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

/*! \file
    \brief Matrix multiply-add operations for PPU (half_t FMA + int8/int16 dp instructions)
*/

#pragma once

#include <hggc_fp16.h>

#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Half-precision FMA Mma operations (from original half-precision FMA)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<2,1,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 1> const &b,
    Array<half_t, 2> const &c
  ) {

    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 B = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB>
struct Mma<
  gemm::GemmShape<1,2,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 2> const &c
  ) {

    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 B = reinterpret_cast<__half2 const &>(b);
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma <
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 Blo = __low2half2(reinterpret_cast<__half2 const &>(b));
    __half2 Bhi = __high2half2(reinterpret_cast<__half2 const &>(b));

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(A, Blo, C[0]);
    __half2 Dhi = __hfma2(A, Bhi, C[1]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> const &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> const &>(Dhi);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

    __half2 Alo = __low2half2(reinterpret_cast<__half2 const &>(a));
    __half2 Ahi = __high2half2(reinterpret_cast<__half2 const &>(a));
    __half2 const & B = reinterpret_cast<__half2 const &>(b);

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(Alo, B, C[0]);
    __half2 Dhi = __hfma2(Ahi, B, C[0]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> &>(Dhi);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Integer dot-product Mma operations (from original integer dot-product)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<1,1,4>,
  1,
  int8_t,
  LayoutA,
  int8_t,
  LayoutB,
  int,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 4>;
  using Operator = OpMultiplyAdd;
  using ElementC = int;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int8_t, 4> const &a,
    Array<int8_t, 4> const &b,
    Array<int, 1> const &c
  ) {

    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("ppu.dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutC>
struct Mma<
  gemm::GemmShape<1, 1, 2>,
  1,
  int16_t,
  layout::RowMajor,
  int16_t,
  layout::ColumnMajor,
  int,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 2>;
  using Operator = OpMultiplyAdd;
  using ElementC = int;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int16_t, 2> const &a,
    Array<int16_t, 2> const &b,
    Array<int, 1> const &c
  ) {

    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 2; ++k) {
      d[0] += a[k] * b[k];
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
