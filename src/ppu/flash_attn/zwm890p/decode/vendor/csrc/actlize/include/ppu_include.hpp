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


#include "accutlass.hpp"
#include "cute/arch/util.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"

#include "cute/ppu_tensor_mix.hpp"

// keep this 2 legacy definition for FA
#include "cute/arch/mma_ppu.hpp"
#include "cute/atom/mma_traits_ppu.hpp"

#include "cute/arch/mma_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/arch/mma_ppu0015.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"

#include "cute/arch/copy_ppu_aiu.hpp"
#include "cute/atom/copy_traits_ppu_aiu.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/gemm/collective/ppu_mma_aiu_multistage.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_overlap_prologue.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_fp8.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_mixed_input.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array_overlap_prologue.hpp"
#include "cutlass/gemm/collective/ppu_mma_cpasync_multistage.hpp"

#include "cutlass/gemm/kernel/ppu_aiu_gemm.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_serial.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_array_persistent_overlap_prologue.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_streamk.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_persistent.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_persistent_overlap_prologue.hpp"
#include "cutlass/gemm/kernel/ppu_gemm.hpp"

#include "cute/ppu_stride.hpp"
#include "cute/algorithm/ppu_copy.hpp"
#include "cute/ppu_layout.hpp"

#include "cutlass/gemm/collective/ppu0015_mma_aiu_multistage_with_scale.hpp"
#include "cutlass/gemm/collective/ppu0015_mma_cpasync_multistage_fp4.hpp"
#include "cutlass/gemm/kernel/ppu0015_cpasync_gemm_with_scale.hpp" // fp4 gemm

#include "cutlass/utils.h"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/config/gemm_configs.hpp"

#include "cutlass/gemm/collective/ppu_mma_twostage_ldmatrix.hpp"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"
#include "cutlass/float4.h"
#include "cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
