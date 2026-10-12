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

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cute/atom/copy_atom.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
#include "cutlass/epilogue/fusion/ppu_callbacks.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp"
#include "cutlass/gemm/kernel/ppu_aiu_gemm_parallel.hpp"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cute/arch/mma_ppu.hpp"
#include "cute/arch/mma_ppu0015.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"

#include "cute/algorithm/functional.hpp"

#include "ppu_include.hpp"
namespace cutlass {
namespace gemm {
namespace config {
using namespace cute;
using cutlass::config::KernelScheduleType;

template<int Stage, KernelScheduleType Schedule, bool UseAiu = true, typename DispatchPolicy_ = void>
struct KernelMetaTypes {
  static constexpr int KERNEL_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::KERNEL_SCHEDULE_MASK));
  static constexpr int MAINLOOP_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::MAINLOOP_DISPATCH_MASK));
  static constexpr int EPILOGUE_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::EPILOGUE_DISPATCH_MASK));

  static constexpr bool IS_PERSIST_OVERLAP = KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE)
                                          || KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE_DYNAMIC);
  static constexpr bool IS_GEMM_ARRAY = (MAINLOOP_SCHEDULE_CHOICE == int(KernelScheduleType::PTR_ARRAY));

  using KernelScheduler = cute::conditional_t<KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::STREAMK),
                                                cutlass::gemm::KernelAiuMultistageStreamK,
                                                cute::conditional_t<IS_PERSIST_OVERLAP,
                                                                    conditional_t<IS_GEMM_ARRAY,
                                                                                  cutlass::gemm::KernelAiuMultistageBatchArrayOverlapPrologue,
                                                                                  cutlass::gemm::KernelAiuMultistagePersistentOverlapPrologue>,
                                                                    cute::conditional_t<(KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::PERSISTENT)),
                                                                                        cutlass::gemm::KernelAiuMultistagePersistent,
                                                                                        cutlass::gemm::KernelAiuMultistage>>>;
  using DispatchPolicy = cute::conditional_t<(MAINLOOP_SCHEDULE_CHOICE == int(KernelScheduleType::FP8_SPLIT_ACC)),
                                                cutlass::gemm::MainloopPPUAiuFP8<Stage, KernelScheduler>,
                                                conditional_t<IS_GEMM_ARRAY,
                                                              cute::conditional_t<IS_PERSIST_OVERLAP,
                                                                                  cutlass::gemm::MainloopPPUAiuBatchArrayPersistOverlapPrologue<Stage>,
                                                                                  cutlass::gemm::MainloopPPUAiuBatchArray<Stage>>,
                                                              conditional_t<IS_PERSIST_OVERLAP,
                                                                            cutlass::gemm::MainloopPPUAiuPersistentOverlapPrologue<Stage, KernelScheduler>,
                                                                            cutlass::gemm::MainloopPPUAiu<Stage, KernelScheduler>>>
                                            >;

  using TileScheduler = cute::conditional_t<KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE),
                                                PersistentSchedulerPPU0015,
                                                cute::conditional_t<KERNEL_SCHEDULE_CHOICE == int(KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE_DYNAMIC),
                                                DynamicPersistentSchedulerPPU0015,
                                                void
                                            >>;

  using EpilogueDispatchPolicy = cute::conditional_t<(MAINLOOP_SCHEDULE_CHOICE == int(KernelScheduleType::PTR_ARRAY)),
                                                cutlass::epilogue::EpiloguePtrArraySimtVectorized,
                                                cutlass::epilogue::EpilogueSimtVectorized>;
};

template<int Stage, KernelScheduleType Schedule, typename DispatchPolicy_>
struct KernelMetaTypes<Stage, Schedule, false, DispatchPolicy_> {
  static constexpr int KERNEL_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::KERNEL_SCHEDULE_MASK));
  static constexpr int MAINLOOP_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::MAINLOOP_DISPATCH_MASK));
  static constexpr int EPILOGUE_SCHEDULE_CHOICE = (int(Schedule) & int(KernelScheduleType::EPILOGUE_DISPATCH_MASK));
  static constexpr bool IS_PERSIST_OVERLAP = false;
  using KernelScheduler = cute::conditional_t<MAINLOOP_SCHEDULE_CHOICE == int(KernelScheduleType::PTR_ARRAY),
                                              KernelMultistageBatchArray,
                                              KernelMultistage
                                             >;
  using DispatchPolicy = cute::conditional_t<MAINLOOP_SCHEDULE_CHOICE != int(KernelScheduleType::PTR_ARRAY),
                                             DispatchPolicy_,
                                             conditional_t<cute::is_same_v<DispatchPolicy_, cutlass::gemm::MainloopPPUTwoStageLdmatrix>,
                                                             cutlass::gemm::MainloopPPUTwoStageLdmatrixBatchArray,
                                                             cutlass::gemm::MainloopPPUCpAsyncBatchArray<Stage, KernelScheduler>>
                                            >;
  using EpilogueDispatchPolicy = cute::conditional_t<(MAINLOOP_SCHEDULE_CHOICE == int(KernelScheduleType::PTR_ARRAY)),
                                                      cutlass::epilogue::EpiloguePtrArraySimtVectorized,
                                                      cutlass::epilogue::EpilogueSimtVectorized>;
};

template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  int AlignmentA,
  typename ElementB,
  typename LayoutB,
  int AlignmentB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseAiu,
  bool ParallelSplitk = false,
  bool UseEpilogueEvt = true, /*cutlass gemm static will not use evt to avoid change device api*/
  KernelScheduleType schedule = KernelScheduleType::DEFAULT,
  typename OperatorClass = arch::OpClassTensorOp,
  typename ScaleGranularityShape = Shape<_0,_0,_0>
> struct GemmKernelConfig;

// normal gemm
// currently async cp + ld matrix
template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  int AlignmentA,
  typename ElementB,
  typename LayoutB,
  int AlignmentB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseEpilogueEvt,
  KernelScheduleType schedule
> struct GemmKernelConfig <
  Arch,
  ElementA,
  LayoutA,
  AlignmentA,
  ElementB,
  LayoutB,
  AlignmentB,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementD,
  AlignmentD,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  false,
  false,
  UseEpilogueEvt,
  schedule
> {

  static_assert(((int(schedule) & int(KernelScheduleType::KERNEL_SCHEDULE_MASK)) == int(KernelScheduleType::DEFAULT)), "only support default kernel schedule when not using AIU");
  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  static constexpr bool DisableCpAsync = (sizeof(ElementA) * AlignmentA < 4) || (sizeof(ElementB) * AlignmentB < 4);
  static constexpr int Alignment = platform::max(AlignmentC, AlignmentD);
  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using KernelScheduler = cute::conditional_t<(int(schedule) & int(KernelScheduleType::MAINLOOP_DISPATCH_MASK)) == int(KernelScheduleType::PTR_ARRAY),
                                              KernelMultistageBatchArray,
                                              KernelMultistage
                                             >;
  using DispatchPolicy_ = typename platform::conditional<
    DisableCpAsync,
    cutlass::gemm::MainloopPPUTwoStageLdmatrix,
    cutlass::gemm::MainloopPPUCpAsyncLegacy<Stage>
  >::type;

  using DispatchPolicy = typename KernelMetaTypes<Stage, schedule, false, DispatchPolicy_>::DispatchPolicy;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using CopyInstA = typename platform::conditional<
    DisableCpAsync,
    UniversalCopy<AlignmentTypeA>,
    PPU_CP_ASYNC_CACHEGLOBAL<AlignmentTypeA>
  >::type;

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using CopyInstB = typename platform::conditional<
    DisableCpAsync,
    UniversalCopy<AlignmentTypeB>,
    PPU_CP_ASYNC_CACHEGLOBAL<AlignmentTypeB>
  >::type;
  using GemmOperandA = DefaultGemm_TensorOpPPU_Operand<ElementA, TransA, AlignmentA, typename platform::conditional<TransA, Int<BlockM>, Int<BlockK>>::type, ThreadNum, CopyInstA>;
  using GemmOperandB = DefaultGemm_TensorOpPPU_Operand<ElementB, TransB, AlignmentB, typename platform::conditional<TransB, Int<BlockN>, Int<BlockK>>::type, ThreadNum, CopyInstB>;

  using MmaInst = typename platform::conditional<
    platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    GetAiuMmaInst<Arch, ElementCompute>,
    GetAiuMmaInst<Arch, ElementA, ElementB, ElementAcc>
  >::type::type;

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    Arch, DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

#if SAIL_EPILOGUE_OPT >= 2
  // enable only for persistent kernel
  using EpilogueInput = typename platform::conditional<(cute::is_same_v<ElementD, cutlass::half_t> || cute::is_same_v<ElementD, cutlass::bfloat16_t>)
                                                    && cute::is_same_v<ElementAcc, float> && cute::is_same_v<ElementD, ElementA> && cute::is_same_v<ElementD, ElementB>
                                                    && UseEpilogueEvt
                                                    && KernelMetaTypes<Stage, schedule>::IS_PERSIST_OVERLAP,
                                                    ElementD,
                                                    ElementAcc>::type;
#else
  using EpilogueInput = ElementAcc;
#endif

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, EpilogueInput, Alignment, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  // EpilogueOp: support passing either a fusion::FusionCallbacks instance or a direct visitor implementation, e.g. fusion::PPULinearCombination
  using EpilogueDispatchPolicy = typename KernelMetaTypes<Stage, schedule, false, DispatchPolicy_>::EpilogueDispatchPolicy;

  using TileShape_MNK = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  using FusionCallbacks =
    typename epilogue::collective::detail::CallbacksBuilder<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape_MNK,
      typename GemmEpilogueConfiguration::EpilogueTile,
      EpilogueInput
    >::Callbacks;
  using CollectiveEpilogue = cutlass::epilogue::collective::detail::ChooseEpilogue<
    UseEpilogueEvt,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    FusionCallbacks,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst, EpilogueInput>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementD) * AlignmentD * 8>, ElementD>,
    EpilogueDispatchPolicy,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementC) * AlignmentC * 8>, ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};

// simt gemm
template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  int AlignmentA,
  typename ElementB,
  typename LayoutB,
  int AlignmentB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseEpilogueEvt,
  KernelScheduleType schedule
> struct GemmKernelConfig <
  Arch,
  ElementA,
  LayoutA,
  AlignmentA,
  ElementB,
  LayoutB,
  AlignmentB,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementD,
  AlignmentD,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  false,
  false,
  UseEpilogueEvt,
  schedule,
  arch::OpClassSimt
> {
  static constexpr bool DisableCpAsync = (sizeof(ElementA) * AlignmentA < 4) || (sizeof(ElementB) * AlignmentB < 4);
  static constexpr int Alignment = platform::max(AlignmentC, AlignmentD);

  using KernelScheduler = cute::conditional_t<(int(schedule) == int(KernelScheduleType::PERSISTENT)),
                          cutlass::gemm::KernelMultistagePersistent,
                          cutlass::gemm::KernelMultistage>;
  using DispatchPolicy = typename platform::conditional<
    DisableCpAsync,
    cutlass::gemm::MainloopPPUTwoStageLdmatrix,
    cutlass::gemm::MainloopPPUCpAsync<Stage, KernelScheduler>
  >::type;

  static constexpr bool UseHfma = platform::is_same<ElementA, cutlass::half_t>::value &&
                                  platform::is_same<ElementB, cutlass::half_t>::value &&
                                  platform::is_same<ElementC, cutlass::half_t>::value &&
                                  platform::is_same<ElementAcc, cutlass::half_t>::value;

  using TileM = typename platform::conditional<
    (!UseHfma) && platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
    Layout<Shape<_16,_2>,Stride<_2,_1>>,
    Underscore
  >::type;

  using TileN = typename platform::conditional<
    (!UseHfma) && platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
    Layout<Shape<_16,_2>,Stride<_2,_1>>,
    Underscore
  >::type;

  using MmaInst = typename platform::conditional<UseHfma,
    typename platform::conditional<
      platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
      PPU_2x1x1_F16F16F16F16,
      PPU_1x2x1_F16F16F16F16,
    >::type,
    PPU_UniversalFMA<ElementAcc, ElementA, ElementB, ElementAcc>
  >::type;

  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<_16, _16, _1>>,                            // 16x16x1 thread group
      Tile<TileM, TileN, Underscore>>;

  static constexpr uint32_t ThreadNum = CUTE_STATIC_V(cute::size(TiledMma{}));

  using SmemLayoutAtomA = typename platform::conditional<
    platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
    Layout<Shape<_128, _16>>,
    Layout<Shape <_128, _16>, Stride<Int<16 + AlignmentA>, _1>>
  >::type;
  using SmemCopyAtomA = Copy_Atom<AutoVectorizingCopy, ElementA>;
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemThrLayoutA = typename platform::conditional<
    platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
    Layout<Shape<Int<128 / AlignmentA>, Int<ThreadNum / 128 * AlignmentA>>>,
    Layout<Shape<Int<ThreadNum / 16 * AlignmentA>, Int<16 / AlignmentA>>, Stride<Int<16 / AlignmentA>, _1>>
  >::type;
  using GmemValLayoutA = typename platform::conditional<
    platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
    Layout<Shape<Int<AlignmentA>, _1>>,
    Layout<Shape<_1, Int<AlignmentA>>>
  >::type;
  using CopyInstA = typename platform::conditional<
    DisableCpAsync,
    UniversalCopy<AlignmentTypeA>,
    PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeA>
  >::type;
  using GmemTiledCopyA = decltype(make_tiled_copy(Copy_Atom<CopyInstA, ElementA>{},
                       GmemThrLayoutA{},
                       GmemValLayoutA{}));

  using SmemLayoutAtomB = typename platform::conditional<
    platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
    Layout<Shape<_128, _16>>,
    Layout<Shape <_128, _16>, Stride<_16, _1>>
  >::type;
  using SmemCopyAtomB = Copy_Atom<AutoVectorizingCopy, ElementB>;
  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemThrLayoutB = typename platform::conditional<
    platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
    Layout<Shape<Int<128 / AlignmentB>, Int<ThreadNum / 128 * AlignmentB>>>,
    Layout<Shape<Int<ThreadNum / 16 * AlignmentB>, Int<16 / AlignmentB>>, Stride<Int<16 / AlignmentB>, _1>>
  >::type;
  using GmemValLayoutB = typename platform::conditional<
    platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
    Layout<Shape<Int<AlignmentB>, _1>>,
    Layout<Shape<_1, Int<AlignmentB>>>
  >::type;
  using CopyInstB = typename platform::conditional<
    DisableCpAsync,
    UniversalCopy<AlignmentTypeB>,
    PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeB>
  >::type;
  using GmemTiledCopyB = decltype(make_tiled_copy(Copy_Atom<CopyInstB, ElementB>{},
                     GmemThrLayoutB{},
                     GmemValLayoutB{}));

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    Arch, DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using WarpOnM = typename platform::conditional<
    platform::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
    Int<BlockM / 32>,
    Int<BlockM / 16>,
  >::type;

#if SAIL_EPILOGUE_OPT >= 2
  // enable only for persistent kernel
  using EpilogueInput = typename platform::conditional<(cute::is_same_v<ElementD, cutlass::half_t> || cute::is_same_v<ElementD, cutlass::bfloat16_t>)
                                                    && cute::is_same_v<ElementAcc, float> && cute::is_same_v<ElementD, ElementA> && cute::is_same_v<ElementD, ElementB>
                                                    && UseEpilogueEvt
                                                    && KernelMetaTypes<Stage, schedule>::IS_PERSIST_OVERLAP,
                                                    ElementD,
                                                    ElementAcc>::type;
#else
  using EpilogueInput = ElementAcc;
#endif

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, EpilogueInput, Alignment, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  // EpilogueOp: support passing either a fusion::FusionCallbacks instance or a direct visitor implementation, e.g. fusion::PPULinearCombination
  using EpilogueDispatchPolicy = typename KernelMetaTypes<Stage, schedule>::EpilogueDispatchPolicy;

  using TileShape_MNK = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  using FusionCallbacks =
    typename epilogue::collective::detail::CallbacksBuilder<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape_MNK,
      typename GemmEpilogueConfiguration::EpilogueTile,
      EpilogueInput
    >::Callbacks;
  using CollectiveEpilogue = cutlass::epilogue::collective::detail::ChooseEpilogue<
    UseEpilogueEvt,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    FusionCallbacks,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,EpilogueInput>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentD * sizeof(ElementD) * 8>, ElementD>,
    EpilogueDispatchPolicy,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>, ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};

// aiu gemm
template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseEpilogueEvt,
  KernelScheduleType schedule
> struct GemmKernelConfig <
  Arch,
  ElementA,
  LayoutA,
  1,
  ElementB,
  LayoutB,
  1,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementD,
  AlignmentD,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  true,
  false,
  UseEpilogueEvt,
  schedule
> {

  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  static constexpr int Alignment = platform::max(AlignmentC, AlignmentD);

  using KernelScheduler = typename KernelMetaTypes<Stage, schedule>::KernelScheduler;
  using DispatchPolicy = typename KernelMetaTypes<Stage, schedule>::DispatchPolicy;
  using TileScheduler = typename KernelMetaTypes<Stage, schedule>::TileScheduler;
  using EpilogueDispatchPolicy = typename KernelMetaTypes<Stage, schedule>::EpilogueDispatchPolicy;

  static constexpr int SmemLayoutStageStride = KernelMetaTypes<Stage, schedule>::IS_PERSIST_OVERLAP ? (BlockM + BlockN) * BlockK : 0;
  static constexpr bool IsGemmArray = cute::is_same_v<DispatchPolicy, cutlass::gemm::MainloopPPUAiuBatchArray<Stage>>
                                      || cute::is_same_v<EpilogueDispatchPolicy, cutlass::epilogue::EpiloguePtrArraySimtVectorized>;
  using GemmOperandA = DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<BlockM>, Int<BlockK>, false, SmemLayoutStageStride>;
  using GemmOperandB = DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<BlockN>, Int<BlockK>, true, SmemLayoutStageStride>;

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using MmaInst = typename platform::conditional<
    platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    GetAiuMmaInst<Arch, ElementCompute>,
    GetAiuMmaInst<Arch, ElementA, ElementB, ElementAcc>
  >::type::type;

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    Arch, DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

#if SAIL_EPILOGUE_OPT >= 2
  // enable only for persistent kernel
  using EpilogueInput = typename platform::conditional<(cute::is_same_v<ElementD, cutlass::half_t> || cute::is_same_v<ElementD, cutlass::bfloat16_t>)
                                                    && cute::is_same_v<ElementAcc, float> && cute::is_same_v<ElementD, ElementA> && cute::is_same_v<ElementD, ElementB>
                                                    && UseEpilogueEvt
                                                    && KernelMetaTypes<Stage, schedule>::IS_PERSIST_OVERLAP,
                                                    ElementD,
                                                    ElementAcc>::type;
#else
  using EpilogueInput = ElementAcc;
#endif

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, EpilogueInput, Alignment, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  // EpilogueOp: support passing either a fusion::FusionCallbacks instance or a direct visitor implementation, e.g. fusion::PPULinearCombination
  using TileShape_MNK = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  using FusionCallbacks =
    typename epilogue::collective::detail::CallbacksBuilder<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape_MNK,
      typename GemmEpilogueConfiguration::EpilogueTile,
      EpilogueInput
    >::Callbacks;
  using CollectiveEpilogue = cutlass::epilogue::collective::detail::ChooseEpilogue<
    UseEpilogueEvt,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    FusionCallbacks,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,EpilogueInput>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentD * sizeof(ElementD) * 8>, ElementD>,
    EpilogueDispatchPolicy,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>, ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    cute::conditional_t<IsGemmArray, cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>, Shape<int,int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler
  >;
};

// aiu gemm parallel
template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseEpilogueEvt,
  KernelScheduleType schedule
> struct GemmKernelConfig <
  Arch,
  ElementA,
  LayoutA,
  1,
  ElementB,
  LayoutB,
  1,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementD,
  AlignmentD,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  true,
  true,
  UseEpilogueEvt,
  schedule
> {

  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

  using KernelScheduler = typename KernelMetaTypes<Stage, schedule>::KernelScheduler;
  using DispatchPolicy = typename KernelMetaTypes<Stage, schedule>::DispatchPolicy;

  static constexpr bool IsGemmArray = cute::is_same_v<DispatchPolicy, cutlass::gemm::MainloopPPUAiuBatchArray<Stage>>;
  using GemmOperandA = DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;
  using GemmOperandB = DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using MmaInst = typename platform::conditional<
    platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    GetAiuMmaInst<Arch, ElementCompute>,
    GetAiuMmaInst<Arch, ElementA, ElementB, ElementAcc>
  >::type::type;

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    Arch, DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, ElementAcc, AlignmentD, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueParallel<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,ElementAcc>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementD) * AlignmentD * 8>, ElementD>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversalParallel<
    cute::conditional_t<IsGemmArray, cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>, Shape<int,int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};

// gemm with blockwise quant
template<
  typename Arch,
  typename ElementA,
  typename LayoutA,
  int AlignmentA,
  typename ElementB,
  typename LayoutB,
  int AlignmentB,
  typename ElementAcc,
  typename ElementC,
  int AlignmentC,
  typename ElementD,
  int AlignmentD,
  typename ElementCompute,
  int BlockM,
  int BlockN,
  int BlockK,
  int WarpM,
  int WarpN,
  int WarpK,
  int Stage,
  typename EpilogueOp,
  bool UseEpilogueEvt,
  KernelScheduleType schedule,
  typename ScaleGranularityShape
> struct GemmKernelConfig <
  Arch,
  ElementA,
  LayoutA,
  AlignmentA,
  ElementB,
  LayoutB,
  AlignmentB,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementD,
  AlignmentD,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  true,
  false,
  UseEpilogueEvt,
  schedule,
  arch::OpClassTensorOp,
  ScaleGranularityShape
> {

  static_assert(size<0>(ScaleGranularityShape{}) != 0, "ScaleGranularityShape must be non-zero");
  using TileShape           = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;                  // Threadblock-level tile size
  using ClusterShape        = Shape<Int<WarpM>, Int<WarpN>, Int<WarpK>>;                     // Shape of the threadblocks in a cluster
  static constexpr int ScaleGranularityM = size<0>(ScaleGranularityShape{});
  static constexpr int ScaleGranularityN = size<1>(ScaleGranularityShape{});
  static constexpr bool K_majorSFA = (ScaleGranularityM != 1);
  static constexpr bool K_majorSFB = (ScaleGranularityN != 1);
  using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
                                     ScaleGranularityShape, K_majorSFA, K_majorSFB, true>(ScaleGranularityShape{}));
  using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
  using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

  using LayoutC             = cutlass::layout::RowMajor;
  using LayoutD             = LayoutC;

  using KernelSchedule      = cutlass::gemm::KernelAiuMultistageWithBlockWiseScale;
  using EpilogueSchedule    = cutlass::epilogue::EpilogueSimtVectorized;
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, arch::OpClassTensorOp,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementCompute, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloopBlockWise = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAcc,
      TileShape, ClusterShape,
      Int<Stage>,
      KernelSchedule,
    >::CollectiveOp;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopBlockWise,
      CollectiveEpilogue
  >;
};

} // namespace config
} // namespace gemm
} // namespace cutlass
