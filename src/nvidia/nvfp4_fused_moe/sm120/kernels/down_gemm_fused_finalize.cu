// CUTLASS grouped GEMM2 with fused finalize (scatter-reduce) epilogue for SM120 NVFP4
// Matches the FlashInfer/TensorRT-LLM GEMM2 FINALIZE tactic:
//   SwapAB=true, tile 256x128x128, ColumnMajor final-output view, and
//   ScaledAccPerRowBiasPerColScaleScatter.
// This fuses the unpermute + weighted-reduce (finalize) into the GEMM2 epilogue,
// eliminating the separate finalizeMoeRoutingKernel.

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

using namespace cute;

// ============================================================================
// SM120 NVFP4 Grouped GEMM2 with fused finalize — type aliases
// ============================================================================

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;

using ElementType   = cutlass::float_e2m1_t;
using ElementSFType = cutlass::float_ue4m3_t;
using ElementA      = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB      = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC      = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = LayoutC;

static constexpr int AlignmentA = 32;
static constexpr int AlignmentB = 32;
// TensorRT-LLM uses the activation alignment for C even though C is nullptr in
// FINALIZE mode. This keeps the SM120 epilogue dispatch identical to FlashInfer.
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementType>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag       = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape   = Shape<_1, _1, _1>;
using MmaTileShape   = Shape<_256, _128, _128>;

using FusionOperation = cutlass::epilogue::fusion::ScaledAccPerRowBiasPerColScaleScatter<
    LayoutD,            // GmemLayoutTagOut
    ElementC,           // ElementOutput (bf16)
    ElementAccumulator, // ElementCompute (float)
    ElementC,           // ElementBias (bf16, unused — nullptr)
    ElementAccumulator, // ElementScale (float — router scales)
    ElementAccumulator  // ElementScalar (float — alpha)
>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        void, LayoutD*, AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized,
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA   = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB   = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

// Stride type for the scatter output tensor. With SwapAB=true the epilogue sees
// a transposed logical output [hidden_size, expanded_rows] in ColumnMajor view.
using StrideOutput = cutlass::gemm::TagToStrideC_t<LayoutD>;

// ============================================================================
// Pointer setup kernel — extends base version with scatter-specific arrays
// ============================================================================

static inline __host__ __device__ int alignUp(int dim, int alignment) {
    return ((dim + alignment - 1) / alignment) * alignment;
}

static bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

__global__ void atrex_setup_fused_finalize_group_ptrs(
    // Base pointer arrays (same as cutlass_gemm.cu)
    ElementType**    a_ptrs,
    ElementType**    b_ptrs,
    ElementSFType**  sfa_ptrs,
    ElementSFType**  sfb_ptrs,
    float**          alpha_ptrs,
    LayoutSFA*       layout_sfa,
    LayoutSFB*       layout_sfb,
    StrideA*         a_strides,     // weight-as-A strides (SwapAB)
    StrideB*         b_strides,     // activation-as-B strides (SwapAB)
    StrideC*         c_strides,
    int64_t*         problem_sizes,
    // Scatter-specific pointer arrays
    float**          scale_ptrs,     // per-expert router scale pointers
    int**            index_ptrs,     // per-expert scatter index pointers
    // Base data pointers
    ElementType*     a_base,
    ElementType*     b_base,
    ElementSFType*   sfa_base,
    ElementSFType*   sfb_base,
    float*           alpha_base,
    float*           scale_base,     // perm_scales [expanded, sorted order]
    int*             index_base,     // permuted_row [expanded], sorted→expanded
    // Expert layout
    int64_t const*   expert_first_token_offset,
    int              N,
    int              K)
{
    int expert = threadIdx.x;
    int num_experts = blockDim.x;
    if (expert >= num_experts) return;

    int64_t offset      = expert_first_token_offset[expert];
    int64_t next_offset = expert_first_token_offset[expert + 1];
    int64_t m           = next_offset - offset;
    int     half_k      = K / 2;
    int     group_k     = K / 16;

    // Keep activation and weight arrays separately, but launch the CUTLASS
    // mainloop as SwapAB: A=weight [N,K], B=activation [M,K].
    a_ptrs[expert]   = a_base + offset * half_k;
    b_ptrs[expert]   = b_base + (int64_t)expert * N * half_k;

    constexpr int MIN_N = 128;
    int64_t psf = alignUp(static_cast<int>(offset + expert * (MIN_N - 1)), MIN_N);

    int padded_N = alignUp(N, MIN_N);
    int padded_K = alignUp(K, 64);
    // SwapAB also swaps the scale-factor roles: SFA belongs to weights and SFB
    // belongs to activations.
    sfa_ptrs[expert] = sfb_base + (int64_t)expert * padded_N * padded_K / 16;
    sfb_ptrs[expert] = sfa_base + psf * group_k;

    alpha_ptrs[expert] = alpha_base + expert;

    // Router scales and scatter index: contiguous slices of sorted-order arrays
    scale_ptrs[expert] = scale_base + offset;
    index_ptrs[expert] = index_base + offset;

    a_strides[expert] = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape((int)N, (int)K, (int)1));
    b_strides[expert] = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape((int)m, (int)K, (int)1));
    c_strides[expert] = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape((int)N, (int)m, (int)1));

    problem_sizes[expert * 3 + 0] = static_cast<int64_t>(N);
    problem_sizes[expert * 3 + 1] = m;
    problem_sizes[expert * 3 + 2] = static_cast<int64_t>(K);

    layout_sfa[expert] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)N, (int)m, (int)K, (int)1));
    layout_sfb[expert] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)N, (int)m, (int)K, (int)1));
}

// ============================================================================
// Workspace size
// ============================================================================

struct FusedFinalizeWorkspaceView {
    ElementType** a_ptrs;
    ElementType** b_ptrs;
    ElementSFType** sfa_ptrs;
    ElementSFType** sfb_ptrs;
    float** alpha_ptrs;
    float** scale_ptrs;
    int** index_ptrs;
    LayoutSFA* layout_sfa;
    LayoutSFB* layout_sfb;
    StrideA* a_strides;
    StrideB* b_strides;
    StrideC* c_strides;
    int64_t* problem_sizes;
};

static int64_t fused_finalize_group_ptrs_bytes(int E) {
    auto align256 = [](int64_t x) -> int64_t { return (x + 255) & ~255LL; };
    int64_t total = 0;
    total += align256(E * sizeof(ElementType*));      // a_ptrs
    total += align256(E * sizeof(ElementType*));      // b_ptrs
    total += align256(E * sizeof(ElementSFType*));     // sfa_ptrs
    total += align256(E * sizeof(ElementSFType*));     // sfb_ptrs
    total += align256(E * sizeof(float*));             // alpha_ptrs
    total += align256(E * sizeof(float*));             // scale_ptrs (new)
    total += align256(E * sizeof(int*));               // index_ptrs (new)
    total += align256(E * sizeof(LayoutSFA));          // layout_sfa
    total += align256(E * sizeof(LayoutSFB));          // layout_sfb
    total += align256(E * sizeof(StrideA));            // a_strides
    total += align256(E * sizeof(StrideB));            // b_strides
    total += align256(E * sizeof(StrideC));            // c_strides
    total += align256(E * 3 * sizeof(int64_t));        // problem_sizes
    return total;
}

static FusedFinalizeWorkspaceView make_fused_finalize_workspace_view(
    void* workspace, int E)
{
    auto align256 = [](int64_t x) -> int64_t { return (x + 255) & ~255LL; };
    uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);
    int64_t off = 0;

    FusedFinalizeWorkspaceView view{};
    view.a_ptrs = reinterpret_cast<ElementType**>(ws + off);
    off += align256(E * sizeof(ElementType*));
    view.b_ptrs = reinterpret_cast<ElementType**>(ws + off);
    off += align256(E * sizeof(ElementType*));
    view.sfa_ptrs = reinterpret_cast<ElementSFType**>(ws + off);
    off += align256(E * sizeof(ElementSFType*));
    view.sfb_ptrs = reinterpret_cast<ElementSFType**>(ws + off);
    off += align256(E * sizeof(ElementSFType*));
    view.alpha_ptrs = reinterpret_cast<float**>(ws + off);
    off += align256(E * sizeof(float*));
    view.scale_ptrs = reinterpret_cast<float**>(ws + off);
    off += align256(E * sizeof(float*));
    view.index_ptrs = reinterpret_cast<int**>(ws + off);
    off += align256(E * sizeof(int*));
    view.layout_sfa = reinterpret_cast<LayoutSFA*>(ws + off);
    off += align256(E * sizeof(LayoutSFA));
    view.layout_sfb = reinterpret_cast<LayoutSFB*>(ws + off);
    off += align256(E * sizeof(LayoutSFB));
    view.a_strides = reinterpret_cast<StrideA*>(ws + off);
    off += align256(E * sizeof(StrideA));
    view.b_strides = reinterpret_cast<StrideB*>(ws + off);
    off += align256(E * sizeof(StrideB));
    view.c_strides = reinterpret_cast<StrideC*>(ws + off);
    off += align256(E * sizeof(StrideC));
    view.problem_sizes = reinterpret_cast<int64_t*>(ws + off);
    return view;
}

static typename GemmKernel::Arguments make_fused_finalize_args(
    const FusedFinalizeWorkspaceView& view,
    int E,
    int N,
    int M,
    void* final_output_bf16)
{
    auto* problem_shapes_ptr =
        reinterpret_cast<UnderlyingProblemShape*>(view.problem_sizes);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = RasterOrderOptions::AlongN;

    // SwapAB mainloop: weights are A, activations are B. This is the same
    // tactic family FlashInfer autotunes to for GEMM2 FINALIZE.
    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<ElementType const**>(static_cast<void*>(view.b_ptrs)),
        view.a_strides,
        static_cast<ElementType const**>(static_cast<void*>(view.a_ptrs)),
        view.b_strides,
        static_cast<ElementSFType const**>(static_cast<void*>(view.sfa_ptrs)),
        view.layout_sfa,
        static_cast<ElementSFType const**>(static_cast<void*>(view.sfb_ptrs)),
        view.layout_sfb};

    // Build scatter output stride using the transposed ColumnMajor view
    // expected by ScaledAccPerRowBiasPerColScaleScatter under SwapAB.
    auto scatter_stride = cutlass::make_cute_packed_stride(
        StrideOutput{}, cute::make_shape(N, M, 1));

    typename GemmKernel::EpilogueArguments epilogue_args{
        {},       // thread params (fusion args, will be set below)
        nullptr,  // ptr_C (no source tensor)
        nullptr,  // stride_C
        nullptr,  // ptr_D (scatter handles output, not standard D)
        nullptr}; // stride_D

    auto& fusion_args = epilogue_args.thread;
    fusion_args.alpha = 1.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.alpha_ptr_array =
        reinterpret_cast<float const* const*>(view.alpha_ptrs);
    fusion_args.dAlpha = {_0{}, _0{}, 1};

    fusion_args.bias_ptr = nullptr;  // no bias for MoE GEMM2
    fusion_args.dBias = {_1{}, _0{}, 0};

    fusion_args.scale_ptr_array =
        reinterpret_cast<float const* const*>(view.scale_ptrs);
    fusion_args.dScale = {_0{}, _1{}, 0};

    fusion_args.ptr_out = reinterpret_cast<ElementC*>(final_output_bf16);
    fusion_args.dOut = scatter_stride;
    fusion_args.ptr_index =
        reinterpret_cast<int const* const*>(view.index_ptrs);
    fusion_args.index_modulo = M;
    fusion_args.shape_override = N;
    fusion_args.use_reduction = true;

    return typename GemmKernel::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {E, problem_shapes_ptr, nullptr},
        mainloop_args,
        epilogue_args,
        hw_info,
        scheduler};
}

extern "C" int64_t cutlass_gemm_fused_finalize_group_ptrs_workspace_bytes(
    int E)
{
    return fused_finalize_group_ptrs_bytes(E);
}

extern "C" int64_t cutlass_gemm_fused_finalize_cutlass_workspace_bytes(
    void* setup_workspace,
    int num_experts,
    int N,
    int K,
    int M)
{
    (void)K;
    auto view = make_fused_finalize_workspace_view(setup_workspace, num_experts);
    auto args = make_fused_finalize_args(view, num_experts, N, M, nullptr);
    return static_cast<int64_t>(Gemm::get_workspace_size(args));
}

extern "C" int cutlass_gemm_fused_finalize_prepare() {
    cudaFuncAttributes attrs;
    cudaError_t err = cudaFuncGetAttributes(
        &attrs, cutlass::device_kernel<GemmKernel>);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return Gemm::maximum_active_blocks() < 0 ? 1 : 0;
}

extern "C" void cutlass_gemm_fused_finalize_debug_info() {
    fprintf(stderr, "[fused_finalize] MmaTileShape: %d x %d x %d\n",
            int(cute::size<0>(MmaTileShape{})),
            int(cute::size<1>(MmaTileShape{})),
            int(cute::size<2>(MmaTileShape{})));
    fprintf(stderr, "[fused_finalize] Epilogue SharedStorage: %zu bytes\n",
            sizeof(typename CollectiveEpilogue::SharedStorage));
    fprintf(stderr, "[fused_finalize] Mainloop SharedStorage: %zu bytes\n",
            sizeof(typename CollectiveMainloop::SharedStorage));
    fprintf(stderr, "[fused_finalize] GemmKernel SharedStorage: %zu bytes\n",
            sizeof(typename GemmKernel::SharedStorage));
    fprintf(stderr, "[fused_finalize] Mainloop stages: %d\n",
            CollectiveMainloop::DispatchPolicy::Stages);
}

// ============================================================================
// CUTLASS grouped GEMM2 with fused finalize launch
// ============================================================================

extern "C" void cutlass_gemm_fused_finalize_setup_group_ptrs(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    int64_t const* expert_first_token_offset,
    int         num_experts,
    int         N,              // hidden_size (output dimension)
    int         K,              // inter_size (reduction dimension)
    int64_t     expanded_num_tokens,
    void*       setup_workspace,
    cudaStream_t stream,
    float*      perm_scales,        // [expanded] sorted-order topk_weights
    int*        unperm_map,         // [expanded] sorted→col_major_expanded mapping
    int         M,                  // num original tokens
    int         topk)
{
    (void)expanded_num_tokens;
    (void)M;
    (void)topk;
    auto view = make_fused_finalize_workspace_view(setup_workspace, num_experts);
    atrex_setup_fused_finalize_group_ptrs<<<1, num_experts, 0, stream>>>(
        view.a_ptrs, view.b_ptrs, view.sfa_ptrs, view.sfb_ptrs,
        view.alpha_ptrs,
        view.layout_sfa, view.layout_sfb,
        view.a_strides, view.b_strides, view.c_strides,
        view.problem_sizes,
        view.scale_ptrs, view.index_ptrs,
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(a_fp4)),
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(b_fp4)),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(sf_a)),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(sf_b)),
        const_cast<float*>(alpha),
        perm_scales,
        unperm_map,
        expert_first_token_offset, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[cutlass_gemm_fused_finalize] setup launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

extern "C" void cutlass_gemm_fused_finalize_forward(
    void*       setup_workspace,
    void*       cutlass_workspace,
    int64_t     cutlass_workspace_bytes,
    int         num_experts,
    int         N,              // hidden_size (output dimension)
    int         K,              // inter_size (reduction dimension)
    int64_t     expanded_num_tokens,
    cudaStream_t stream,
    // Fused finalize parameters
    void*       final_output_bf16,  // [M, N] bf16, must be pre-zeroed
    int         M,                  // num original tokens
    int         topk)
{
    (void)K;
    (void)topk;
    int E = num_experts;
    auto view = make_fused_finalize_workspace_view(setup_workspace, E);
    auto args = make_fused_finalize_args(view, E, N, M, final_output_bf16);

    Gemm gemm_op;

    size_t needed_ws = Gemm::get_workspace_size(args);
    if (static_cast<int64_t>(needed_ws) > cutlass_workspace_bytes) {
        fprintf(stderr,
                "[cutlass_gemm_fused_finalize] workspace too small: need %zu, have %lld\n",
                needed_ws, (long long)cutlass_workspace_bytes);
        return;
    }

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[cutlass_gemm_fused_finalize] can_implement failed: %d\n",
                static_cast<int>(status));
        return;
    }

    bool capturing = stream_is_capturing(stream);
    status = capturing
        ? gemm_op.update(args, cutlass_workspace)
        : gemm_op.initialize(args, cutlass_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[cutlass_gemm_fused_finalize] %s failed: %d "
                "(E=%d, M=%d, N=%d, K=%d, expanded=%lld, ws=%lld needed=%zu)\n",
                capturing ? "update" : "initialize", static_cast<int>(status),
                E, M, N, K,
                (long long)expanded_num_tokens,
                (long long)cutlass_workspace_bytes, needed_ws);
        return;
    }

    gemm_op.run(stream);
}
