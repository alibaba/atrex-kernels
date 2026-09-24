// CUTLASS grouped GEMM wrapper for SM120 NVFP4
// Replaces custom PTX GEMM with CUTLASS 3.x CollectiveBuilder approach.
// Reference: vLLM nvfp4_blockwise_moe_kernel.cu SM120 path

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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

using namespace cute;

// ============================================================================
// SM120 NVFP4 Grouped GEMM type aliases
// ============================================================================

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;

using ElementType   = cutlass::float_e2m1_t;
using ElementSFType = cutlass::float_ue4m3_t;
using ElementA      = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB      = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC      = cutlass::bfloat16_t;  // SM120 hardcoded
using ElementD      = ElementC;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

static constexpr int AlignmentA = 32;
static constexpr int AlignmentB = 32;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag       = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape   = Shape<_1, _1, _1>;
using MmaTileShape   = Shape<_128, _128, _128>;

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
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

// ============================================================================
// Pointer setup kernel — converts expert_first_token_offset to per-expert
// pointer arrays compatible with CUTLASS Ptr-Array grouped GEMM.
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

__global__ void atrex_setup_cutlass_group_ptrs(
    // Output pointer arrays
    ElementType**    a_ptrs,
    ElementType**    b_ptrs,
    ElementC**       out_ptrs,
    ElementSFType**  sfa_ptrs,
    ElementSFType**  sfb_ptrs,
    float**          alpha_ptrs,
    LayoutSFA*       layout_sfa,
    LayoutSFB*       layout_sfb,
    int64_t*         a_strides,
    int64_t*         b_strides,
    int64_t*         c_strides,
    int64_t*         problem_sizes,
    // Base data pointers
    ElementType*     a_base,
    ElementType*     b_base,
    ElementC*        out_base,
    ElementSFType*   sfa_base,
    ElementSFType*   sfb_base,
    float*           alpha_base,
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

    a_ptrs[expert]   = a_base   + offset  * half_k;
    b_ptrs[expert]   = b_base   + (int64_t)expert * N * half_k;
    out_ptrs[expert] = out_base + offset  * N;

    constexpr int MIN_N = 128;
    int64_t psf = alignUp(static_cast<int>(offset + expert * (MIN_N - 1)), MIN_N);
    sfa_ptrs[expert] = sfa_base + psf * group_k;

    int padded_N = alignUp(N, MIN_N);
    int padded_K = alignUp(K, 64);
    sfb_ptrs[expert] = sfb_base +
        (int64_t)expert * padded_N * padded_K / 16;

    alpha_ptrs[expert] = alpha_base + expert;

    a_strides[expert] = K;
    b_strides[expert] = K;
    c_strides[expert] = N;

    problem_sizes[expert * 3 + 0] = m;
    problem_sizes[expert * 3 + 1] = static_cast<int64_t>(N);
    problem_sizes[expert * 3 + 2] = static_cast<int64_t>(K);

    // SF layouts (CuTe layouts for TMA)
    layout_sfa[expert] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)m, (int)N, (int)K, (int)1));
    layout_sfb[expert] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)m, (int)N, (int)K, (int)1));
}

// ============================================================================
// CUTLASS grouped GEMM workspace size
// ============================================================================

struct CutlassGemmWorkspaceView {
    ElementType** a_ptrs;
    ElementType** b_ptrs;
    ElementC** out_ptrs;
    ElementSFType** sfa_ptrs;
    ElementSFType** sfb_ptrs;
    float** alpha_ptrs;
    LayoutSFA* layout_sfa;
    LayoutSFB* layout_sfb;
    int64_t* a_strides;
    int64_t* b_strides;
    int64_t* c_strides;
    int64_t* problem_sizes;
};

static int64_t cutlass_group_ptrs_bytes(int E) {
    auto align256 = [](int64_t x) -> int64_t { return (x + 255) & ~255LL; };
    int64_t total = 0;
    total += align256(E * sizeof(ElementType*));      // a_ptrs
    total += align256(E * sizeof(ElementType*));      // b_ptrs
    total += align256(E * sizeof(ElementC*));          // out_ptrs
    total += align256(E * sizeof(ElementSFType*));     // sfa_ptrs
    total += align256(E * sizeof(ElementSFType*));     // sfb_ptrs
    total += align256(E * sizeof(float*));             // alpha_ptrs
    total += align256(E * sizeof(LayoutSFA));          // layout_sfa
    total += align256(E * sizeof(LayoutSFB));          // layout_sfb
    total += align256(E * sizeof(int64_t));            // a_strides
    total += align256(E * sizeof(int64_t));            // b_strides
    total += align256(E * sizeof(int64_t));            // c_strides
    total += align256(E * 3 * sizeof(int64_t));        // problem_sizes
    return total;
}

static CutlassGemmWorkspaceView make_cutlass_gemm_workspace_view(
    void* workspace, int E)
{
    auto align256 = [](int64_t x) -> int64_t { return (x + 255) & ~255LL; };
    uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);
    int64_t off = 0;

    CutlassGemmWorkspaceView view{};
    view.a_ptrs = reinterpret_cast<ElementType**>(ws + off);
    off += align256(E * sizeof(ElementType*));
    view.b_ptrs = reinterpret_cast<ElementType**>(ws + off);
    off += align256(E * sizeof(ElementType*));
    view.out_ptrs = reinterpret_cast<ElementC**>(ws + off);
    off += align256(E * sizeof(ElementC*));
    view.sfa_ptrs = reinterpret_cast<ElementSFType**>(ws + off);
    off += align256(E * sizeof(ElementSFType*));
    view.sfb_ptrs = reinterpret_cast<ElementSFType**>(ws + off);
    off += align256(E * sizeof(ElementSFType*));
    view.alpha_ptrs = reinterpret_cast<float**>(ws + off);
    off += align256(E * sizeof(float*));
    view.layout_sfa = reinterpret_cast<LayoutSFA*>(ws + off);
    off += align256(E * sizeof(LayoutSFA));
    view.layout_sfb = reinterpret_cast<LayoutSFB*>(ws + off);
    off += align256(E * sizeof(LayoutSFB));
    view.a_strides = reinterpret_cast<int64_t*>(ws + off);
    off += align256(E * sizeof(int64_t));
    view.b_strides = reinterpret_cast<int64_t*>(ws + off);
    off += align256(E * sizeof(int64_t));
    view.c_strides = reinterpret_cast<int64_t*>(ws + off);
    off += align256(E * sizeof(int64_t));
    view.problem_sizes = reinterpret_cast<int64_t*>(ws + off);
    return view;
}

static typename GemmKernel::Arguments make_cutlass_gemm_args(
    const CutlassGemmWorkspaceView& view,
    int E)
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

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<ElementType const**>(static_cast<void*>(view.a_ptrs)),
        reinterpret_cast<StrideA*>(view.a_strides),
        static_cast<ElementType const**>(static_cast<void*>(view.b_ptrs)),
        reinterpret_cast<StrideB*>(view.b_strides),
        static_cast<ElementSFType const**>(static_cast<void*>(view.sfa_ptrs)),
        view.layout_sfa,
        static_cast<ElementSFType const**>(static_cast<void*>(view.sfb_ptrs)),
        view.layout_sfb};

    typename GemmKernel::EpilogueArguments epilogue_args{
        {},        // thread params placeholder
        nullptr,   // ptr_C
        reinterpret_cast<StrideC*>(view.c_strides),
        reinterpret_cast<ElementD**>(view.out_ptrs),
        reinterpret_cast<StrideD*>(view.c_strides)};
    auto& fusion_args = epilogue_args.thread;
    fusion_args.alpha_ptr_array =
        reinterpret_cast<float const* const*>(view.alpha_ptrs);
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.beta = 0.0f;

    return typename GemmKernel::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {E, problem_shapes_ptr, nullptr},
        mainloop_args,
        epilogue_args,
        hw_info,
        scheduler};
}

// ============================================================================
// CUTLASS grouped GEMM launch
// ============================================================================

extern "C" int64_t cutlass_gemm_group_ptrs_workspace_bytes(int E) {
    return cutlass_group_ptrs_bytes(E);
}

extern "C" int64_t cutlass_gemm_cutlass_workspace_bytes(
    void*       setup_workspace,
    int         num_experts,
    int         N,
    int         K,
    int         M)
{
    (void)N;
    (void)K;
    (void)M;
    auto view = make_cutlass_gemm_workspace_view(setup_workspace, num_experts);
    auto args = make_cutlass_gemm_args(view, num_experts);
    return static_cast<int64_t>(Gemm::get_workspace_size(args));
}

extern "C" void cutlass_gemm_setup_group_ptrs(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void*       output_bf16,
    int64_t const* expert_first_token_offset,
    int         num_experts,
    int         N,
    int         K,
    int64_t     expanded_num_tokens,
    void*       setup_workspace,
    cudaStream_t stream)
{
    (void)expanded_num_tokens;
    auto view = make_cutlass_gemm_workspace_view(setup_workspace, num_experts);
    atrex_setup_cutlass_group_ptrs<<<1, num_experts, 0, stream>>>(
        view.a_ptrs, view.b_ptrs, view.out_ptrs,
        view.sfa_ptrs, view.sfb_ptrs, view.alpha_ptrs,
        view.layout_sfa, view.layout_sfb,
        view.a_strides, view.b_strides, view.c_strides,
        view.problem_sizes,
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(a_fp4)),
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(b_fp4)),
        reinterpret_cast<ElementC*>(output_bf16),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(sf_a)),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(sf_b)),
        const_cast<float*>(alpha),
        expert_first_token_offset, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[cutlass_gemm] setup launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

extern "C" void cutlass_gemm_forward(
    void*       setup_workspace,
    void*       cutlass_workspace,
    int64_t     cutlass_workspace_bytes,
    int         num_experts,
    int         N,
    int         K,
    int64_t     expanded_num_tokens,
    cudaStream_t stream)
{
    int E = num_experts;
    auto view = make_cutlass_gemm_workspace_view(setup_workspace, E);
    auto args = make_cutlass_gemm_args(view, E);

    Gemm gemm_op;

    size_t needed_ws = Gemm::get_workspace_size(args);
    if (static_cast<int64_t>(needed_ws) > cutlass_workspace_bytes) {
        fprintf(stderr,
                "[cutlass_gemm] workspace too small: need %zu, have %lld\n",
                needed_ws, (long long)cutlass_workspace_bytes);
        return;
    }

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[cutlass_gemm] can_implement failed: %d\n",
                static_cast<int>(status));
        return;
    }

    bool capturing = stream_is_capturing(stream);
    status = capturing
        ? gemm_op.update(args, cutlass_workspace)
        : gemm_op.initialize(args, cutlass_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[cutlass_gemm] %s failed: %d "
                "(E=%d, N=%d, K=%d, expanded=%lld, ws=%lld needed=%zu)\n",
                capturing ? "update" : "initialize", static_cast<int>(status),
                E, N, K, (long long)expanded_num_tokens,
                (long long)cutlass_workspace_bytes, needed_ws);
        return;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[cutlass_gemm] run failed: %d\n",
                static_cast<int>(status));
    }
}

extern "C" int cutlass_gemm_prepare() {
    cudaFuncAttributes attrs;
    cudaError_t err = cudaFuncGetAttributes(
        &attrs, cutlass::device_kernel<GemmKernel>);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    if (Gemm::maximum_active_blocks() < 0) {
        return 1;
    }

    constexpr int E = 1;
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int64_t expanded = M;

    void* a = nullptr;
    void* b = nullptr;
    void* sfa = nullptr;
    void* sfb = nullptr;
    float* alpha = nullptr;
    void* out = nullptr;
    int64_t* expert_offset = nullptr;
    void* setup_workspace = nullptr;
    void* cutlass_workspace = nullptr;

    int64_t sfa_bytes = alignUp(M + E * (128 - 1), 128) * (K / 16);
    int64_t sfb_bytes = E * alignUp(N, 128) * alignUp(K, 64) / 16;
    int64_t setup_workspace_bytes = cutlass_group_ptrs_bytes(E);

    auto cleanup = [&]() {
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (sfa) cudaFree(sfa);
        if (sfb) cudaFree(sfb);
        if (alpha) cudaFree(alpha);
        if (out) cudaFree(out);
        if (expert_offset) cudaFree(expert_offset);
        if (setup_workspace) cudaFree(setup_workspace);
        if (cutlass_workspace) cudaFree(cutlass_workspace);
    };

    if (cudaMalloc(&a, expanded * K / 2) != cudaSuccess ||
        cudaMalloc(&b, E * N * K / 2) != cudaSuccess ||
        cudaMalloc(&sfa, sfa_bytes) != cudaSuccess ||
        cudaMalloc(&sfb, sfb_bytes) != cudaSuccess ||
        cudaMalloc(&alpha, E * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&out, expanded * N * sizeof(ElementC)) != cudaSuccess ||
        cudaMalloc(&expert_offset, (E + 1) * sizeof(int64_t)) != cudaSuccess ||
        cudaMalloc(&setup_workspace, setup_workspace_bytes) != cudaSuccess) {
        cleanup();
        cudaGetLastError();
        return 1;
    }

    int64_t h_offsets[E + 1] = {0, expanded};
    float h_alpha[E] = {1.0f};
    cudaMemset(a, 0, expanded * K / 2);
    cudaMemset(b, 0, E * N * K / 2);
    cudaMemset(sfa, 0, sfa_bytes);
    cudaMemset(sfb, 0, sfb_bytes);
    cudaMemset(out, 0, expanded * N * sizeof(ElementC));
    cudaMemcpy(alpha, h_alpha, E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(expert_offset, h_offsets, (E + 1) * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cleanup();
        cudaGetLastError();
        return 1;
    }

    cutlass_gemm_setup_group_ptrs(
        a, b, sfa, sfb, alpha, out, expert_offset,
        E, N, K, expanded, setup_workspace, stream);
    int64_t cutlass_workspace_bytes = cutlass_gemm_cutlass_workspace_bytes(
        setup_workspace, E, N, K, M);
    int64_t cutlass_alloc_bytes =
        cutlass_workspace_bytes > 0 ? cutlass_workspace_bytes : 1;
    if (cudaMalloc(&cutlass_workspace, cutlass_alloc_bytes) != cudaSuccess) {
        cudaStreamDestroy(stream);
        cleanup();
        cudaGetLastError();
        return 1;
    }
    cutlass_gemm_forward(
        setup_workspace, cutlass_workspace, cutlass_alloc_bytes,
        E, N, K, expanded, stream);
    err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cleanup();
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return 0;
}
