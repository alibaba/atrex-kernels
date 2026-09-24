#include "pybind_common.h"

namespace py = pybind11;

extern "C" int64_t cutlass_gemm_group_ptrs_workspace_bytes(int E);

extern "C" int64_t cutlass_gemm_cutlass_workspace_bytes(
    void* setup_workspace,
    int num_experts,
    int N,
    int K,
    int M);

extern "C" void cutlass_gemm_setup_group_ptrs(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_bf16,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* setup_workspace,
    cudaStream_t stream);

extern "C" void cutlass_gemm_forward(
    void* setup_workspace,
    void* cutlass_workspace,
    int64_t cutlass_workspace_bytes,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream);

extern "C" int64_t cutlass_gemm_fused_finalize_group_ptrs_workspace_bytes(
    int E);

extern "C" int64_t cutlass_gemm_fused_finalize_cutlass_workspace_bytes(
    void* setup_workspace,
    int num_experts,
    int N,
    int K,
    int M);

extern "C" void cutlass_gemm_fused_finalize_setup_group_ptrs(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* setup_workspace,
    cudaStream_t stream,
    float* perm_scales,
    int* unperm_map,
    int M,
    int topk);

extern "C" void cutlass_gemm_fused_finalize_forward(
    void* setup_workspace,
    void* cutlass_workspace,
    int64_t cutlass_workspace_bytes,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    cudaStream_t stream,
    void* final_output_bf16,
    int M,
    int topk);

extern "C" void finalize_moe_routing(
    void const* expanded_permuted_rows,
    void* reduced_unpermuted_output,
    void const* bias,
    float const* scales,
    int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts,
    int64_t num_rows,
    int64_t padded_cols,
    int64_t unpadded_cols,
    int64_t experts_per_token,
    int num_experts_per_node,
    int start_expert_id,
    cudaStream_t stream);

extern "C" int64_t atrex_task30_gemm2_small_m_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens);

extern "C" void atrex_task30_gemm2_small_m_forward(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* final_output_bf16,
    int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row,
    float const* sorted_scales,
    int num_experts,
    int N,
    int K,
    int M,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream);

int64_t get_workspace_size_setup_gemm2_group_ptrs(int E)
{
    return cutlass_gemm_group_ptrs_workspace_bytes(E);
}

int64_t get_workspace_size_task30_gemm2_small_m(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_task30_gemm2_small_m_workspace_bytes(
        E, hidden_size, inter_size, (int64_t)M * topk);
}

int64_t get_workspace_size_gemm2(
    const torch::Tensor& setup_ws,
    int E,
    int M,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_dtype(setup_ws, at::kByte, "setup_ws");
    require_numel_at_least(
        setup_ws,
        get_workspace_size_setup_gemm2_group_ptrs(E),
        "setup_ws");
    return cutlass_gemm_cutlass_workspace_bytes(
        raw_data_ptr_mut(setup_ws),
        E,
        hidden_size,
        inter_size,
        M);
}

int64_t get_workspace_size_setup_fused_finalize_group_ptrs(int E)
{
    return cutlass_gemm_fused_finalize_group_ptrs_workspace_bytes(E);
}

int64_t get_workspace_size_gemm2_fused_finalize(
    const torch::Tensor& setup_ws,
    int E,
    int M,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_dtype(setup_ws, at::kByte, "setup_ws");
    require_numel_at_least(setup_ws,
                           get_workspace_size_setup_fused_finalize_group_ptrs(E),
                           "setup_ws");
    return cutlass_gemm_fused_finalize_cutlass_workspace_bytes(
        raw_data_ptr_mut(setup_ws),
        E,
        hidden_size,
        inter_size,
        M);
}

void wait_for_shared_event_py(uint64_t shared_event)
{
    wait_for_shared_event(shared_event, current_stream());
}

void setup_gemm2_group_ptrs_py(
    const torch::Tensor& act_out,
    const torch::Tensor& w2_fp4,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& w2_sf,
    const torch::Tensor& gemm2_alpha,
    const torch::Tensor& gemm2_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& setup_ws,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(w2_fp4, "w2_fp4");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(w2_sf, "w2_sf");
    require_cuda_contiguous(gemm2_alpha, "gemm2_alpha");
    require_cuda_contiguous(gemm2_out, "gemm2_out");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(w2_fp4, at::kByte, "w2_fp4");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(w2_sf, at::kByte, "w2_sf");
    require_dtype(gemm2_alpha, at::kFloat, "gemm2_alpha");
    require_dtype(gemm2_out, at::kBFloat16, "gemm2_out");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(setup_ws, at::kByte, "setup_ws");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(w2_fp4,
                           (int64_t)E * hidden_size * inter_size / 2,
                           "w2_fp4");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(w2_sf,
                           (int64_t)E * hidden_size * inter_size / 16,
                           "w2_sf");
    require_numel_at_least(gemm2_alpha, E, "gemm2_alpha");
    require_numel_at_least(gemm2_out, expanded * hidden_size,
                           "gemm2_out");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        setup_ws,
        get_workspace_size_setup_gemm2_group_ptrs(E),
        "setup_ws");

    cutlass_gemm_setup_group_ptrs(
        raw_data_ptr(act_out),
        raw_data_ptr(w2_fp4),
        raw_data_ptr(fc2_act_sf),
        raw_data_ptr(w2_sf),
        gemm2_alpha.data_ptr<float>(),
        raw_data_ptr_mut(gemm2_out),
        expert_offset.data_ptr<int64_t>(),
        E,
        hidden_size,
        inter_size,
        expanded,
        raw_data_ptr_mut(setup_ws),
        current_stream());
}

void cutlass_gemm_forward_py(
    const torch::Tensor& act_out,
    const torch::Tensor& w2_fp4,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& w2_sf,
    const torch::Tensor& gemm2_alpha,
    const torch::Tensor& gemm2_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& setup_ws,
    const torch::Tensor& cutlass_ws,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(w2_fp4, "w2_fp4");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(w2_sf, "w2_sf");
    require_cuda_contiguous(gemm2_alpha, "gemm2_alpha");
    require_cuda_contiguous(gemm2_out, "gemm2_out");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_cuda_contiguous(cutlass_ws, "cutlass_ws");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(w2_fp4, at::kByte, "w2_fp4");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(w2_sf, at::kByte, "w2_sf");
    require_dtype(gemm2_alpha, at::kFloat, "gemm2_alpha");
    require_dtype(gemm2_out, at::kBFloat16, "gemm2_out");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(setup_ws, at::kByte, "setup_ws");
    require_dtype(cutlass_ws, at::kByte, "cutlass_ws");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(w2_fp4,
                           (int64_t)E * hidden_size * inter_size / 2,
                           "w2_fp4");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(w2_sf,
                           (int64_t)E * hidden_size * inter_size / 16,
                           "w2_sf");
    require_numel_at_least(gemm2_alpha, E, "gemm2_alpha");
    require_numel_at_least(gemm2_out, expanded * hidden_size,
                           "gemm2_out");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        setup_ws,
        get_workspace_size_setup_gemm2_group_ptrs(E),
        "setup_ws");
    require_numel_at_least(
        cutlass_ws,
        get_workspace_size_gemm2(setup_ws, E, M, hidden_size, inter_size),
        "cutlass_ws");

    cutlass_gemm_forward(
        raw_data_ptr_mut(setup_ws),
        raw_data_ptr_mut(cutlass_ws),
        cutlass_ws.numel(),
        E,
        hidden_size,
        inter_size,
        expanded,
        current_stream());
}

void setup_fused_finalize_group_ptrs_py(
    const torch::Tensor& act_out,
    const torch::Tensor& w2_fp4,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& w2_sf,
    const torch::Tensor& gemm2_alpha,
    const torch::Tensor& expert_offset,
    const torch::Tensor& setup_ws,
    const torch::Tensor& perm_scales,
    const torch::Tensor& permuted_row,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(w2_fp4, "w2_fp4");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(w2_sf, "w2_sf");
    require_cuda_contiguous(gemm2_alpha, "gemm2_alpha");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_cuda_contiguous(perm_scales, "perm_scales");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(w2_fp4, at::kByte, "w2_fp4");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(w2_sf, at::kByte, "w2_sf");
    require_dtype(gemm2_alpha, at::kFloat, "gemm2_alpha");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(setup_ws, at::kByte, "setup_ws");
    require_dtype(perm_scales, at::kFloat, "perm_scales");
    require_dtype(permuted_row, at::kInt, "permuted_row");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(w2_fp4,
                           (int64_t)E * hidden_size * inter_size / 2,
                           "w2_fp4");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(w2_sf,
                           (int64_t)E * hidden_size * inter_size / 16,
                           "w2_sf");
    require_numel_at_least(gemm2_alpha, E, "gemm2_alpha");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        setup_ws,
        get_workspace_size_setup_fused_finalize_group_ptrs(E),
        "setup_ws");
    require_numel_at_least(perm_scales, expanded, "perm_scales");
    require_numel_at_least(permuted_row, expanded, "permuted_row");

    cutlass_gemm_fused_finalize_setup_group_ptrs(
        raw_data_ptr(act_out),
        raw_data_ptr(w2_fp4),
        raw_data_ptr(fc2_act_sf),
        raw_data_ptr(w2_sf),
        gemm2_alpha.data_ptr<float>(),
        expert_offset.data_ptr<int64_t>(),
        E,
        hidden_size,
        inter_size,
        expanded,
        raw_data_ptr_mut(setup_ws),
        current_stream(),
        perm_scales.data_ptr<float>(),
        permuted_row.data_ptr<int>(),
        M,
        topk);
}

void cutlass_gemm_fused_finalize_forward_py(
    const torch::Tensor& act_out,
    const torch::Tensor& w2_fp4,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& w2_sf,
    const torch::Tensor& gemm2_alpha,
    const torch::Tensor& expert_offset,
    const torch::Tensor& setup_ws,
    const torch::Tensor& cutlass_ws,
    const torch::Tensor& output,
    const torch::Tensor& perm_scales,
    const torch::Tensor& permuted_row,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(w2_fp4, "w2_fp4");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(w2_sf, "w2_sf");
    require_cuda_contiguous(gemm2_alpha, "gemm2_alpha");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(setup_ws, "setup_ws");
    require_cuda_contiguous(cutlass_ws, "cutlass_ws");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(perm_scales, "perm_scales");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(w2_fp4, at::kByte, "w2_fp4");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(w2_sf, at::kByte, "w2_sf");
    require_dtype(gemm2_alpha, at::kFloat, "gemm2_alpha");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(setup_ws, at::kByte, "setup_ws");
    require_dtype(cutlass_ws, at::kByte, "cutlass_ws");
    require_dtype(output, at::kBFloat16, "output");
    require_dtype(perm_scales, at::kFloat, "perm_scales");
    require_dtype(permuted_row, at::kInt, "permuted_row");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(w2_fp4,
                           (int64_t)E * hidden_size * inter_size / 2,
                           "w2_fp4");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(w2_sf,
                           (int64_t)E * hidden_size * inter_size / 16,
                           "w2_sf");
    require_numel_at_least(gemm2_alpha, E, "gemm2_alpha");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        setup_ws,
        get_workspace_size_setup_fused_finalize_group_ptrs(E),
        "setup_ws");
    require_numel_at_least(
        cutlass_ws,
        get_workspace_size_gemm2_fused_finalize(
            setup_ws, E, M, hidden_size, inter_size),
        "cutlass_ws");
    require_numel_at_least(output, (int64_t)M * hidden_size, "output");
    require_numel_at_least(perm_scales, expanded, "perm_scales");
    require_numel_at_least(permuted_row, expanded, "permuted_row");

    cutlass_gemm_fused_finalize_forward(
        raw_data_ptr_mut(setup_ws),
        raw_data_ptr_mut(cutlass_ws),
        cutlass_ws.numel(),
        E,
        hidden_size,
        inter_size,
        expanded,
        current_stream(),
        raw_data_ptr_mut(output),
        M,
        topk);
}

void atrex_task30_gemm2_small_m_forward_py(
    const torch::Tensor& act_out,
    const torch::Tensor& w2_fp4,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& w2_sf,
    const torch::Tensor& gemm2_alpha,
    const torch::Tensor& output,
    const torch::Tensor& expert_offset,
    const torch::Tensor& permuted_row,
    const torch::Tensor& perm_scales,
    const torch::Tensor& workspace,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(w2_fp4, "w2_fp4");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(w2_sf, "w2_sf");
    require_cuda_contiguous(gemm2_alpha, "gemm2_alpha");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_cuda_contiguous(perm_scales, "perm_scales");
    require_cuda_contiguous(workspace, "workspace");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(w2_fp4, at::kByte, "w2_fp4");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(w2_sf, at::kByte, "w2_sf");
    require_dtype(gemm2_alpha, at::kFloat, "gemm2_alpha");
    require_dtype(output, at::kBFloat16, "output");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(permuted_row, at::kInt, "permuted_row");
    require_dtype(perm_scales, at::kFloat, "perm_scales");
    require_dtype(workspace, at::kByte, "workspace");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(w2_fp4,
                           (int64_t)E * hidden_size * inter_size / 2,
                           "w2_fp4");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(w2_sf,
                           (int64_t)E * hidden_size * inter_size / 16,
                           "w2_sf");
    require_numel_at_least(gemm2_alpha, E, "gemm2_alpha");
    require_numel_at_least(output, (int64_t)M * hidden_size, "output");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(permuted_row, expanded, "permuted_row");
    require_numel_at_least(perm_scales, expanded, "perm_scales");
    require_numel_at_least(
        workspace,
        get_workspace_size_task30_gemm2_small_m(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    atrex_task30_gemm2_small_m_forward(
        raw_data_ptr(act_out),
        raw_data_ptr(w2_fp4),
        raw_data_ptr(fc2_act_sf),
        raw_data_ptr(w2_sf),
        gemm2_alpha.data_ptr<float>(),
        raw_data_ptr_mut(output),
        expert_offset.data_ptr<int64_t>(),
        permuted_row.data_ptr<int>(),
        perm_scales.data_ptr<float>(),
        E,
        hidden_size,
        inter_size,
        M,
        expanded,
        raw_data_ptr_mut(workspace),
        current_stream());
}

void finalize_moe_routing_py(
    const torch::Tensor& gemm2_out,
    const torch::Tensor& output,
    const torch::Tensor& topk_weights,
    const torch::Tensor& unperm_map,
    const torch::Tensor& topk_ids,
    int M,
    int E,
    int topk,
    int hidden_size,
    bool output_preinitialized)
{
    require_cuda_contiguous(gemm2_out, "gemm2_out");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(topk_weights, "topk_weights");
    require_cuda_contiguous(unperm_map, "unperm_map");
    require_cuda_contiguous(topk_ids, "topk_ids");
    require_dtype(gemm2_out, at::kBFloat16, "gemm2_out");
    require_dtype(output, at::kBFloat16, "output");
    require_dtype(topk_weights, at::kFloat, "topk_weights");
    require_dtype(unperm_map, at::kInt, "unperm_map");
    require_dtype(topk_ids, at::kInt, "topk_ids");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(gemm2_out, expanded * hidden_size, "gemm2_out");
    require_numel_at_least(output, (int64_t)M * hidden_size, "output");
    require_numel_at_least(topk_weights, expanded, "topk_weights");
    require_numel_at_least(unperm_map, expanded, "unperm_map");
    require_numel_at_least(topk_ids, expanded, "topk_ids");

    set_output_preinitialized(output_preinitialized);
    finalize_moe_routing(
        raw_data_ptr(gemm2_out),
        raw_data_ptr_mut(output),
        nullptr,
        topk_weights.data_ptr<float>(),
        unperm_map.data_ptr<int>(),
        topk_ids.data_ptr<int>(),
        M,
        hidden_size,
        hidden_size,
        topk,
        E,
        0,
        current_stream());
    set_output_preinitialized(false);
}

void register_down(py::module_& m)
{
    m.def("wait_for_shared_event", &wait_for_shared_event_py,
          py::arg("shared_event"));
    m.def("get_workspace_size_task30_gemm2_small_m",
          &get_workspace_size_task30_gemm2_small_m,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_setup_gemm2_group_ptrs",
          &get_workspace_size_setup_gemm2_group_ptrs,
          py::arg("E"));
    m.def("get_workspace_size_gemm2", &get_workspace_size_gemm2,
          py::arg("setup_ws"), py::arg("E"), py::arg("M"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_setup_fused_finalize_group_ptrs",
          &get_workspace_size_setup_fused_finalize_group_ptrs,
          py::arg("E"));
    m.def("get_workspace_size_gemm2_fused_finalize",
          &get_workspace_size_gemm2_fused_finalize,
          py::arg("setup_ws"), py::arg("E"), py::arg("M"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("setup_gemm2_group_ptrs", &setup_gemm2_group_ptrs_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("gemm2_out"),
          py::arg("expert_offset"), py::arg("setup_ws"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("cutlass_gemm_forward", &cutlass_gemm_forward_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("gemm2_out"),
          py::arg("expert_offset"), py::arg("setup_ws"),
          py::arg("cutlass_ws"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("setup_fused_finalize_group_ptrs",
          &setup_fused_finalize_group_ptrs_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("expert_offset"),
          py::arg("setup_ws"),
          py::arg("perm_scales"), py::arg("permuted_row"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("cutlass_gemm_fused_finalize_forward",
          &cutlass_gemm_fused_finalize_forward_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("expert_offset"),
          py::arg("setup_ws"), py::arg("cutlass_ws"), py::arg("output"),
          py::arg("perm_scales"), py::arg("permuted_row"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("task30_gemm2_small_m_forward",
          &atrex_task30_gemm2_small_m_forward_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("output"),
          py::arg("expert_offset"), py::arg("permuted_row"),
          py::arg("perm_scales"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("finalize_moe_routing", &finalize_moe_routing_py,
          py::arg("gemm2_out"), py::arg("output"),
          py::arg("topk_weights"), py::arg("unperm_map"),
          py::arg("topk_ids"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"),
          py::arg("output_preinitialized") = false);
}
