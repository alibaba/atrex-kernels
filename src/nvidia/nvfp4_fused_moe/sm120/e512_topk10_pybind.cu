#include "pybind_common.h"

namespace py = pybind11;

// ============================================================================
// e512_topk10 (E=512, topk=10, hidden=2560, inter=320) — additive bindings.
//
// These wrap the atrex_e512t10_* extern "C" entries defined in
// e512_topk10_gemm1.cu / e512_topk10_gemm2.cu / e512_topk10_activation.cu.
// Python-facing names are prefixed with e512_topk10_ so they coexist with the
// dev topk=8 bindings (do_activation / task29_* / task30_*) registered by
// register_up_gate / register_down without any name collision. New optional
// parameters carry defaults so existing callers are unaffected.
//
// The extern "C" declarations below MUST match the definitions in the .cu
// files byte-for-byte (param count / type / order); a mismatch would silently
// corrupt the ABI rather than fail at link time.
// ============================================================================

extern "C" int64_t atrex_e512t10_task29_fused_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k);

extern "C" void atrex_e512t10_task29_forward_fused(
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
    int split_k,
    void* workspace,
    cudaStream_t stream);

extern "C" int64_t atrex_e512t10_task29_grouped_m16_fused_act_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens);

extern "C" void atrex_e512t10_task29_forward_grouped_m16_fused_act(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    bool use_shared_sf_staging,
    void* workspace,
    cudaStream_t stream);

extern "C" void atrex_e512t10_do_activation(
    void* output,
    void const* gemm_result,
    float const* fp8_quant,
    void const* bias,
    bool bias_is_broadcast,
    int64_t const* expert_first_token_offset,
    int num_experts_per_node,
    int64_t inter_size,
    int64_t expanded_num_tokens,
    float const* fc2_act_global_scale,
    bool use_per_expert_act_scale,
    uint8_t* fc2_act_sf_flat,
    int const* token_expert_ids,
    cudaStream_t stream,
    bool interleaved,
    bool skip_sf_padding,
    bool round_post_act_bf16);

extern "C" int64_t atrex_e512t10_task30_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens);

extern "C" void atrex_e512t10_task30_forward_fixed_e512t10(
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
    bool use_shared_sf_staging,
    void* workspace,
    void* expert_rows_bf16,
    float const* topk_weights,
    int* completion_counters,
    bool output_preinitialized,
    int const* topk_ids,
    cudaStream_t stream);

// ============================================================================
// Workspace size helpers
// ============================================================================

int64_t get_workspace_size_e512_topk10_task29_fused(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_e512t10_task29_fused_workspace_bytes(
        E, 2 * inter_size, hidden_size, (int64_t)M * topk, 4);
}

int64_t get_workspace_size_e512_topk10_task29_grouped_m16_fused_act(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_e512t10_task29_grouped_m16_fused_act_workspace_bytes(
        E, 2 * inter_size, hidden_size, (int64_t)M * topk);
}

int64_t get_workspace_size_e512_topk10_task30(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_e512t10_task30_workspace_bytes(
        E, hidden_size, inter_size, (int64_t)M * topk);
}

// ============================================================================
// Torch-level wrappers
// ============================================================================

void e512_topk10_task29_forward_fused_py(
    const torch::Tensor& expand_out,
    const torch::Tensor& w1_fp4,
    const torch::Tensor& fc1_act_sf,
    const torch::Tensor& w1_sf,
    const torch::Tensor& gemm1_alpha,
    const torch::Tensor& gemm1_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& workspace,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(expand_out, "expand_out");
    require_cuda_contiguous(w1_fp4, "w1_fp4");
    require_cuda_contiguous(fc1_act_sf, "fc1_act_sf");
    require_cuda_contiguous(w1_sf, "w1_sf");
    require_cuda_contiguous(gemm1_alpha, "gemm1_alpha");
    require_cuda_contiguous(gemm1_out, "gemm1_out");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(workspace, "workspace");
    require_dtype(expand_out, at::kByte, "expand_out");
    require_dtype(w1_fp4, at::kByte, "w1_fp4");
    require_dtype(fc1_act_sf, at::kByte, "fc1_act_sf");
    require_dtype(w1_sf, at::kByte, "w1_sf");
    require_dtype(gemm1_alpha, at::kFloat, "gemm1_alpha");
    require_dtype(gemm1_out, at::kBFloat16, "gemm1_out");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(workspace, at::kByte, "workspace");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(expand_out, expanded * hidden_size / 2,
                           "expand_out");
    require_numel_at_least(w1_fp4,
                           (int64_t)E * 2 * inter_size * hidden_size / 2,
                           "w1_fp4");
    require_numel_at_least(fc1_act_sf,
                           get_fc1_act_sf_size(M, E, topk, hidden_size),
                           "fc1_act_sf");
    require_numel_at_least(w1_sf,
                           (int64_t)E * 2 * inter_size * hidden_size / 16,
                           "w1_sf");
    require_numel_at_least(gemm1_alpha, E, "gemm1_alpha");
    require_numel_at_least(gemm1_out, expanded * 2 * inter_size,
                           "gemm1_out");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        workspace,
        get_workspace_size_e512_topk10_task29_fused(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    atrex_e512t10_task29_forward_fused(
        raw_data_ptr(expand_out),
        raw_data_ptr(w1_fp4),
        raw_data_ptr(fc1_act_sf),
        raw_data_ptr(w1_sf),
        gemm1_alpha.data_ptr<float>(),
        raw_data_ptr_mut(gemm1_out),
        expert_offset.data_ptr<int64_t>(),
        E,
        2 * inter_size,
        hidden_size,
        expanded,
        4,
        raw_data_ptr_mut(workspace),
        current_stream());
}

void e512_topk10_task29_forward_grouped_m16_fused_act_py(
    const torch::Tensor& expand_out,
    const torch::Tensor& w1_fp4,
    const torch::Tensor& fc1_act_sf,
    const torch::Tensor& w1_sf,
    const torch::Tensor& gemm1_alpha,
    const torch::Tensor& act_out,
    const torch::Tensor& fc2_act_global_scale,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& expert_offset,
    const torch::Tensor& workspace,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size,
    bool use_shared_sf_staging)
{
    require_cuda_contiguous(expand_out, "expand_out");
    require_cuda_contiguous(w1_fp4, "w1_fp4");
    require_cuda_contiguous(fc1_act_sf, "fc1_act_sf");
    require_cuda_contiguous(w1_sf, "w1_sf");
    require_cuda_contiguous(gemm1_alpha, "gemm1_alpha");
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(fc2_act_global_scale, "fc2_act_global_scale");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(workspace, "workspace");
    require_dtype(expand_out, at::kByte, "expand_out");
    require_dtype(w1_fp4, at::kByte, "w1_fp4");
    require_dtype(fc1_act_sf, at::kByte, "fc1_act_sf");
    require_dtype(w1_sf, at::kByte, "w1_sf");
    require_dtype(gemm1_alpha, at::kFloat, "gemm1_alpha");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(fc2_act_global_scale, at::kFloat,
                  "fc2_act_global_scale");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(workspace, at::kByte, "workspace");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(expand_out, expanded * hidden_size / 2,
                           "expand_out");
    require_numel_at_least(w1_fp4,
                           (int64_t)E * 2 * inter_size * hidden_size / 2,
                           "w1_fp4");
    require_numel_at_least(fc1_act_sf,
                           get_fc1_act_sf_size(M, E, topk, hidden_size),
                           "fc1_act_sf");
    require_numel_at_least(w1_sf,
                           (int64_t)E * 2 * inter_size * hidden_size / 16,
                           "w1_sf");
    require_numel_at_least(gemm1_alpha, E, "gemm1_alpha");
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(fc2_act_global_scale, E,
                           "fc2_act_global_scale");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        workspace,
        get_workspace_size_e512_topk10_task29_grouped_m16_fused_act(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    atrex_e512t10_task29_forward_grouped_m16_fused_act(
        raw_data_ptr(expand_out),
        raw_data_ptr(w1_fp4),
        raw_data_ptr(fc1_act_sf),
        raw_data_ptr(w1_sf),
        gemm1_alpha.data_ptr<float>(),
        raw_data_ptr_mut(act_out),
        fc2_act_global_scale.data_ptr<float>(),
        raw_data_ptr_mut(fc2_act_sf),
        expert_offset.data_ptr<int64_t>(),
        E,
        2 * inter_size,
        hidden_size,
        expanded,
        use_shared_sf_staging,
        raw_data_ptr_mut(workspace),
        current_stream());
}

void e512_topk10_do_activation_py(
    const torch::Tensor& act_out,
    const torch::Tensor& gemm1_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& fc2_act_global_scale,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& permuted_token_selected_experts,
    int M,
    int E,
    int topk,
    int inter_size,
    bool skip_sf_padding,
    bool round_post_act_bf16)
{
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(gemm1_out, "gemm1_out");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(fc2_act_global_scale, "fc2_act_global_scale");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(permuted_token_selected_experts,
                            "permuted_token_selected_experts");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(gemm1_out, at::kBFloat16, "gemm1_out");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(fc2_act_global_scale, at::kFloat,
                  "fc2_act_global_scale");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(permuted_token_selected_experts, at::kInt,
                  "permuted_token_selected_experts");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(gemm1_out, expanded * 2 * inter_size,
                           "gemm1_out");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(fc2_act_global_scale, E,
                           "fc2_act_global_scale");
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(permuted_token_selected_experts, expanded,
                           "permuted_token_selected_experts");

    atrex_e512t10_do_activation(
        raw_data_ptr_mut(act_out),
        raw_data_ptr(gemm1_out),
        nullptr,
        nullptr,
        false,
        expert_offset.data_ptr<int64_t>(),
        E,
        inter_size,
        expanded,
        fc2_act_global_scale.data_ptr<float>(),
        true,
        fc2_act_sf.data_ptr<uint8_t>(),
        permuted_token_selected_experts.data_ptr<int>(),
        current_stream(),
        true,
        skip_sf_padding,
        round_post_act_bf16);
}

void e512_topk10_task30_forward_fixed_py(
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
    const torch::Tensor& fixed_expert_rows,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const torch::Tensor& completion_counters,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size,
    bool use_shared_sf_staging,
    bool output_preinitialized)
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
    require_cuda_contiguous(fixed_expert_rows, "fixed_expert_rows");
    require_cuda_contiguous(topk_weights, "topk_weights");
    require_cuda_contiguous(topk_ids, "topk_ids");
    require_cuda_contiguous(completion_counters, "completion_counters");
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
    require_dtype(fixed_expert_rows, at::kBFloat16, "fixed_expert_rows");
    require_dtype(topk_weights, at::kFloat, "topk_weights");
    require_dtype(topk_ids, at::kInt, "topk_ids");
    require_dtype(completion_counters, at::kInt, "completion_counters");

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
        get_workspace_size_e512_topk10_task30(
            M, E, topk, hidden_size, inter_size),
        "workspace");
    require_numel_at_least(fixed_expert_rows, expanded * hidden_size,
                           "fixed_expert_rows");
    require_numel_at_least(topk_weights, expanded, "topk_weights");
    require_numel_at_least(topk_ids, expanded, "topk_ids");
    int64_t const n_tiles = (hidden_size + 127) / 128;
    require_numel_at_least(completion_counters, M * n_tiles,
                           "completion_counters");

    atrex_e512t10_task30_forward_fixed_e512t10(
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
        use_shared_sf_staging,
        raw_data_ptr_mut(workspace),
        raw_data_ptr_mut(fixed_expert_rows),
        topk_weights.data_ptr<float>(),
        completion_counters.data_ptr<int>(),
        output_preinitialized,
        topk_ids.data_ptr<int>(),
        current_stream());
}

void register_e512_topk10(py::module_& m)
{
    m.def("get_workspace_size_e512_topk10_task29_fused",
          &get_workspace_size_e512_topk10_task29_fused,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_e512_topk10_task29_grouped_m16_fused_act",
          &get_workspace_size_e512_topk10_task29_grouped_m16_fused_act,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_e512_topk10_task30",
          &get_workspace_size_e512_topk10_task30,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("e512_topk10_task29_forward_fused",
          &e512_topk10_task29_forward_fused_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("gemm1_out"),
          py::arg("expert_offset"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("e512_topk10_task29_forward_grouped_m16_fused_act",
          &e512_topk10_task29_forward_grouped_m16_fused_act_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("act_out"),
          py::arg("fc2_act_global_scale"), py::arg("fc2_act_sf"),
          py::arg("expert_offset"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"),
          py::arg("use_shared_sf_staging") = false);
    m.def("e512_topk10_do_activation",
          &e512_topk10_do_activation_py,
          py::arg("act_out"), py::arg("gemm1_out"),
          py::arg("expert_offset"), py::arg("fc2_act_global_scale"),
          py::arg("fc2_act_sf"),
          py::arg("permuted_token_selected_experts"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("inter_size"),
          py::arg("skip_sf_padding") = false,
          py::arg("round_post_act_bf16") = false);
    m.def("e512_topk10_task30_forward_fixed",
          &e512_topk10_task30_forward_fixed_py,
          py::arg("act_out"), py::arg("w2_fp4"),
          py::arg("fc2_act_sf"), py::arg("w2_sf"),
          py::arg("gemm2_alpha"), py::arg("output"),
          py::arg("expert_offset"), py::arg("permuted_row"),
          py::arg("perm_scales"), py::arg("workspace"),
          py::arg("fixed_expert_rows"), py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("completion_counters"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"),
          py::arg("use_shared_sf_staging") = false,
          py::arg("output_preinitialized") = false);
}
