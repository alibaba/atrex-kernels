#include "pybind_common.h"

namespace py = pybind11;

extern "C" void gemm_forward_v20_fused_act(
    void const* a_fp4,
    void const* b_fp4,
    void const* sf_a,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    void* act_sf_flat,
    float const* fc2_act_global_scale,
    int64_t const* expert_first_token_offset,
    int num_experts,
    int N_gemm,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream);

extern "C" void gemm_forward_v20(
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
    void* workspace,
    cudaStream_t stream);

extern "C" void do_activation(
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
    bool interleaved);

extern "C" int64_t task13_gemm1_gather_fused_act_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens);

extern "C" int64_t task13_gather_fc2_act_sf_bytes(
    int num_experts,
    int inter_size,
    int64_t expanded_num_tokens);

extern "C" void task13_gemm1_gather_fused_act_forward(
    void const* hidden_states,
    void const* input_sf,
    void const* b_fp4,
    void const* sf_b,
    float const* alpha,
    void* output_fp4,
    float const* fc2_act_global_scale,
    void* fc2_act_sf,
    float const* fc1_act_global_scale,
    int const* permuted_source_rows,
    int64_t const* expert_first_token_offset,
    int num_tokens,
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    void* workspace,
    cudaStream_t stream,
    bool input_is_nvfp4);

extern "C" int64_t atrex_task29_gemm1_small_m_fused_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens,
    int split_k);

extern "C" void atrex_task29_gemm1_small_m_forward_fused(
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

extern "C" int64_t atrex_task29_gemm1_small_m_grouped_m16_fused_act_workspace_bytes(
    int num_experts,
    int N,
    int K,
    int64_t expanded_num_tokens);

extern "C" void atrex_task29_gemm1_small_m_forward_grouped_m16_fused_act(
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
    void* workspace,
    cudaStream_t stream);

int64_t get_workspace_size_gemm1(int M, int E, int topk,
                                 int hidden_size, int inter_size)
{
    (void)hidden_size;
    int64_t expanded = (int64_t)M * topk;
    int max_gemm_tiles = ((int)expanded / 128 + E) *
                         (2 * inter_size / 128 + 1);
    int64_t tile_info_bytes = align256(4 + (int64_t)max_gemm_tiles * 12);
    int64_t tma_bytes = align256(256 + (int64_t)E * 2 * 128);
    return tile_info_bytes + tma_bytes;
}

int64_t get_workspace_size_task13_gather_fused_act(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return task13_gemm1_gather_fused_act_workspace_bytes(
        E, 2 * inter_size, hidden_size, (int64_t)M * topk);
}

int64_t get_task13_fc2_act_sf_size(
    int M, int E, int topk, int inter_size)
{
    return task13_gather_fc2_act_sf_bytes(
        E, inter_size, (int64_t)M * topk);
}

int64_t get_workspace_size_task29_gemm1_small_m_fused(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_task29_gemm1_small_m_fused_workspace_bytes(
        E, 2 * inter_size, hidden_size, (int64_t)M * topk, 4);
}

int64_t get_workspace_size_task29_gemm1_small_m_grouped_m16_fused_act(
    int M, int E, int topk, int hidden_size, int inter_size)
{
    return atrex_task29_gemm1_small_m_grouped_m16_fused_act_workspace_bytes(
        E, 2 * inter_size, hidden_size, (int64_t)M * topk);
}

void gemm_forward_v20_fused_act_py(
    const torch::Tensor& expand_out,
    const torch::Tensor& w1_fp4,
    const torch::Tensor& fc1_act_sf,
    const torch::Tensor& w1_sf,
    const torch::Tensor& gemm1_alpha,
    const torch::Tensor& act_out,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& fc2_act_global_scale,
    const torch::Tensor& expert_offset,
    const torch::Tensor& gemm_ws,
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
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(fc2_act_global_scale, "fc2_act_global_scale");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(gemm_ws, "gemm_ws");
    require_dtype(expand_out, at::kByte, "expand_out");
    require_dtype(w1_fp4, at::kByte, "w1_fp4");
    require_dtype(fc1_act_sf, at::kByte, "fc1_act_sf");
    require_dtype(w1_sf, at::kByte, "w1_sf");
    require_dtype(gemm1_alpha, at::kFloat, "gemm1_alpha");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(fc2_act_global_scale, at::kFloat,
                  "fc2_act_global_scale");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(gemm_ws, at::kByte, "gemm_ws");

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
    require_numel_at_least(fc2_act_sf,
                           get_fc2_act_sf_size(M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(fc2_act_global_scale, E,
                           "fc2_act_global_scale");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(gemm_ws,
                           get_workspace_size_gemm1(
                               M, E, topk, hidden_size, inter_size),
                           "gemm_ws");

    gemm_forward_v20_fused_act(
        raw_data_ptr(expand_out),
        raw_data_ptr(w1_fp4),
        raw_data_ptr(fc1_act_sf),
        raw_data_ptr(w1_sf),
        gemm1_alpha.data_ptr<float>(),
        raw_data_ptr_mut(act_out),
        raw_data_ptr_mut(fc2_act_sf),
        fc2_act_global_scale.data_ptr<float>(),
        expert_offset.data_ptr<int64_t>(),
        E,
        2 * inter_size,
        hidden_size,
        expanded,
        raw_data_ptr_mut(gemm_ws),
        current_stream());
}

void gemm_forward_v20_py(
    const torch::Tensor& expand_out,
    const torch::Tensor& w1_fp4,
    const torch::Tensor& fc1_act_sf,
    const torch::Tensor& w1_sf,
    const torch::Tensor& gemm1_alpha,
    const torch::Tensor& gemm1_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& gemm_ws,
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
    require_cuda_contiguous(gemm_ws, "gemm_ws");
    require_dtype(expand_out, at::kByte, "expand_out");
    require_dtype(w1_fp4, at::kByte, "w1_fp4");
    require_dtype(fc1_act_sf, at::kByte, "fc1_act_sf");
    require_dtype(w1_sf, at::kByte, "w1_sf");
    require_dtype(gemm1_alpha, at::kFloat, "gemm1_alpha");
    require_dtype(gemm1_out, at::kBFloat16, "gemm1_out");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(gemm_ws, at::kByte, "gemm_ws");

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
    require_numel_at_least(gemm_ws,
                           get_workspace_size_gemm1(
                               M, E, topk, hidden_size, inter_size),
                           "gemm_ws");

    gemm_forward_v20(
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
        raw_data_ptr_mut(gemm_ws),
        current_stream());
}

void do_activation_py(
    const torch::Tensor& act_out,
    const torch::Tensor& gemm1_out,
    const torch::Tensor& expert_offset,
    const torch::Tensor& fc2_act_global_scale,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& permuted_token_selected_experts,
    int M,
    int E,
    int topk,
    int inter_size)
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

    do_activation(
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
        true);
}

void task13_gemm1_gather_fused_act_forward_py(
    const torch::Tensor& hidden_states,
    const py::object& input_sf,
    const torch::Tensor& w1_fp4,
    const torch::Tensor& w1_sf,
    const torch::Tensor& gemm1_alpha,
    const torch::Tensor& act_out,
    const torch::Tensor& fc2_act_global_scale,
    const torch::Tensor& fc2_act_sf,
    const torch::Tensor& fc1_act_global_scale,
    const torch::Tensor& permuted_source_rows,
    const torch::Tensor& expert_offset,
    const torch::Tensor& workspace,
    int M,
    int E,
    int topk,
    int hidden_size,
    int inter_size)
{
    require_cuda_contiguous(hidden_states, "hidden_states");
    require_cuda_contiguous(w1_fp4, "w1_fp4");
    require_cuda_contiguous(w1_sf, "w1_sf");
    require_cuda_contiguous(gemm1_alpha, "gemm1_alpha");
    require_cuda_contiguous(act_out, "act_out");
    require_cuda_contiguous(fc2_act_global_scale, "fc2_act_global_scale");
    require_cuda_contiguous(fc2_act_sf, "fc2_act_sf");
    require_cuda_contiguous(fc1_act_global_scale, "fc1_act_global_scale");
    require_cuda_contiguous(permuted_source_rows, "permuted_source_rows");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(workspace, "workspace");

    bool input_is_nvfp4 = !input_sf.is_none();
    const uint8_t* input_sf_ptr = nullptr;
    if (input_is_nvfp4) {
        require_dtype(hidden_states, at::kByte, "hidden_states");
        require_numel_at_least(hidden_states, (int64_t)M * hidden_size / 2,
                               "hidden_states");
        torch::Tensor input_sf_tensor = input_sf.cast<torch::Tensor>();
        require_cuda_contiguous(input_sf_tensor, "input_sf");
        require_dtype(input_sf_tensor, at::kByte, "input_sf");
        require_numel_at_least(input_sf_tensor,
                               (int64_t)M * hidden_size / 16,
                               "input_sf");
        input_sf_ptr = reinterpret_cast<const uint8_t*>(
            input_sf_tensor.data_ptr());
    } else {
        require_dtype(hidden_states, at::kBFloat16, "hidden_states");
        require_numel_at_least(hidden_states, (int64_t)M * hidden_size,
                               "hidden_states");
    }

    require_dtype(w1_fp4, at::kByte, "w1_fp4");
    require_dtype(w1_sf, at::kByte, "w1_sf");
    require_dtype(gemm1_alpha, at::kFloat, "gemm1_alpha");
    require_dtype(act_out, at::kByte, "act_out");
    require_dtype(fc2_act_global_scale, at::kFloat,
                  "fc2_act_global_scale");
    require_dtype(fc2_act_sf, at::kByte, "fc2_act_sf");
    require_dtype(fc1_act_global_scale, at::kFloat,
                  "fc1_act_global_scale");
    require_dtype(permuted_source_rows, at::kInt, "permuted_source_rows");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(workspace, at::kByte, "workspace");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(w1_fp4,
                           (int64_t)E * 2 * inter_size * hidden_size / 2,
                           "w1_fp4");
    require_numel_at_least(w1_sf,
                           (int64_t)E * 2 * inter_size * hidden_size / 16,
                           "w1_sf");
    require_numel_at_least(gemm1_alpha, E, "gemm1_alpha");
    require_numel_at_least(act_out, expanded * inter_size / 2, "act_out");
    require_numel_at_least(fc2_act_global_scale, E,
                           "fc2_act_global_scale");
    require_numel_at_least(fc2_act_sf,
                           get_task13_fc2_act_sf_size(
                               M, E, topk, inter_size),
                           "fc2_act_sf");
    require_numel_at_least(fc1_act_global_scale, E,
                           "fc1_act_global_scale");
    require_numel_at_least(permuted_source_rows, expanded,
                           "permuted_source_rows");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(
        workspace,
        get_workspace_size_task13_gather_fused_act(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    task13_gemm1_gather_fused_act_forward(
        raw_data_ptr(hidden_states),
        input_sf_ptr,
        raw_data_ptr(w1_fp4),
        raw_data_ptr(w1_sf),
        gemm1_alpha.data_ptr<float>(),
        raw_data_ptr_mut(act_out),
        fc2_act_global_scale.data_ptr<float>(),
        raw_data_ptr_mut(fc2_act_sf),
        fc1_act_global_scale.data_ptr<float>(),
        permuted_source_rows.data_ptr<int>(),
        expert_offset.data_ptr<int64_t>(),
        M,
        E,
        2 * inter_size,
        hidden_size,
        expanded,
        raw_data_ptr_mut(workspace),
        current_stream(),
        input_is_nvfp4);
}

void atrex_task29_gemm1_small_m_forward_fused_py(
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
        get_workspace_size_task29_gemm1_small_m_fused(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    atrex_task29_gemm1_small_m_forward_fused(
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

void atrex_task29_gemm1_small_m_forward_grouped_m16_fused_act_py(
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
    int inter_size)
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
        get_workspace_size_task29_gemm1_small_m_grouped_m16_fused_act(
            M, E, topk, hidden_size, inter_size),
        "workspace");

    atrex_task29_gemm1_small_m_forward_grouped_m16_fused_act(
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
        raw_data_ptr_mut(workspace),
        current_stream());
}

void register_up_gate(py::module_& m)
{
    m.def("get_workspace_size_gemm1", &get_workspace_size_gemm1,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_task13_gather_fused_act",
          &get_workspace_size_task13_gather_fused_act,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_task13_fc2_act_sf_size",
          &get_task13_fc2_act_sf_size,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("inter_size"));
    m.def("get_workspace_size_task29_gemm1_small_m_fused",
          &get_workspace_size_task29_gemm1_small_m_fused,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("get_workspace_size_task29_gemm1_small_m_grouped_m16_fused_act",
          &get_workspace_size_task29_gemm1_small_m_grouped_m16_fused_act,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("gemm_forward_v20_fused_act", &gemm_forward_v20_fused_act_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("act_out"),
          py::arg("fc2_act_sf"), py::arg("fc2_act_global_scale"),
          py::arg("expert_offset"), py::arg("gemm_ws"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("gemm_forward_v20", &gemm_forward_v20_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("gemm1_out"),
          py::arg("expert_offset"), py::arg("gemm_ws"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("do_activation", &do_activation_py,
          py::arg("act_out"), py::arg("gemm1_out"),
          py::arg("expert_offset"), py::arg("fc2_act_global_scale"),
          py::arg("fc2_act_sf"),
          py::arg("permuted_token_selected_experts"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("inter_size"));
    m.def("task13_gemm1_gather_fused_act_forward",
          &task13_gemm1_gather_fused_act_forward_py,
          py::arg("hidden_states"), py::arg("input_sf"),
          py::arg("w1_fp4"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("act_out"),
          py::arg("fc2_act_global_scale"), py::arg("fc2_act_sf"),
          py::arg("fc1_act_global_scale"),
          py::arg("permuted_source_rows"),
          py::arg("expert_offset"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("task29_gemm1_small_m_forward_fused",
          &atrex_task29_gemm1_small_m_forward_fused_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("gemm1_out"),
          py::arg("expert_offset"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
    m.def("task29_gemm1_small_m_forward_grouped_m16_fused_act",
          &atrex_task29_gemm1_small_m_forward_grouped_m16_fused_act_py,
          py::arg("expand_out"), py::arg("w1_fp4"),
          py::arg("fc1_act_sf"), py::arg("w1_sf"),
          py::arg("gemm1_alpha"), py::arg("act_out"),
          py::arg("fc2_act_global_scale"), py::arg("fc2_act_sf"),
          py::arg("expert_offset"), py::arg("workspace"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("inter_size"));
}
