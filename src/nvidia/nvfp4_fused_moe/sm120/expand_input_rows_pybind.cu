#include "pybind_common.h"

namespace py = pybind11;

extern "C" void expand_input_rows(
    void const* unpermuted_input,
    void* permuted_output,
    float const* unpermuted_scales,
    float* permuted_scales,
    int const* permuted_row_to_unpermuted_row,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t k,
    float const* fc1_act_global_scale,
    bool use_per_expert_act_scale,
    int64_t const* expert_first_token_offset,
    uint8_t* fc1_act_sf_flat,
    uint8_t const* input_sf,
    bool swizzled_input_sf,
    int64_t num_experts_per_node,
    int const* token_expert_ids,
    bool skip_sf_padding,
    cudaStream_t stream);

void expand_input_rows_py(
    const torch::Tensor& hidden_states,
    const torch::Tensor& expand_out,
    const torch::Tensor& topk_weights,
    const torch::Tensor& perm_scales,
    const torch::Tensor& permuted_row,
    const torch::Tensor& fc1_act_global_scale,
    const torch::Tensor& expert_offset,
    const torch::Tensor& fc1_act_sf,
    const py::object& input_sf,
    const torch::Tensor& permuted_token_selected_experts,
    int M,
    int E,
    int topk,
    int hidden_size,
    bool skip_sf_padding)
{
    require_cuda_contiguous(hidden_states, "hidden_states");
    require_cuda_contiguous(expand_out, "expand_out");
    require_cuda_contiguous(topk_weights, "topk_weights");
    require_cuda_contiguous(perm_scales, "perm_scales");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_cuda_contiguous(fc1_act_global_scale, "fc1_act_global_scale");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(fc1_act_sf, "fc1_act_sf");
    require_cuda_contiguous(permuted_token_selected_experts,
                            "permuted_token_selected_experts");
    require_dtype(expand_out, at::kByte, "expand_out");
    require_dtype(topk_weights, at::kFloat, "topk_weights");
    require_dtype(perm_scales, at::kFloat, "perm_scales");
    require_dtype(permuted_row, at::kInt, "permuted_row");
    require_dtype(fc1_act_global_scale, at::kFloat, "fc1_act_global_scale");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(fc1_act_sf, at::kByte, "fc1_act_sf");
    require_dtype(permuted_token_selected_experts, at::kInt,
                  "permuted_token_selected_experts");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(expand_out, expanded * hidden_size / 2,
                           "expand_out");
    require_numel_at_least(topk_weights, expanded, "topk_weights");
    require_numel_at_least(perm_scales, expanded, "perm_scales");
    require_numel_at_least(permuted_row, expanded, "permuted_row");
    require_numel_at_least(fc1_act_global_scale, E, "fc1_act_global_scale");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(fc1_act_sf,
                           get_fc1_act_sf_size(M, E, topk, hidden_size),
                           "fc1_act_sf");
    require_numel_at_least(permuted_token_selected_experts, expanded,
                           "permuted_token_selected_experts");

    const uint8_t* input_sf_ptr = optional_input_sf_ptr(input_sf);
    if (input_sf_ptr == nullptr) {
        require_dtype(hidden_states, at::kBFloat16, "hidden_states");
        require_numel_at_least(hidden_states, (int64_t)M * hidden_size,
                               "hidden_states");
    } else {
        require_dtype(hidden_states, at::kByte, "hidden_states");
        require_numel_at_least(hidden_states, (int64_t)M * hidden_size / 2,
                               "hidden_states");
    }

    expand_input_rows(
        raw_data_ptr(hidden_states),
        raw_data_ptr_mut(expand_out),
        topk_weights.data_ptr<float>(),
        perm_scales.data_ptr<float>(),
        permuted_row.data_ptr<int>(),
        M,
        hidden_size,
        topk,
        fc1_act_global_scale.data_ptr<float>(),
        true,
        expert_offset.data_ptr<int64_t>(),
        fc1_act_sf.data_ptr<uint8_t>(),
        input_sf_ptr,
        false,
        E,
        permuted_token_selected_experts.data_ptr<int>(),
        skip_sf_padding,
        current_stream());
}

void register_expand_input_rows(py::module_& m)
{
    m.def("expand_input_rows", &expand_input_rows_py,
          py::arg("hidden_states"), py::arg("expand_out"),
          py::arg("topk_weights"), py::arg("perm_scales"),
          py::arg("permuted_row"), py::arg("fc1_act_global_scale"),
          py::arg("expert_offset"), py::arg("fc1_act_sf"),
          py::arg("input_sf"), py::arg("permuted_token_selected_experts"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"), py::arg("skip_sf_padding") = false);
}
