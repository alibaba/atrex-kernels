#include "pybind_common.h"

#include <stdexcept>

namespace py = pybind11;

extern "C" void routing_sort(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row,
    int64_t* expert_first_token_offset,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t num_tokens,
    int64_t num_experts_per_node,
    int64_t num_experts_per_token,
    int start_expert_id,
    uint16_t* output_to_zero,
    int64_t output_numel,
    int* completion_counters_to_zero,
    int64_t completion_counter_numel,
    float const* a1_global_scale,
    float const* w1_global_scale,
    float* gemm1_alpha,
    float const* a2_global_scale,
    float const* w2_global_scale,
    float* gemm2_alpha,
    cudaStream_t stream);

extern "C" void routing_sort_with_scales(
    int const* token_selected_experts,
    float const* topk_weights,
    int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row,
    int64_t* expert_first_token_offset,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    float* permuted_scales,
    int64_t num_tokens,
    int64_t num_experts_per_node,
    int64_t num_experts_per_token,
    int start_expert_id,
    cudaStream_t stream);

int64_t get_workspace_size_routing(int M, int E, int topk)
{
    (void)topk;
    int64_t num_blocks_per_seq = get_num_blocks_per_seq(M, E);
    int64_t off = 0;
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    off += align256((int64_t)E * M * sizeof(int));
    return off;
}

void routing_sort_py(
    const torch::Tensor& topk_ids,
    const torch::Tensor& routing_ws,
    const torch::Tensor& expert_offset,
    const torch::Tensor& permuted_token_selected_experts,
    const torch::Tensor& permuted_row,
    const torch::Tensor& unperm_map,
    int M,
    int E,
    int topk,
    const py::object& output_to_zero_obj,
    const py::object& a1_global_scale_obj,
    const py::object& w1_global_scale_obj,
    const py::object& gemm1_alpha_obj,
    const py::object& a2_global_scale_obj,
    const py::object& w2_global_scale_obj,
    const py::object& gemm2_alpha_obj,
    const py::object& completion_counters_to_zero_obj)
{
    require_cuda_contiguous(topk_ids, "topk_ids");
    require_cuda_contiguous(routing_ws, "routing_ws");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(permuted_token_selected_experts,
                            "permuted_token_selected_experts");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_cuda_contiguous(unperm_map, "unperm_map");
    require_dtype(topk_ids, at::kInt, "topk_ids");
    require_dtype(routing_ws, at::kByte, "routing_ws");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(permuted_token_selected_experts, at::kInt,
                  "permuted_token_selected_experts");
    require_dtype(permuted_row, at::kInt, "permuted_row");
    require_dtype(unperm_map, at::kInt, "unperm_map");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(topk_ids, expanded, "topk_ids");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(permuted_token_selected_experts, expanded,
                           "permuted_token_selected_experts");
    require_numel_at_least(permuted_row, expanded, "permuted_row");
    require_numel_at_least(unperm_map, expanded, "unperm_map");
    require_numel_at_least(routing_ws,
                           get_workspace_size_routing(M, E, topk),
                           "routing_ws");

    uint16_t* output_to_zero = nullptr;
    int64_t output_numel = 0;
    if (!output_to_zero_obj.is_none()) {
        torch::Tensor output = output_to_zero_obj.cast<torch::Tensor>();
        require_cuda_contiguous(output, "output_to_zero");
        require_dtype(output, at::kBFloat16, "output_to_zero");
        output_to_zero = reinterpret_cast<uint16_t*>(output.data_ptr());
        output_numel = output.numel();
    }

    int* completion_counters_to_zero = nullptr;
    int64_t completion_counter_numel = 0;
    if (!completion_counters_to_zero_obj.is_none()) {
        torch::Tensor counters =
            completion_counters_to_zero_obj.cast<torch::Tensor>();
        require_cuda_contiguous(counters, "completion_counters_to_zero");
        require_dtype(counters, at::kInt, "completion_counters_to_zero");
        completion_counters_to_zero = counters.data_ptr<int>();
        completion_counter_numel = counters.numel();
    }

    bool const fuse_alpha = !gemm1_alpha_obj.is_none();
    if (fuse_alpha != !a1_global_scale_obj.is_none() ||
        fuse_alpha != !w1_global_scale_obj.is_none() ||
        fuse_alpha != !a2_global_scale_obj.is_none() ||
        fuse_alpha != !w2_global_scale_obj.is_none() ||
        fuse_alpha != !gemm2_alpha_obj.is_none()) {
        throw std::invalid_argument(
            "routing fused alpha requires all six scale/alpha tensors");
    }
    if (fuse_alpha && (M < 1 || E != 512 || topk != 10)) {
        throw std::invalid_argument(
            "routing fused alpha currently requires M>=1, E=512, topk=10");
    }

    torch::Tensor a1_global_scale;
    torch::Tensor w1_global_scale;
    torch::Tensor gemm1_alpha;
    torch::Tensor a2_global_scale;
    torch::Tensor w2_global_scale;
    torch::Tensor gemm2_alpha;
    if (fuse_alpha) {
        a1_global_scale = a1_global_scale_obj.cast<torch::Tensor>();
        w1_global_scale = w1_global_scale_obj.cast<torch::Tensor>();
        gemm1_alpha = gemm1_alpha_obj.cast<torch::Tensor>();
        a2_global_scale = a2_global_scale_obj.cast<torch::Tensor>();
        w2_global_scale = w2_global_scale_obj.cast<torch::Tensor>();
        gemm2_alpha = gemm2_alpha_obj.cast<torch::Tensor>();
        auto validate_scale = [E](torch::Tensor const& tensor,
                                  char const* name) {
            require_cuda_contiguous(tensor, name);
            require_dtype(tensor, at::kFloat, name);
            require_numel_at_least(tensor, E, name);
        };
        validate_scale(a1_global_scale, "a1_global_scale");
        validate_scale(w1_global_scale, "w1_global_scale");
        validate_scale(gemm1_alpha, "gemm1_alpha");
        validate_scale(a2_global_scale, "a2_global_scale");
        validate_scale(w2_global_scale, "w2_global_scale");
        validate_scale(gemm2_alpha, "gemm2_alpha");
    }

    uint8_t* ws = routing_ws.data_ptr<uint8_t>();
    int64_t off = 0;
    int64_t num_blocks_per_seq = get_num_blocks_per_seq(M, E);
    int* blocked_expert_counts = reinterpret_cast<int*>(ws + off);
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    int* blocked_expert_counts_cumsum = reinterpret_cast<int*>(ws + off);
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    int* blocked_row_to_unpermuted_row = reinterpret_cast<int*>(ws + off);

    routing_sort(
        topk_ids.data_ptr<int>(),
        blocked_expert_counts,
        blocked_expert_counts_cumsum,
        blocked_row_to_unpermuted_row,
        expert_offset.data_ptr<int64_t>(),
        permuted_token_selected_experts.data_ptr<int>(),
        permuted_row.data_ptr<int>(),
        unperm_map.data_ptr<int>(),
        M, E, topk, 0, output_to_zero, output_numel,
        completion_counters_to_zero, completion_counter_numel,
        fuse_alpha ? a1_global_scale.data_ptr<float>() : nullptr,
        fuse_alpha ? w1_global_scale.data_ptr<float>() : nullptr,
        fuse_alpha ? gemm1_alpha.data_ptr<float>() : nullptr,
        fuse_alpha ? a2_global_scale.data_ptr<float>() : nullptr,
        fuse_alpha ? w2_global_scale.data_ptr<float>() : nullptr,
        fuse_alpha ? gemm2_alpha.data_ptr<float>() : nullptr,
        current_stream());
}

void routing_sort_with_scales_py(
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& routing_ws,
    const torch::Tensor& expert_offset,
    const torch::Tensor& permuted_token_selected_experts,
    const torch::Tensor& permuted_row,
    const torch::Tensor& unperm_map,
    const torch::Tensor& perm_scales,
    int M,
    int E,
    int topk)
{
    require_cuda_contiguous(topk_ids, "topk_ids");
    require_cuda_contiguous(topk_weights, "topk_weights");
    require_cuda_contiguous(routing_ws, "routing_ws");
    require_cuda_contiguous(expert_offset, "expert_offset");
    require_cuda_contiguous(permuted_token_selected_experts,
                            "permuted_token_selected_experts");
    require_cuda_contiguous(permuted_row, "permuted_row");
    require_cuda_contiguous(unperm_map, "unperm_map");
    require_cuda_contiguous(perm_scales, "perm_scales");
    require_dtype(topk_ids, at::kInt, "topk_ids");
    require_dtype(topk_weights, at::kFloat, "topk_weights");
    require_dtype(routing_ws, at::kByte, "routing_ws");
    require_dtype(expert_offset, at::kLong, "expert_offset");
    require_dtype(permuted_token_selected_experts, at::kInt,
                  "permuted_token_selected_experts");
    require_dtype(permuted_row, at::kInt, "permuted_row");
    require_dtype(unperm_map, at::kInt, "unperm_map");
    require_dtype(perm_scales, at::kFloat, "perm_scales");

    int64_t expanded = (int64_t)M * topk;
    require_numel_at_least(topk_ids, expanded, "topk_ids");
    require_numel_at_least(topk_weights, expanded, "topk_weights");
    require_numel_at_least(expert_offset, E + 1, "expert_offset");
    require_numel_at_least(permuted_token_selected_experts, expanded,
                           "permuted_token_selected_experts");
    require_numel_at_least(permuted_row, expanded, "permuted_row");
    require_numel_at_least(unperm_map, expanded, "unperm_map");
    require_numel_at_least(perm_scales, expanded, "perm_scales");
    require_numel_at_least(routing_ws,
                           get_workspace_size_routing(M, E, topk),
                           "routing_ws");

    uint8_t* ws = routing_ws.data_ptr<uint8_t>();
    int64_t off = 0;
    int64_t num_blocks_per_seq = get_num_blocks_per_seq(M, E);
    int* blocked_expert_counts = reinterpret_cast<int*>(ws + off);
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    int* blocked_expert_counts_cumsum = reinterpret_cast<int*>(ws + off);
    off += align256((int64_t)E * num_blocks_per_seq * sizeof(int));
    int* blocked_row_to_unpermuted_row = reinterpret_cast<int*>(ws + off);

    routing_sort_with_scales(
        topk_ids.data_ptr<int>(),
        topk_weights.data_ptr<float>(),
        blocked_expert_counts,
        blocked_expert_counts_cumsum,
        blocked_row_to_unpermuted_row,
        expert_offset.data_ptr<int64_t>(),
        permuted_token_selected_experts.data_ptr<int>(),
        permuted_row.data_ptr<int>(),
        unperm_map.data_ptr<int>(),
        perm_scales.data_ptr<float>(),
        M, E, topk, 0, current_stream());
}

void register_routing(py::module_& m)
{
    m.def("get_workspace_size_routing", &get_workspace_size_routing,
          py::arg("M"), py::arg("E"), py::arg("topk"));
    m.def("routing_sort", &routing_sort_py,
          py::arg("topk_ids"), py::arg("routing_ws"),
          py::arg("expert_offset"),
          py::arg("permuted_token_selected_experts"),
          py::arg("permuted_row"), py::arg("unperm_map"),
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("output_to_zero") = py::none(),
          py::arg("a1_global_scale") = py::none(),
          py::arg("w1_global_scale") = py::none(),
          py::arg("gemm1_alpha") = py::none(),
          py::arg("a2_global_scale") = py::none(),
          py::arg("w2_global_scale") = py::none(),
          py::arg("gemm2_alpha") = py::none(),
          py::arg("completion_counters_to_zero") = py::none());
    m.def("routing_sort_with_scales", &routing_sort_with_scales_py,
          py::arg("topk_ids"), py::arg("topk_weights"),
          py::arg("routing_ws"), py::arg("expert_offset"),
          py::arg("permuted_token_selected_experts"),
          py::arg("permuted_row"), py::arg("unperm_map"),
          py::arg("perm_scales"),
          py::arg("M"), py::arg("E"), py::arg("topk"));
}
