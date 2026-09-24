#include "pybind_common.h"

#include <ATen/cuda/CUDAContext.h>

bool g_enable_pdl = true;

static thread_local int g_output_preinitialized = 0;

extern "C" int fused_moe_get_output_preinitialized()
{
    return g_output_preinitialized;
}

void set_output_preinitialized(bool output_preinitialized)
{
    g_output_preinitialized = output_preinitialized ? 1 : 0;
}

int64_t align256(int64_t x)
{
    return (x + 255) & ~255LL;
}

static int64_t compute_num_tokens_per_block(int64_t num_tokens,
                                            int64_t num_experts_per_node)
{
    for (int64_t n = 32; n <= 1024; n *= 2) {
        int64_t num_blocks = ceilDiv(num_tokens, n);
        if (num_blocks * num_experts_per_node <= n) return n;
    }
    return 1024;
}

int64_t get_num_blocks_per_seq(int M, int E)
{
    int64_t num_tokens_per_block = compute_num_tokens_per_block(M, E);
    return ceilDiv((int64_t)M, num_tokens_per_block);
}

int64_t get_fc1_act_sf_size(int M, int E, int topk, int hidden_size)
{
    int64_t expanded = (int64_t)M * topk;
    int64_t padded_hidden = TmaConst::alignToSfDim(
        hidden_size, (int)TmaConst::MinKDimAlignmentNVFP4);
    int64_t min_n = TmaConst::MinNDimAlignmentNVFP4;
    int64_t padded_expanded_sf = TmaConst::alignToSfDim(
        (int)(expanded + (int64_t)E * (min_n - 1)), (int)min_n);
    return align256(padded_expanded_sf * padded_hidden / 16);
}

int64_t get_fc2_act_sf_size(int M, int E, int topk, int inter_size)
{
    int64_t expanded = (int64_t)M * topk;
    int64_t padded_inter = TmaConst::alignToSfDim(
        inter_size, (int)TmaConst::MinKDimAlignmentNVFP4);
    int64_t min_n = TmaConst::MinNDimAlignmentNVFP4;
    int64_t padded_expanded_sf = TmaConst::alignToSfDim(
        (int)(expanded + (int64_t)E * (min_n - 1)), (int)min_n);
    return align256(padded_expanded_sf * padded_inter / 16);
}

cudaStream_t current_stream()
{
    return at::cuda::getCurrentCUDAStream().stream();
}

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_dtype(const torch::Tensor& tensor,
                   at::ScalarType dtype,
                   const char* name)
{
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has dtype ",
                tensor.scalar_type(), ", expected ", dtype);
}

void require_numel_at_least(const torch::Tensor& tensor,
                            int64_t numel,
                            const char* name)
{
    TORCH_CHECK(tensor.numel() >= numel, name, " has ", tensor.numel(),
                " elements, expected at least ", numel);
}

const void* raw_data_ptr(const torch::Tensor& tensor)
{
    return tensor.data_ptr();
}

void* raw_data_ptr_mut(const torch::Tensor& tensor)
{
    return const_cast<void*>(tensor.data_ptr());
}

const uint8_t* optional_input_sf_ptr(const pybind11::object& input_sf_obj)
{
    if (input_sf_obj.is_none()) {
        return nullptr;
    }
    torch::Tensor input_sf = input_sf_obj.cast<torch::Tensor>();
    require_cuda_contiguous(input_sf, "input_sf");
    return reinterpret_cast<const uint8_t*>(input_sf.data_ptr());
}

void wait_for_shared_event(uint64_t shared_event, cudaStream_t stream)
{
    if (shared_event != 0) {
        cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(shared_event)),
            0);
    }
}
