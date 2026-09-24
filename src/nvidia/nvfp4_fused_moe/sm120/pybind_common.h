#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "moe_common.cuh"

int64_t align256(int64_t x);
int64_t get_num_blocks_per_seq(int M, int E);
int64_t get_fc1_act_sf_size(int M, int E, int topk, int hidden_size);
int64_t get_fc2_act_sf_size(int M, int E, int topk, int inter_size);

cudaStream_t current_stream();
void require_cuda_contiguous(const torch::Tensor& tensor, const char* name);
void require_dtype(const torch::Tensor& tensor, at::ScalarType dtype,
                   const char* name);
void require_numel_at_least(const torch::Tensor& tensor, int64_t numel,
                            const char* name);
const void* raw_data_ptr(const torch::Tensor& tensor);
void* raw_data_ptr_mut(const torch::Tensor& tensor);
const uint8_t* optional_input_sf_ptr(const pybind11::object& input_sf_obj);
void wait_for_shared_event(uint64_t shared_event, cudaStream_t stream);
void set_output_preinitialized(bool output_preinitialized);

void register_routing(pybind11::module_& m);
void register_expand_input_rows(pybind11::module_& m);
void register_up_gate(pybind11::module_& m);
void register_down(pybind11::module_& m);
void register_e512_topk10(pybind11::module_& m);
