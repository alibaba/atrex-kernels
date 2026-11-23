#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <climits>

// Metadata only: the original prep and attention kernels consume each group.
// At least minimum requests per non-final group bounds the number of graph slots.
__global__ void atrex_ppu_prefill_group_plan(
    const int* cu, const int* lengths, int64_t* cost, int* groups, int* next,
    int* starts, int batch, int capacity, int minimum, int slots,
    int query_multiplier, int query_tile, int head_groups, int splits, int resident) {
  if (threadIdx.x != 0) return;
  // Minimize estimated CTA waves across contiguous groups, with fewer groups
  // breaking ties. Rebuild the plan from current device metadata on replay.
  cost[batch] = 0;
  groups[batch] = 0;
  next[batch] = batch;
  for (int begin = batch - 1; begin >= 0; --begin) {
    int64_t best = INT64_MAX, blocks = 0;
    int best_groups = INT_MAX, best_end = batch, pages = 0;
    for (int end = begin + 1; end <= min(batch, begin + capacity); ++end) {
      const int query = cu[end] - cu[end - 1];
      blocks += ((int64_t(query) * query_multiplier + query_tile - 1) / query_tile) * head_groups * splits;
      if (query > 0) pages = max(pages, (lengths[end - 1] + 63) / 64);
      if (end - begin < minimum && end != batch) continue;
      const int64_t waves = (blocks + resident - 1) / resident;
      const int64_t value = waves * ((pages + splits - 1) / splits) + cost[end];
      const int count = 1 + groups[end];
      if (value < best || (value == best && count <= best_groups)) {
        best = value;
        best_groups = count;
        best_end = end;
      }
    }
    cost[begin] = best;
    groups[begin] = best_groups;
    next[begin] = best_end;
  }
  int request = 0;
  for (int slot = 0; slot <= slots; ++slot) {
    starts[slot] = request;
    request = next[request];
  }
}

__global__ void atrex_ppu_prefill_group_metadata(
    const int* table, const int* cu, const int* lengths, const float* scales,
    const int* starts, int* group_table, int* group_cu, int* group_lengths,
    float* group_scales, int slot, int capacity, int batch, int heads,
    int pages, int table_stride, int virtual_stride, int subpages) {
  const int begin = starts[slot], end = starts[slot + 1];
  const int count = end - begin;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // Keep global Q offsets so attention writes directly into the caller's output.
  // Padding with repeated offsets and zero KV lengths makes unused slots empty.
  for (int i = tid; i <= capacity; i += stride)
    group_cu[i] = cu[begin + min(i, count)];
  for (int i = tid; i < capacity; i += stride)
    group_lengths[i] = i < count && cu[begin + i + 1] > cu[begin + i] ? lengths[begin + i] : 0;
  for (int64_t i = tid; i < int64_t(capacity) * pages; i += stride) {
    const int request = i / pages, page = i % pages;
    const int original = begin + request;
    int mapped = 0;
    if (request < count && page * 64 < lengths[original] && cu[original + 1] > cu[original])
      mapped = table[int64_t(original) * table_stride + page / subpages] * virtual_stride + page % subpages;
    group_table[i] = mapped;
  }
  if (scales) {
    for (int i = tid; i < 3 * capacity * heads; i += stride) {
      const int scale = i / (capacity * heads), request = (i / heads) % capacity, head = i % heads;
      group_scales[i] = request < count ? scales[(scale * batch + begin + request) * heads + head] : 1.f;
    }
  }
}

void plan(torch::Tensor cu, torch::Tensor lengths, torch::Tensor cost,
          torch::Tensor groups, torch::Tensor next, torch::Tensor starts,
          int capacity, int minimum, int query_multiplier, int query_tile,
          int head_groups, int splits, int resident) {
  auto stream = at::cuda::getCurrentCUDAStream();
  atrex_ppu_prefill_group_plan<<<1, 32, 0, stream>>>(
      cu.data_ptr<int>(), lengths.data_ptr<int>(), cost.data_ptr<int64_t>(),
      groups.data_ptr<int>(), next.data_ptr<int>(), starts.data_ptr<int>(),
      cu.numel() - 1, capacity, minimum, starts.numel() - 1,
      query_multiplier, query_tile, head_groups, splits, resident);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare(torch::Tensor table, torch::Tensor cu, torch::Tensor lengths,
             torch::Tensor scales, torch::Tensor starts, torch::Tensor group_table,
             torch::Tensor group_cu, torch::Tensor group_lengths,
             torch::Tensor group_scales, int slot, int virtual_stride, int subpages) {
  const int capacity = group_lengths.numel(), heads = scales.numel() ? scales.size(2) : 1;
  const int blocks = std::max<int64_t>(1, std::min<int64_t>(256, (group_table.numel() + 255) / 256));
  auto stream = at::cuda::getCurrentCUDAStream();
  atrex_ppu_prefill_group_metadata<<<blocks, 256, 0, stream>>>(
      table.data_ptr<int>(), cu.data_ptr<int>(), lengths.data_ptr<int>(),
      scales.numel() ? scales.data_ptr<float>() : nullptr, starts.data_ptr<int>(),
      group_table.data_ptr<int>(), group_cu.data_ptr<int>(), group_lengths.data_ptr<int>(),
      group_scales.numel() ? group_scales.data_ptr<float>() : nullptr, slot, capacity,
      lengths.numel(), heads, group_table.size(1), table.stride(0), virtual_stride, subpages);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &plan);
  m.def("prepare", &prepare);
}
