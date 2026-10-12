#pragma once

#include "cute/container/array.hpp"
#include "cute/container/tuple.hpp"
#include "cute/util/type_traits.hpp"

constexpr cute::array<int, 7> b8_tile_mappings[] = {
#include "cutlass/gemm/collective/builders/int8_tile_list.inl"
};
constexpr cute::array<int, 7> b16_tile_mappings_ppu0010[] = {
#include "cutlass/gemm/collective/builders/fp16_tile_list_ppu0010.inl"
};
constexpr cute::array<int, 7> b16_tile_mappings_ppu0015[] = {
#include "cutlass/gemm/collective/builders/fp16_tile_list_ppu0015.inl"
};
constexpr cute::array<int, 7> b32_tile_mappings_ppu0010[] = {
#include "cutlass/gemm/collective/builders/fp32_tile_list_ppu0010.inl"
};
constexpr cute::array<int, 7> b32_tile_mappings_ppu0015[] = {
#include "cutlass/gemm/collective/builders/fp32_tile_list_ppu0015.inl"
};

template <typename Arch, typename ComputeType, int Block_M, int Block_N, int Block_K>
constexpr cute::tuple<int, int, int, int> MapBlockShapeToWarpShapeStage() {
  auto find_tile_mapping = [&](auto& mappings) {
    for (const auto& row : mappings) {
      if (row[0] == Block_M && row[1] == Block_N && row[2] == Block_K) {
          return cute::tuple<int, int, int, int>{row[3], row[4], row[5], row[6]};
      }
    }
    return cute::tuple<int, int, int, int>{Block_M/2, Block_N/2, Block_K/2, 2};
  };
  constexpr int bit_num = sizeof(ComputeType) * 8;
  if (bit_num == 8) {
      return find_tile_mapping(b8_tile_mappings);
  } else if (bit_num == 16) {
      return (cute::is_same_v<Arch, cutlass::arch::PPU0015>) ?
        find_tile_mapping(b16_tile_mappings_ppu0015) : find_tile_mapping(b16_tile_mappings_ppu0010);
  } else if (bit_num == 32) {
      return (cute::is_same_v<Arch, cutlass::arch::PPU0015>) ?
        find_tile_mapping(b32_tile_mappings_ppu0015) : find_tile_mapping(b32_tile_mappings_ppu0010);
  } else {
    printf("illegal ComputeType for mapping !!!\n");
    return cute::tuple<int, int, int, int>{0, 0, 0, 0};
  }
};
