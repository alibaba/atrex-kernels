#pragma once

#include <stdint.h>

namespace flash {

__device__ __forceinline__ void decode_aiu_two_planes(void* destination, void const* source,
                                                     int rows, int stride, int row_start = 0) {
    uint64_t row_stride = stride;
    uint64_t plane_stride = 128;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b8"
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(destination), "l"(source), "r"(128), "r"(rows), "r"(2),
           "r"(128), "r"(32), "r"(2), "l"(row_stride), "l"(plane_stride),
           "r"(0), "r"(row_start), "r"(0), "r"(0));
}

}
