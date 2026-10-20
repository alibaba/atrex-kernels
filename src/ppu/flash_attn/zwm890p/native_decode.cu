#include <hggc_runtime.h>
#include <stdint.h>
#include <math.h>

constexpr int NativeKeyRows = 32;
constexpr int NativeKeyHalves = NativeKeyRows / 16;
constexpr int NativeTileBytes = NativeKeyRows * 256;

template<int Query, int Group>
struct NativeGeometry {
    static constexpr int Rows = Query == 1 ? 16 : ((Query * Group + 15) / 16) * 16;
    static constexpr int Threads = Query == 1 ? 64 : Rows * 2;
    static constexpr int KeyOffset = Rows * 256 > NativeTileBytes * 2 ? Rows * 256 : NativeTileBytes * 2;
    static constexpr int SharedBytes = KeyOffset + NativeTileBytes;
};

template<bool Value>
struct NativeFlag {
    static constexpr bool value = Value;
};

struct DecodeParams {
    const unsigned char* __restrict__ queries;
    const unsigned char* __restrict__ keys;
    const unsigned char* __restrict__ values;
    unsigned short* __restrict__ output;
    const int* __restrict__ offsets;
    const int* __restrict__ lengths;
    const int* __restrict__ page_table;
    const float* __restrict__ query_descales;
    const float* __restrict__ key_descales;
    const float* __restrict__ value_descales;
    float* __restrict__ partial_output;
    float* __restrict__ partial_lse;
    int* __restrict__ plan;
    int batch;
    int total_q;
    int max_query;
    int max_kv;
    int cache_pages;
    int page_size;
    int64_t page_table_stride;
    int64_t query_row_stride;
    int64_t query_head_stride;
    int64_t key_page_stride;
    int64_t key_row_stride;
    int64_t key_head_stride;
    int64_t value_page_stride;
    int64_t value_row_stride;
    int64_t value_head_stride;
    int64_t output_row_stride;
    int64_t output_head_stride;
    int64_t query_descale_batch_stride;
    int64_t query_descale_head_stride;
    int64_t key_descale_batch_stride;
    int64_t key_descale_head_stride;
    int64_t value_descale_batch_stride;
    int64_t value_descale_head_stride;
    int num_cu;
    int query_heads;
    int kv_heads;
    float scale;
    int splits;
    bool automatic;
    int average_splits;
    int page_shift;
};

int native_average_splits(const DecodeParams& params) {
    int groups = params.batch * params.kv_heads;
    int occupancy = params.max_query == 1 ? 16 : (params.query_heads / params.kv_heads == 8 ? 8 : 4);
    int capacity = params.num_cu * occupancy;
    float waves = static_cast<float>(groups) / capacity;
    if ((occupancy > 8 && groups >= .8f * capacity)
        || (occupancy <= 8 && waves / ceilf(waves) > .9f)) return 1;
    int tiles = (params.max_kv + NativeKeyRows - 1) / NativeKeyRows;
    if (tiles <= 4) return 1;
    int limit = min(32, min(capacity, tiles));
    float maximum = 0.f;
    for (int split = 1; split <= limit; ++split) {
        float count = static_cast<float>(groups * split) / capacity;
        maximum = fmaxf(maximum, count / ceilf(count));
    }
    for (int split = 1; split <= limit; ++split) {
        float count = static_cast<float>(groups * split) / capacity;
        if (count / ceilf(count) >= .9f * maximum) return split;
    }
    return 1;
}

__device__ __forceinline__ int native_warp() {
    return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x >> 5), 0);
}

__device__ __forceinline__ unsigned int native_bfloat_pair(float first, float second) {
    unsigned int packed;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
        : "=r"(packed) : "f"(second), "f"(first));
    return packed;
}

__device__ __forceinline__ int native_transpose_offset(int dimension, int slot) {
    return (dimension * NativeKeyRows + slot) ^ ((dimension & 28) * 4);
}

__device__ __forceinline__ int native_query_offset(int row, int dimension) {
    return (row * 256 + dimension) ^ ((row & 7) * 16);
}

__device__ __forceinline__ void native_copy(void* destination, const void* source, bool valid) {
    unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
    int bytes = valid ? 16 : 0;
    asm volatile("ppu.cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(source), "r"(bytes));
}

__device__ __forceinline__ void native_mma(float (&accumulator)[8],
                                        const unsigned int (&first)[4],
                                        const unsigned int (&second)[4]) {
    asm volatile(
        "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};"
        : "=r"(accumulator[0]), "=r"(accumulator[1]), "=r"(accumulator[2]), "=r"(accumulator[3]),
          "=r"(accumulator[4]), "=r"(accumulator[5]), "=r"(accumulator[6]), "=r"(accumulator[7])
        : "r"(first[0]), "r"(first[1]), "r"(first[2]), "r"(first[3]),
          "r"(second[0]), "r"(second[1]), "r"(second[2]), "r"(second[3]),
          "r"(accumulator[0]), "r"(accumulator[1]), "r"(accumulator[2]), "r"(accumulator[3]),
          "r"(accumulator[4]), "r"(accumulator[5]), "r"(accumulator[6]), "r"(accumulator[7]));
}

__device__ __forceinline__ unsigned int native_pack(float first, float second,
                                                  float third, float fourth) {
    unsigned int packed;
    asm volatile(
        "{\n.reg .b16 low;\n.reg .b16 high;\n"
        "ppu.cvt.rtte.satfinite.e4m3x2.f32 low, %2, %1;\n"
        "ppu.cvt.rtte.satfinite.e4m3x2.f32 high, %4, %3;\n"
        "ppu.mov.b32 %0, {low, high};\n}"
        : "=r"(packed) : "f"(first), "f"(second), "f"(third), "f"(fourth));
    return packed;
}

template<int Rows>
__device__ __forceinline__ void native_pack_scores(const float (&scores)[NativeKeyHalves][8],
                                                  unsigned int (&probabilities)[NativeKeyRows / 32][4]) {
    #pragma unroll
    for (int part = 0; part < NativeKeyRows / 32; ++part) {
        probabilities[part][0] = native_pack(scores[part * 2][0], scores[part * 2][1], scores[part * 2][4], scores[part * 2][5]);
        probabilities[part][1] = Rows == 2 ? native_pack(scores[part * 2][2], scores[part * 2][3], scores[part * 2][6], scores[part * 2][7]) : 0;
        probabilities[part][2] = native_pack(scores[part * 2 + 1][0], scores[part * 2 + 1][1], scores[part * 2 + 1][4], scores[part * 2 + 1][5]);
        probabilities[part][3] = Rows == 2 ? native_pack(scores[part * 2 + 1][2], scores[part * 2 + 1][3], scores[part * 2 + 1][6], scores[part * 2 + 1][7]) : 0;
    }
}

__device__ __forceinline__ void native_load_key(const unsigned char* keys, int channel_tile,
                                              int key_half, unsigned int (&fragment)[4]) {
    unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(keys));
    int cube = (channel_tile >> 2) * NativeKeyHalves + key_half;
    unsigned long long address = static_cast<unsigned long long>(shared >> 4) + cube * 128 + (channel_tile & 3) * 2;
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0,%1,%2,%3}, [%4], %5, %6, %7;"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "l"(address), "r"(64), "r"(1), "r"(0));
}

template<int Threads, bool Keys, bool Values, bool FullTile = false>
__device__ __forceinline__ const unsigned char* native_stage(const DecodeParams& params, const int* page_table,
                                            const unsigned char* source, int tile, int length,
                                            unsigned char* keys, unsigned char* values,
                                            const unsigned char* value_source = nullptr) {
    static_assert(Keys != Values);
    const unsigned char* next_value = nullptr;
    if (native_warp() == 0) {
        int base = tile * NativeKeyRows;
        const unsigned char* copy_source = source;
        bool page_valid = source != nullptr;
        if constexpr (Keys) {
            int page = __ldg(page_table + (base >> params.page_shift));
            page_valid = static_cast<unsigned int>(page) < static_cast<unsigned int>(params.cache_pages);
            if (page_valid) {
                int row_start = base & (params.page_size - 1);
                copy_source += page * params.key_page_stride + row_start * params.key_row_stride;
                next_value = value_source + page * params.value_page_stride + row_start * params.value_row_stride;
            }
        }
        unsigned char* target = Keys ? keys : values;
        int64_t row_stride = Keys ? params.key_row_stride : params.value_row_stride;
        int valid_rows = FullTile ? NativeKeyRows : min(NativeKeyRows, max(0, length - base));
        #pragma unroll
        for (int slab = 0; slab < 2; ++slab) {
            if (page_valid && valid_rows > 0) {
                unsigned int destination = static_cast<unsigned int>(__cvta_generic_to_shared(target + slab * NativeKeyRows * 128));
                unsigned long long stride = row_stride;
                unsigned long long tensor_stride = stride * valid_rows;
                asm volatile(
                    "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b8 "
                    "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;"
                    :: "r"(destination), "l"(copy_source), "r"(256), "r"(valid_rows), "r"(1),
                       "r"(128), "r"(NativeKeyRows), "r"(1), "l"(stride), "l"(tensor_stride),
                       "r"(slab * 128), "r"(0), "r"(0), "r"(0));
            } else {
                #pragma unroll
                for (int chunk = 0; chunk < NativeKeyRows / 4; ++chunk) {
                    *reinterpret_cast<uint4*>(target + slab * NativeKeyRows * 128 + (threadIdx.x & 31) * 16 + chunk * 512)
                        = make_uint4(0, 0, 0, 0);
                }
            }
        }
    }
    asm volatile("ppu.cp.async.commit_group;");
    return next_value;
}

__device__ __forceinline__ uint4 native_transpose_quad(uint4 values) {
    unsigned int low_xy = __byte_perm(values.x, values.y, 0x5410);
    unsigned int high_xy = __byte_perm(values.x, values.y, 0x7632);
    unsigned int low_zw = __byte_perm(values.z, values.w, 0x5410);
    unsigned int high_zw = __byte_perm(values.z, values.w, 0x7632);
    return make_uint4(__byte_perm(low_xy, low_zw, 0x6420), __byte_perm(low_xy, low_zw, 0x7531),
        __byte_perm(high_xy, high_zw, 0x6420), __byte_perm(high_xy, high_zw, 0x7531));
}

template<int Row>
__device__ __forceinline__ unsigned int native_transpose_source(const unsigned int* source, int column) {
    return source[Row * 32 + (column ^ ((Row & 7) * 4))];
}

template<int Threads>
__device__ __forceinline__ void native_transpose(const unsigned char* __restrict__ source, unsigned char* __restrict__ destination) {
    constexpr int CopyThreads = Threads >= 128 ? 128 : 64;
    if (threadIdx.x >= CopyThreads) return;
    #pragma unroll
    for (int copy = 0; copy < NativeKeyRows * 4 / CopyThreads; ++copy) {
        int index = threadIdx.x + copy * CopyThreads;
        int slot = (index >> 6) * 16;
        int dimension = (index & 63) * 4;
        const unsigned int* words = reinterpret_cast<const unsigned int*>(source + (dimension >> 7) * NativeKeyRows * 128
            + slot * 128);
        int column = (dimension & 127) / 4;
        uint4 first = native_transpose_quad(make_uint4(native_transpose_source<0>(words, column), native_transpose_source<1>(words, column),
            native_transpose_source<8>(words, column), native_transpose_source<9>(words, column)));
        uint4 second = native_transpose_quad(make_uint4(native_transpose_source<2>(words, column), native_transpose_source<3>(words, column),
            native_transpose_source<10>(words, column), native_transpose_source<11>(words, column)));
        uint4 third = native_transpose_quad(make_uint4(native_transpose_source<4>(words, column), native_transpose_source<5>(words, column),
            native_transpose_source<12>(words, column), native_transpose_source<13>(words, column)));
        uint4 fourth = native_transpose_quad(make_uint4(native_transpose_source<6>(words, column), native_transpose_source<7>(words, column),
            native_transpose_source<14>(words, column), native_transpose_source<15>(words, column)));
        *reinterpret_cast<uint4*>(destination + native_transpose_offset(dimension, slot))
            = make_uint4(first.x, second.x, third.x, fourth.x);
        *reinterpret_cast<uint4*>(destination + native_transpose_offset(dimension + 1, slot))
            = make_uint4(first.y, second.y, third.y, fourth.y);
        *reinterpret_cast<uint4*>(destination + native_transpose_offset(dimension + 2, slot))
            = make_uint4(first.z, second.z, third.z, fourth.z);
        *reinterpret_cast<uint4*>(destination + native_transpose_offset(dimension + 3, slot))
            = make_uint4(first.w, second.w, third.w, fourth.w);
    }
}

template<int Row, int Tile, int Count, int Rows>
__device__ __forceinline__ void native_rescale_row(float (&output)[Count][8], const float (&alpha)[2]) {
    output[Tile][Row * 2] *= alpha[Row];
    output[Tile][Row * 2 + 1] *= alpha[Row];
    output[Tile][Row * 2 + 4] *= alpha[Row];
    output[Tile][Row * 2 + 5] *= alpha[Row];
    if constexpr (Tile + 1 < Count) native_rescale_row<Row, Tile + 1, Count, Rows>(output, alpha);
    else if constexpr (Row + 1 < Rows) native_rescale_row<Row + 1, 0, Count, Rows>(output, alpha);
}

template<int Tile, int Stride = 16>
__device__ __forceinline__ void native_load_value(const unsigned char* transposed, int output_base,
                                                 int lane, unsigned int (&values)[4]) {
    int dimension = output_base + Tile * Stride + (lane & 7) + ((lane >> 4) & 1) * 8;
    int slot = ((lane >> 3) & 1) * 16;
    unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(transposed + native_transpose_offset(dimension, slot)));
    asm volatile("ppu.tc02.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]) : "r"(address));
}

template<int Tile, int Count>
__device__ __forceinline__ void native_pv_scoped(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                                const unsigned char* transposed, int output_base, int lane) {
    __ppu_sched_bound();
    unsigned int first[4], second[4];
    native_load_value<Tile, 32>(transposed, output_base, lane, first);
    native_load_value<Tile + 1, 32>(transposed, output_base, lane, second);
    #pragma hggc mmatiestrictly
    {
        native_mma(output[Tile], probabilities, first);
        native_mma(output[Tile + 1], probabilities, second);
    }
    __ppu_sched_bound();
    if constexpr (Tile + 2 < Count) native_pv_scoped<Tile + 2, Count>(output, probabilities, transposed, output_base, lane);
}

template<int Tile, int Window, int Offset = 0>
__device__ __forceinline__ void native_value_window(const unsigned char* transposed, int output_base,
                                                   int lane, unsigned int (&values)[Window][4]) {
    native_load_value<Tile + Offset>(transposed, output_base, lane, values[Offset]);
    if constexpr (Offset + 1 < Window) native_value_window<Tile, Window, Offset + 1>(transposed, output_base, lane, values);
}

template<int Tile, int Window, int Count, int Offset = 0>
__device__ __forceinline__ void native_mma_window(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                                 const unsigned int (&values)[Window][4]) {
    native_mma(output[Tile + Offset], probabilities, values[Offset]);
    if constexpr (Offset + 1 < Window) native_mma_window<Tile, Window, Count, Offset + 1>(output, probabilities, values);
}

template<int Tile, int Count, int Window>
__device__ __forceinline__ void native_pv_window(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                                const unsigned char* transposed, int output_base, int lane,
                                                unsigned int (&first)[Window][4], unsigned int (&second)[Window][4]) {
    if constexpr (Tile + Window < Count) native_value_window<Tile + Window, Window>(transposed, output_base, lane, second);
    __ppu_sched_bound();
    #pragma hggc mmatiestrictly
    {
        native_mma_window<Tile, Window, Count>(output, probabilities, first);
    }
    __ppu_sched_bound();
    if constexpr (Tile + Window < Count) native_pv_window<Tile + Window, Count, Window>(output, probabilities,
        transposed, output_base, lane, second, first);
}

template<int Count>
__device__ __forceinline__ void native_pv(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                        const unsigned char* transposed, int output_base, int lane) {
    if constexpr (Count == 8) {
        native_pv_scoped<0, Count>(output, probabilities, transposed, output_base, lane);
    } else {
        constexpr int Window = 4;
        unsigned int first[Window][4], second[Window][4];
        native_value_window<0, Window>(transposed, output_base, lane, first);
        native_pv_window<0, Count, Window>(output, probabilities, transposed, output_base, lane, first, second);
    }
}

template<int Channel, int KeyPair = 0>
__device__ __forceinline__ void native_qk_direct(float (&scores)[NativeKeyHalves][8], const unsigned int (&queries)[8][4],
                                                const unsigned char* keys) {
    __ppu_sched_bound();
    unsigned int fragment[2][4];
    native_load_key(keys, Channel, KeyPair * 2, fragment[0]);
    native_load_key(keys, Channel, KeyPair * 2 + 1, fragment[1]);
    #pragma hggc mmatiestrictly
    {
        native_mma(scores[KeyPair * 2], queries[Channel], fragment[0]);
        native_mma(scores[KeyPair * 2 + 1], queries[Channel], fragment[1]);
    }
    __ppu_sched_bound();
    if constexpr (KeyPair + 1 < NativeKeyHalves / 2) native_qk_direct<Channel, KeyPair + 1>(scores, queries, keys);
    else if constexpr (Channel + 1 < 8) native_qk_direct<Channel + 1>(scores, queries, keys);
}

template<int Channel>
__device__ __forceinline__ void native_qk_ring(float (&scores)[NativeKeyHalves][8], const unsigned int (&queries)[8][4],
                                              const unsigned char* keys, unsigned int (&ring)[3][2][4]) {
    static_assert(NativeKeyHalves == 2);
    __ppu_sched_bound();
    #pragma hggc mmatiestrictly
    {
        native_mma(scores[0], queries[Channel], ring[Channel % 3][0]);
        native_mma(scores[1], queries[Channel], ring[Channel % 3][1]);
    }
    __ppu_sched_bound();
    if constexpr (Channel + 3 < 8) {
        native_load_key(keys, Channel + 3, 0, ring[Channel % 3][0]);
        native_load_key(keys, Channel + 3, 1, ring[Channel % 3][1]);
    }
    if constexpr (Channel + 1 < 8) native_qk_ring<Channel + 1>(scores, queries, keys, ring);
}

template<int Tile, int Count, int Group>
__device__ __forceinline__ void native_store(const DecodeParams& params, const float (&output)[Count][8],
                                           const float (&multipliers)[2], const float (&lse)[2],
                                           int query_begin, int query_count, int head, int split,
                                           int row_base, int output_base, int lane) {
    unsigned short* destination = static_cast<unsigned short*>(params.output);
    float* partial = static_cast<float*>(params.partial_output);
    float* partial_lse = static_cast<float*>(params.partial_lse);
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        int row = pair & 1;
        int packed_row = row_base + (lane >> 2) + row * 8;
        int query_index = packed_row / Group;
        int query_head = head * Group + packed_row % Group;
        int dimension = output_base + Tile * (Count == 8 ? 32 : 16) + (lane & 3) * 2 + (pair >> 1) * 8;
        if (query_index < query_count) {
            int token = query_begin + query_index;
            float first = output[Tile][pair * 2] * multipliers[row];
            float second = output[Tile][pair * 2 + 1] * multipliers[row];
            if (params.splits == 1) {
                *reinterpret_cast<unsigned int*>(destination + token * params.output_row_stride
                    + query_head * params.output_head_stride + dimension)
                    = native_bfloat_pair(first, second);
            } else {
                int64_t offset = ((static_cast<int64_t>(split) * params.query_heads + query_head) * params.total_q + token) * 256 + dimension;
                *reinterpret_cast<float2*>(partial + offset) = make_float2(first, second);
                if (dimension == 0) {
                    int64_t lse_offset = (static_cast<int64_t>(split) * params.query_heads + query_head) * params.total_q + token;
                    partial_lse[lse_offset] = lse[row];
                }
            }
        }
    }
    if constexpr (Tile + 1 < Count) native_store<Tile + 1, Count, Group>(params, output, multipliers, lse,
        query_begin, query_count, head, split, row_base, output_base, lane);
}

template<int Query, int Group>
__device__ __forceinline__ void native_decode_task(const DecodeParams& params, int split, int head, int request,
                                                  int active_splits, unsigned char* shared_keys,
                                                  unsigned char* shared_values, unsigned char* shared_transposed) {
    constexpr int Threads = NativeGeometry<Query, Group>::Threads;
    constexpr int OutputTiles = Query == 1 ? 8 : 16;
    constexpr int ActiveRows = Query == 1 && Group == 8 ? 1 : 2;
    int lane = threadIdx.x & 31;
    int warp = native_warp();
    const int* offsets = static_cast<const int*>(params.offsets);
    const int* lengths = static_cast<const int*>(params.lengths);
    int query_begin = __ldg(offsets + request);
    int query_count = __ldg(offsets + request + 1) - query_begin;
    int length = __ldg(lengths + request);
    const int* page_table = params.page_table + request * params.page_table_stride;
    const unsigned char* key_source = params.keys + head * params.key_head_stride;
    const unsigned char* value_source = params.values + head * params.value_head_stride;
    int row_base = Query == 1 ? 0 : warp * 16;
    int output_base = Query == 1 ? warp * 16 : 0;
    int tile_count = (length + NativeKeyRows - 1) / NativeKeyRows;
    int begin = tile_count * split / active_splits;
    int end = tile_count * (split + 1) / active_splits;
    unsigned int query_fragments[8][4];
    const unsigned char* queries = static_cast<const unsigned char*>(params.queries);
    __syncthreads();
    constexpr int QueryRows = NativeGeometry<Query, Group>::Rows;
    #pragma unroll
    for (int copy = 0; copy < QueryRows * 16 / Threads; ++copy) {
        int vector = threadIdx.x + copy * Threads;
        int row = vector >> 4;
        int dimension = (vector & 15) * 16;
        int query_index = row / Group;
        int query_head = head * Group + row % Group;
        int64_t offset = (query_begin + query_index) * params.query_row_stride
            + query_head * params.query_head_stride + dimension;
        native_copy(shared_keys + native_query_offset(row, dimension),
            queries + (query_index < query_count ? offset : 0), query_index < query_count);
    }
    asm volatile("ppu.cp.async.commit_group;");
    const unsigned char* current_value = nullptr;
    if (begin < end) current_value = native_stage<Threads, true, false>(params, page_table, key_source, end - 1, length, shared_transposed, shared_values, value_source);
    else asm volatile("ppu.cp.async.commit_group;");
    asm volatile("ppu.cp.async.wait_group 1;");
    __syncthreads();
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        int row = row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
        int dimension = tile * 32 + (lane >> 4) * 16;
        unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(shared_keys + native_query_offset(row, dimension)));
        asm volatile("ppu.tc02.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(query_fragments[tile][0]), "=r"(query_fragments[tile][1]),
              "=r"(query_fragments[tile][2]), "=r"(query_fragments[tile][3]) : "r"(address));
    }
    __syncthreads();
    const float* query_descales = static_cast<const float*>(params.query_descales);
    const float* key_descales = static_cast<const float*>(params.key_descales);
    const float* value_descales = static_cast<const float*>(params.value_descales);
    float log_scale = params.scale * 1.4426950408889634f
        * __ldg(query_descales + request * params.query_descale_batch_stride + head * params.query_descale_head_stride)
        * __ldg(key_descales + request * params.key_descale_batch_stride + head * params.key_descale_head_stride);
    float value_scale = __ldg(value_descales + request * params.value_descale_batch_stride + head * params.value_descale_head_stride);
    float output[OutputTiles][8] = {};
    float running_max[2] = {-INFINITY, -INFINITY};
    float running_sum[2] = {0.f, 0.f};
    unsigned char* current = shared_transposed;
    unsigned char* next = shared_keys;
    const unsigned char* next_value = nullptr;
    __syncthreads();
    if (begin < end) {
        native_stage<Threads, false, true>(params, page_table, current_value, end - 1, length, current, shared_values);
        if (begin + 1 < end) next_value = native_stage<Threads, true, false, true>(params, page_table, key_source, end - 2, length, next, shared_values, value_source);
        else asm volatile("ppu.cp.async.commit_group;");
    }
    auto step = [&](int tile, auto first_flag, auto masked_flag) {
        constexpr bool First = decltype(first_flag)::value;
        constexpr bool Masked = decltype(masked_flag)::value;
        asm volatile("ppu.cp.async.wait_group 1;");
        __syncthreads();
        if constexpr (!First) {
            if (tile > begin) next_value = native_stage<Threads, true, false, true>(params, page_table, key_source, tile - 1, length, next, shared_values, value_source);
            else asm volatile("ppu.cp.async.commit_group;");
        }
        float scores[NativeKeyHalves][8] = {};
        if constexpr (Query == 1) {
            native_qk_direct<0>(scores, query_fragments, current);
        } else {
            unsigned int ring[3][2][4];
            #pragma unroll
            for (int channel = 0; channel < 3; ++channel) {
                native_load_key(current, channel, 0, ring[channel][0]);
                native_load_key(current, channel, 1, ring[channel][1]);
            }
            native_qk_ring<0>(scores, query_fragments, current, ring);
        }
        if constexpr (Masked) {
            #pragma unroll
            for (int key_half = 0; key_half < NativeKeyHalves; ++key_half) {
                #pragma unroll
                for (int value = 0; value < 8; ++value) {
                    int packed_row = row_base + (lane >> 2) + ((value >> 1) & 1) * 8;
                    int key = tile * NativeKeyRows + key_half * 16 + (lane & 3) * 2 + (value & 1) + (value >> 2) * 8;
                    if (key >= length - query_count + packed_row / Group + 1) scores[key_half][value] = -INFINITY;
                }
            }
        }
        float alpha[2] = {1.f, 1.f};
        float exponent_shift[2] = {};
        #pragma unroll
        for (int row = 0; row < ActiveRows; ++row) {
            float maximum = scores[0][row * 2];
            #pragma unroll
            for (int key_half = 0; key_half < NativeKeyHalves; ++key_half) {
                #pragma unroll
                for (int column = 0; column < 4; ++column) {
                    int index = (column & 1) + row * 2 + (column >> 1) * 4;
                    if (key_half != 0 || column != 0) maximum = fmaxf(maximum, scores[key_half][index]);
                }
            }
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
            float next_max = First ? maximum : fmaxf(running_max[row], maximum);
            float safe_max = Masked && next_max == -INFINITY ? 0.f : next_max;
            if constexpr (First) alpha[row] = 0.f;
            else if constexpr (Masked) alpha[row] = running_max[row] == -INFINITY ? 0.f : __exp2f((running_max[row] - safe_max) * log_scale);
            else alpha[row] = __exp2f((running_max[row] - safe_max) * log_scale);
            exponent_shift[row] = fmaf(-safe_max, log_scale, 8.f);
            running_sum[row] *= alpha[row];
            running_max[row] = next_max;
        }
        #pragma unroll
        for (int key_half = 0; key_half < NativeKeyHalves; ++key_half) {
            #pragma unroll
            for (int column = 0; column < 4; ++column) {
                #pragma unroll
                for (int row = 0; row < ActiveRows; ++row) {
                    int index = (column & 1) + row * 2 + (column >> 1) * 4;
                    scores[key_half][index] = __exp2f(fmaf(scores[key_half][index], log_scale, exponent_shift[row]));
                }
            }
        }
        #pragma unroll
        for (int key_half = 0; key_half < NativeKeyHalves; ++key_half) {
            #pragma unroll
            for (int column = 0; column < 4; ++column) {
                #pragma unroll
                for (int row = 0; row < ActiveRows; ++row) {
                    int index = (column & 1) + row * 2 + (column >> 1) * 4;
                    running_sum[row] += scores[key_half][index];
                }
            }
        }
        if constexpr (Query != 1 && !First) {
            if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) native_rescale_row<0, 0, OutputTiles, ActiveRows>(output, alpha);
        }
        unsigned int probabilities[NativeKeyRows / 32][4];
        if constexpr (Query != 1) native_pack_scores<ActiveRows>(scores, probabilities);
        asm volatile("ppu.cp.async.wait_group 1;");
        __syncthreads();
        native_transpose<Threads>(shared_values, current);
        if constexpr (Query == 1) native_pack_scores<ActiveRows>(scores, probabilities);
        if constexpr (Query == 1 && !First) {
            if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) {
                native_rescale_row<0, 0, OutputTiles, ActiveRows>(output, alpha);
            }
        }
        __syncthreads();
        if (tile > begin) native_stage<Threads, false, true, true>(params, page_table, next_value, tile - 1, length, current, shared_values);
        else asm volatile("ppu.cp.async.commit_group;");
        #pragma unroll
        for (int part = 0; part < NativeKeyRows / 32; ++part) {
            native_pv<OutputTiles>(output, probabilities[part], current + part * 32, output_base, lane);
        }
        unsigned char* previous = current;
        current = next;
        next = previous;
    };
    if (begin < end) {
        int tile = end - 1;
        step(tile, NativeFlag<true>{}, NativeFlag<true>{});
        --tile;
        for (; tile >= begin && tile * NativeKeyRows + NativeKeyRows - 1 >= length - query_count + 1; --tile) {
            step(tile, NativeFlag<false>{}, NativeFlag<true>{});
        }
        for (; tile >= begin; --tile) {
            step(tile, NativeFlag<false>{}, NativeFlag<false>{});
        }
    }
    asm volatile("ppu.cp.async.wait_group 0;");
    float multipliers[2] = {};
    float lse[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int row = 0; row < ActiveRows; ++row) {
        float sum = running_sum[row];
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        if (sum > 0.f) {
            multipliers[row] = value_scale / sum;
            if (params.splits > 1) lse[row] = fmaf(running_max[row], log_scale, log2f(sum) - 8.f);
        }
    }
    native_store<0, OutputTiles, Group>(params, output, multipliers, lse,
        query_begin, query_count, head, split, row_base, output_base, lane);
}

__global__ void atrex_ppu_native_decode_prepare(DecodeParams params) {
    __shared__ int tiles[1024];
    __shared__ int starts[1025];
    __shared__ int order[1024];
    __shared__ int split_counts[1024];
    __shared__ long long sums[32];
    __shared__ int warp_maximum_splits[32];
    __shared__ int warp_job_counts[32];
    __shared__ int chunk_shared;
    __shared__ int force_average;
    int request = threadIdx.x;
    int batch = params.batch;
    int heads = params.kv_heads;
    int* plan = static_cast<int*>(params.plan);
    const int* lengths = static_cast<const int*>(params.lengths);
    int lane = threadIdx.x & 31;
    int warp = native_warp();
    int warp_count = (batch + 31) / 32;
    int count = request < batch ? (__ldg(lengths + request) + NativeKeyRows - 1) / NativeKeyRows : 0;
    tiles[request] = count;
    long long total = static_cast<long long>(count) * heads;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (lane == 0) sums[warp] = total;
    __syncthreads();
    int occupancy = params.max_query == 1 ? 16 : (params.query_heads / heads == 8 ? 8 : 4);
    if (request == 0) {
        long long tile_total = 0;
        for (int index = 0; index < warp_count; ++index) tile_total += sums[index];
        long long capacity = params.num_cu * occupancy * 5;
        chunk_shared = max(1, static_cast<int>((tile_total * 6 + capacity - 1) / capacity));
    }
    __syncthreads();
    int chunk = chunk_shared;
    int active_splits = request < batch ? (params.splits == 1 ? 1 : min(32, max(1, (count + chunk - 1) / chunk))) : 0;
    int maximum = active_splits;
    int jobs = active_splits * heads;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        jobs += __shfl_down_sync(0xffffffff, jobs, offset);
    }
    if (lane == 0) {
        warp_maximum_splits[warp] = maximum;
        warp_job_counts[warp] = jobs;
    }
    __syncthreads();
    if (request == 0) {
        int maximum_splits = 0;
        int dynamic_blocks = 0;
        for (int index = 0; index < warp_count; ++index) {
            maximum_splits = max(maximum_splits, warp_maximum_splits[index]);
            dynamic_blocks += warp_job_counts[index];
        }
        int average = max(1, params.average_splits);
        int resident_capacity = params.num_cu * occupancy;
        force_average = maximum_splits < average || (maximum_splits < average + 3 && dynamic_blocks > resident_capacity
            && batch * heads * average < dynamic_blocks);
    }
    __syncthreads();
    if (request < batch) {
        int selected_splits = force_average ? max(1, params.average_splits) : active_splits;
        split_counts[request] = selected_splits;
        tiles[request] = (count + selected_splits - 1) / selected_splits;
    }
    __syncthreads();
    if (request < batch) {
        int work = tiles[request];
        int rank = 0;
        for (int other = 0; other < batch; ++other) {
            int other_work = tiles[other];
            rank += other_work > work || (other_work == work && other < request);
        }
        order[rank] = request;
    }
    __syncthreads();
    int selected_jobs = request < batch ? split_counts[order[request]] * heads : 0;
    int inclusive = selected_jobs;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int previous = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset) inclusive += previous;
    }
    if (lane == 31) warp_job_counts[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int warp_jobs = warp_job_counts[lane];
        int prefix = warp_jobs;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int previous = __shfl_up_sync(0xffffffff, prefix, offset);
            if (lane >= offset) prefix += previous;
        }
        warp_maximum_splits[lane] = prefix - warp_jobs;
        if (lane == 31) starts[batch] = prefix;
    }
    __syncthreads();
    if (request < batch) starts[request] = warp_maximum_splits[warp] + inclusive - selected_jobs;
    __syncthreads();
    if (request < batch) plan[4 + request] = split_counts[request];
    if (request == 0) {
        plan[0] = 0;
        plan[1] = starts[batch];
        plan[2] = chunk;
        plan[3] = max(1, params.average_splits);
    }
    for (int rank = warp; rank < batch; rank += 32) {
        int selected_request = __shfl_sync(0xffffffff, order[rank], 0);
        int splits = __shfl_sync(0xffffffff, split_counts[selected_request], 0);
        int start = __shfl_sync(0xffffffff, starts[rank], 0);
        for (int head = 0; head < heads; ++head) {
            if (lane < splits) {
                int work = start + head * splits + lane;
                plan[4 + batch + work] = (selected_request * heads + head) * 32 + lane;
            }
        }
    }
}

template<int Query, int Group, bool Persistent>
__global__ __launch_bounds__((NativeGeometry<Query, Group>::Threads))
void atrex_ppu_native_decode_kernel(DecodeParams params) {
    __shared__ __align__(128) unsigned char storage[NativeGeometry<Query, Group>::SharedBytes];
    unsigned char* shared_keys = storage;
    unsigned char* shared_values = storage + NativeTileBytes;
    unsigned char* shared_transposed = storage + NativeGeometry<Query, Group>::KeyOffset;
    if constexpr (Persistent) {
        int* plan = static_cast<int*>(params.plan);
        volatile int* task_mailbox = reinterpret_cast<volatile int*>(shared_transposed);
        int total_jobs = __ldg(plan + 1);
        int task = blockIdx.x;
        while (task < total_jobs) {
            int packed = __ldg(plan + 4 + params.batch + task);
            int split = packed & 31;
            int request_head = packed >> 5;
            int head = request_head % params.kv_heads;
            int request = request_head / params.kv_heads;
            native_decode_task<Query, Group>(params, split, head, request, __ldg(plan + 4 + request),
                shared_keys, shared_values, shared_transposed);
            __syncthreads();
            if (threadIdx.x == 0) *task_mailbox = atomicAdd(plan, 1) + static_cast<int>(gridDim.x);
            __syncthreads();
            task = __shfl_sync(0xffffffff, static_cast<int>(*task_mailbox), 0);
        }
    } else {
        native_decode_task<Query, Group>(params, blockIdx.x, blockIdx.y, blockIdx.z, params.splits,
            shared_keys, shared_values, shared_transposed);
    }
}

__global__ void atrex_ppu_native_decode_combine(DecodeParams params) {
    constexpr int Width = DECODE_Q1_ONLY ? 32 : 16;
    constexpr int Parts = 256 / (Width * 4);
    int token = blockIdx.x * 8 + threadIdx.x / Width;
    int head = blockIdx.y;
    int lane = threadIdx.x & (Width - 1);
    if (token >= params.total_q) return;
    unsigned int active_mask = DECODE_Q1_ONLY ? 0xffffffff : __activemask();
    const float* partial = params.partial_output;
    const float* partial_lse = params.partial_lse;
    float maximum = -INFINITY;
    int request = token;
#if !DECODE_Q1_ONLY
    if (params.automatic) {
        const int* offsets = params.offsets;
        int upper = params.batch;
        request = 0;
        while (request + 1 < upper) {
            int middle = (request + upper) / 2;
            if (__ldg(offsets + middle) <= token) request = middle;
            else upper = middle;
        }
    }
#endif
    int active_splits = params.automatic ? __ldg(params.plan + 4 + request) : params.splits;
    for (int split = 0; split < active_splits; ++split) {
        int64_t offset = (static_cast<int64_t>(split) * params.query_heads + head) * params.total_q + token;
        float value = lane == 0 ? __ldg(partial_lse + offset) : -INFINITY;
        value = __shfl_sync(active_mask, value, 0, Width);
        maximum = fmaxf(maximum, value);
    }
    float numerator[Parts][4] = {};
    float denominator = 0.f;
    for (int split = 0; split < active_splits; ++split) {
        int64_t offset = (static_cast<int64_t>(split) * params.query_heads + head) * params.total_q + token;
        float value = lane == 0 ? __ldg(partial_lse + offset) : -INFINITY;
        value = __shfl_sync(active_mask, value, 0, Width);
        float weight = maximum == -INFINITY ? 0.f : __exp2f(value - maximum);
        #pragma unroll
        for (int part = 0; part < Parts; ++part) {
            int dimension = lane * 4 + part * Width * 4;
            float4 values = *reinterpret_cast<const float4*>(partial + offset * 256 + dimension);
            numerator[part][0] += weight * values.x;
            numerator[part][1] += weight * values.y;
            numerator[part][2] += weight * values.z;
            numerator[part][3] += weight * values.w;
        }
        denominator += weight;
    }
    float inverse = denominator > 0.f ? 1.f / denominator : 0.f;
    #pragma unroll
    for (int part = 0; part < Parts; ++part) {
        int dimension = lane * 4 + part * Width * 4;
        uint2 values = make_uint2(native_bfloat_pair(numerator[part][0] * inverse,
            numerator[part][1] * inverse), native_bfloat_pair(numerator[part][2] * inverse,
            numerator[part][3] * inverse));
        *reinterpret_cast<uint2*>(params.output + token * params.output_row_stride
            + head * params.output_head_stride + dimension) = values;
    }
}

template<int Query, int Group>
int native_launch(DecodeParams params, hggcStream_t stream) {
    constexpr int Threads = NativeGeometry<Query, Group>::Threads;
    dim3 grid(params.splits, params.kv_heads, params.batch);
    if (params.automatic) {
        params.average_splits = native_average_splits(params);
        atrex_ppu_native_decode_prepare<<<1, 1024, 0, stream>>>(params);
        auto status = hggcGetLastError();
        if (status != hggcSuccess) return static_cast<int>(status);
        int blocks_per_cu = 0;
        status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu,
            atrex_ppu_native_decode_kernel<Query, Group, true>, Threads, 0);
        if (status != hggcSuccess) return static_cast<int>(status);
        grid = dim3(min(params.num_cu * blocks_per_cu, params.batch * params.kv_heads * params.splits));
        atrex_ppu_native_decode_kernel<Query, Group, true><<<grid, Threads, 0, stream>>>(params);
    } else {
        atrex_ppu_native_decode_kernel<Query, Group, false><<<grid, Threads, 0, stream>>>(params);
    }
    auto status = hggcGetLastError();
    if (status != hggcSuccess) return static_cast<int>(status);
    if (params.splits > 1) {
        atrex_ppu_native_decode_combine<<<dim3((params.total_q + 7) / 8, params.query_heads), DECODE_Q1_ONLY ? 256 : 128, 0, stream>>>(params);
        status = hggcGetLastError();
    }
    return static_cast<int>(status);
}

extern "C" int atrex_ppu_decode(void** pointers, int64_t* metadata, float scale,
                                int splits, void* stream_pointer) {
    DecodeParams params{};
    if (metadata[0] < 0 || metadata[0] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[1] < 0 || metadata[1] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[2] < 0 || metadata[2] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[3] < 0 || metadata[3] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[4] < 0 || metadata[4] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[5] < 0 || metadata[5] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[23] < 0 || metadata[23] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[24] < 0 || metadata[24] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    if (metadata[25] < 0 || metadata[25] > INT32_MAX) return static_cast<int>(hggcErrorInvalidValue);
    params.queries = static_cast<const unsigned char*>(pointers[0]);
    params.keys = static_cast<const unsigned char*>(pointers[1]);
    params.values = static_cast<const unsigned char*>(pointers[2]);
    params.output = static_cast<unsigned short*>(pointers[3]);
    params.offsets = static_cast<const int*>(pointers[4]);
    params.lengths = static_cast<const int*>(pointers[5]);
    params.page_table = static_cast<const int*>(pointers[6]);
    params.query_descales = static_cast<const float*>(pointers[7]);
    params.key_descales = static_cast<const float*>(pointers[8]);
    params.value_descales = static_cast<const float*>(pointers[9]);
    params.partial_output = static_cast<float*>(pointers[10]);
    params.partial_lse = static_cast<float*>(pointers[11]);
    params.plan = static_cast<int*>(pointers[12]);
    params.batch = metadata[0];
    params.total_q = metadata[1];
    params.max_query = metadata[2];
    params.max_kv = metadata[3];
    params.cache_pages = metadata[4];
    params.page_size = metadata[5];
    params.page_table_stride = metadata[6];
    params.query_row_stride = metadata[7];
    params.query_head_stride = metadata[8];
    params.key_page_stride = metadata[9];
    params.key_row_stride = metadata[10];
    params.key_head_stride = metadata[11];
    params.value_page_stride = metadata[12];
    params.value_row_stride = metadata[13];
    params.value_head_stride = metadata[14];
    params.output_row_stride = metadata[15];
    params.output_head_stride = metadata[16];
    params.query_descale_batch_stride = metadata[17];
    params.query_descale_head_stride = metadata[18];
    params.key_descale_batch_stride = metadata[19];
    params.key_descale_head_stride = metadata[20];
    params.value_descale_batch_stride = metadata[21];
    params.value_descale_head_stride = metadata[22];
    params.num_cu = metadata[23];
    params.query_heads = metadata[24];
    params.kv_heads = metadata[25];
    params.scale = scale;
    params.page_shift = __builtin_ctz(static_cast<unsigned int>(metadata[5]));
    params.automatic = splits == 0;
    params.splits = splits ? splits : 32;
#if DECODE_Q1_ONLY
    if (metadata[24] / metadata[25] == 8) return native_launch<1, 8>(params, static_cast<hggcStream_t>(stream_pointer));
    return native_launch<1, 16>(params, static_cast<hggcStream_t>(stream_pointer));
#else
    if (params.max_query == 5) {
        if (metadata[24] / metadata[25] == 8) return native_launch<5, 8>(params, static_cast<hggcStream_t>(stream_pointer));
        return native_launch<5, 16>(params, static_cast<hggcStream_t>(stream_pointer));
    }
    if (metadata[24] / metadata[25] == 8) return native_launch<4, 8>(params, static_cast<hggcStream_t>(stream_pointer));
    return native_launch<4, 16>(params, static_cast<hggcStream_t>(stream_pointer));
#endif
}

extern "C" int atrex_ppu_decode_resources(int* values) {
    hggcFuncAttributes attributes{};
#if DECODE_Q1_ONLY
    auto function = atrex_ppu_native_decode_kernel<1, 16, true>;
#else
    auto function = atrex_ppu_native_decode_kernel<4, 16, true>;
#endif
    auto status = hggcFuncGetAttributes(&attributes, function);
    if (status != hggcSuccess) return static_cast<int>(status);
    int blocks = 0;
#if DECODE_Q1_ONLY
    status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, 64, 0);
#else
    status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, 128, 0);
#endif
    values[0] = attributes.numRegs;
    values[1] = attributes.sharedSizeBytes;
    values[2] = attributes.localSizeBytes;
    values[3] = blocks;
    return static_cast<int>(status);
}
