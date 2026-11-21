#include <hggc_runtime.h>
#include <stdint.h>
#include <math.h>

enum class DecodeVariant { q1, q4, q4_wide, q16_dual, q16_history };

constexpr auto NativeVariant = DecodeVariant::DECODE_VARIANT;
constexpr bool NativeQ1Only = NativeVariant == DecodeVariant::q1;
constexpr bool NativeQ16History = NativeVariant == DecodeVariant::q16_history;
constexpr bool NativeQ16Dual = NativeQ16History || NativeVariant == DecodeVariant::q16_dual;
constexpr bool NativeWide = NativeVariant == DecodeVariant::q4_wide || NativeQ16Dual;
constexpr int NativeKeyRows = NativeWide ? 64 : 32;
constexpr int NativePlanRows = 32;
constexpr int NativeMinSortKV = 64 * 1024;
constexpr int NativeMinSortHistory = 32 * 1024;
constexpr int NativeMinShortSortHistory = 16 * 1024;
constexpr int NativeMaxShortSortPages = 513;
constexpr int NativeKeyHalves = NativeKeyRows / 16;
constexpr int NativeTileBytes = NativeKeyRows * 256;
constexpr int NativeKeyWindow = NativeKeyHalves == 2 ? 3 : 2;

template<int Query, int Group>
struct NativeGeometry {
    static constexpr int Rows = Query == 1 ? 16 : ((Query * Group + 15) / 16) * 16;
    static constexpr int Threads = Query == 1 ? 64 : Rows * 2;
    static constexpr int KeyOffset = Rows * 256 > NativeTileBytes * 2 ? Rows * 256 : NativeTileBytes * 2;
    static constexpr int SharedBytes = NativeQ16Dual ? NativeTileBytes * 4 : KeyOffset + NativeTileBytes;
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
    bool sort_history;
    int average_splits;
    int page_shift;
    int key_stage_offset;
};

int native_average_splits(const DecodeParams& params, int resident_blocks = 0) {
    int groups = params.batch * params.kv_heads;
    int occupancy = params.max_query == 1 ? 16 : (params.query_heads / params.kv_heads <= 8 ? 8 : 4);
    if (resident_blocks) occupancy = resident_blocks;
    int capacity = params.num_cu * occupancy;
    float waves = static_cast<float>(groups) / capacity;
    if ((occupancy > 8 && groups >= .8f * capacity)
        || (occupancy <= 8 && waves / ceilf(waves) > .9f)) return 1;
    int tiles = (params.max_kv + NativePlanRows - 1) / NativePlanRows;
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

__device__ __forceinline__ float native_descale(
    const float* descales, int request, int head, int64_t batch_stride, int64_t head_stride) {
    return descales ? __ldg(descales + request * batch_stride + head * head_stride) : 1.f;
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

template<int Threads, bool Keys, bool Values, bool FullTile = false, bool CommitGroup = true>
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
        // The wide decode stages both 128-channel slabs with one AIU request.
        constexpr int SlabsPerTransfer = NativeWide ? 2 : 1;
        #pragma unroll
        for (int slab = 0; slab < 2; slab += SlabsPerTransfer) {
            if (page_valid && valid_rows > 0) {
                unsigned int destination = static_cast<unsigned int>(__cvta_generic_to_shared(target + slab * NativeKeyRows * 128));
                unsigned long long stride = row_stride;
                unsigned long long slab_stride = SlabsPerTransfer == 2 ? 128 : stride * valid_rows;
                #define ATREX_DECODE_AIUCOPY(cache) \
                    asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global" cache \
                        ".2d.tile.LLC::128B.padz.swzl.b8 " \
                        "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;" \
                        :: "r"(destination), "l"(copy_source), "r"(256 / SlabsPerTransfer), \
                           "r"(valid_rows), "r"(SlabsPerTransfer), "r"(128), "r"(NativeKeyRows), \
                           "r"(SlabsPerTransfer), "l"(stride), "l"(slab_stride), \
                           "r"(slab * 128), "r"(0), "r"(0), "r"(0))
                if constexpr (SlabsPerTransfer == 2) {
                    ATREX_DECODE_AIUCOPY(".cg");
                } else {
                    ATREX_DECODE_AIUCOPY("");
                }
                #undef ATREX_DECODE_AIUCOPY
            } else {
                #pragma unroll
                for (int chunk = 0; chunk < NativeKeyRows * SlabsPerTransfer / 4; ++chunk) {
                    *reinterpret_cast<uint4*>(target + slab * NativeKeyRows * 128 + (threadIdx.x & 31) * 16 + chunk * 512)
                        = make_uint4(0, 0, 0, 0);
                }
            }
        }
    }
    if constexpr (CommitGroup) asm volatile("ppu.cp.async.commit_group;");
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

template<int Tile>
__device__ __forceinline__ void native_load_value(const unsigned char* transposed, int output_base,
                                                 int lane, unsigned int (&values)[4]) {
    int dimension = output_base + Tile * 32 + (lane & 7) + ((lane >> 4) & 1) * 8;
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
    native_load_value<Tile>(transposed, output_base, lane, first);
    native_load_value<Tile + 1>(transposed, output_base, lane, second);
    #pragma hggc mmatiestrictly
    {
        native_mma(output[Tile], probabilities, first);
        native_mma(output[Tile + 1], probabilities, second);
    }
    __ppu_sched_bound();
    if constexpr (Tile + 2 < Count) native_pv_scoped<Tile + 2, Count>(output, probabilities, transposed, output_base, lane);
}

template<int Tile, int Part = 0>
__device__ __forceinline__ void native_value_pair(const unsigned char* values, unsigned int (&fragments)[2][4]) {
    unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(values));
    int column = Tile * 16;
    unsigned long long address = static_cast<unsigned long long>(shared >> 4) + (column / 128) * (NativeKeyRows * 8) + Part * 256 + ((column / 32) % 4) * 2;
    unsigned int lower[4], upper[4];
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0,%1,%2,%3}, [%4], %5, %6, %7;"
        : "=r"(lower[0]), "=r"(lower[1]), "=r"(lower[2]), "=r"(lower[3])
        : "l"(address), "r"(1), "r"(64), "r"(0));
    address += 128;
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0,%1,%2,%3}, [%4], %5, %6, %7;"
        : "=r"(upper[0]), "=r"(upper[1]), "=r"(upper[2]), "=r"(upper[3])
        : "l"(address), "r"(1), "r"(64), "r"(0));
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        fragments[half][0] = __byte_perm(lower[half * 2], lower[half * 2 + 1], 0x6420);
        fragments[half][1] = __byte_perm(upper[half * 2], upper[half * 2 + 1], 0x6420);
        fragments[half][2] = __byte_perm(lower[half * 2], lower[half * 2 + 1], 0x7531);
        fragments[half][3] = __byte_perm(upper[half * 2], upper[half * 2 + 1], 0x7531);
    }
}

template<int Tile, int Count, int Part>
__device__ __forceinline__ void native_pv_direct_ring(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                                    const unsigned char* values, unsigned int (&first)[2][4], unsigned int (&second)[2][4]) {
    if constexpr (Tile + 2 < Count) native_value_pair<Tile + 2, Part>(values, second);
    __ppu_sched_bound();
    #pragma hggc mmatiestrictly
    {
        native_mma(output[Tile], probabilities, first[0]);
        native_mma(output[Tile + 1], probabilities, first[1]);
    }
    __ppu_sched_bound();
    if constexpr (Tile + 2 < Count) native_pv_direct_ring<Tile + 2, Count, Part>(output, probabilities, values, second, first);
}

template<int Tile, int Count, int Part = 0>
__device__ __forceinline__ void native_pv_direct(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                               const unsigned char* values) {
    unsigned int first[2][4], second[2][4];
    native_value_pair<0, Part>(values, first);
    native_pv_direct_ring<0, Count, Part>(output, probabilities, values, first, second);
}

template<int Count, int Part>
__device__ __forceinline__ void native_pv_atoms(float (&output)[Count][8], const unsigned int (&probabilities)[4],
                                              const unsigned char* values) {
    __ppu_sched_bound();
    unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(values));
    #pragma hggc mmatiestrictly
    {
        #pragma unroll
        for (int pair = 0; pair < Count / 2; ++pair) {
            unsigned long long address = static_cast<unsigned long long>(shared >> 4)
                + (pair / 4) * (NativeKeyRows * 8) + Part * 256 + (pair % 4) * 2;
            unsigned int lower[4], upper[4];
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0,%1,%2,%3}, [%4], %5, %6, %7;"
                : "=r"(lower[0]), "=r"(lower[1]), "=r"(lower[2]), "=r"(lower[3])
                : "l"(address), "r"(1), "r"(64), "r"(0));
            address += 128;
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0,%1,%2,%3}, [%4], %5, %6, %7;"
                : "=r"(upper[0]), "=r"(upper[1]), "=r"(upper[2]), "=r"(upper[3])
                : "l"(address), "r"(1), "r"(64), "r"(0));
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned int fragment[4] = {
                    __byte_perm(lower[half * 2], lower[half * 2 + 1], 0x6420),
                    __byte_perm(upper[half * 2], upper[half * 2 + 1], 0x6420),
                    __byte_perm(lower[half * 2], lower[half * 2 + 1], 0x7531),
                    __byte_perm(upper[half * 2], upper[half * 2 + 1], 0x7531),
                };
                native_mma(output[pair * 2 + half], probabilities, fragment);
            }
        }
    }
    __ppu_sched_bound();
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
                                              const unsigned char* keys, unsigned int (&ring)[NativeKeyWindow][NativeKeyHalves][4]) {
    static_assert(NativeKeyHalves >= 2);
    __ppu_sched_bound();
    #pragma hggc mmatiestrictly
    {
        #pragma unroll
        for (int half = 0; half < NativeKeyHalves; ++half) native_mma(scores[half], queries[Channel], ring[Channel % NativeKeyWindow][half]);
    }
    __ppu_sched_bound();
    if constexpr (Channel + NativeKeyWindow < 8) {
        #pragma unroll
        for (int half = 0; half < NativeKeyHalves; ++half) native_load_key(keys, Channel + NativeKeyWindow, half, ring[Channel % NativeKeyWindow][half]);
    }
    if constexpr (Channel + 1 < 8) native_qk_ring<Channel + 1>(scores, queries, keys, ring);
}

template<int Tile, int Count, int Group>
__device__ __forceinline__ void native_store(const DecodeParams& params, const float (&output)[Count][8],
                                           const float (&multipliers)[2], const float (&lse)[2],
                                           int query_begin, int query_count, int head, int split,
                                           int row_base, int output_base, int lane, bool single_split) {
    unsigned short* destination = static_cast<unsigned short*>(params.output);
    float* partial = static_cast<float*>(params.partial_output);
    float* partial_lse = static_cast<float*>(params.partial_lse);
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        int row = pair & 1;
        int packed_row = row_base + (lane >> 2) + row * 8;
        int query_index = packed_row / Group;
        int query_head = head * Group + packed_row % Group;
        int dimension = NativeQ1Only
            ? output_base + Tile * 32 + (lane & 3) * 2 + (pair >> 1) * 8
            : output_base + Tile * 16 + (lane & 3) * 4 + (pair >> 1) * 2;
        if (query_index < query_count) {
            int token = query_begin + query_index;
            int first_index = NativeQ1Only ? pair * 2 : row * 2 + (pair >> 1);
            int second_index = NativeQ1Only ? pair * 2 + 1 : first_index + 4;
            float first = output[Tile][first_index] * multipliers[row];
            float second = output[Tile][second_index] * multipliers[row];
            if (single_split) {
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
        query_begin, query_count, head, split, row_base, output_base, lane, single_split);
}

template<int Tile, int Count, int Group>
__device__ __forceinline__ void native_scatter_output(unsigned short* shared_output, const float (&output)[Count][8],
                                                    const float (&multipliers)[2], int query_count, int row_base, int lane) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        int row = pair & 1;
        int packed_row = row_base + (lane >> 2) + row * 8;
        int dimension = Tile * 16 + (lane & 3) * 4 + (pair >> 1) * 2;
        if (packed_row / Group < query_count) {
            int index = row * 2 + (pair >> 1);
            float first = output[Tile][index] * multipliers[row];
            float second = output[Tile][index + 4] * multipliers[row];
            int offset = packed_row * 256 + (dimension ^ ((packed_row & 7) * 32));
            *reinterpret_cast<unsigned int*>(shared_output + offset) = native_bfloat_pair(first, second);
        }
    }
    if constexpr (Tile + 1 < Count) native_scatter_output<Tile + 1, Count, Group>(shared_output, output, multipliers, query_count, row_base, lane);
}

template<int Query, int Group>
__device__ __forceinline__ void native_decode_task(const DecodeParams& params, int split, int head, int request,
                                                  int active_splits, unsigned char* shared_keys,
                                                  unsigned char* shared_values, unsigned char* shared_transposed) {
    constexpr int Threads = NativeGeometry<Query, Group>::Threads;
    constexpr int OutputTiles = Query == 1 ? 8 : 16;
    constexpr int ActiveRows = Query == 1 && Group <= 8 ? 1 : 2;
    int lane = threadIdx.x & 31;
    int warp = native_warp();
    const int* offsets = static_cast<const int*>(params.offsets);
    const int* lengths = static_cast<const int*>(params.lengths);
    int query_begin = __ldg(offsets + request);
    int query_count = __ldg(offsets + request + 1) - query_begin;
    int length = __ldg(lengths + request);
    const int* page_table = params.page_table + request * params.page_table_stride;
    const unsigned char* key_source = params.keys + head * params.key_head_stride;
    int output_base = Query == 1 ? warp * 16 : 0;
    const unsigned char* value_source = params.values + head * params.value_head_stride;
    int row_base = Query == 1 ? 0 : warp * 16;
    int tile_count = (length + NativeKeyRows - 1) / NativeKeyRows;
    bool single_split = params.splits == 1 || (Query != 1 && params.automatic && active_splits == 1);
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
    if constexpr (NativeQ16Dual) {
        asm volatile("ppu.cp.async.wait_group 0;");
    } else {
        if (begin < end) current_value = native_stage<Threads, true, false>(params, page_table, key_source, end - 1, length, shared_transposed, shared_values, value_source);
        else asm volatile("ppu.cp.async.commit_group;");
        asm volatile("ppu.cp.async.wait_group 1;");
    }
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
    const float log_scale = NativeQ16History ? 0.0625f * 1.4426950408889634f :
        params.scale * 1.4426950408889634f
        * native_descale(params.query_descales, request, head,
                         params.query_descale_batch_stride, params.query_descale_head_stride)
        * native_descale(params.key_descales, request, head,
                         params.key_descale_batch_stride, params.key_descale_head_stride);
    float value_scale = native_descale(params.value_descales, request, head,
                                       params.value_descale_batch_stride, params.value_descale_head_stride);
    float output[OutputTiles][8] = {};
    float running_max[2] = {-INFINITY, -INFINITY};
    float running_sum[2] = {0.f, 0.f};
    float history_limit[2] = {-INFINITY, -INFINITY};
    float history_shift[2] = {8.f, 8.f};
    auto refresh_history = [&](int row, float maximum, float shift) {
        // P is scaled by 256, so a 0.75 log2 margin stays below E4M3's maximum of 448.
        constexpr float ReferenceMargin = 0.75f / (0.0625f * 1.4426950408889634f);
        // Preserve the usual max update for very large logits, where the
        // ULP of the scale/shift pair consumes more of E4M3's headroom.
        history_limit[row] = fabsf(maximum) <= 262144.f ? maximum + ReferenceMargin : maximum;
        history_shift[row] = shift;
    };
    unsigned char* current = NativeQ16Dual ? shared_keys : shared_transposed;
    unsigned char* next = NativeQ16Dual ? shared_keys + NativeTileBytes * 2 : shared_keys;
    const unsigned char* next_value = nullptr;
    auto stage_pair = [&](int tile, unsigned char* key, auto full_flag) {
        constexpr bool FullTile = decltype(full_flag)::value;
        unsigned char* value = key + NativeTileBytes;
        // One wait_group 0 already covers the complete tile, so issue both copies in one group.
        const auto* source = native_stage<Threads, true, false, FullTile, false>(
            params, page_table, key_source, tile, length, key, value, value_source);
        native_stage<Threads, false, true, FullTile>(params, page_table, source, tile, length, key, value);
    };
    __syncthreads();
    if (begin < end) {
        if constexpr (NativeQ16Dual) {
            stage_pair(end - 1, current, NativeFlag<false>{});
        } else {
            native_stage<Threads, false, true>(params, page_table, current_value, end - 1, length, current, shared_values);
            if (begin + 1 < end) next_value = native_stage<Threads, true, false, true>(params, page_table, key_source, end - 2, length, next, shared_values, value_source);
            else asm volatile("ppu.cp.async.commit_group;");
        }
    }
    auto step = [&](int tile, auto first_flag, auto masked_flag) {
        constexpr bool First = decltype(first_flag)::value;
        constexpr bool Masked = decltype(masked_flag)::value;
        if constexpr (NativeQ16Dual) {
            asm volatile("ppu.cp.async.wait_group 0;");
            __syncthreads();
            // Each K/V pair has its own space while the other pair feeds QK and PV.
            if (tile > begin) stage_pair(tile - 1, next, NativeFlag<true>{});
        } else {
            asm volatile("ppu.cp.async.wait_group 1;");
            __syncthreads();
            if constexpr (!First) {
                if (tile > begin) next_value = native_stage<Threads, true, false, true>(params, page_table, key_source, tile - 1, length, next, shared_values, value_source);
                else asm volatile("ppu.cp.async.commit_group;");
            }
        }
        float scores[NativeKeyHalves][8] = {};
        if constexpr (Query == 1 || (NativeKeyRows > 32 && !NativeQ16Dual)) {
            native_qk_direct<0>(scores, query_fragments, current);
        } else {
            unsigned int ring[NativeKeyWindow][NativeKeyHalves][4];
            #pragma unroll
            for (int channel = 0; channel < NativeKeyWindow; ++channel) {
                #pragma unroll
                for (int half = 0; half < NativeKeyHalves; ++half) native_load_key(current, channel, half, ring[channel][half]);
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
        constexpr bool GuardedHistory = NativeQ16History && !First && !Masked;
        float alpha[2] = {1.f, 1.f};
        float tile_maximum[2];
        float exponent_shift[2] = {};
        #pragma unroll
        for (int row = 0; row < ActiveRows; ++row) {
            float maximum;
            if constexpr (NativeKeyRows == 64 && (Query == 4 || Query == 16) && Group == 16) {
                // Reduce each half-tile independently to shorten the softmax dependency chain.
                float half_max[NativeKeyHalves];
                #pragma unroll
                for (int half = 0; half < NativeKeyHalves; ++half) {
                    const float* lane_scores = scores[half] + row * 2;
                    half_max[half] = fmaxf(fmaxf(lane_scores[0], lane_scores[1]),
                                           fmaxf(lane_scores[4], lane_scores[5]));
                }
                maximum = fmaxf(fmaxf(half_max[0], half_max[1]), fmaxf(half_max[2], half_max[3]));
            } else {
                maximum = scores[0][row * 2];
                #pragma unroll
                for (int key_half = 0; key_half < NativeKeyHalves; ++key_half) {
                    #pragma unroll
                    for (int column = 0; column < 4; ++column) {
                        int index = (column & 1) + row * 2 + (column >> 1) * 4;
                        if (key_half != 0 || column != 0) maximum = fmaxf(maximum, scores[key_half][index]);
                    }
                }
            }
            if constexpr (GuardedHistory) {
                tile_maximum[row] = maximum;
            } else {
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
                if constexpr (NativeQ16History) refresh_history(row, next_max, exponent_shift[row]);
            }
        }
        if constexpr (GuardedHistory) {
            // A warp-wide vote on lane-local maxima avoids cross-lane reductions on steady pages.
            const bool any_exceeds_reference =
                tile_maximum[0] > history_limit[0] || tile_maximum[1] > history_limit[1];
            if (__any_sync(0xffffffff, any_exceeds_reference)) {
                #pragma unroll
                for (int row = 0; row < ActiveRows; ++row) {
                    float maximum = tile_maximum[row];
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
                    tile_maximum[row] = maximum;
                }
                if (tile_maximum[0] > history_limit[0]) {
                    alpha[0] = __exp2f((running_max[0] - tile_maximum[0]) * log_scale);
                    running_sum[0] *= alpha[0];
                    running_max[0] = tile_maximum[0];
                    refresh_history(0, running_max[0],
                                    fmaf(-running_max[0], log_scale, 8.f));
                }
                if (tile_maximum[1] > history_limit[1]) {
                    alpha[1] = __exp2f((running_max[1] - tile_maximum[1]) * log_scale);
                    running_sum[1] *= alpha[1];
                    running_max[1] = tile_maximum[1];
                    refresh_history(1, running_max[1],
                                    fmaf(-running_max[1], log_scale, 8.f));
                }
                native_rescale_row<0, 0, OutputTiles, ActiveRows>(output, alpha);
            }
            exponent_shift[0] = history_shift[0];
            exponent_shift[1] = history_shift[1];
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
        if constexpr (Query != 1 && !First && !GuardedHistory) {
            if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) native_rescale_row<0, 0, OutputTiles, ActiveRows>(output, alpha);
        }
        unsigned int probabilities[NativeKeyRows / 32][4];
        if constexpr (Query != 1) native_pack_scores<ActiveRows>(scores, probabilities);
        if constexpr (!NativeQ16Dual) {
            asm volatile("ppu.cp.async.wait_group 1;");
            __syncthreads();
        }
        if constexpr (Query == 1) native_transpose<Threads>(shared_values, current);
        if constexpr (Query == 1) native_pack_scores<ActiveRows>(scores, probabilities);
        if constexpr (Query == 1 && !First) {
            if (__any_sync(0xffffffff, alpha[0] != 1.f || alpha[1] != 1.f)) {
                native_rescale_row<0, 0, OutputTiles, ActiveRows>(output, alpha);
            }
        }
        if constexpr (Query == 1) __syncthreads();
        if constexpr (!NativeQ16Dual) {
            unsigned char* value_destination = Query == 1 ? shared_values : current;
            if (tile > begin) native_stage<Threads, false, true, true>(params, page_table, next_value, tile - 1, length, current, value_destination);
            else asm volatile("ppu.cp.async.commit_group;");
        }
        #pragma unroll
        for (int part = 0; part < NativeKeyRows / 32; ++part) {
            if constexpr (Query == 1) native_pv_scoped<0, OutputTiles>(output, probabilities[part], current + part * 32, output_base, lane);
            else if constexpr (NativeKeyRows > 32 && (Group == 16 || Group == 12)) {
                if (part == 0) native_pv_atoms<OutputTiles, 0>(output, probabilities[0], shared_values);
                else native_pv_atoms<OutputTiles, 1>(output, probabilities[1], shared_values);
            }
            else if (part == 0) native_pv_direct<0, OutputTiles, 0>(output, probabilities[0], shared_values);
            else if (part == 1) native_pv_direct<0, OutputTiles, 1>(output, probabilities[1], shared_values);
        }
        unsigned char* previous = current;
        current = next;
        if constexpr (NativeQ16Dual) {
            next = previous;
            shared_values = current + NativeTileBytes;
        } else if constexpr (Query == 1) next = previous;
        else {
            next = shared_values;
            shared_values = previous;
        }
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
            if (!single_split) lse[row] = fmaf(running_max[row], log_scale, log2f(sum) - 8.f);
        }
    }
    if constexpr (NativeKeyRows > 32 && Query != 1
        && sizeof(unsigned short) * QueryRows * 256 <= NativeGeometry<Query, Group>::SharedBytes) {
        if (single_split) {
            auto shared_output = reinterpret_cast<unsigned short*>(shared_keys);
            __syncthreads();
            native_scatter_output<0, OutputTiles, Group>(shared_output, output, multipliers, query_count, row_base, lane);
            __syncthreads();
            #pragma unroll
            for (int copy = 0; copy < QueryRows * 32 / Threads; ++copy) {
                int vector = threadIdx.x + copy * Threads;
                int packed_row = vector / 32;
                int dimension = (vector & 31) * 8;
                int query_index = packed_row / Group;
                if (query_index < query_count) {
                    int token = query_begin + query_index;
                    int query_head = head * Group + packed_row % Group;
                    int offset = packed_row * 256 + (dimension ^ ((packed_row & 7) * 32));
                    *reinterpret_cast<uint4*>(params.output + token * params.output_row_stride
                        + query_head * params.output_head_stride + output_base + dimension)
                        = *reinterpret_cast<const uint4*>(shared_output + offset);
                }
            }
        } else native_store<0, OutputTiles, Group>(params, output, multipliers, lse,
            query_begin, query_count, head, split, row_base, output_base, lane, single_split);
    } else native_store<0, OutputTiles, Group>(params, output, multipliers, lse,
        query_begin, query_count, head, split, row_base, output_base, lane, single_split);
}

template<int Capacity>
__global__ __launch_bounds__(256)
void atrex_ppu_native_decode_sort_history(DecodeParams params, int* sorted_table, int width) {
    __shared__ unsigned int keys[Capacity];
    constexpr unsigned int Pad = UINT32_MAX;
    int request = blockIdx.x;
    const int* source = params.page_table + static_cast<int64_t>(request) * params.page_table_stride;
    int* destination = sorted_table + request * width;
    int64_t query_count = static_cast<int64_t>(__ldg(params.offsets + request + 1))
        - __ldg(params.offsets + request);
    int length = __ldg(params.lengths + request);
    constexpr int MinimumLength = Capacity <= 512 ? NativeMinShortSortHistory : NativeMinSortHistory;
    int history = 0;
    if (query_count > 0 && query_count <= params.max_query
        && length >= MinimumLength && length <= params.max_kv) {
        // For the earliest query, the visible prefix contains length - query_count + 1 tokens.
        // Leave the partial or causally masked page and every later entry in its original slot.
        int visible = max(0, length - static_cast<int>(query_count) + 1);
        history = min(min(Capacity, width), visible >> params.page_shift);
    }
    bool reorder = false;
    if (history >= 2) {
        bool valid_scale = true;
        if (threadIdx.x < params.kv_heads) {
            int head = threadIdx.x;
            float scale = native_descale(params.query_descales, request, head,
                                         params.query_descale_batch_stride, params.query_descale_head_stride)
                * native_descale(params.key_descales, request, head,
                                 params.key_descale_batch_stride, params.key_descale_head_stride);
            unsigned int bits = __float_as_uint(scale);
            valid_scale = bits != 0 && bits < 0x7f800000u;
        }
        if (!__syncthreads_and(valid_scale)) history = 0;
    }
    if (history >= 2) {
        int network_size = 32;
        while (network_size < history) network_size *= 2;
        for (int index = threadIdx.x; index < network_size; index += blockDim.x) {
            keys[index] = index < history ? static_cast<unsigned int>(__ldg(source + index)) : Pad;
        }
        __syncthreads();
        bool valid = true;
        bool ordered = true;
        for (int index = threadIdx.x; index < history; index += blockDim.x) {
            unsigned int page = keys[index];
            valid &= page < static_cast<unsigned int>(params.cache_pages);
            ordered &= index == 0 || keys[index - 1] <= page;
        }
        bool all_valid = __syncthreads_and(valid);
        bool all_ordered = __syncthreads_and(ordered);
        reorder = all_valid && !all_ordered;
        if (reorder) {
            // Sort alternating ascending/descending runs inside each warp first.
            for (int index = threadIdx.x; index < network_size; index += blockDim.x) {
                unsigned int page = keys[index];
                #pragma unroll
                for (int span = 2; span <= 32; span *= 2) {
                    #pragma unroll
                    for (int distance = span / 2; distance > 0; distance /= 2) {
                        unsigned int other = __shfl_xor_sync(0xffffffff, page, distance);
                        bool take_min = ((index & span) == 0) == ((index & distance) == 0);
                        page = take_min ? min(page, other) : max(page, other);
                    }
                }
                keys[index] = page;
            }
            __syncthreads();
            for (int span = 64; span <= network_size; span *= 2) {
                for (int distance = span / 2; distance >= 32; distance /= 2) {
                    for (int pair = threadIdx.x; pair < network_size / 2; pair += blockDim.x) {
                        int low = ((pair & ~(distance - 1)) << 1) | (pair & (distance - 1));
                        int high = low + distance;
                        unsigned int first = keys[low], second = keys[high];
                        unsigned int smaller = min(first, second), larger = max(first, second);
                        bool ascending = (low & span) == 0;
                        keys[low] = ascending ? smaller : larger;
                        keys[high] = ascending ? larger : smaller;
                    }
                    __syncthreads();
                }
                for (int index = threadIdx.x; index < network_size; index += blockDim.x) {
                    unsigned int page = keys[index];
                    #pragma unroll
                    for (int distance = 16; distance > 0; distance /= 2) {
                        unsigned int other = __shfl_xor_sync(0xffffffff, page, distance);
                        bool take_min = ((index & span) == 0) == ((index & distance) == 0);
                        page = take_min ? min(page, other) : max(page, other);
                    }
                    keys[index] = page;
                }
                __syncthreads();
            }
        }
    }
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        destination[index] = reorder && index < history ? static_cast<int>(keys[index]) : __ldg(source + index);
    }
    // Prepare reads this count before replacing it with the request's split count.
    if (threadIdx.x == 0) params.plan[4 + request] = reorder ? history : 0;
}

template<int Buckets>
__device__ __forceinline__ int native_page_bucket(const DecodeParams& params, int request,
                                                   int history, int split, int splits,
                                                   int bucket_width) {
    if (history == 0 || split >= splits) return Buckets;
    int length = __ldg(params.lengths + request);
    int tiles = (length - 1) / 64 + 1;
    int begin = tiles * split / splits;
    int end = min(history, tiles * (split + 1) / splits);
    if (end <= begin) return Buckets;
    const int* pages = params.page_table + request * params.page_table_stride;
    int page = __ldg(pages + begin + (end - begin - 1) / 2);
    if (static_cast<unsigned int>(page) >= static_cast<unsigned int>(params.cache_pages)) return Buckets;
    return min(Buckets - 1, page / bucket_width);
}

template<int Buckets>
__device__ __forceinline__ void native_write_bucketed_plan(const DecodeParams& params,
    const int* order, const int* split_counts, const int* histories, int chunk) {
    __shared__ int warp_bins[32][Buckets + 1];
    __shared__ int bin_starts[Buckets + 1];
    int lane = threadIdx.x & 31;
    int warp = native_warp();
    int ranks_per_warp = (params.batch + 31) / 32;
    int rank_begin = warp * ranks_per_warp;
    int rank_end = min(rank_begin + ranks_per_warp, params.batch);
    int bucket_width = (params.cache_pages - 1) / Buckets + 1;
    int count = 0;
    for (int rank = rank_begin; rank < rank_end; ++rank) {
        int request = order[rank];
        int splits = split_counts[request];
        int bucket = native_page_bucket<Buckets>(params, request, histories[request], lane,
                                                splits, bucket_width);
        #pragma unroll
        for (int bin = 0; bin <= Buckets; ++bin) {
            unsigned int mask = __ballot_sync(0xffffffff, lane < splits && bucket == bin);
            if (lane == bin) count += __popc(mask) * params.kv_heads;
        }
    }
    if (lane <= Buckets) warp_bins[warp][lane] = count;
    __syncthreads();
    if (warp == 0 && lane <= Buckets) {
        int prefix = 0;
        for (int worker = 0; worker < 32; ++worker) {
            int count = warp_bins[worker][lane];
            warp_bins[worker][lane] = prefix;
            prefix += count;
        }
        bin_starts[lane] = prefix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prefix = 0;
        for (int bin = 0; bin <= Buckets; ++bin) {
            int count = bin_starts[bin];
            bin_starts[bin] = prefix;
            prefix += count;
        }
        int* plan = params.plan;
        plan[0] = 0;
        plan[1] = prefix;
        plan[2] = chunk;
        plan[3] = max(1, params.average_splits);
    }
    __syncthreads();
    if (threadIdx.x < params.batch) {
        params.plan[4 + threadIdx.x] = split_counts[threadIdx.x];
    }
    int cursor = lane <= Buckets ? bin_starts[lane] + warp_bins[warp][lane] : 0;
    unsigned int lower = (1u << lane) - 1u;
    int* jobs = params.plan + 4 + params.batch;
    for (int rank = rank_begin; rank < rank_end; ++rank) {
        int request = order[rank];
        int splits = split_counts[request];
        int bucket = native_page_bucket<Buckets>(params, request, histories[request], lane,
                                                splits, bucket_width);
        #pragma unroll
        for (int bin = 0; bin <= Buckets; ++bin) {
            unsigned int mask = __ballot_sync(0xffffffff, lane < splits && bucket == bin);
            int base = __shfl_sync(0xffffffff, cursor, bin);
            int size = __popc(mask);
            if (lane < splits && bucket == bin) {
                int index = __popc(mask & lower);
                for (int head = 0; head < params.kv_heads; ++head) {
                    jobs[base + head * size + index] = (request * params.kv_heads + head) * 32 + lane;
                }
            }
            if (lane == bin) cursor += size * params.kv_heads;
        }
    }
}

__global__ void atrex_ppu_native_decode_prepare(DecodeParams params, int resident_blocks) {
    __shared__ int tiles[1024];
    __shared__ int starts[1025];
    __shared__ int order[1024];
    __shared__ int split_counts[1024];
    __shared__ long long sums[32];
    __shared__ int warp_maximum_splits[32];
    __shared__ int warp_job_counts[32];
    __shared__ int chunk_shared;
    __shared__ int force_average;
    __shared__ bool split_major;
    int request = threadIdx.x;
    int batch = params.batch;
    int heads = params.kv_heads;
    int* plan = static_cast<int*>(params.plan);
    const int* lengths = static_cast<const int*>(params.lengths);
    int lane = threadIdx.x & 31;
    int warp = native_warp();
    int warp_count = (batch + 31) / 32;
    int count = request < batch ? (__ldg(lengths + request) + NativePlanRows - 1) / NativePlanRows : 0;
    tiles[request] = count;
    if (!NativeQ1Only && params.max_query == 4) {
        int reference_count = (__ldg(lengths) + NativePlanRows - 1) / NativePlanRows;
        unsigned int same = __ballot_sync(0xffffffff, request >= batch || count == reference_count);
        if (lane == 0 && warp < warp_count) warp_maximum_splits[warp] = same == 0xffffffff;
        if (params.sort_history) {
            bool reordered = request < batch && __ldg(plan + 4 + request) != 0;
            unsigned int reordered_mask = __ballot_sync(0xffffffff, reordered);
            if (lane == 0 && warp < warp_count) warp_job_counts[warp] = __popc(reordered_mask);
        }
        __syncthreads();
        if (request == 0) {
            bool uniform = true;
            for (int index = 0; index < warp_count; ++index) uniform &= warp_maximum_splits[index] != 0;
            force_average = 0;
            split_major = false;
            if (uniform) {
                if (params.sort_history) {
                    int reordered = 0;
                    for (int index = 0; index < warp_count; ++index) reordered += warp_job_counts[index];
                    split_major = reordered * 2 >= batch;
                }
                int occupancy = params.query_heads / heads <= 8 ? 8 : 4;
                if (resident_blocks) occupancy = resident_blocks;
                int capacity = params.num_cu * occupancy;
                long long tile_total = static_cast<long long>(reference_count) * batch * heads;
                long long work_capacity = static_cast<long long>(capacity) * 5;
                int chunk = max(1, static_cast<int>((tile_total * 6 + work_capacity - 1) / work_capacity));
                chunk_shared = chunk;
                int selected = params.splits == 1 ? 1 : min(32, max(1, (reference_count + chunk - 1) / chunk));
                int average = max(1, params.average_splits);
                int dynamic_blocks = batch * heads * selected;
                int fixed_blocks = batch * heads * average;
                bool balanced = false;
                if (resident_blocks && dynamic_blocks > capacity) {
                    int dynamic_waves = (dynamic_blocks + capacity - 1) / capacity;
                    int fixed_waves = (fixed_blocks + capacity - 1) / capacity;
                    balanced = static_cast<long long>(fixed_blocks) * dynamic_waves * 9
                        > static_cast<long long>(dynamic_blocks) * fixed_waves * 10;
                }
                if (selected < average || (selected < average + 3 && dynamic_blocks > capacity
                    && (fixed_blocks < dynamic_blocks || balanced))) selected = average;
                selected = min(selected, max(1, reference_count));
                if (params.query_heads / heads == 16
                    && reference_count <= params.max_query * (params.query_heads / heads) * 2) selected = 1;
                force_average = selected;
            }
        }
        __syncthreads();
        int uniform_splits = force_average;
        if (uniform_splits > 0) {
            int groups = batch * heads;
            int jobs = groups * uniform_splits;
            if (request < batch) plan[4 + request] = uniform_splits;
            if (request == 0) {
                plan[0] = 0;
                plan[1] = jobs;
                plan[2] = chunk_shared;
                plan[3] = max(1, params.average_splits);
            }
            for (int work = request; work < jobs; work += blockDim.x) {
                if (split_major) {
                    // Dispatch the same KV split across requests together.
                    plan[4 + batch + work] = (work % groups) * 32 + work / groups;
                } else {
                    plan[4 + batch + work] = (work / uniform_splits) * 32 + work % uniform_splits;
                }
            }
            return;
        }
    }
    long long total = static_cast<long long>(count) * heads;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (lane == 0) sums[warp] = total;
    __syncthreads();
    int occupancy = params.max_query == 1 ? 16 : (params.query_heads / heads <= 8 ? 8 : 4);
    if (resident_blocks) occupancy = resident_blocks;
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
        int fixed_blocks = batch * heads * average;
        bool balanced_average = false;
        if (resident_blocks && dynamic_blocks > resident_capacity) {
            int dynamic_waves = (dynamic_blocks + resident_capacity - 1) / resident_capacity;
            int fixed_waves = (fixed_blocks + resident_capacity - 1) / resident_capacity;
            balanced_average = static_cast<long long>(fixed_blocks) * dynamic_waves * 9
                > static_cast<long long>(dynamic_blocks) * fixed_waves * 10;
        }
        force_average = maximum_splits < average || (maximum_splits < average + 3 && dynamic_blocks > resident_capacity
            && (fixed_blocks < dynamic_blocks || balanced_average));
    }
    __syncthreads();
    if (request < batch) {
        int selected_splits = force_average ? max(1, params.average_splits) : active_splits;
        selected_splits = min(selected_splits, max(1, count));
        if (params.max_query == 4 && params.query_heads / heads == 16) {
            int minimum_tiles = params.max_query * (params.query_heads / heads) * 2;
            if (count <= minimum_tiles) selected_splits = 1;
        }
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
    if constexpr (NativeVariant == DecodeVariant::q4_wide) {
        const int* reordered_history = params.sort_history ? params.plan + 4 : nullptr;
        if (reordered_history && heads == 2 && params.page_size == 64 && params.cache_pages > 0) {
            int history = request < batch ? __ldg(reordered_history + request) : 0;
            if (request < batch) starts[request] = history;
            unsigned int eligible = __ballot_sync(0xffffffff, history > 0);
            if (lane == 0) warp_job_counts[warp] = __popc(eligible);
            __syncthreads();
            if (request == 0) {
                int eligible_requests = 0;
                for (int index = 0; index < warp_count; ++index) eligible_requests += warp_job_counts[index];
                force_average = eligible_requests >= 32 && eligible_requests * 2 >= batch;
            }
            __syncthreads();
            if (force_average) {
                // Reorder dispatch only. Each request keeps the original split intervals.
                native_write_bucketed_plan<16>(params, order, split_counts, starts, chunk);
                return;
            }
        }
    }
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
        int warp_jobs = lane < warp_count ? warp_job_counts[lane] : 0;
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
    for (int rank = warp; rank < batch; rank += blockDim.x / 32) {
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
    unsigned char* shared_transposed = storage + (NativeKeyRows > 32 ? params.key_stage_offset : NativeGeometry<Query, Group>::KeyOffset);
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
            if constexpr (Query != 1) {
                if (total_jobs <= static_cast<int>(gridDim.x)) break;
            }
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
    constexpr int Width = NativeQ1Only ? 32 : 16;
    constexpr int Parts = 256 / (Width * 4);
    int token = blockIdx.x * 8 + threadIdx.x / Width;
    int head = blockIdx.y;
    int lane = threadIdx.x & (Width - 1);
    if (token >= params.total_q) return;
    unsigned int active_mask = NativeQ1Only ? 0xffffffff : __activemask();
    const float* partial = params.partial_output;
    const float* partial_lse = params.partial_lse;
    float maximum = -INFINITY;
    int request = token;
    if (!NativeQ1Only && params.automatic) {
        const int* offsets = params.offsets;
        int upper = params.batch;
        request = 0;
        while (request + 1 < upper) {
            int middle = (request + upper) / 2;
            if (__ldg(offsets + middle) <= token) request = middle;
            else upper = middle;
        }
    }
    int active_splits = params.automatic ? __ldg(params.plan + 4 + request) : params.splits;
    if (!NativeQ1Only && params.automatic && active_splits == 1) return;
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

template<int SplitCapacity>
__global__ void atrex_ppu_native_decode_combine_tree(DecodeParams params) {
    constexpr int Warps = 4;
    constexpr int Width = 32;
    constexpr int Vectors = 256 / 4;
    constexpr int SplitGroups = SplitCapacity / Width;
    __shared__ float weights[SplitCapacity];
    __shared__ float inverse_denominator;
    __shared__ float4 warp_numerators[Warps - 1][Vectors];

    int token = blockIdx.x;
    int head = blockIdx.y;
    int lane = threadIdx.x & (Width - 1);
    int warp = threadIdx.x / Width;
    int64_t row = static_cast<int64_t>(head) * params.total_q + token;
    int64_t split_stride = static_cast<int64_t>(params.query_heads) * params.total_q;

    if (warp == 0) {
        float lse[SplitGroups];
        float maximum = -INFINITY;
        #pragma unroll
        for (int group = 0; group < SplitGroups; ++group) {
            int split = lane + group * Width;
            lse[group] = split < params.splits ? __ldg(params.partial_lse + row + split * split_stride) : -INFINITY;
            maximum = fmaxf(maximum, lse[group]);
        }
        #pragma unroll
        for (int delta = Width / 2; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
        }
        float denominator = 0.f;
        #pragma unroll
        for (int group = 0; group < SplitGroups; ++group) {
            float weight = maximum == -INFINITY ? 0.f : __exp2f(lse[group] - maximum);
            weights[lane + group * Width] = weight;
            denominator += weight;
        }
        #pragma unroll
        for (int delta = Width / 2; delta > 0; delta /= 2) {
            denominator += __shfl_down_sync(0xffffffff, denominator, delta);
        }
        if (lane == 0) inverse_denominator = denominator > 0.f ? 1.f / denominator : 0.f;
    }
    __syncthreads();

    constexpr int VectorsPerLane = Vectors / Width;
    float4 numerator[VectorsPerLane] = {};
    for (int split = warp; split < params.splits; split += Warps) {
        float weight = weights[split];
        if (weight == 0.f) continue;  // Empty splits need only initialize their LSE.
        int64_t offset = (row + split * split_stride) * 256;
        const float4* values = reinterpret_cast<const float4*>(params.partial_output + offset);
        #pragma unroll
        for (int part = 0; part < VectorsPerLane; ++part) {
            float4 value = values[lane + part * Width];
            numerator[part].x = fmaf(weight, value.x, numerator[part].x);
            numerator[part].y = fmaf(weight, value.y, numerator[part].y);
            numerator[part].z = fmaf(weight, value.z, numerator[part].z);
            numerator[part].w = fmaf(weight, value.w, numerator[part].w);
        }
    }
    if (warp != 0) {
        #pragma unroll
        for (int part = 0; part < VectorsPerLane; ++part) {
            warp_numerators[warp - 1][lane + part * Width] = numerator[part];
        }
    }
    __syncthreads();

    if (warp != 0) return;
    unsigned short* output = params.output + token * params.output_row_stride + head * params.output_head_stride;
    #pragma unroll
    for (int part = 0; part < VectorsPerLane; ++part) {
        int vector = lane + part * Width;
        float4 first = numerator[part];
        float4 second = warp_numerators[0][vector];
        float4 third = warp_numerators[1][vector];
        float4 fourth = warp_numerators[2][vector];
        float x = ((first.x + second.x) + (third.x + fourth.x)) * inverse_denominator;
        float y = ((first.y + second.y) + (third.y + fourth.y)) * inverse_denominator;
        float z = ((first.z + second.z) + (third.z + fourth.z)) * inverse_denominator;
        float w = ((first.w + second.w) + (third.w + fourth.w)) * inverse_denominator;
        reinterpret_cast<uint2*>(output)[vector] = make_uint2(native_bfloat_pair(x, y), native_bfloat_pair(z, w));
    }
}

template<int Query, int Group>
int native_launch(DecodeParams params, hggcStream_t stream) {
    constexpr int Threads = NativeGeometry<Query, Group>::Threads;
    params.key_stage_offset = NativeGeometry<Query, Group>::KeyOffset;
    if constexpr (NativeVariant == DecodeVariant::q4_wide && Query == 4 && Group == 16) {
        if (params.sort_history) {
            int width = params.max_kv >> params.page_shift;
            int* sorted_table = params.plan + 4 + params.batch + params.batch * params.kv_heads * params.splits;
            dim3 rows(params.batch);
            if (width <= 257) atrex_ppu_native_decode_sort_history<256><<<rows, 256, 0, stream>>>(params, sorted_table, width);
            else if (width <= NativeMaxShortSortPages) atrex_ppu_native_decode_sort_history<512><<<rows, 256, 0, stream>>>(params, sorted_table, width);
            else if (width <= 1024) atrex_ppu_native_decode_sort_history<1024><<<rows, 256, 0, stream>>>(params, sorted_table, width);
            else if (width <= 2048) atrex_ppu_native_decode_sort_history<2048><<<rows, 256, 0, stream>>>(params, sorted_table, width);
            else atrex_ppu_native_decode_sort_history<4096><<<rows, 256, 0, stream>>>(params, sorted_table, width);
            auto status = hggcGetLastError();
            if (status != hggcSuccess) return static_cast<int>(status);
            params.page_table = sorted_table;
            params.page_table_stride = width;
        }
    }
    if constexpr (Query == 4 && Group == 16) {
        constexpr int single_split_limit = NativePlanRows * Query * Group * 2;
        if (params.automatic && params.max_query == 4 && params.max_kv <= single_split_limit) {
            params.automatic = false;
            params.splits = 1;
        }
    }
    dim3 grid(params.splits, params.kv_heads, params.batch);
    if (params.automatic) {
        int blocks_per_cu = 0;
        auto status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu,
            atrex_ppu_native_decode_kernel<Query, Group, true>, Threads, 0);
        if (status != hggcSuccess) return static_cast<int>(status);
        // Plan against the resident CTAs for the tuned high-concurrency Q1 shapes.
        // The fallback counts warps, while each Q1 CTA uses two warps.
        const bool q1_residency = Query == 1 && Group == 16 && params.kv_heads == 2
            && params.page_size == 64 && (params.batch == 64 || params.batch == 128)
            && params.max_kv >= 16 * 1024;
        const int resident_blocks = q1_residency || (Query == 4 && Group == 12) ? blocks_per_cu : 0;
        params.average_splits = native_average_splits(params, resident_blocks);
        atrex_ppu_native_decode_prepare<<<1, 1024, 0, stream>>>(params, resident_blocks);
        status = hggcGetLastError();
        if (status != hggcSuccess) return static_cast<int>(status);
        grid = dim3(min(params.num_cu * blocks_per_cu, params.batch * params.kv_heads * params.splits));
        atrex_ppu_native_decode_kernel<Query, Group, true><<<grid, Threads, 0, stream>>>(params);
    } else {
        atrex_ppu_native_decode_kernel<Query, Group, false><<<grid, Threads, 0, stream>>>(params);
    }
    auto status = hggcGetLastError();
    if (status != hggcSuccess) return static_cast<int>(status);
    if (params.splits > 1) {
        if (!params.automatic && params.batch == 1 && params.query_heads == 32 && params.kv_heads == 2
            && params.splits >= 16 && params.splits <= 128
            && (params.max_query == 1 || params.max_query == 4)) {
            dim3 combine_grid(params.total_q, params.query_heads);
            if (params.splits <= 32) {
                atrex_ppu_native_decode_combine_tree<32><<<combine_grid, 128, 0, stream>>>(params);
            } else {
                atrex_ppu_native_decode_combine_tree<128><<<combine_grid, 128, 0, stream>>>(params);
            }
        } else {
            atrex_ppu_native_decode_combine<<<dim3((params.total_q + 7) / 8, params.query_heads), NativeQ1Only ? 256 : 128, 0, stream>>>(params);
        }
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
    params.sort_history = splits == -1;
    const bool short_sort = params.batch >= 64 && params.max_kv >= NativeMinShortSortHistory
        && params.max_kv <= NativeMaxShortSortPages * 64;
    if (splits < 0 && (!params.sort_history || NativeVariant != DecodeVariant::q4_wide
        || params.max_query != 4 || params.query_heads != 32 || params.kv_heads != 2
        || params.batch < 8 || (params.max_kv < NativeMinSortKV && !short_sort) || params.plan == nullptr
        || params.page_size != 64
        || params.max_kv % params.page_size || params.max_kv / params.page_size > 4096)) {
        return static_cast<int>(hggcErrorInvalidValue);
    }
    params.page_shift = __builtin_ctz(static_cast<unsigned int>(metadata[5]));
    params.automatic = splits <= 0;
    params.splits = splits > 0 ? splits : 32;
    if constexpr (NativeQ16History) {
        if (scale != 0.0625f || params.query_descales || params.key_descales || params.value_descales) {
            return static_cast<int>(hggcErrorInvalidValue);
        }
    }
    if constexpr (NativeQ16Dual) {
        if (params.max_query != 16 || params.query_heads != 32 || params.kv_heads != 2
            || params.page_size != 64 || splits < 1 || splits > 32) {
            return static_cast<int>(hggcErrorInvalidValue);
        }
        return native_launch<16, 16>(params, static_cast<hggcStream_t>(stream_pointer));
    } else if constexpr (NativeQ1Only) {
        if (metadata[24] / metadata[25] == 6) return native_launch<1, 6>(params, static_cast<hggcStream_t>(stream_pointer));
        if (metadata[24] / metadata[25] == 12) return native_launch<1, 12>(params, static_cast<hggcStream_t>(stream_pointer));
        if (metadata[24] / metadata[25] == 8) return native_launch<1, 8>(params, static_cast<hggcStream_t>(stream_pointer));
        return native_launch<1, 16>(params, static_cast<hggcStream_t>(stream_pointer));
    } else {
        if (params.max_query == 5) {
            if (metadata[24] / metadata[25] == 6) return native_launch<5, 6>(params, static_cast<hggcStream_t>(stream_pointer));
            if (metadata[24] / metadata[25] == 12) return native_launch<5, 12>(params, static_cast<hggcStream_t>(stream_pointer));
            if (metadata[24] / metadata[25] == 8) return native_launch<5, 8>(params, static_cast<hggcStream_t>(stream_pointer));
            return native_launch<5, 16>(params, static_cast<hggcStream_t>(stream_pointer));
        }
        if (metadata[24] / metadata[25] == 6) return native_launch<4, 6>(params, static_cast<hggcStream_t>(stream_pointer));
        if (metadata[24] / metadata[25] == 12) return native_launch<4, 12>(params, static_cast<hggcStream_t>(stream_pointer));
        if (metadata[24] / metadata[25] == 8) return native_launch<4, 8>(params, static_cast<hggcStream_t>(stream_pointer));
        return native_launch<4, 16>(params, static_cast<hggcStream_t>(stream_pointer));
    }
}

extern "C" int atrex_ppu_decode_resources(int* values) {
    hggcFuncAttributes attributes{};
    constexpr int Query = NativeQ1Only ? 1 : (NativeQ16Dual ? 16 : 4);
    constexpr int Threads = NativeGeometry<Query, 16>::Threads;
    auto function = atrex_ppu_native_decode_kernel<Query, 16, !NativeQ16Dual>;
    auto status = hggcFuncGetAttributes(&attributes, function);
    if (status != hggcSuccess) return static_cast<int>(status);
    int blocks = 0;
    status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, Threads, 0);
    values[0] = attributes.numRegs;
    values[1] = attributes.sharedSizeBytes;
    values[2] = attributes.localSizeBytes;
    values[3] = blocks;
    return static_cast<int>(status);
}
