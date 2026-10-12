#pragma once
#if DECODE_Q1_ONLY
using DecodeMainloop = flash::CollectiveMainloopFwdSm80<
    2, 2, true, cute::Shape<cute::Int<16>, cute::Int<32>, cute::Int<256>>,
    256, cutlass::float_e4m3_t, float, cutlass::arch::PPU0015,
    false, false, false, true, true, true, 32, false, true, true, cutlass::bfloat16_t>;
using DecodeEpilogue = flash::CollectiveEpilogueFwd<
    cute::Shape<cute::Int<16>, cute::Int<256>, cute::Int<32>>,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    cutlass::bfloat16_t, cutlass::arch::PPU0015, 64, true, true, true, false>;
using DecodeScheduler = flash::DynamicPersistentTileSchedulerSM80<true, true, true, 16>;
#else
using DecodeMainloop = flash::CollectiveMainloopFwdSm80<
    4, 2, true, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>,
    256, cutlass::float_e4m3_t, float, cutlass::arch::PPU0015,
    true, false, false, true, true, true, 32, false, true, true, cutlass::bfloat16_t>;
using DecodeEpilogue = flash::CollectiveEpilogueFwd<
    cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<32>>,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    cutlass::bfloat16_t, cutlass::arch::PPU0015, 128, true, true, true, false>;
using DecodeScheduler = flash::DynamicPersistentTileSchedulerSM80<true, true, true, 64>;
#endif
using DecodeKernel = flash::enable_sm89<flash::FlashAttnFwdSm80<DecodeMainloop, DecodeEpilogue, DecodeScheduler>>;

extern "C" int atrex_ppu_decode_resources(int* values) {
    auto function = cutlass::device_kernel<DecodeKernel>;
    hggcFuncAttributes attributes{};
    auto status = hggcFuncGetAttributes(&attributes, function);
    if (status != hggcSuccess) return static_cast<int>(status);
    int blocks = 0;
    status = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, function, DecodeKernel::NumThreads, DecodeKernel::SharedStorageSize);
    if (status != hggcSuccess) return static_cast<int>(status);
    values[0] = attributes.numRegs;
    values[1] = attributes.sharedSizeBytes + DecodeKernel::SharedStorageSize;
    values[2] = attributes.localSizeBytes;
    values[3] = blocks;
    return 0;
}
