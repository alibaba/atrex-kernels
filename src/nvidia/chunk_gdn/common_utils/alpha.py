"""Alpha processing helpers shared by NVIDIA Chunk-GDN kernels."""

import cutlass
import cutlass.cute as cute


# ─── AlphaProcessor ──────────────────────────────────────────────────────────
# Translates FlatMainloopTmaWarpSpecializedDeltaRule::AlphaProcessor.


class AlphaProcessor:
    CUMSUM_LOG = 0
    CUMPROD = 1
    # Channel 2 is intentionally reused by two mutually exclusive constexpr
    # paths: scaled cumulative products or negative end-relative reciprocals.
    CUMPROD_SCALE = 2
    CUMPROD_NEG_END_RCP = 2
    NUM_CHANNELS = 3

    @cute.jit
    def run(self, vecs: cute.Tensor, scale: cutlass.Float32, channel2_neg_end_rcp: cutlass.Constexpr = False):
        WARP_SIZE = 32
        blk_q = cute.size(vecs.shape[0])
        num_iters = blk_q // WARP_SIZE

        lane_id = cute.arch.lane_idx()

        # vecs shape (blk_q, NUM_CHANNELS) col-major.
        vecs_32 = cute.flat_divide(vecs, (WARP_SIZE,))

        frag = cute.make_rmem_tensor(num_iters, cutlass.Float32)
        for i in cutlass.range_constexpr(num_iters):
            raw = cutlass.Float32(vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG])
            # The public path supplies exp(log-gates), so alpha is nonnegative.
            # Clamp underflowed zeros before log2; 1e-10 is far below the
            # meaningful BF16 gate range and only prevents -inf propagation.
            alpha_floor = cutlass.Float32(1e-10)
            if raw < alpha_floor:
                raw = alpha_floor
            frag[i] = cute.math.log2(raw, fastmath=True)

        for log_off in cutlass.range_constexpr(5):  # offsets 1, 2, 4, 8, 16
            off = 1 << log_off
            for i in cutlass.range_constexpr(num_iters):
                v = cute.arch.shuffle_sync_up(frag[i], off, mask_and_clamp=0)
                if lane_id >= off:
                    frag[i] = frag[i] + v

        for i in cutlass.range_constexpr(1, num_iters):
            carry = cute.arch.shuffle_sync(frag[i - 1], 31)
            frag[i] = frag[i] + carry

        # This constexpr branch specializes the whole dependent path away when
        # false, so end_log is defined exactly in the variants that consume it.
        if cutlass.const_expr(channel2_neg_end_rcp):
            end_log = cute.arch.shuffle_sync(frag[num_iters - 1], 31)
        for i in cutlass.range_constexpr(num_iters):
            vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG] = frag[i]
            cumprod = cute.math.exp2(frag[i], fastmath=True)
            vecs_32[lane_id, i, AlphaProcessor.CUMPROD] = cumprod
            if cutlass.const_expr(channel2_neg_end_rcp):
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_NEG_END_RCP] = -cute.math.exp2(end_log - frag[i], fastmath=True)
            else:
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_SCALE] = cumprod * scale
