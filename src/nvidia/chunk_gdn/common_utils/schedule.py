"""Persistent-work schedule shared by NVIDIA Chunk-GDN kernels."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute


@dataclass
class WorkDesc:
    # coord
    seq_idx: cutlass.Int32
    private_q_head_idx: cutlass.Int32
    private_v_head_idx: cutlass.Int32
    tok_offset: cutlass.Int32
    split_v_idx: cutlass.Int32
    v_row_offset: cutlass.Int32

    # shape
    seq_len: cutlass.Int32

    # update by mainloop
    tile_idx: cutlass.Int32

    def is_valid(self, num_seqs: cutlass.Int32):
        return self.seq_idx >= cutlass.Int32(0) and self.seq_idx < num_seqs

    def q_head_idx(self):
        # Plain field accessors are resolved directly while CuTeDSL traces the
        # caller; only accessors containing runtime DSL control flow need jit.
        return self.private_q_head_idx

    @staticmethod
    @cute.jit
    def is_gva(num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        return num_v_heads > num_q_heads

    @cute.jit
    def k_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        k_head_idx = self.private_v_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            k_head_idx = self.private_q_head_idx
        return k_head_idx

    def v_head_idx(self):
        return self.private_v_head_idx

    @cute.jit
    def o_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        o_head_idx = self.private_q_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            o_head_idx = self.private_v_head_idx
        return o_head_idx
