"""Validated tile layouts, launch descriptors and shared device helpers."""
from __future__ import annotations


import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import BFloat16, Float32, Int32

from ... import qsa_sm_ptx

from ...device import DeviceHelpers, _elem_pointer, _shfl as _shfl, _shfl32 as _shfl32

LOG2_E = 1.4426950408889634


class QSAForward(DeviceHelpers):
    SOFTMAX_WARPS = 8      # warps 0..7: two column-split softmax warpgroups
    MMA_WARP = 8           # warp 8: issues every tcgen05 MMA
    KLOAD_WARP = 9         # warp 9: K ring producer
    VLOAD_WARP = 10        # warp 10: V ring producer
    NWARPS = 12

    def __init__(
        self, group: int, head_dim: int, page_size: int, sel_width: int, bt_width: int, br: int,
        maxe: int, hkv: int=2, m_tile: int=128, n_tile: int=64, ent_tile: int=16,
        k_stages: int=4, v_stages: int=3, s_stages: int=2, use_pdl: bool=False,
        arrive_on: bool=False, pair_bar: bool=False, aohyb: bool=False,
        emask: bool=False, peel: int=2, gw: int=2, sum_tree: bool=False, rowidx: bool=False,
        sopt: int=0, spin: int=0, drain: int=0, rsplit: bool=False,
        mmasched: bool=False, qdepth: int=2, qpf: int=0, gpf: int=0, ordered: bool=False,
        kvmajor: bool=False, partial_bound: bool=False, single_bt: bool=False,
        compact_ws: bool=False,
    ):
        """Configure validated layouts; fixed policy flags remain device-visible attributes."""
        self.compact_ws = compact_ws
        assert not compact_ws or (single_bt and not rowidx)
        self.single_bt = single_bt
        self.ordered = ordered
        self.kvmajor = kvmajor
        self.partial_bound = partial_bound
        assert n_tile == 128 and ent_tile == 32 and (head_dim == 256)
        assert n_tile == 4 * ent_tile, 'one union entry expands to 4 tokens'
        assert ent_tile <= 32, 'one gather-warp lane resolves one union entry'
        assert n_tile * s_stages + head_dim <= 512, 'TMEM column overflow'
        assert k_stages >= 2 and v_stages >= 2, 'lagged arrival needs >=2 stages'
        smem_est = (k_stages + v_stages) * n_tile * head_dim + m_tile * head_dim + 2048 + 1024
        assert smem_est <= 232448, f'SMEM budget overflow: {smem_est}'
        self.group = group
        self.head_dim = head_dim
        self.page_size = page_size
        self.hkv = int(hkv)
        assert self.hkv >= 1, 'need at least one kv head'
        assert br * group <= m_tile, 'row block lanes must fit the M tile'
        self.sel_width = sel_width
        self.bt_width = bt_width
        self.br = br
        self.maxe = maxe
        self.m_tile = m_tile
        self.n_tile = n_tile
        self.ent_tile = ent_tile
        self.k_stages = k_stages
        self.v_stages = v_stages
        self.s_stages = s_stages
        self.use_pdl = bool(use_pdl)
        self.arrive_on = bool(arrive_on)
        self.aohyb = bool(aohyb)
        assert not (self.aohyb and self.arrive_on), 'QSA_AOHYB replaces the pure QSA_ARRIVEON form'
        self.pair_bar = bool(pair_bar)
        self.emask = bool(emask)
        _k = (br + 2) // 4 + 1
        self.peel = max(int(peel), _k // ent_tile + 1)
        self.sum_tree = bool(sum_tree)
        self.rowidx = bool(rowidx)
        self.sopt = int(sopt)
        self.spin = int(spin)
        self.gw = int(gw)
        assert self.gw in (2, 4), 'gather warps must be 2 or 4'
        assert int(qdepth) in (2, 3)
        self.qdepth = int(qdepth)
        assert int(qpf) in (0, 1, 2)
        assert int(gpf) in (0, 1)
        self.qpf = int(qpf)
        self.gpf = int(gpf)
        assert int(drain) in (0, 1, 2, 3, 4, 5, 6)
        self.drain = int(drain)
        self.rsplit = bool(rsplit)
        self.gwr = self.gw // 2
        self.mmasched = bool(mmasched)
        assert self.mmasched, 'QSA requires the folded scheduler'
        assert self.gw == 2, 'folded scheduling requires two gather warps'
        self.MMA_WARP = 8
        self.KLOAD_WARP = 9
        self.VLOAD_WARP = 10
        self.NWARPS = 12
        self.KLOAD_WARP2 = 11
        self.VLOAD_WARP2 = -1
        self.tmax = (maxe + 15) // 16
        self.RB = 2 * self.tmax
        self.RM = br * self.RB
        self.take_warps = 11 if self.mmasched else self.NWARPS - 1
        self.threads = 32 * self.NWARPS
        self.dtype = cutlass.Float8E4M3FN
        self.acc = Float32
        self.arch_str = 'sm_100'
        self.copy_elems = 16
        self.gptr = 8



    @cute.jit
    def _drain_split16(self,tOtO,wg,sg,L,row0,kvh,group,lane_rows,inv,mGate,mOut,sG=None,gstaged=None):
        ptr=cute.make_ptr(self.acc,(tOtO.iterator+(wg*16+(sg//32)*32)*65536).toint(),
            cute.AddressSpace.tmem,assumed_align=16)
        r=cute.make_rmem_tensor((32,),self.acc)
        rg=cute.make_rmem_tensor((32,),Float32 if self.gate_factor else BFloat16);rw=cute.make_rmem_tensor((32,),BFloat16)
        if cutlass.const_expr(self.decode_gate):self._mwait(gstaged,0)
        for chunk in cutlass.range_constexpr(4):
            qsa_sm_ptx.split_load(Int32((ptr+chunk*32).toint()),r,32,128)
            cute.arch.fence_view_async_tmem_load()
            if L<lane_rows:
                row=row0+L//group;head=kvh*group+L%group;col=(sg%32//16)*128+chunk*32
                gp=cute.make_ptr(BFloat16,_elem_pointer(mGate,(row,head,col)).toint(),cute.AddressSpace.gmem,assumed_align=32)
                op=cute.make_ptr(BFloat16,_elem_pointer(mOut,(row,head,col)).toint(),cute.AddressSpace.gmem,assumed_align=32)
                if cutlass.const_expr(self.decode_gate):
                    for part in cutlass.range_constexpr(4):
                        gate_warp = 0 if cutlass.const_expr(self.br==1) else sg//32
                        if cutlass.const_expr(self.gate_factor):
                            offset=(chunk*4+part)*(2*self.br*group*8)+((sg%32//16)*self.br*group+L)*8
                        else:
                            offset=gate_warp*4096+(chunk*4+part)*256+(sg%32)*8
                        src=cute.make_tensor(sG.iterator+offset,cute.make_layout((8,)))
                        dst=cute.make_tensor(rg.iterator+part*8,cute.make_layout((8,)))
                        cute.autovec_copy(src,dst)
                else:
                    cute.autovec_copy(cute.make_tensor(gp,cute.make_layout((32,))),rg)
                zero2=(Float32(0),Float32(0));half2=(Float32(.5),Float32(.5))
                for pair in cutlass.range_constexpr(16):
                    j=pair*2
                    o0,o1=cute.arch.fma_packed_f32x2((Float32(r[j]),Float32(r[j+1])),(inv,inv),zero2)
                    a0=Float32(cutlass.select_(inv!=Float32(0),o0,Float32(0))).to(BFloat16)
                    a1=Float32(cutlass.select_(inv!=Float32(0),o1,Float32(0))).to(BFloat16)
                    if cutlass.const_expr(self.gate_factor):
                        s0,s1=Float32(rg[j]),Float32(rg[j+1])
                    else:
                        g0,g1=cute.arch.fma_packed_f32x2((Float32(rg[j]),Float32(rg[j+1])),half2,zero2)
                        t0=qsa_sm_ptx.tanh_approx(g0);t1=qsa_sm_ptx.tanh_approx(g1)
                        s0,s1=cute.arch.fma_packed_f32x2((t0,t1),half2,half2)
                    y0,y1=cute.arch.fma_packed_f32x2((a0.to(Float32),a1.to(Float32)),(s0,s1),zero2)
                    rw[j]=y0.to(BFloat16);rw[j+1]=y1.to(BFloat16)
                cute.autovec_copy(rw,cute.make_tensor(op,cute.make_layout((32,))))
