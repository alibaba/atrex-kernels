#pragma once

// Raw-E4M3 paged causal FlashAttention prefill for T-Head SAIL PPU (ppu0015).
//
// Q, K, and V arrive as unscaled E4M3. The prep path only rearranges K/V into
// the tc02 fragment layouts consumed by the attention mainloop; it performs no
// cast or dynamic quantisation. Output remains BF16, with FP32 accumulators and
// online-softmax state.
//
// Every causal row uses the E4M3 mainloop; the BF16-only short-row fallback is
// intentionally outside this raw-input ABI.
//
// FRAGMENT LAYOUTS -- measured, not assumed (one-hot + random-GEMM checks over
// all 32 lanes and all 1024
// (lq, la, tile, i), and end-to-end in workspace/v26/pv_probe.cu at max_rel 1.14e-7 with 0/4096
// accumulator mismatches against an exact int64 host contraction):
//      A fragment (lane L, reg r, sub-element e):  row = 2*(L>>2) + (r&1)
//                                                  k   = KPL*(L&3) + (KPL/2)*(r>>1) + e
//      B fragment (lane L, reg r, sub-element e):  row = ((L>>2)&1) + 2*(r>>1) + 4*(L>>3)
//                                                  k   = KPL*(L&3) + (KPL/2)*(r&1)  + e
//      accumulator (lane L, slot i):               m   = 2*(L>>2) + ((i>>1)&1)
//                                                  n   = 4*(L&3) + 2*(i>>2) + (i&1)
//      KPL = 4 for bf16 k16 and 8 for e4m3 k32.
// The accumulator map is BIT-IDENTICAL to the A-fragment map. P packs into the
// PV A operand with zero cross-lane movement at k32, but only under
// the FREE k RELABEL `kv_of_k(k) = (k&3) + 4*(k>>3) + 16*((k>>2)&1)`: at KPL = 8 a lane's eight k
// slots are 8*la + 4*(r>>1) + b, while its two QK tiles hand it 4*la + {0..3} twice, so the
// contraction index has to be permuted.  It is free because the SAME permutation applied to the B
// operand cancels it -- and B is built by atrex_ppu_prep, so the permutation costs nothing at run time.
// `slot_of_kv` below is that inverse.  Get it wrong and the MMA computes a permuted GEMM that still
// runs (v26's BMODE=1 control: every P code correct, 4080/4096 accumulator values wrong, rel_l2 1.002).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef unsigned int u32;
typedef unsigned short u16;

// v89 ARMS (-DAR=N at compile time). Two independent knobs plus one
// diagnostic; every arm except LAGA is numerically EXACT.
//
//   ONEB  ONE __syncthreads() per page instead of two, at CONSTANT smem footprint and CONSTANT
//         prefetch distance. The trailing barrier only ever covered the WAR hazard on the buffer the
//         NEXT iteration stages into; moving the STAGE below the leading barrier makes the leading
//         barrier cover it instead. TARGET.md 26k's A2 arm halved the barrier rate too, but it also
//         doubled the smem request and the commit group, and 26k says that pair was never
//         decomposed -- so the clean version had never been measured.  MEASURED -1.5%.
//   VPF   how many of the group's 16 V d-tiles have their two 8 B B-fragment halves loaded BEFORE
//         the softmax. The V address depends only on the page and `g`, never on P, so these loads
//         are legal anywhere after the barrier. 33's TSM queue is ~6 deep and 66c's census puts
//         tsmcnt(5) as the commonest wait form, and the whole softmax stretch -- mask, max tree,
//         redux, branch, phase A, 16 exp2, 16 cvt, 12 prmt -- currently has ZERO TSM loads in
//         flight. This is the load-GROUPING lever rather than the load-COUNT one (56f showed count
//         at constant bytes is exactly neutral). Costs 4 vregs per d-tile.  MEASURED -1.6% at VPF=2.
//   VPFE  hoist the VPF loads above the QK MMA loop instead of below it.
//   LAGA  DIAGNOSTIC, numerically wrong: PV consumes the PREVIOUS group's A. Every instruction and
//         every byte is unchanged -- only the exp2 -> cvt -> prmt -> prmt -> MMA dependency is cut.
//
//   AR:    0 control  1 ONEB  2 VPF2 only  3 ONEB+VPF2  4 ONEB+VPF4  5 LAGA
//          6 ONEB+VPF3  7 ONEB+VPF2 early  8 ONEB+VPF6  9 ONEB+VPF8 (SPILLS, do not use)
//          10 ONEB+VPF4 early  11 ONEB+VPF2 as four NAMED SCALARS  <-- SHIPPED, the default
//          12 ONEB+VPS4  13 ONEB+VPS3  14 ONEB+VPS2 (new spelling, consistency check on 11)
//          15 ONEB+VPS2 STAGGERED: d-tiles 0,1 above phase A and d-tiles 2,3 below the rescale
//             branch, so four loads sit in the chain at all times without four being in flight at
//             once -- separates "how deep" from "how many concurrently" (12/13 say depth alone loses)
//
// MEASURED, every arm n=14 with aggregate_valid_for_target=true.
//   0  control          17814.1 / 17817.3 / 17817.1  = 17816.2   (v82's own record: 17834.8)
//   11 SHIPPED          17258.1 / 17257.4 / 17257.5  = 17257.7   -> 1.0324x over the control
//   1  ONEB alone       17549.5     -1.50%   the barrier half is most of the first point
//   3  ONEB+VPF2 array  17426.2 / 17418.3 / 17434.8  = 17426.4   the SAME four loads, 0.97% worse
//   2  VPF2 without ONEB 17996.6    +1.02%   SLOWER -- the two only pay TOGETHER
//   13 ONEB+VPS3        17394.0     6 ONEB+VPF3 17559.4
//   12 ONEB+VPS4        17497.5     4 ONEB+VPF4 17438.0     8 ONEB+VPF6 18045.6
//   7  ONEB+VPF2 early  17429.3    10 ONEB+VPF4 early 17364.9   (above the QK MMAs: worse)
//   15 ONEB+VPS2 staggered 17632.1 / 17637.7            (splitting the 4 loads apart: worse)
//   5  LAGA diagnostic  18612.0     +4.5% -- the exp2 -> A -> MMA JOIN IS NOT THE COST. Cutting the
//      dependency at constant instructions and constant bytes makes the kernel SLOWER, so 64b's
//      residual 3.09 cyc convert+pack is not that serialisation either.
// So: depth 2 is a strict optimum, tighter is better than staggered, below the QK MMAs beats above
// them, and the prefetch is worthless until the trailing barrier is gone. 9 (VPF8) SPILLS 12 B.
#ifndef AR
#define AR 11
#endif

#define ONEB (AR >= 1 && AR != 2 && AR != 5)
#define LAGA (AR == 5)
#define VPF2S (AR == 11 || AR == 15)
#define VSTAG (AR == 15)
#if AR == 12
#define VPS 4
#elif AR == 13
#define VPS 3
#elif AR == 14
#define VPS 2
#else
#define VPS 0
#endif
#if AR == 2 || AR == 3 || AR == 7
#define VPF 2
#elif AR == 6
#define VPF 3
#elif AR == 4 || AR == 10
#define VPF 4
#elif AR == 8
#define VPF 6
#elif AR == 9
#define VPF 8
#else
#define VPF 0
#endif
#define VPFE (AR == 7 || AR == 10)

// ================================================================================ v93 ARMS (-DNA=n)
// v89 filled the load-free softmax stretch with FOUR V loads and depth 2 was a strict optimum, which
// TARGET.md 71 read as "4 in flight + ~2 outstanding == 33's ~6-deep tsmcnt ceiling, so that gap is
// saturated". Four independent families are tried on top, all numerically EXACT:
//
//   KPF   the NEXT GROUP's K fragment(s) prefetched into the same stretch. v89 only hoisted V. A K
//         fragment is one 16 B load = 4 vregs, and the next group's kv is different data so the load
//         is independent. Only group 0 has a successor inside the page (NKVG == 2); group 1's
//         successor is the next page, behind the barrier. Costs 4 vregs per fragment against 248/256.
//   SP2   FOUR page buffers staged TWO pages at a time, so __syncthreads() runs once per 128 kv
//         instead of once per 64. v89's ONEB (2 -> 1 barrier per page) was worth -1.50% and 71c says
//         "a barrier is not intrinsically expensive here; barrier-induced lockstep is", so halving the
//         rate again is the same lever applied twice. 4 x (16384 + 16384) = 128 KB/CTA, which 57 says
//         is exactly the budget that still keeps 2 CTAs/SM (2 x 128 KB = 256 KB). ABANDON if occupancy
//         drops to 1. Prefetch distance also grows 1 page -> 2, which 18 measured as WORSE at the old
//         barrier rate -- the two effects are deliberately NOT separable here, and SP2B separates them
//         by keeping 4 buffers and the 2-page commit while syncing every page.
//   VG    the V burst re-grouped: NDT/VG iterations of (2*VG loads, then VG MMAs) instead of 16
//         iterations of (2 loads, 1 MMA). Pure reordering, exact. v89's depth-2 optimum was found
//         under (load, load, MMA) triples; the regime changed when the barrier rate halved.
//   MIX   REGISTER-NEUTRAL control for KPF: one V d-tile (2 loads, 4 vregs) is given up to pay for one
//         K fragment (1 load, 4 vregs). Same 8 vregs, 3 loads in flight instead of 4. If KPF wins only
//         because of a fourth/fifth load rather than because K is the useful one, MIX loses.
//
//   NA:   0  control (== v89 AR=11, must reproduce 17257)
//         1  KPF1            2  KPF2            3  KPF1 LATE (below the rescale branch)
//         4  SP2             5  SP2 + KPF1      6  VG2             7  VG4
//         8  VG2 + KPF1      9  KPF2 LATE      10  MIX1           11  VG8
//        12  SP2B (4 buffers, 2-page commit, barrier EVERY page -- isolates depth from barrier rate)
//        13  SP2 + VG2      14  KPF1 + VG4     15  KPF1T1 (next group's t=1 fragment, not t=0)
//
// ROUND 1 MEASURED (n=14, valid, control 17257.3 which reproduces v89's 17256.9 to 0.002%):
//        1  KPF1   17448.3  +1.11%   4  SP2   17191.4  -0.38%
//        6  VG2    17403.9  +0.85%  10  MIX1  17585.0  +1.90%
// So the load-free stretch really is SATURATED at v89's depth 2: ADDING a K fragment costs 1.11%, and
// TRADING a V d-tile for one (MIX1, register-neutral, 3 loads instead of 4) costs 1.90%. Re-grouping
// the V burst into (4 loads, 2 MMAs) triples costs 0.85%. The ONLY winner is the extra page buffers.
//
// ROUND 2 -- everything below builds on SP2, the only lever that moved:
//   PGU2  the page loop UNROLLED BY 2. This is SP2's real structural payoff and round 1 did not take
//         it: under SP2 there is NO barrier between page 2k and page 2k+1, so page 2k's group 1 and
//         page 2k+1's group 0 are separated only by the loop back-edge. Unrolling removes it and lets
//         the same PV/QK overlap that 47's note credits for the intra-page unroll span the pair.
//         Risk: 47's regbisect found "the cost is the SECOND BODY ITSELF" -- watch numRegs.
//   VPD   the V-prefetch DEPTH re-swept under the halved barrier rate, since v89 tuned it at
//         2 barriers... 1 barrier per page and the regime has moved twice since. Reuses v89's own
//         VPS3/VPS4 named-scalar spellings and MIX1's depth-1 spelling so no new spelling is
//         introduced (51c/71: spelling alone is worth ~1%).
//
//   NA:  16  SP2 + PGU2     17  SP2 + VPD3     18  SP2 + VPD1     20  SP2 + VPD4
//        21  SP2 + PGU2 + VPD3                 22  SP2 + PGU2 + VPD1
//        23  SP2 + SPG4     26  SP2 + VPD5     27  SP2 + VPD6     28  SP2 + VPD4 + VG2
//        29  SP2C          30/31/32 SP2E + VPD5/4/2               33/34 SP2 + VPD5/6 EARLY
//
// ROUND 2 MEASURED (all n=14, valid; PQ=0 unless stated):
//        4  SP2            17192.8  -0.37%     <- the barrier rate halved
//       20  SP2+VPD4       17151.2  -0.61%     26 SP2+VPD5  17097.4  -0.92%   <-- new optimum
//       18  SP2+VPD1       17521.5  +1.53%     27 SP2+VPD6  17642.2  +2.23%
//       23  SP2+SPG4       17398.3  +0.82%     28 SP2+VPD4+VG2 17235.2 -0.13%
//       12  SP2B           18132.5  +5.07%  <- NOT a decomposition: `_lead` is true every iteration so
//                                              every page is staged TWICE. It prices DOUBLED HBM->smem
//                                              staging traffic at +5.07%, nothing else.
//       29  SP2C           17417.2  +0.93%  <- the REAL decomposition: 4 buffers, barrier every page,
//                                              ONE page staged per iteration at distance 2, constant
//                                              traffic. So distance 1 -> 2 costs +0.93% and halving the
//                                              barrier rate is worth about -1.30%; SP2 is their sum.
//       30  SP2E+VPD5      17138.2  (-0.69% w/ PQ=3, i.e. +0.8% vs SP2+VPD5) 31 +0.36%  32 -0.11%
//                                           <- issuing the pair's stage one page LATER (distance
//                                              {1,2} instead of {2,3}) LOSES, so the +0.93% of SP2C is
//                                              not a distance effect. Deeper staging is WANTED here.
//       16/21/22 PGU2      SPILL 68 B, not benched (47: "the cost is the SECOND BODY ITSELF")
//       33  SP2+VPD5 EARLY 17035.9 (w/ PQ=3, vs 17002.6 late)   34 SP2+VPD6 EARLY 17669.8
//                                           <- early placement is still worse at the new optimum, and
//                                              v89's depth-4 sign flip does not extend to depth 5/6.
//   PQ (atrex_ppu_prep):  1 fma 17103.4   2 unroll 17104.5   3 both 17060.4  (all on NA=20)
//                 3 on the UNMODIFIED atrex_ppu_attn (NA=0): 17165.9, atrex_ppu_prep 179.9 -> 88.0 us
//   SHIPPED: NA=26 PQ=3  ->  17002.6  = 1.0149x over v89
#ifndef NA
#define NA 26      // Raw-FP8 scale folding makes V-prefetch depth FIVE fit at 248/248 registers with
                   // zero spill. The BF16-input source previously spilled 12 B here after adding HDR,
                   // which is why v102 originally shipped NA=20 / depth FOUR.
#endif
#if NA == 1 || NA == 3 || NA == 5 || NA == 8 || NA == 10 || NA == 14 || NA == 15
#define KPF 1
#elif NA == 2 || NA == 9
#define KPF 2
#else
#define KPF 0
#endif
#define KPFL (NA == 3 || NA == 9)          // K prefetch placed BELOW the rescale branch
#define KPFT1 (NA == 15)                   // prefetch the successor group's t=1 chunk, not its t=0
#define MIX1 (NA == 10 || NA == 18 || NA == 22 || NA == 24)  // V prefetch depth 1 (NA=10 also adds the K frag)
#define SP2 (NA == 4 || NA == 5 || NA == 13 || (NA >= 16 && NA <= 28) || NA == 33 || NA == 34)
#define SP2B (NA == 12)
#define PGU2 (NA == 16 || NA == 21 || NA == 22)
// SPG4: the SUPER-PAGE. PGU2 duplicates the page body and SPILLS 68 B (measured); this reaches the
// SAME cross-page overlap with ONE page prologue by making the fully-unrolled GROUP loop span both
// pages of the pair -- 4 groups of 32 kv instead of 2. It is only expressible because SP2's four
// buffers put pages 2k and 2k+1 in ADJACENT buffers ((2k)&3 is 0 or 2), so the pair is CONTIGUOUS in
// smem and K's 4096-byte chunk stride walks across the boundary by itself. Only V needs the explicit
// +VPAGE_BYTES (its d-tile is the top index) and only `aa` needs the per-half kscale.
#define SPG4 (NA == 23 || NA == 24 || NA == 25)
#define GRPS (SPG4 ? 4 : NKVG)
#define PGSTEP2 (PGU2 || SPG4)
#define NBUF ((SP2 || SP2B || SP2C || SP2E) ? 4 : 2)
#if NA == 17 || NA == 21 || NA == 25
#undef VPS
#define VPS 3
#elif NA == 20 || NA == 28
#undef VPS
#define VPS 4
#elif NA == 26 || NA == 30 || NA == 33
#undef VPS
#define VPS 5
#elif NA == 34
#undef VPS
#define VPS 6
#elif NA == 31
#undef VPS
#define VPS 4
#elif NA == 27
#undef VPS
#define VPS 6
#endif
// SP2C: 4 buffers, barrier EVERY page, but exactly ONE page staged per iteration at distance 2. This
// is the CLEAN decomposition SP2B failed to be -- SP2B leaves `_lead` true every iteration and so
// stages every page TWICE, i.e. it doubles the HBM->smem traffic, and its +5.07% measures that and not
// the prefetch depth. cp_wait<1> is what makes distance 2 correct: two commits are outstanding and
// only the older one (the page being read now) has to have landed.
#define SP2C (NA == 29)
// SP2E: the barrier still runs once per PAGE PAIR, but the staging of the pair moves from the EVEN
// iteration to the ODD one. Both pages of a pair must land in the SAME commit group (there is no wait
// at the odd page), and that group may be issued any time after the PREVIOUS pair's barrier -- so the
// latest legal issue point is the odd iteration, which cuts the prefetch distance from {2,3} pages to
// {1,2}. SP2C prices one page of extra distance at +0.93%, so this is where that goes back.
#define SP2E (NA >= 30 && NA <= 32)
#define VPSE (NA == 33 || NA == 34)   // the V prefetch block ABOVE the QK MMAs instead of below
// v89's shipped four-named-scalar form yields to any arm that re-picks the depth, and VPBLK is the
// gate on the declaration block so the depth-1 and depth-3/4 arms still get pf0/pf1.
#undef VPF2S
#define VPF2S ((AR == 11 || AR == 15) && VPS == 0 && !MIX1)
#define VPBLK (VPF2S || VPS > 0 || MIX1)
#if NA == 6 || NA == 8 || NA == 13 || NA == 28
#define VG 2
#elif NA == 7 || NA == 14
#define VG 4
#elif NA == 11
#define VG 8
#else
#define VG 1
#endif

// ------------------------------------------------------------------- atrex_ppu_prep quantiser arms (-DPQ=n)
// atrex_ppu_prep is 1.04% of the target and runs at 22% of the HBM roofline, so ~0.8% of the whole gate is
// sitting in it. PQFMA hoists the per-(request, channel) reciprocal out of the page loop so the
// correctly-rounded divide -- whose 148-instruction slow path the ISA shows specialised INTO atrex_ppu_prep --
// collapses to one `fmaf` that also fuses the error-feedback add. PQUNR fully unrolls the 64-step
// quantise loop so `slot_of_kv(j)` is compile-time and `pack[s >> 2]` stops being a RUNTIME index into
// a 16-element local array. PQFMA is the only arm here that touches numerics, and only sub-ulp: it
// trades a correctly-rounded quotient for a reciprocal multiply that is then FUSED with the error
// feedback, against the E4M3 quantiser used by the production path.
#ifndef PQ
#define PQ 3       // SHIPPED: both atrex_ppu_prep quantiser fixes (hoisted reciprocal + full unroll)
#endif
#define PQFMA (PQ == 1 || PQ == 3)
#define PQUNR (PQ == 2 || PQ == 3)

// =========================================================== v103 ARMS: the v97 pool, ported (77h)
// v97 (TARGET.md 77) attacked the AUXILIARY kernels from the v96 base and took the non-atrex_ppu_attn pool
// from 336 to 187 us. v102 (76) attacked atrex_ppu_attn's page loop from the v89/v93 base. The two touch
// DISJOINT code, and 77h is the port list. v102 ALREADY has the atrex_ppu_prep full unroll (its PQUNR ==
// v97's PU=1) and ALREADY has the vamax memset deleted (v97's MZ), so neither is ported here. The
// rest are below, each independently switchable so the per-item coefficient can be re-measured on
// THIS base -- several were within v97's noise floor and are not guaranteed to survive.
//
// KS -- atrex_ppu_prep's K output store width. `*(u32*)dst = ..; *(u32*)(dst+4) = ..;` was NOT folded
// (v96/v102 binary: 16 x `vmem.st.b32` per page, not 8 x `vmem.st.b32x2`) because `dst` is an
// a byte pointer whose 8 B alignment the compiler cannot see. k_blk_off() is a multiple of 16 and
// khalf is 0 or 8, so the uint2 spelling is provably aligned. BIT-EXACT (same bytes, one instruction).
// v97: part of atrex_ppu_prep's 132.9 -> 83.6 together with the full unroll.  KS 0: v102 verbatim.
#ifndef KS
#define KS 1
#endif
// CF -- split-KV partial record format, and the ONE numerics change in this port set.
//   CF 0: v102 -- 272 fp32 = 1088 B, UNNORMALISED O_s.
//   CF 1: the slice's NORMALISED output O_s/l_s in fp16 (512 B) + (m_s, l_s) fp32 replicated over the
//         four `la` lanes (32 B), in a 544 B record. Exactly half. Storing RAW O_s in fp16 would
//         OVERFLOW (O_s ~ l_s*out ~ 2.3e5 against 65504) but O_s/l_s is a weighted average of V, so
//         |O_s/l_s| <= max|V| ~ 6 and is normally O(0.1). The merge is algebraically identical:
//             out = sum_s O_s w_s / sum_s l_s w_s = sum_s (O_s/l_s)(l_s w_s) / sum_s (l_s w_s)
//         so atrex_ppu_comb's per-slice weight becomes l_s*w_s and the denominator is its sum. An empty slice
//         publishes m_s = -inf / l_s = 0, so w_s = 0, so lw = 0 and its (zeroed, because inv == 0)
//         normalised O never reaches the sum -- no NaN.
//         544 and NOT the minimal 520: the record must be a whole number of 32 B sectors or 3 of
//         every 4 records start mid-sector and the epilogue's 32 B per-row store straddles two,
//         handing the whole saving back. 544 = 17 * 32, and the store address stays the single
//         expression `rec + 32*dt + 8*la` for every dt including the (m_s, l_s) one at dt == NDT.
//   v97 measured -102 us in atrex_ppu_attn<1> and -9.5 in atrex_ppu_comb (the byte count pays INSIDE the main kernel,
//   cf. 73d), at +1.0e-5 .. +1.4e-5 of rel_l2 and EXACTLY ZERO on the four nsplit == 1 shapes.
//   THE HAZARD ON THIS BASE (76b): it changes the epilogue's store shape from float4 to uint2, and
//   v102 ships at 248/248 with V-prefetch depth 4 where ONE extra statement at depth 5 spills 12 B.
//   Register census re-run: see 78. The runtime partial-record size must track PRECB.
#ifndef CF
#define CF 0
#endif
// CD -- atrex_ppu_comb bytes per thread. At CF 1 the record read per thread per slice falls from 16 B (one
// float4) to 8 B (one uint2) while the NUMBER of load instructions stays the same, so atrex_ppu_comb drops
// from 1.73 TB/s to 1.03 and half the byte saving evaporates: it is limited by outstanding requests,
// not by bytes. CD restores 16 B/lane by giving each thread more dims and putting one warp on one
// whole row.  v97: CD 0 105.2, CD 1 68.5, CD 2 74.5 us -- 8 dims is the optimum, 16 is worse.
//   CD 0: 4 dims/thread, 64 threads/row, CROWS 4   (v102's shape)
//   CD 1: 8 dims/thread, 32 threads/row, CROWS 8   -- one warp = one row, 16 B/lane
//   CD 2: 16 dims/thread, 16 threads/row, CROWS 16 -- 32 B/lane
#ifndef CD
#define CD 1
#endif
// SP -- atrex_ppu_short's shared-memory bank pattern. atrex_ppu_short is ~9 us of the mean but 48-57 us on the two
// kv_len == q_len shapes, where it is the only kernel that computes those rows. Its two hot shared
// reads are both bank-degenerate at a 256 u16 row stride (512 B = bank 0 for every row):
//   * QK: Ks[(cc + 8*jj)*STRIDE + d0], 16 B/lane, address depends only on cc -> a warp's 8 distinct
//     addresses all land in banks 0..3, so the 128 B/cycle port needs 8 phases instead of 1.
//   * PV: Vs[j*STRIDE + 32*cc + 8*u] -> 16*cc mod 32 is 0 or 16, i.e. 2 bank groups, 4 phases.
// SP 1 pads the row stride to 264 u16 (528 B, bank stride 4) AND re-maps the PV dim assignment from
// 32*cc (a contiguous 32) to 8*cc + 64*u, so both accesses cover all 32 banks exactly once. The
// values computed are IDENTICAL -- only which thread holds which dim, and where in shared the same
// element lives, change. v97: -1.6 us of the mean (sid 0: 56.9 -> 48.0), i.e. bank conflicts are NOT
// atrex_ppu_short's main cost; it runs 8 CTAs on 39 SMs and the grid is the wall.
#ifndef SP
#define SP 1
#endif
// VW -- atrex_ppu_prep's V shared-read width. The V quantise loop issues 64 x `tsm.ld.b16` per page per
// thread, i.e. 64 B per warp instruction against a 128 B port. The error feedback is serial along kv
// so the channel->thread map cannot change, but the STAGED LAYOUT can: VW 1 stores the page as 2048
// uint4 units, unit (ch, t) holding kv 8t..8t+7 of channel ch, so one thread reads its whole 64-kv
// column in EIGHT `tsm.ld.b32x4`.
//   * The unit index is swizzled, `ch*8 + (t ^ (ch&7) ^ ((ch>>3)&7))`. Without the XOR a warp's 32
//     lanes are 128 B apart and all land in the same 4 banks -- 32 phases for 512 B instead of 4.
//   * The staging global read stays one fully coalesced 512 B per warp instruction; the 8x8 u16
//     transpose it now needs is 32 `prmt` per thread per page.
// BIT-EXACT: the same u16 reaches the same thread in the same kv order, so `err`, `qv` and the packed
// words are identical (proved by v97/vqcheck.py, re-run for v103). v97: atrex_ppu_prep 84.16 -> 82.86, i.e.
// consistent in sign on all fourteen shapes but below the mean's noise floor -- optional.
#ifndef VW
#define VW 1
#endif

#define HEAD_DIM 256
#define NHEADS 8
#define PAGE 64
#define WARPS 8
#define NTHREADS (WARPS * 32)
#define MQ 32          // q rows per CTA q-tile (one warp PAIR)
#define MW 16          // q rows per WARP = 1 MMA row tile
#define DW 256         // output dims per warp -- the whole head
#define NDT (DW / 16)  // 16 MMA col tiles; acc is MW x DW FP32 = 128 vregs
#define KVC 16         // kv per QK step (one e4m3 m16n16k32 -> a 16-wide n tile)
#define KVG 32         // kv per PV step -- the E4M3 k32 MMA contracts 32 kv at once
#define NKVG (PAGE / KVG)
#define KSTEPS (HEAD_DIM / 32)  // e4m3 k-steps of 32

// ---------------------------------------------------------------------- softmax reference policy
// The online softmax is mathematically INVARIANT to the subtracted reference, so v19..v40 PINNED it,
// proved `S*aa - mm <= EXP_LIM` once per page from the e4m3 amax bound, and routed the pages that
// could not be proven safe to an exact slow lane.  v47 deletes that whole apparatus -- EXP_LIM, the
// per-page bound and the second lane -- because the reference is now also P's QUANTISATION scale, and
#define PVT 256
#define L255 8.f
#define VQ_STEP (1.f / (float)PVT)
#define SHORT_T 64              // vscalar.py's T: q-blocks seeing fewer keys than this go to atrex_ppu_short

// ======================================================================== v63 ARMS (-D selected)
// TARGET.md 26/35b/39/42 plus workspace/v58/itemise.py localise the whole remaining intercept gap in
// the page loop's VALU: 38.75/chunk on v40 against 70.00 on v60, and +35 x 0.274 = 9.6 cyc IS the
// 8.7 cyc intercept gap. Three of the four named items are attacked here, each independently
// switchable so the per-item coefficient can be read off one arm at a time. 0 == v60 EXACTLY.
//
//   V63_MASK  the causal mask (13.50/chunk -> ~1). v40's fast lane paid nothing because it was a
//             separate mask-free body; v47 merged the lanes to fit 248 registers (see the ONE LANE
//             note below -- that argument is measured and still binding, so the second body does NOT
//             come back). Instead the exact 8 cmp + 8 csel go behind `if (mlim < 3)`, which this ISA
//             compiles to `s.lop.emsk` + `s.cbr.az` -- a real skip. `mlim`'s lane-varying part is
//             2*lq - 4*la in [-12, 14] and the chunk stride is 16 kv, so at most TWO chunks of a
//             whole CTA sweep are divergent; every other chunk is warp-uniformly all-visible (skip)
//             or all-masked (run once). NUMERICALLY IDENTICAL -- the fmaf is hoisted out and the
//             select becomes an overwrite, which is the same value.
//   V63_MQU   one warp-uniform reference instead of 16 (group max 12.00/chunk unchanged, but the
//             RESCALE block's firing rate falls ~16x). 42 measured the block as already branch-
//             skipped by `s.cbr.az emsk`, but the guard is warp-WIDE over 8 (lp,lb) lane groups x 2
//             `pr` = 16 independent monotone maxima, so it fires whenever any of 16 rises: 12.8% of
//             groups at kv=15360 == 7.4 cyc/chunk. One maximum makes that ~0.5. Costs 3 more
//             shuffles per group and shares P's quantisation scale across 16 q rows -- the only arm
//             here that moves `rel_l2`, so it is validated on its own.
//   V63_RDX   the cross-lane max in ONE instruction. `v.red.max.u32` exists in the ISA (it is
//             `redux.sync`, i.e. `__reduce_max_sync`) and replaces 2 (or 5) shuffle+fmax pairs.
//             Integer redux on a FLOAT is legal here because the value is clamped at 0 first: for
//             x >= 0 the IEEE-754 encoding is monotone in x, and the clamp is a numerical NO-OP --
//             a group whose max is below the reference gives nm = fmaxf(mq, mq - log2(255)) = mq,
//             i.e. the reference does not move, which is what an all-below-reference group must do.
#ifndef V63_MASK
#define V63_MASK 1   // v68: SHIPPED ON (v63 arm m1q1r1 measured 18300.4 on v60's base)
#endif
#ifndef V63_MQU
#define V63_MQU 1   // v68: SHIPPED ON (v63 arm m1q1r1 measured 18300.4 on v60's base)
#endif
#ifndef V63_RDX
#define V63_RDX 1   // v68: SHIPPED ON (v63 arm m1q1r1 measured 18300.4 on v60's base)
#endif

// Split-KV partial record: 256 fp32 O values + m_s + l_s + 14 floats of pad. The pad exists so the
// record is a whole number of 16 B lanes (272*4 = 68*16) AND so the m/l pair can ride in the same
// float4 store stream as O -- see the register-cliff note in the epilogue.
#define PREC 272
// CF 1 halves it: 256 fp16 (NORMALISED) O + 4 x (m_s, l_s) fp32 = 512 + 32 = 544 B = 272 HALVES.
// The (m_s, l_s) pair is replicated across the four `la` lanes -- they hold bit-identical values by
// construction (mq is warp-uniform under V63_MQU, and ls is a commutative butterfly sum) -- so every
// lane writes its own 8 B slot, the warp's 32 B per row is one whole sector, and atrex_ppu_comb reads la 0's.
// runtime.py's `_PARTIAL_FLOATS_PER_TOKEN_HEAD` is PRECB / sizeof(float) and must track it.
#if CF
#define PRECB 544
#else
#define PRECB (PREC * 4)
#endif

#define KPAGE_BYTES 16384  // 64 kv x 256 d, e4m3, swizzled
#define VPAGE_BYTES 16384  // 256 d x 64 kv, E4M3 bytes, swizzled

// ---------------------------------------------------------------------------------------- swizzles
// Both are "one 16 B TSM block == exactly one lane's B fragment half-pair", so every B fragment is a
// single conflict-free ld.shared.b128 and the four registers come out in natural uint4 order.
//
// K (e4m3): block = kv pair {kv, kv+2} x 8 d, indexed (KP = (kv>>2)*2 + (kv&1), Bd = d>>3);
//   byte = ((kv>>1)&1)*8 + (d&7). XOR by (KP&1)*4 spreads the 8 lanes of one bank phase (which
//   straddle KP even / KP odd) over all 32 banks instead of colliding 2-way.
__device__ __forceinline__ int k_blk_off(int KP, int Bd) {
  return ((KP * 32) + (Bd ^ ((KP & 1) * 4))) * 16;
}
__device__ __forceinline__ int k_ldmatrix_tile_halfword(int row, int dimension) {
  const int row_group = 2 * (row >> 2) + (row & 1);
  int index = 72 * row_group + 2 * (dimension >> 3) + ((dimension >> 1) & 1);
  if ((dimension >> 2) & 1) index += (row_group & 1) ? -8 : 8;
  if ((row >> 1) & 1) index += 512;
  return index;
}
__device__ __forceinline__ int k_ldmatrix_page_halfword(int row, int dimension) {
  const int cube = 2 * (row >> 5) + (dimension >> 7);
  const int row_half = (row >> 4) & 1;
  const int column_tile = (dimension >> 5) & 3;
  const int local = k_ldmatrix_tile_halfword(row & 15, dimension & 31);
  return cube * 2048 + row_half * 1024 + (local ^ (column_tile * 16));
}
__device__ __forceinline__ void load_k_ldmatrix(u32 (&fragment)[4], u32 block_address) {
  const int address = static_cast<int>(block_address);
  const int lbo = 64;
  const int sbo = 1;
  const int mode = 0;
  asm volatile(
      "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "
      "{%0,%1,%2,%3}, [%4], %5, %6, %7;"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "l"(address), "r"(lbo), "r"(sbo), "r"(mode));
}
// V^T (E4M3 bytes), split-load form. An earlier layout used one 16 B block
//   v8_off(DP, KB) = ((DP*8) + (KB ^ ((DP&1)*4))) * 16,   half = ((d>>1)&1)*8
// and read it with a single b32x4. That is the load profile v53 measured as a TRAP: at IDENTICAL
// bytes, 16 wide loads cost +2.55 cyc per 16-kv warp-chunk against 24 narrow ones (the v44/v44b
// pair), and only ~12-13% of MMA issue is exposed, so the k32 MMA saving does NOT pay for a wider
// load. v53c kept the k32 MMA and REFUSED the wide load by splitting the B fragment into its two 8 B
// halves -- B regs 0,1 (row d) and B regs 2,3 (row d+2) -- and measured 1.0495x. This is that change.
//
// The separation has to be in the LAYOUT, not in the load intrinsic: v44b already proved the
// vectoriser folds two ADJACENT uint2 straight back into one b32x4. So the half bit moves out of the
// block (v47 had it at byte 8, inside the 16 B unit) and up to 512 B:
//   unit (half in 0..1, DP = 8*dt + dp in 0..127, KB = slot>>3 in 0..7), 8 B each
//     byte = 1024*dt + 512*half + 64*dp + 8*(KB ^ xor) + e     -> 2*16*2*8*8*8 = 16384 = VPAGE_BYTES
//     half 0 = row d, B regs 0,1     half 1 = row d+2, B regs 2,3     (both over the same 8 slots)
// so bit 9 is the half, bits 6..8 are dp, bits 3..5 are the swizzled KB and bits 0..2 the byte.
//
// TWO THINGS HAD TO BE RE-DERIVED, and both are checked element by element in workspace/v60/layout.py.
//
// (1) THE XOR FIELD. 32 banks x 4 B = 128 B, so a 16 B access phase is 8 lanes but an 8 B access
// phase is SIXTEEN. A unit index s lands on banks (2s)%32 and (2s+1)%32, so the 16 lanes of a phase
// need 16 distinct values of s%16. Within a phase lp&1, lb and la all vary. Carrying v47's
// (DP&1)*4 = lb*4 gives s%16 = 8*lb + 4*(sc^lb) + la, INDEPENDENT of lp&1 -- only 8 distinct values,
// a 2-way conflict on every V load (layout.py's CONTROL check confirms it: 16 of 32 banks). XOR-ing by
// 4*((DP&1) ^ ((DP>>1)&1)) = 4*(lb ^ (lp&1)) gives s%16 = 8*lb + 4*(sc ^ lb ^ (lp&1)) + la, a
// bijection on the 16 lanes. Both bits come from dp and 8*dt contributes nothing to DP bits 0..2, so
// the term is still INDEPENDENT of the d-tile and stays loop-invariant in the dt loop.
//
// (2) THE TSM 512 B STRIDE WINDOW (TARGET.md 36b). TSM sub-array select uses address bits in the
// 512 B - 4 KB window, and a sibling experiment that moved a warp's 8 lane groups to a 2048 B stride
// paid +1.85 cyc per load -- roughly DOUBLE -- while removing 19 instructions. Here the lane-varying
// bits of one load phase are exactly bits 3..8: the 8 lane groups span 480 B, INSIDE the window's
// floor. Note v47's own V phase spanned 1024 B (lane-varying up to bit 9), so the split does not
// merely avoid the penalty, it takes v47's lane variation OUT of the sub-array select bits.
// This is also why the half sits at 512 B and not at 8192 B as first tried: same bank residue either
// way (64*half and 1024*dt are both multiples of 16 units), but the whole d-tile sweep then fits in a
// far smaller signed offset window, which is what a 248-register fit needs.
__device__ __forceinline__ int v_unit_off(int half, int DP, int KB) {
  return ((128 * (DP >> 3)) + (64 * half) + (8 * (DP & 7))
          + (KB ^ (((DP & 1) ^ ((DP >> 1) & 1)) * 4))) * 8;
}
__device__ __forceinline__ int v_ldmatrix_tile_halfword(int row, int dimension) {
  const int row_group = 2 * (row >> 2) + (row & 1);
  int index = 32 * row_group + 8 * (row_group >> 1);
  index += 2 * (dimension >> 3) + ((dimension >> 1) & 1);
  if ((dimension >> 2) & 1) index += ((row_group >> 1) & 1) ? -8 : 8;
  if ((row >> 1) & 1) index += 512;
  return index;
}
__device__ __forceinline__ int v_ldmatrix_page_halfword(int row, int dimension) {
  const int row_tile = row >> 4;
  const int column_tile = dimension >> 5;
  const int local = v_ldmatrix_tile_halfword(row & 15, dimension & 31);
  return (row_tile >> 1) * 1024 + (row_tile & 1) * 256 + (local ^ (column_tile * 16));
}
__device__ __forceinline__ void load_v_ldmatrix(u32 (&fragment)[4], u32 block_address) {
  const int address = static_cast<int>(block_address);
  const int lbo = 64;
  const int sbo = 1;
  const int mode = 1;
  asm volatile(
      "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "
      "{%0,%1,%2,%3}, [%4], %5, %6, %7;"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "l"(address), "r"(lbo), "r"(sbo), "r"(mode));
}
__device__ __forceinline__ void store_v_ldmatrix_row(
    unsigned char* destination, int row, const u32 (&pack)[16]) {
  const int local_row = row & 15;
  const int row_quarter = local_row >> 2;
  int base = (row >> 5) * 1024 + ((row >> 4) & 1) * 256;
  base += 64 * row_quarter + 32 * (local_row & 1) + 512 * ((local_row >> 1) & 1);
#pragma unroll
  for (int chunk = 0; chunk < 8; ++chunk) {
    int first_word = 8 * (chunk >> 2) + 4 * (chunk & 1) + ((chunk >> 1) & 1);
    first_word ^= (row_quarter & 1) | ((row_quarter & 2) << 2);
    *reinterpret_cast<uint2*>(destination + 2 * (base + 4 * chunk)) =
        make_uint2(pack[first_word], pack[first_word + 2]);
  }
}
// The free k relabel, inverted: which of the page's 64 fragment slots carries true kv?
// k = 8*la + 4*(r>>1) + b  <->  kv_rel = b + 4*la + 16*(r>>1), so slot = 8*la + 4*tile + b.
__device__ __forceinline__ int slot_of_kv(int kv) {
  const int g = kv >> 5, l = kv & 31;
  return 32 * g + 8 * ((l >> 2) & 3) + 4 * (l >> 4) + (l & 3);
}

// ------------------------------------------------------------------------------------------- MMA
// `.row.col` must stay immediately after the shape; the trailing 1,1,1 is tc02's negation triple.
#define MMA_E4M3(d, a, b)                                                     \
  asm volatile(                                                               \
      "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.f32.e4m3.e4m3.f32 "         \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "        \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, 1, 1, 1;"                                    \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),                        \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])                         \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                            \
        "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]))

#define MMA_PV(d, a, b) MMA_E4M3(d, a, b)
#define ACCF(x) (x)
#define ACC_STORE(dst, v) ((dst) = (v))

// ------------------------------------------------------------------------------------- primitives
__device__ __forceinline__ float bf_lo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }

__device__ __forceinline__ u16 cvt_e4m3x2(float hi, float lo) {
  u16 r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 cvt_bf16x2(float hi, float lo) {
  u32 r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// fp16, for CF 1's partial record. Same shape as cvt_bf16x2: the LOW half of the result is `lo`.
// Written as asm rather than __float2half so the emitted instruction is CHECKED, not assumed --
// TARGET.md's standing rule after several "parsed then silently dropped" constructs.
// Round a POSITIVE float onto e4m3's 3-bit mantissa, round-to-nearest-even, by truncating the
// low 20 bits of the f32 significand. The point is that the value summed into `ll` is EXACTLY
// the value the MMA multiplies: `p` is otherwise rounded on the numerator side only, and
// e4m3's 6.25% relative step turns that into a systematic gain error on the output.
// Subnormal e4m3 inputs are not modelled here -- they are probabilities below 2^-9 of the row
// max and contribute nothing to either sum.
__device__ __forceinline__ float e4m3_grid(float p) {
  const u32 b = __float_as_uint(p);
  const u32 add = 0x0007FFFFu + ((b >> 20) & 1u);
  return __uint_as_float((b + add) & 0xFFF00000u);
}

__device__ __forceinline__ u32 cvt_f16x2(float hi, float lo) {
  u32 r;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float h2f(u16 h) {
  float r;
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
  return r;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 r;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
// exp2 -- USE THE INTRINSIC, NOT INLINE ASM. Measured on this box (workspace/vex, 8 independent
// exp2 per kernel, ISA counted with hgobjdump):
//     ex2.approx.ftz.f32 asm   90 instr : exp2 + abs + cmp.lt + and + csel   (4-op tail)
//     ex2.approx.f32     asm  100 instr : exp2 + cmp.lt + 2 csel + 2 mul     (5-op fixup)
//     exp2f()                 100 instr : same fixup
//     __exp2f()                58 instr : BARE v.exp2.f32, nothing else       <-- this one
// Both asm forms are worse than they look: the fixup writes and reads the SINGLE `vcc` predicate
// register, so N exp2 in a row become a STRICTLY SERIAL chain through vcc, each one exposing the
// full SFU latency. __exp2f() emits the naked instruction, so 16 of them pipeline.
__device__ __forceinline__ float ex2(float x) { return __exp2f(x); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                   static_cast<u32>(__cvta_generic_to_shared(dst))),
               "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// Pack four u8 codes (each already in byte 0 of its own register) into one A-fragment word:
// 3 prmt, versus 3 shifts + 3 ors for the arithmetic spelling.
__device__ __forceinline__ u32 pack4(u32 b0, u32 b1, u32 b2, u32 b3) {
  return prmt(prmt(b0, b1, 0x0040u), prmt(b2, b3, 0x0040u), 0x5410u);
}
