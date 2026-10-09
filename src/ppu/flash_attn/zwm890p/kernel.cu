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

// ==================================================================================== KERNEL 0
// GLOBAL per-(request, channel) amax of V.  This pass exists so that `v_scale` does NOT depend on the
// page, so the same V scale is valid across the full KV sweep.
// It is a pure streaming read of V -- one extra pass over the value cache.  atrex_ppu_prep is 0.47% of op time
// (114 us of 24407 on the 14-shape sweep), so this is ~0.2%, against the 8 loads/chunk it removes from
// the hot loop.
// The amax lands via atomicMax on the raw bit pattern: for x >= 0 the IEEE-754 encoding is monotone in
// x, so an unsigned integer max IS a float max.  `vamax` is memset to 0 by the launcher first (0.0f is
// the identity), which also makes the pass idempotent under the harness's repeated calls.
__global__ __launch_bounds__(NTHREADS) void atrex_ppu_vamax(
    const __nv_bfloat16* __restrict__ value_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ vamax, int bt_stride, int max_pages) {
  __shared__ float part[WARPS][HEAD_DIM];
  const int req = blockIdx.y;
  const int kv_len = seq_lens[req];
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int tid = threadIdx.x, wrp = tid >> 5, lch = tid & 31;

  // thread `tid` reads uint4s whose channel offset is always 8*lch, so it owns channels 8*lch+0..7
  // and sweeps kv = 8*c + wrp for c = 0..7 -- 512 B per warp per kv row, fully coalesced.
  float lam[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) lam[j] = 0.f;
  for (int p = blockIdx.x; p < n_pages; p += gridDim.x) {
    const int pid = block_table[req * bt_stride + p];
    const __nv_bfloat16* vp = value_cache + (size_t)pid * PAGE * HEAD_DIM;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 t = *reinterpret_cast<const uint4*>(vp + (c * NTHREADS + tid) * 8);
      const u32* w = reinterpret_cast<const u32*>(&t);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        lam[2 * j] = fmaxf(lam[2 * j], fabsf(bf_lo(w[j])));
        lam[2 * j + 1] = fmaxf(lam[2 * j + 1], fabsf(bf_hi(w[j])));
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) part[wrp][8 * lch + j] = lam[j];
  __syncthreads();
  float am = part[0][tid];
#pragma unroll
  for (int w = 1; w < WARPS; ++w) am = fmaxf(am, part[w][tid]);
  if (am > 0.f)
    atomicMax(reinterpret_cast<unsigned int*>(&vamax[(size_t)req * HEAD_DIM + tid]),
              __float_as_uint(am));
}

// ==================================================================================== KERNEL 1
// Per KV page: quantise K to E4M3 with a per-page amax and quantise+transpose V to
// E4M3 with the global per-channel scale from atrex_ppu_vamax, both written straight into the TSM fragment
// layout atrex_ppu_attn consumes, so the main loop's staging is a plain contiguous cp.async.
//
// The V path is workspace/v23's validated transpose (bit-exact against pvquant/vgroup.py's `shape_v`
// on 4 arms x 3 seeds x 40 pages) with the per-page amax replaced by the global one.  Two details are
// load-bearing and were each a measured bug:
//   * The former integer arm used order-1 noise shaping. E4M3 must not use it, so ``err`` remains zero.
//   * `rintf`, i.e. round-half-to-EVEN, matching torch.round.  `roundf` is half-AWAY-from-zero and
//     disagrees on ~half of the reachable ties (v26's RMODE=1 control: 4066/4096 accumulator values
//     wrong on a tie sweep, yet 0 wrong on gaussian data -- the trap is silent).
__global__ __launch_bounds__(NTHREADS) void atrex_ppu_prep(
    const __nv_bfloat16* __restrict__ key_cache, const __nv_bfloat16* __restrict__ value_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    unsigned char* __restrict__ k_fp8, unsigned char* __restrict__ v_fp8,
    float* __restrict__ kscale,
    const float* __restrict__ vamax, int bt_stride, int max_pages) {
  const int r = blockIdx.y;
  const int kv_len = seq_lens[r];
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int tid = threadIdx.x;
  __shared__ float red[WARPS + 1];
  __shared__ u16 Vs[PAGE * HEAD_DIM];

  // Channel `tid` of V belongs to fragment column pair DP with fragment HALF `half`; both are
  // constants. v47 had `half` as a byte offset (0 or 8) inside a 16 B block; in v60 the half is a
  // 0/1 index that `v_unit_off` places at 512 B, so the two halves are separate 8 B loads.
  // reciprocal-multiply, NOT __fdiv_rn: the studies write `amax / 127.0`, a tensor / PYTHON SCALAR,
  // which torch lowers to a multiply by the reciprocal.
  const float vsc = fmaxf(vamax[(size_t)r * HEAD_DIM + tid] * VQ_STEP, 1e-30f);
#if PQFMA
  // ONE divide per thread for the whole kernel, instead of one per V element. `vsc` depends on the
  // request and the channel only, and a thread owns exactly one channel of one request.
  const float rvsc = 1.0f / vsc;
#endif

  for (int p = blockIdx.x; p < n_pages; p += gridDim.x) {
    const int pid = block_table[r * bt_stride + p];
    const __nv_bfloat16* kp = key_cache + (size_t)pid * PAGE * HEAD_DIM;
    const __nv_bfloat16* vp = value_cache + (size_t)pid * PAGE * HEAD_DIM;
    unsigned char* kdst = k_fp8 + ((size_t)r * max_pages + p) * KPAGE_BYTES;
    unsigned char* vdst = v_fp8 + ((size_t)r * max_pages + p) * VPAGE_BYTES;

    // ---- K pass 1: page amax ----------------------------------------------------------------
    const int kv = tid >> 2, ka = tid & 3;
    float amax = 0.f;
    uint4 kreg[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      kreg[j] = *reinterpret_cast<const uint4*>(kp + kv * HEAD_DIM + ka * 8 + j * 32);
      const u32* w = reinterpret_cast<const u32*>(&kreg[j]);
#pragma unroll
      for (int t = 0; t < 4; ++t)
        amax = fmaxf(amax, fmaxf(fabsf(bf_lo(w[t])), fabsf(bf_hi(w[t]))));
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
    if ((tid & 31) == 0) red[tid >> 5] = amax;
    __syncthreads();
    if (tid == 0) {
      float m = red[0];
#pragma unroll
      for (int i = 1; i < WARPS; ++i) m = fmaxf(m, red[i]);
      m = (m > 0.f) ? m : 1.f;
      red[WARPS] = m;
      kscale[(size_t)r * max_pages + p] = m * (1.f / 448.f);
    }
    __syncthreads();
    const float qmul = 448.f / red[WARPS];

    // ---- K pass 2: quantise + scatter into the fragment layout ------------------------------
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const u32* w = reinterpret_cast<const u32*>(&kreg[j]);
      u16 h[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) h[t] = cvt_e4m3x2(bf_hi(w[t]) * qmul, bf_lo(w[t]) * qmul);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int dimension = 32 * j + 8 * ka + 2 * pair;
        reinterpret_cast<u16*>(kdst)[k_ldmatrix_page_halfword(kv, dimension)] = h[pair];
      }
    }

    // ---- V: stage the page, then quantise column `tid` down kv with error feedback -----------
    __syncthreads();
#if VW
    // VW 1: unit (ch, t) = kv 8t..8t+7 of channel ch, at uint4 index ch*8 + (t ^ (ch&7) ^ ((ch>>3)&7)).
    // Thread (t8 = tid>>5, c8 = tid&31) owns kv 8*t8..8*t8+7 for channels 8*c8..8*c8+7: eight
    // coalesced 512 B/warp global reads in, an 8x8 u16 transpose (one prmt per output word), eight
    // swizzled uint4 out. Both the store here and the read below hit 8 distinct four-bank groups with
    // 4 lanes each -- 4 port phases for 512 B, which is the floor.
    {
      uint4* Vu = reinterpret_cast<uint4*>(Vs);
      const int t8 = tid >> 5, c8 = tid & 31;
      u32 iw[32];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const uint4 g = *reinterpret_cast<const uint4*>(vp + (8 * t8 + u) * HEAD_DIM + 8 * c8);
        iw[4 * u + 0] = g.x;
        iw[4 * u + 1] = g.y;
        iw[4 * u + 2] = g.z;
        iw[4 * u + 3] = g.w;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // word k of channel i = { u16 i of in[2k]  (kv 8t+2k, LOW),
        //                         u16 i of in[2k+1](kv 8t+2k+1, HIGH) } -- one prmt.
        const u32 sel = (i & 1) ? 0x7632u : 0x5410u;
        const int h = i >> 1;
        uint4 o;
        o.x = prmt(iw[0 + h], iw[4 + h], sel);
        o.y = prmt(iw[8 + h], iw[12 + h], sel);
        o.z = prmt(iw[16 + h], iw[20 + h], sel);
        o.w = prmt(iw[24 + h], iw[28 + h], sel);
        const int ch = 8 * c8 + i;
        Vu[ch * 8 + (t8 ^ (ch & 7) ^ ((ch >> 3) & 7))] = o;
      }
    }
#else
    // The staging read is contiguous; the shared read below is Vs[j*256 + tid], i.e. 256 adjacent
    // u16 across the CTA -- conflict-free, but only 64 B per warp instruction.
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int e = (c * NTHREADS + tid) * 8;
      *reinterpret_cast<uint4*>(Vs + e) = *reinterpret_cast<const uint4*>(vp + e);
    }
#endif
    __syncthreads();

    u32 pack[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) pack[j] = 0u;
    float err = 0.f;
#if VW
    // Eight wide reads instead of 64 narrow ones. kv order is preserved EXACTLY: unit t8 holds
    // kv 8*t8..8*t8+7, word k holds kv 8*t8+2k in its LOW u16 and 8*t8+2k+1 in its HIGH u16, which is
    // how the staging prmt built it. Fully unrolled, so every `slot_of_kv` is still a literal.
    // (`rvsc` is hoisted above the page loop on this base -- see PQFMA.)
    {
      const uint4* Vu = reinterpret_cast<const uint4*>(Vs) + tid * 8;
      const int xk = (tid & 7) ^ ((tid >> 3) & 7);
#pragma unroll
      for (int t8 = 0; t8 < 8; ++t8) {
        const uint4 g = Vu[t8 ^ xk];
        u32 gw[4];
        gw[0] = g.x;
        gw[1] = g.y;
        gw[2] = g.z;
        gw[3] = g.w;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int j = 8 * t8 + 2 * k + hh;
            const float v = __uint_as_float(hh ? (gw[k] & 0xFFFF0000u) : (gw[k] << 16));
#if PQFMA
            const float tgt = fmaf(v, rvsc, err);
#else
            const float tgt = __fdiv_rn(v, vsc) + err;
#endif
            // E4M3 is a relative quantiser; `err` stays zero and `tgt` is a
            // pure scaled value.
            const u32 code = (u32)(cvt_e4m3x2(0.f, tgt) & 0xFFu);
            err = 0.f;
            const int s = slot_of_kv(j);
            pack[s >> 2] |= code << (8 * (s & 3));
          }
        }
      }
    }
#else
#if PQUNR
    // FULL unroll: `j` compile-time makes `slot_of_kv(j)` a constant, so `pack[s >> 2]` is a named
    // register and `8 * (s & 3)` an immediate. At `unroll 4` the compiler knows j only modulo 4, so
    // the index is dynamic and the 16-word array cannot stay in registers.
#pragma unroll
#else
#pragma unroll 4
#endif
    for (int j = 0; j < PAGE; ++j) {
      const float v = __uint_as_float((u32)Vs[j * HEAD_DIM + tid] << 16);
#if PQFMA
      const float tgt = fmaf(v, rvsc, err);
#else
      const float tgt = __fdiv_rn(v, vsc) + err;  // tensor/tensor IS correctly rounded
#endif
      // E4M3 is a relative quantiser; `err` stays zero and `tgt` is a pure
      // scaled value.
      const u32 code = (u32)(cvt_e4m3x2(0.f, tgt) & 0xFFu);
      err = 0.f;
      const int s = slot_of_kv(j);
      pack[s >> 2] |= code << (8 * (s & 3));
    }
#endif  // VW
#pragma unroll
    for (int pair = 0; pair < 32; ++pair) {
      const u16 value = static_cast<u16>(pack[pair >> 1] >> (16 * (pair & 1)));
      reinterpret_cast<u16*>(vdst)[v_ldmatrix_page_halfword(tid, 2 * pair)] = value;
    }
    __syncthreads();
  }
}

// Raw-FP8 prep performs layout conversion only. Identity K/V scales are folded
// into the attention math, so no scale metadata is materialised here.
__global__ __launch_bounds__(NTHREADS) void atrex_ppu_prep_raw(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    unsigned char* __restrict__ k_fp8, unsigned char* __restrict__ v_fp8,
    float* __restrict__ kscale, float* __restrict__ vamax,
    int bt_stride, int max_pages) {
  const int req = blockIdx.y;
  const int kv_len = seq_lens[req];
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int tid = threadIdx.x;
  __align__(16) __shared__ unsigned char Vs[PAGE * HEAD_DIM];

  for (int p = blockIdx.x; p < n_pages; p += gridDim.x) {
    const int pid = block_table[req * bt_stride + p];
    const unsigned char* kp = key_cache + (size_t)pid * PAGE * HEAD_DIM;
    const unsigned char* vp = value_cache + (size_t)pid * PAGE * HEAD_DIM;
    unsigned char* kdst = k_fp8 + ((size_t)req * max_pages + p) * KPAGE_BYTES;
    unsigned char* vdst = v_fp8 + ((size_t)req * max_pages + p) * VPAGE_BYTES;

    const int kv = tid >> 2, ka = tid & 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint2 code = *reinterpret_cast<const uint2*>(
          kp + kv * HEAD_DIM + ka * 8 + j * 32);
      const int dimension = 32 * j + 8 * ka;
      reinterpret_cast<u32*>(kdst)[k_ldmatrix_page_halfword(kv, dimension) / 2] = code.x;
      reinterpret_cast<u32*>(kdst)[k_ldmatrix_page_halfword(kv, dimension + 4) / 2] = code.y;
    }
#pragma unroll
    for (int c = 0; c < (PAGE * HEAD_DIM) / (NTHREADS * 16); ++c) {
      const int e = (c * NTHREADS + tid) * 16;
      *reinterpret_cast<uint4*>(Vs + e) = *reinterpret_cast<const uint4*>(vp + e);
    }
    __syncthreads();
    u32 pack[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) pack[j] = 0u;
#pragma unroll
    for (int j = 0; j < PAGE; ++j) {
      const u32 code = (u32)Vs[j * HEAD_DIM + tid];
      const int slot = slot_of_kv(j);
      pack[slot >> 2] |= code << (8 * (slot & 3));
    }
    store_v_ldmatrix_row(vdst, tid, pack);
    __syncthreads();
  }
}

// ==================================================================================== KERNEL 2
// TILING: warp 2u owns q rows [q0,q0+16) over ALL 256 dims, warp 2u+1 owns [q0+16,q0+32). The
// accumulator is 16 q x 256 d = 128 vregs -- s32 now, fp32 in v40, SAME COUNT. Nothing is duplicated:
// 16 e4m3 MMAs and 16 exp2 per pair per chunk, no cross-warp exchange and no extra barrier.
// SPLIT-KV (v27..v40): with SPLIT=1 the grid gains blockIdx.z = the KV slice and a CTA owns one
// (q-block, slice) pair, snapped to PAGE granularity because the page loop carries a barrier. Each
// slice keeps its own reference max and writes fp32 (O_s, m_s, l_s); atrex_ppu_comb merges them. SPLIT=0
// constant-folds every line of that away.
template <int SPLIT, bool GROUPED = false, int QK_BUNDLE = 4>
__global__ __launch_bounds__(NTHREADS, 2) void atrex_ppu_attn(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ k_fp8,
    const unsigned char* __restrict__ v_fp8, const float* __restrict__ kscale,
    const float* __restrict__ vamax, const int* __restrict__ cu_q,
    const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out,
    float* __restrict__ o_part, int n_req, int max_pages, int n_tok, int nsplit) {
  extern __shared__ unsigned char sm[];
  unsigned char* Ks = sm;
  unsigned char* Vs = sm + NBUF * KPAGE_BYTES;

  int bx = GROUPED ? blockIdx.x / (NHEADS / 4) : blockIdx.x;
  int req = -1, qbi = 0, nb_req = 0;
  for (int r = 0; r < n_req; ++r) {
    const int nb = ((cu_q[r + 1] - cu_q[r]) + MQ - 1) / MQ;
    if (bx < nb) { req = r; qbi = bx; nb_req = nb; break; }
    bx -= nb;
  }
  if (req < 0) return;

  const int q_start = cu_q[req], q_len = cu_q[req + 1] - q_start;
  const int kv_len = seq_lens[req];
  // CTA ORDER: longest-processing-time-first, but ONLY when the KV cache is empty. q-tile qb sweeps
  // ceil((qb*MQ + kv_len - q_len + MQ)/PAGE) pages, so CTA work is LINEAR and INCREASING in qb, which
  // is the worst case for greedy list scheduling; reversing is LPT and is worth 4.4% when the profile
  // is steepest (kv_len == q_len). It is deliberately NOT applied when kv_len > q_len: an isolated
  // probe says LPT is still positive there, but under the harness's cold-L2 protocol the same change
  // measures -2.4% at kv=2q, because the first wave then sweeps the FULL page range at once.
  // v95/72a: LPT applies UNCONDITIONALLY. CTA work is linear and INCREASING in qbi, so in-order
  // dispatch is shortest-first, the worst order for greedy list scheduling. 66a's "-2.4% at kv=2q" did
  // not reproduce: three interleaved control/LPT pairs gave 17256.6 vs 17192.1, -0.374%, with LPT below
  // the control in EVERY pairing. Pure permutation of independent CTAs, so numerically exact.
  const int qb = nb_req - 1 - qbi;
  const int q0 = qb * MQ;
  const int base_pos = q0 + kv_len - q_len;  // CTA-wide: key j visible to CTA row rr iff j<=base+rr
  const int hi_cta = min(base_pos + MQ - 1, kv_len - 1);
  // The BF16-only short-row fallback cannot consume the raw-FP8 cache. Keep
  // every query block on the E4M3 mainloop.

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int unit = warp >> 1, wq = warp & 1;   // head within the CTA / which 16-row half
  const int head = (GROUPED ? blockIdx.x % (NHEADS / 4) : blockIdx.y) * 4 + unit;
  const int lq = lane >> 2, la = lane & 3;         // A side: row group / k group
  const int lp = lane >> 3, lb = (lane >> 2) & 1;  // B side: row index is lb + 2*(r>>1) + 4*lp

  const int bp_abs = base_pos + wq * MW;      // this warp's own 16 rows start here
  const int hi_abs = min(bp_abs + MW - 1, kv_len - 1);
  // Per-lane part of the causal limit, hoisted: row rr = 2*lq + ((i>>1)&1) may see key
  // kvj = kv_base + 4*la + 2*(i>>2) + (i&1) iff kvj <= bp + rr, i.e. iff
  //   2*(i>>2) + (i&1) - ((i>>1)&1)  <=  bp + 2*lq - 4*la - kv_base
  // The left side is a compile-time constant per accumulator slot, so the whole causal test reduces
  // to one integer subtract per chunk plus 8 compare/selects -- no branch. (The k relabel does NOT
  // touch this: the QK n index still runs 0..15 inside its own 16-kv chunk; only the PV contraction
  // index is permuted, and that permutation is absorbed by atrex_ppu_prep.)
  const int mlim_abs = bp_abs + 2 * lq - 4 * la;
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int last_page = min(n_pages - 1, hi_cta / PAGE);  // CTA-wide: the page loop has a barrier

  // ------------------------------------------------------------------------ this CTA's KV slice
  // The slice is a page range: pps = ceil(n_pages / nsplit) pages each, slice sl starting at sl*pps.
  // That still tiles [0, n_pages) disjointly. p_hi is clamped to last_page because pages past the
  // q-tile's causal limit hold no visible key; when the WHOLE slice is past it the range is empty and
  // the epilogue publishes m_s = -inf / l_s = 0 / O_s = 0 (necessary, not cosmetic: the partial
  // buffers are reused workspace, so a stale finite l_s would otherwise be merged in).
  // ALL 32-BIT AND ONE DIVISION, deliberately: this kernel sits ON a register-allocation cliff, and
  // the split path at 240 VRegs runs the inner loop 2.18x slower than the same work at 248.
  const int sl = SPLIT ? (int)blockIdx.z : 0;
  int p_lo = 0, p_hi = last_page;
  if (SPLIT) {
    const int pps = (n_pages + nsplit - 1) / nsplit;
    p_lo = sl * pps;
    p_hi = min(p_lo + pps - 1, last_page);
  }
  // THE PAGE LOOP RUNS ON A RELATIVE INDEX pg = p - p_lo, with the K/V/scale bases pre-offset by p_lo
  // and every causal constant shifted down by p_lo*PAGE: indexing by the absolute page makes the
  // compiler keep p_lo-offset copies of three bases live across the loop, which costs exactly the
  // 8 VRegs that drop it from 248 to 240. With SPLIT=0 every shift is by zero.
  const int kv_lo = p_lo * PAGE;
  const int npg = p_hi - p_lo;  // last relative page; < 0 means an empty slice
  const int bp = bp_abs - kv_lo;
  const int hi_w = hi_abs - kv_lo;
  const int mlim_base = mlim_abs - kv_lo;
  const int first_mask_page = bp >= 0 ? min(bp / PAGE, (kv_len - kv_lo) / PAGE) : -1;

  // ------------------------------------------------------------- Q: raw E4M3 fragment loads
  const unsigned char* qbase = q + (size_t)q_start * NHEADS * HEAD_DIM + head * HEAD_DIM;
  const int rowa = min(q0 + wq * MW + 2 * lq, q_len - 1);
  const int rowb = min(q0 + wq * MW + 2 * lq + 1, q_len - 1);

  u32 qf[KSTEPS][4];
#pragma unroll
  for (int ks = 0; ks < KSTEPS; ++ks) {
    const int d0 = ks * 32 + la * 8;
    const uint2 a = *reinterpret_cast<const uint2*>(
        qbase + (size_t)rowa * NHEADS * HEAD_DIM + d0);
    const uint2 b = *reinterpret_cast<const uint2*>(
        qbase + (size_t)rowb * NHEADS * HEAD_DIM + d0);
    qf[ks][0] = a.x;  // row even, d&7 = 0..3
    qf[ks][1] = b.x;  // row odd,  d&7 = 0..3
    qf[ks][2] = a.y;  // row even, d&7 = 4..7
    qf[ks][3] = b.y;  // row odd,  d&7 = 4..7
  }

  // ------------------------------------------------------------------------------ accumulators
  // FP8 P x V accumulates directly into FP32 registers.
  float acc[NDT][8];
#pragma unroll
  for (int dt = 0; dt < NDT; ++dt)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[dt][i] = 0.f;
  // mq = the MONOTONE running max of the scaled logit, per row of the pair. -inf is the "nothing
  // accumulated yet" state and needs no special case: ex2(-inf - nm) is exactly 0, and acc/ll are
  // zero then anyway, so the first refresh's rescale is a no-op by arithmetic.
  float mq[2] = {-INFINITY, -INFINITY};
  float ll[2] = {0.f, 0.f};
#if LAGA
  u32 Aprev[4] = {0u, 0u, 0u, 0u};  // LAGA diagnostic only
#endif

  // bases pre-offset to this slice's first page, so the loop indexes them by the relative page
  const unsigned char* kg = k_fp8 + ((size_t)req * max_pages + p_lo) * KPAGE_BYTES;
  const unsigned char* vg = v_fp8 + ((size_t)req * max_pages + p_lo) * VPAGE_BYTES;
#define STAGE_NC(buf, page)                                                            \
  do {                                                                                 \
    const unsigned char* _ks = kg + (size_t)(npg - (page))*KPAGE_BYTES;                 \
    const unsigned char* _vs = vg + (size_t)(npg - (page))*VPAGE_BYTES;                 \
    unsigned char* _kd = Ks + (buf)*KPAGE_BYTES;                                       \
    unsigned char* _vd = Vs + (buf)*VPAGE_BYTES;                                       \
    _Pragma("unroll") for (int c = 0; c < KPAGE_BYTES / (NTHREADS * 16); ++c) {         \
      const int _o = (c * NTHREADS + tid) * 16;                                        \
      cp_async16(_kd + _o, _ks + _o);                                                  \
    }                                                                                  \
    _Pragma("unroll") for (int c = 0; c < VPAGE_BYTES / (NTHREADS * 16); ++c) {         \
      const int _o = (c * NTHREADS + tid) * 16;                                        \
      cp_async16(_vd + _o, _vs + _o);                                                  \
    }                                                                                  \
  } while (0)

#define STAGE(buf, page)                                                               \
  do {                                                                                 \
    STAGE_NC(buf, page);                                                               \
    cp_commit();                                                                       \
  } while (0)

  // The whole-accumulator rescale: acc *= 2^(mq_old - mq_new), exactly, via one fp32 round trip per
  // vreg -- 128 x (cvt, mul, cvt). It fires only when the running max actually rises, which for a
  // monotone max is O(log) times per q-block, plus once per OVFN keys for the s32 bound. The fp32
  // detour costs 2^-24 relative on a value carrying ~30 bits: 5 orders under the gate.
#define RESCALE(f0, f1)                                                                \
  do {                                                                                 \
    _Pragma("unroll") for (int _dt = 0; _dt < NDT; ++_dt) {                             \
      _Pragma("unroll") for (int _i = 0; _i < 8; ++_i) {                                \
        const float _v = ACCF(acc[_dt][_i]) * (((_i >> 1) & 1) ? (f1) : (f0));    \
        ACC_STORE(acc[_dt][_i], _v);                                         \
      }                                                                                \
    }                                                                                  \
    ll[0] *= (f0);                                                                      \
    ll[1] *= (f1);                                                                      \
  } while (0)

  // An empty slice (npg < 0) iterates zero times and falls straight through to the epilogue with
  // acc = 0, ll = 0. Only the pipeline prime is guarded -- NOT with a block around the loop, whose
  // __syncthreads() must stay unconditional.
#if SP2E
  // Pages 0 and 1 in ONE commit group, exactly as SP2: the pair is always waited as a unit.
  if (npg >= 0) {
    STAGE_NC(0, 0);
    if (npg >= 1) STAGE_NC(1, 1);
    cp_commit();
  }
#elif SP2C
  // SP2C primes pages 0 and 1 as SEPARATE commit groups, so cp_wait<1> in the loop can retire exactly
  // one of them per iteration and the steady-state distance is two pages at ONE page of traffic.
  if (npg >= 0) STAGE(0, 0);
  if (npg >= 1) STAGE(1, 1);
#elif SP2 || SP2B
  // TWO pages primed into ONE commit group. Buffer b holds relative page pg with b == pg & 3, so the
  // pair staged at iteration pg lands in buffers (pg+2)&3 and (pg+3)&3, both of which were last READ
  // at iterations pg-2 and pg-1 -- i.e. strictly before this iteration's barrier. cp_wait<0> is
  // therefore correct with exactly one commit group outstanding, and the prefetch distance is TWO
  // pages of compute rather than one.
  if (npg >= 0) {
    STAGE_NC(0, 0);
    if (npg >= 1) STAGE_NC(1, 1);
    cp_commit();
  }
#else
  if (npg >= 0) STAGE(0, 0);
#endif
#if PGSTEP2
  // SP2 puts a barrier only on EVEN pages, so page 2k and page 2k+1 are separated by nothing but the
  // loop back-edge. `#pragma unroll 2` on the page loop is SILENTLY DROPPED (the census came back
  // byte-identical to SP2's -- 71b's trap again), so the pair is unrolled in the SOURCE. PGU2 does it
  // by duplicating the page body (SPSTEP 2) and SPILLS; SPG4 does it by lengthening the group loop.
#define PGSTEP 2
#else
#define PGSTEP 1
#endif
#if PGU2
#define SPSTEP 2
#else
#define SPSTEP 1
#endif
  for (int pp = 0; pp <= npg; pp += PGSTEP) {
#if SP2E
    if ((pp & 1) == 0) {
      cp_wait<0>();
      __syncthreads();
    } else if (pp + 1 <= npg) {
      // buffers (pp+1)&3 and (pp+2)&3 held pages pp-3 and pp-2, last read at iterations pp-3 and pp-2,
      // both before the barrier at pp-1. Safe to overwrite here, and one page LATER than SP2 does.
      STAGE_NC((pp + 1) & 3, pp + 1);
      if (pp + 2 <= npg) STAGE_NC((pp + 2) & 3, pp + 2);
      cp_commit();
    }
#elif SP2C
    cp_wait<1>();
    __syncthreads();
    if (pp + 2 <= npg) STAGE((pp + 2) & 3, pp + 2);
#elif SP2 || SP2B
    // `pp` is CTA-uniform, so this predicate is uniform and the barrier inside it is legal. (The
    // hggcErrorLaunchFailure of 21's note is about CROSS-LANE ops under a predicate, not barriers.)
#if SP2B
    const bool _lead = true;   // barrier EVERY page: isolates depth-2 staging from the barrier rate
#else
    const bool _lead = (PGSTEP == 2) || ((pp & 1) == 0);
#endif
    if (_lead) {
      cp_wait<0>();
      __syncthreads();
      if (pp + 2 <= npg) {
        STAGE_NC((pp + 2) & 3, pp + 2);
        if (pp + 3 <= npg) STAGE_NC((pp + 3) & 3, pp + 3);
        cp_commit();
      }
    }
#elif ONEB
    // ONE barrier per page. The buffer this iteration stages into, (pp+1)&1, was last READ in
    // iteration pp-1, so the barrier that has to separate those two accesses is THIS iteration's --
    // the trailing one was redundant once the STAGE moves below it. cp_wait<0> is right because the
    // only group outstanding at this point is the STAGE issued one iteration ago, which had the
    // whole of page pp-1's compute to complete in: the prefetch distance is unchanged at one page.
    cp_wait<0>();
    __syncthreads();
    if (pp < npg) STAGE((pp + 1) & 1, pp + 1);
#else
    if (pp < npg) {
      STAGE((pp + 1) & 1, pp + 1);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
#endif

#pragma unroll
    for (int _sp = 0; _sp < SPSTEP; ++_sp) {
      const int pg = pp + _sp;
      if (pg > npg) continue;
    const unsigned char* Kb = Ks + (pg & (NBUF - 1)) * KPAGE_BYTES;
    const unsigned char* Vb = Vs + (pg & (NBUF - 1)) * VPAGE_BYTES;
    const u32 key_block = static_cast<u32>(reinterpret_cast<uintptr_t>(Kb) / 16);
    const u32 value_block = static_cast<u32>(reinterpret_cast<uintptr_t>(Vb) / 16);
#if !SPG4
    const float aa = 0.0625f * 1.4426950408889634f;
#endif
    float page_scores[4][8];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int element = 0; element < 8; ++element) page_scores[tile][element] = 0.f;
    }
#pragma unroll
    for (int step = 0; step < KSTEPS; ++step) {
      const int base_offset = 8192 + 4096 * (step >> 2) + 32 * (step & 3);
      u32 first_key[4];
      u32 second_key[4];
      load_k_ldmatrix(first_key, key_block + base_offset / 16);
      load_k_ldmatrix(second_key, key_block + base_offset / 16 + 128);
      MMA_E4M3(page_scores[2], qf[step], first_key);
      MMA_E4M3(page_scores[3], qf[step], second_key);
    }
    const int physical_page = npg - pg;
    if (physical_page >= first_mask_page) {
      const int page_mask_limit = mlim_base - physical_page * PAGE;
#pragma unroll
      for (int tile = 2; tile < 4; ++tile) {
        const int mask_limit = page_mask_limit - tile * 16;
#pragma unroll
        for (int element = 0; element < 8; ++element)
          if (2 * (element >> 2) + (element & 1) - ((element >> 1) & 1) > mask_limit ||
              4 * la + 2 * (element >> 2) + (element & 1) >= kv_len - kv_lo - physical_page * PAGE - tile * 16)
            page_scores[tile][element] = -INFINITY;
      }
    }

    u32 future_first_key[4];
    u32 future_second_key[4];
    load_k_ldmatrix(future_first_key, key_block);
    load_k_ldmatrix(future_second_key, key_block + 128);
    u32 first_probability[4];
    {
      const int group = 1;
    float maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
#pragma unroll
      for (int tile = 2 * group; tile < 2 * group + 2; ++tile) {
        const int offset = 2 * row_half;
        const float lower = fmaxf(page_scores[tile][offset], page_scores[tile][offset + 1]);
        const float upper = fmaxf(page_scores[tile][offset + 4], page_scores[tile][offset + 5]);
        maximum[row_half] = fmaxf(maximum[row_half], fmaxf(lower, upper));
      }
      maximum[row_half] = fmaxf(maximum[row_half], __shfl_xor_sync(0xffffffffu, maximum[row_half], 1));
      maximum[row_half] = fmaxf(maximum[row_half], __shfl_xor_sync(0xffffffffu, maximum[row_half], 2));
    }
    float next_reference[2] = {
        fmaxf(mq[0], maximum[0] * aa),
        fmaxf(mq[1], maximum[1] * aa)
    };
    if (next_reference[0] > mq[0] || next_reference[1] > mq[1]) {
      const float first_scale = next_reference[0] > mq[0] ? ex2(mq[0] - next_reference[0]) : 1.f;
      const float second_scale = next_reference[1] > mq[1] ? ex2(mq[1] - next_reference[1]) : 1.f;
      RESCALE(first_scale, second_scale);
      mq[0] = next_reference[0];
      mq[1] = next_reference[1];
    }

#pragma unroll
      for (int tile_half = 0; tile_half < 2; ++tile_half) {
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
          const int tile = 2 * group + tile_half;
          const int lower = 2 * row_half;
          const int upper = lower + 4;
          const float shift = L255 - (next_reference[row_half] == -INFINITY ? 0.f : next_reference[row_half]);
          const float first = ex2(fmaf(page_scores[tile][lower], aa, shift));
          const float second = ex2(fmaf(page_scores[tile][lower + 1], aa, shift));
          const float third = ex2(fmaf(page_scores[tile][upper], aa, shift));
          const float fourth = ex2(fmaf(page_scores[tile][upper + 1], aa, shift));
          ll[row_half] += (first + second) + (third + fourth);
          first_probability[2 * tile_half + row_half] =
              (u32)cvt_e4m3x2(second, first) | ((u32)cvt_e4m3x2(fourth, third) << 16);
          if (true) {
#pragma unroll
            for (int substep = 0; substep < 2; ++substep) {
              const int step = 2 * (2 * tile_half + row_half) + substep;
              const int base_offset = 4096 * (step >> 2) + 32 * (step & 3);
              u32 first_key[4];
              u32 second_key[4];
              if (step == 0) {
#pragma unroll
                for (int word = 0; word < 4; ++word) {
                  first_key[word] = future_first_key[word];
                  second_key[word] = future_second_key[word];
                }
              } else {
                load_k_ldmatrix(first_key, key_block + base_offset / 16);
                load_k_ldmatrix(second_key, key_block + base_offset / 16 + 128);
              }
              MMA_E4M3(page_scores[0], qf[step], first_key);
              MMA_E4M3(page_scores[1], qf[step], second_key);
            }
          }
        }
      }
    }
    if (physical_page >= first_mask_page) {
      const int page_mask_limit = mlim_base - physical_page * PAGE;
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
        const int mask_limit = page_mask_limit - tile * 16;
#pragma unroll
        for (int element = 0; element < 8; ++element)
          if (2 * (element >> 2) + (element & 1) - ((element >> 1) & 1) > mask_limit ||
              4 * la + 2 * (element >> 2) + (element & 1) >= kv_len - kv_lo - physical_page * PAGE - tile * 16)
            page_scores[tile][element] = -INFINITY;
      }
    }
    float future_scale[2] = {1.f, 1.f};
    bool future_rescale = false;
    u32 second_probability[4];
    {
      const int group = 0;
    float maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
#pragma unroll
      for (int tile = 2 * group; tile < 2 * group + 2; ++tile) {
        const int offset = 2 * row_half;
        const float lower = fmaxf(page_scores[tile][offset], page_scores[tile][offset + 1]);
        const float upper = fmaxf(page_scores[tile][offset + 4], page_scores[tile][offset + 5]);
        maximum[row_half] = fmaxf(maximum[row_half], fmaxf(lower, upper));
      }
      maximum[row_half] = fmaxf(maximum[row_half], __shfl_xor_sync(0xffffffffu, maximum[row_half], 1));
      maximum[row_half] = fmaxf(maximum[row_half], __shfl_xor_sync(0xffffffffu, maximum[row_half], 2));
    }
    float next_reference[2] = {
        fmaxf(mq[0], maximum[0] * aa),
        fmaxf(mq[1], maximum[1] * aa)
    };
    future_rescale = next_reference[0] > mq[0] || next_reference[1] > mq[1];
    if (future_rescale) {
      future_scale[0] = next_reference[0] > mq[0] ? ex2(mq[0] - next_reference[0]) : 1.f;
      future_scale[1] = next_reference[1] > mq[1] ? ex2(mq[1] - next_reference[1]) : 1.f;
      ll[0] *= future_scale[0];
      ll[1] *= future_scale[1];
      mq[0] = next_reference[0];
      mq[1] = next_reference[1];
    }
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < NDT; ++dimension_tile) {
        u32 value_fragment[4];
        load_v_ldmatrix(value_fragment, value_block + 128 * (dimension_tile >> 1) + 32 * (dimension_tile & 1) + 2);
        MMA_PV(acc[dimension_tile], first_probability, value_fragment);
        if ((dimension_tile & 3) == 3) {
          const int tile_half = dimension_tile >> 3;
          const int row_half = (dimension_tile >> 2) & 1;
          const int tile = 2 * group + tile_half;
          const int lower = 2 * row_half;
          const int upper = lower + 4;
          const float shift = L255 - (next_reference[row_half] == -INFINITY ? 0.f : next_reference[row_half]);
          const float first = ex2(fmaf(page_scores[tile][lower], aa, shift));
          const float second = ex2(fmaf(page_scores[tile][lower + 1], aa, shift));
          const float third = ex2(fmaf(page_scores[tile][upper], aa, shift));
          const float fourth = ex2(fmaf(page_scores[tile][upper + 1], aa, shift));
          ll[row_half] += (first + second) + (third + fourth);
          second_probability[2 * tile_half + row_half] =
              (u32)cvt_e4m3x2(second, first) | ((u32)cvt_e4m3x2(fourth, third) << 16);
          first_probability[0] |= (__float_as_uint(first) | __float_as_uint(second) |
                                   __float_as_uint(third) | __float_as_uint(fourth)) & 0x80000000u;
        }
      }
    }
    if (future_rescale) {
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < NDT; ++dimension_tile) {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          const float scaled = ACCF(acc[dimension_tile][element]) * future_scale[(element >> 1) & 1];
          ACC_STORE(acc[dimension_tile][element], scaled);
        }
      }
    }
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < NDT; ++dimension_tile) {
      u32 value_fragment[4];
      load_v_ldmatrix(value_fragment, value_block + 128 * (dimension_tile >> 1) + 32 * (dimension_tile & 1));
      MMA_PV(acc[dimension_tile], second_probability, value_fragment);
    }
    }
#if !ONEB
    __syncthreads();
#endif
  }
#undef PGSTEP
#undef SPSTEP
#undef STAGE
#undef STAGE_NC
#undef RESCALE

  // ------------------------------------------------------------------------------- epilogue
  // THE ONE AND ONLY s32 -> fp32 CONVERSION, once per q-block: O = acc * v_scale[ch]. The common
  // 1/255 of p's full scale cancels between O and ll (out = O/ll), so it is never applied at all.
  // v_scale is read here and NOWHERE ELSE -- 16 float4 per q-block, against v25's 8 loads per CHUNK.
  // ll is held per `la` lane group; both epilogues need the true row sum, so reduce it first. mq is
  // la-uniform by construction (the group max is reduced across la), so no re-weighting is needed.
  float ls[2];
#pragma unroll
  for (int pr = 0; pr < 2; ++pr) {
    float s = ll[pr];
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
    ls[pr] = s;
  }
  if (SPLIT) {
    // ---- partial write: O_s with m_s / l_s folded into the SAME store stream -------------------
    // ONE STORE DESTINATION AND ONE STORE SHAPE IN THE WHOLE KERNEL. That is a hard requirement, not
    // style: giving atrex_ppu_attn a SECOND store site -- any second site, global or shared, live or dead --
    // drops the allocation from 248 to 240 VRegs, which costs 2.18x on the inner loop (measured: v21's
    // epilogue plus ONE DEAD 4-byte store runs sid13 in 9058.8 us instead of 4147.1). Reshaping the
    // ONE stream is free, so m_s and l_s ride in the dt == NDT iteration of the same store. That is
    // also why CF 1 uses ONE mixed-type buffer rather than a separate fp16 array and fp32 array.
#if CF
    // v103 CF 1 (ported from v97/77b): publish the NORMALISED slice output O_s/l_s in fp16 -- byte for
    // byte the same value the nsplit == 1 epilogue writes as bf16, just at fp16's 11-bit mantissa
    // instead of bf16's 8. RAW O_s IN FP16 OVERFLOWS (O_s ~ l_s*out ~ 2.3e5 vs 65504); O_s/l_s is a
    // weighted average of V, so |O_s/l_s| <= max|V| ~ 6 and is normally O(0.1). atrex_ppu_comb undoes it with
    // the weight l_s*w_s, which is algebraically the same merge. An empty slice (ls == 0) publishes
    // m_s = -inf, l_s = 0 and, because inv == 0, an all-zero O -- it contributes nothing, not a NaN.
    float inv[2];
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) inv[pr] = (ls[pr] > 0.f) ? (1.f / ls[pr]) : 0.f;
    unsigned char* row0 =
        reinterpret_cast<unsigned char*>(o_part) +
        ((size_t)sl * n_tok * NHEADS + (size_t)(q_start + q0) * NHEADS + head) * PRECB;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      unsigned char* row = row0 + (size_t)rr * NHEADS * PRECB;
      const bool nul = !(ls[pr] > 0.f);
      const float sc = inv[pr];
#pragma unroll
      for (int dt = 0; dt <= NDT; ++dt) {
        const int d = (dt < NDT) ? dt : 0;  // dt == NDT carries (m_s, l_s), not a dim group
        uint2 v;
        v.x = (dt < NDT) ? cvt_f16x2(ACCF(acc[d][2 * pr + 1]) * sc,
                                     ACCF(acc[d][2 * pr]) * sc)
                         : __float_as_uint(nul ? -INFINITY : mq[pr]);
        v.y = (dt < NDT) ? cvt_f16x2(ACCF(acc[d][5 + 2 * pr]) * sc,
                                     ACCF(acc[d][4 + 2 * pr]) * sc)
                         : __float_as_uint(nul ? 0.f : ls[pr]);
        *reinterpret_cast<uint2*>(row + dt * 32 + la * 8) = v;
      }
    }
#else
    // O_s is UNNORMALISED on purpose -- the combine divides by the merged l. A row with l_s == 0 saw
    // no key in this slice; publishing m_s = -inf lets the combine's guarded weight drop it outright.
    float* row0 = o_part +
                  ((size_t)sl * n_tok * NHEADS + (size_t)(q_start + q0) * NHEADS + head) * PREC;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      float* row = row0 + (size_t)rr * NHEADS * PREC;
      const bool nul = !(ls[pr] > 0.f);
#pragma unroll
      for (int dt = 0; dt <= NDT; ++dt) {
        const int d = (dt < NDT) ? dt : 0;  // dt == NDT carries (m_s, l_s), not a dim group
        float4 v;
        v.x = (dt < NDT) ? ACCF(acc[d][2 * pr]) : (nul ? -INFINITY : mq[pr]);
        v.y = (dt < NDT) ? ACCF(acc[d][2 * pr + 1]) : (nul ? 0.f : ls[pr]);
        v.z = (dt < NDT) ? ACCF(acc[d][4 + 2 * pr]) : 0.f;
        v.w = (dt < NDT) ? ACCF(acc[d][5 + 2 * pr]) : 0.f;
        *reinterpret_cast<float4*>(row + dt * 16 + 4 * la) = v;
      }
    }
#endif
  } else {
    // ---- direct-to-bf16 epilogue --------------------------------------------------------------
    float inv[2];
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) inv[pr] = (ls[pr] > 0.f) ? (1.f / ls[pr]) : 0.f;
    __nv_bfloat16* obase = out + (size_t)q_start * NHEADS * HEAD_DIM + head * HEAD_DIM;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      __nv_bfloat16* row = obase + (size_t)(q0 + rr) * NHEADS * HEAD_DIM;
      const float sc = inv[pr];
#pragma unroll
      for (int dt = 0; dt < NDT; ++dt) {
        const int dbase = dt * 16 + 4 * la;
        *reinterpret_cast<u32*>(row + dbase) =
            cvt_bf16x2(ACCF(acc[dt][2 * pr + 1]) * sc,
                       ACCF(acc[dt][2 * pr]) * sc);
        *reinterpret_cast<u32*>(row + dbase + 2) =
            cvt_bf16x2(ACCF(acc[dt][5 + 2 * pr]) * sc,
                       ACCF(acc[dt][4 + 2 * pr]) * sc);
      }
    }
  }
}

// ==================================================================================== KERNEL 3
// COMBINE -- only launched when nsplit > 1. Merges the per-slice (O_s, m_s, l_s) triples of one
// (token, head) row, validated numerically in workspace/pvquant/splitkv.py at ~6e-5 rel_l2:
//     m = max_s m_s ;  O = sum_s O_s*ex2(m_s - m) ;  l = sum_s l_s*ex2(m_s - m) ;  out = O/l
// mq lives in log2 units (aa carries the 1/ln2), so the merge weight is ex2, matching the kernel.
// The (m_s == -inf) guard is load-bearing: a dead slice would otherwise compute -inf - -inf = NaN
// BEFORE the weight multiplied it by zero.
// CF 1 stores O_s/l_s instead of O_s (fp16, half the bytes -- and this kernel is bandwidth-bound, so
// bytes are the only lever). The identity is exact:
//     out = sum_s O_s w_s / sum_s l_s w_s = sum_s (O_s/l_s)(l_s w_s) / sum_s (l_s w_s)
// so the per-slice weight becomes lw = l_s*w_s and the denominator is its sum. A dead slice has
// w_s = 0, hence lw = 0, hence no contribution -- exactly as before.
#if CD == 0
#define NDPT 4
#elif CD == 1
#define NDPT 8
#else
#define NDPT 16
#endif
#define TPR (HEAD_DIM / NDPT)   // threads per row: 64 / 32 / 16
#define CROWS (256 / TPR)       // rows per CTA:    4 / 8 / 16
template <int SplitCount>
__global__ __launch_bounds__(256) void atrex_ppu_comb(const float* __restrict__ o_part,
                                             __nv_bfloat16* __restrict__ out, int nrow,
                                             int nsplit) {
  const int split_count = SplitCount > 0 ? SplitCount : nsplit;
  const int tid = threadIdx.x;
  const int row = blockIdx.x * CROWS + tid / TPR;
  if (row >= nrow) return;
  const int d0 = (tid % TPR) * NDPT;
  float acc[NDPT];
#pragma unroll
  for (int i = 0; i < NDPT; ++i) acc[i] = 0.f;
#if CF
  const unsigned char* rec = reinterpret_cast<const unsigned char*>(o_part) + (size_t)row * PRECB;
  const size_t sstride = (size_t)nrow * PRECB;

  float m = -INFINITY;
#pragma unroll
  for (int s = 0; s < split_count; ++s)
    m = fmaxf(m, *reinterpret_cast<const float*>(rec + s * sstride + HEAD_DIM * 2));

  float lw = 0.f;
#pragma unroll
  for (int s = 0; s < split_count; ++s) {
    const unsigned char* r = rec + s * sstride;
    const float2 ml = *reinterpret_cast<const float2*>(r + HEAD_DIM * 2);
    const float w = (ml.x == -INFINITY) ? 0.f : ex2(ml.x - m);
    const float ws = ml.y * w;
    lw += ws;
    // Adjacent uint2 are folded back into the widest load by the vectoriser (v44b), which is what is
    // wanted here: NDPT 8 becomes one 16 B load, NDPT 16 two of them.
#pragma unroll
    for (int u = 0; u < NDPT / 4; ++u) {
      const uint2 o = *reinterpret_cast<const uint2*>(r + 2 * (d0 + 4 * u));
      acc[4 * u + 0] = fmaf(h2f((u16)(o.x & 0xFFFFu)), ws, acc[4 * u + 0]);
      acc[4 * u + 1] = fmaf(h2f((u16)(o.x >> 16)), ws, acc[4 * u + 1]);
      acc[4 * u + 2] = fmaf(h2f((u16)(o.y & 0xFFFFu)), ws, acc[4 * u + 2]);
      acc[4 * u + 3] = fmaf(h2f((u16)(o.y >> 16)), ws, acc[4 * u + 3]);
    }
  }
  const float inv = (lw > 0.f) ? (1.f / lw) : 0.f;
#else
  const float* rec = o_part + (size_t)row * PREC;  // slice s adds s*nrow*PREC
  const size_t sstride = (size_t)nrow * PREC;

  float m = -INFINITY;
#pragma unroll
  for (int s = 0; s < split_count; ++s) m = fmaxf(m, rec[s * sstride + HEAD_DIM]);

  float l = 0.f;
#pragma unroll
  for (int s = 0; s < split_count; ++s) {
    const float* r = rec + s * sstride;
    const float ms = r[HEAD_DIM];
    const float w = (ms == -INFINITY) ? 0.f : ex2(ms - m);
    l = fmaf(r[HEAD_DIM + 1], w, l);
#pragma unroll
    for (int u = 0; u < NDPT / 4; ++u) {
      const float4 o = *reinterpret_cast<const float4*>(r + d0 + 4 * u);
      acc[4 * u + 0] = fmaf(o.x, w, acc[4 * u + 0]);
      acc[4 * u + 1] = fmaf(o.y, w, acc[4 * u + 1]);
      acc[4 * u + 2] = fmaf(o.z, w, acc[4 * u + 2]);
      acc[4 * u + 3] = fmaf(o.w, w, acc[4 * u + 3]);
    }
  }
  const float inv = (l > 0.f) ? (1.f / l) : 0.f;
#endif
  __nv_bfloat16* dst = out + (size_t)row * HEAD_DIM + d0;
  u32 ow[NDPT / 2];
#pragma unroll
  for (int i = 0; i < NDPT / 2; ++i) ow[i] = cvt_bf16x2(acc[2 * i + 1] * inv, acc[2 * i] * inv);
#if CD == 0
  // 4 dims = 8 B: one uint2.
  *reinterpret_cast<uint2*>(dst) = make_uint2(ow[0], ow[1]);
#else
  // 8 or 16 dims: whole 16 B chunks. d0 is a multiple of NDPT >= 8 bf16, so dst is 16 B aligned.
#pragma unroll
  for (int i = 0; i < NDPT / 8; ++i)
    *reinterpret_cast<uint4*>(dst + 8 * i) =
        make_uint4(ow[4 * i], ow[4 * i + 1], ow[4 * i + 2], ow[4 * i + 3]);
#endif
}

// ==================================================================================== KERNEL 4
// SHORT ROWS, EXACTLY. Owns up to two q-blocks per request whose rows all see fewer than
// SHORT_T keys -- the regime where E4M3 V
// quantisation error is not diluted (an earlier study measured 4.417e-3 on rows seeing <= 32 keys,
// which carry 48.5% of
// ||ref||^2 on the kv_len == q_len shapes).
//
// It is deliberately a plain fp32 kernel over the ORIGINAL bf16 cache: exact, hence strictly better
// than the bf16-V path the accuracy study modelled, and -- the reason it is a separate kernel rather
// than a branch -- it puts no accumulator, no second store site and no extra register anywhere near
// atrex_ppu_attn. Every visible key is < 64, so the whole q-block lives in page 0 and the CTA touches exactly
// one page. NHEADS*n_req CTAs of 256 threads, ~2k fma each.
//
// It runs LAST in the stream so its rows overwrite whatever atrex_ppu_attn/atrex_ppu_comb left there; atrex_ppu_attn skips
// this q-block outright, so the two never disagree about a row.
#if SP
// 264 u16 = 528 B, so the bank stride between consecutive kv rows is 4 words instead of 0. See the SP
// arm note. Shared grows from 64 to 66 KB for the pair, against 256 KB/SM.
#define KSTR 264
#else
#define KSTR HEAD_DIM
#endif
__global__ __launch_bounds__(256) void atrex_ppu_short(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache, const int* __restrict__ block_table,
    const int* __restrict__ cu_q, const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out, int bt_stride) {
  extern __shared__ unsigned char ssm[];
  u16* Ks = reinterpret_cast<u16*>(ssm);
  u16* Vs = Ks + PAGE * KSTR;
  float* Ss = reinterpret_cast<float*>(Vs + PAGE * KSTR);  // [MQ][PAGE]
  float* Rd = Ss + MQ * PAGE;                              // [MQ][8] reduction scratch

  const int head = blockIdx.x, req = blockIdx.y;
  const int q0 = blockIdx.z * MQ;
  const int q_start = cu_q[req], q_len = cu_q[req + 1] - q_start;
  if (q0 >= q_len) return;
  const int kv_len = seq_lens[req];
  const int base_pos = kv_len - q_len;
  if (min(base_pos + q0 + MQ, kv_len) >= SHORT_T) return;  // atrex_ppu_attn owns it

  const int tid = threadIdx.x, r = tid >> 3, cc = tid & 7;
  const int pid = block_table[req * bt_stride];
  const __nv_bfloat16* kp = key_cache + (size_t)pid * PAGE * HEAD_DIM;
  const __nv_bfloat16* vp = value_cache + (size_t)pid * PAGE * HEAD_DIM;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int e = (c * 256 + tid) * 8;
    // e = kv*HEAD_DIM + d, so the padded index is e + (KSTR-HEAD_DIM)*kv. Both terms are multiples of
    // 8 u16, so the uint4 store stays 16 B aligned and the warp still writes 512 B contiguous.
    const int es = e + (KSTR - HEAD_DIM) * (e >> 8);
    *reinterpret_cast<uint4*>(Ks + es) = *reinterpret_cast<const uint4*>(kp + e);
    *reinterpret_cast<uint4*>(Vs + es) = *reinterpret_cast<const uint4*>(vp + e);
  }
  __syncthreads();

  // ---- S for this thread's eight keys j = cc + 8*jj ----------------------------------------
  const int rq = min(q0 + r, q_len - 1);
  const __nv_bfloat16* qrow = q + ((size_t)(q_start + rq) * NHEADS + head) * HEAD_DIM;
  float s[8];
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) s[jj] = 0.f;
  for (int d0 = 0; d0 < HEAD_DIM; d0 += 8) {
    const uint4 qq = *reinterpret_cast<const uint4*>(qrow + d0);
    const u32* wq = reinterpret_cast<const u32*>(&qq);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const uint4 kk = *reinterpret_cast<const uint4*>(Ks + (cc + 8 * jj) * KSTR + d0);
      const u32* wk = reinterpret_cast<const u32*>(&kk);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        s[jj] = fmaf(bf_lo(wq[t]), bf_lo(wk[t]), s[jj]);
        s[jj] = fmaf(bf_hi(wq[t]), bf_hi(wk[t]), s[jj]);
      }
    }
  }
  const int qpos = base_pos + q0 + r;  // this row's causal limit
  float mx = -INFINITY;
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    s[jj] = ((cc + 8 * jj) <= qpos) ? (s[jj] * 0.0625f * 1.4426950408889634f) : -INFINITY;
    mx = fmaxf(mx, s[jj]);
  }
  Rd[r * 8 + cc] = mx;
  __syncthreads();
  float m = Rd[r * 8];
#pragma unroll
  for (int i = 1; i < 8; ++i) m = fmaxf(m, Rd[r * 8 + i]);
  float sm = 0.f;
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    const float e = ex2(s[jj] - m);
    Ss[r * PAGE + cc + 8 * jj] = e;
    sm += e;
  }
  __syncthreads();
  Rd[r * 8 + cc] = sm;
  __syncthreads();
  float den = Rd[r * 8];
#pragma unroll
  for (int i = 1; i < 8; ++i) den += Rd[r * 8 + i];
  const float inv = (den > 0.f) ? (1.f / den) : 0.f;

  // ---- O for this thread's 32 dims -------------------------------------------------------
  // SP 1: thread cc owns dims 8*cc + 64*u for u = 0..3 rather than the contiguous block 32*cc, which
  // is what makes the Vs read spread over all 32 banks at the padded stride. Same 32 dims per thread,
  // same arithmetic, only the assignment differs.
#if SP
#define ODIM(u) (8 * cc + 64 * (u))
#else
#define ODIM(u) (32 * cc + 8 * (u))
#endif
  float o[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;
  for (int j = 0; j < PAGE; ++j) {
    const float p = Ss[r * PAGE + j];
    const u16* vr = Vs + j * KSTR;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint4 vv = *reinterpret_cast<const uint4*>(vr + ODIM(u));
      const u32* w = reinterpret_cast<const u32*>(&vv);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        o[8 * u + 2 * t] = fmaf(p, bf_lo(w[t]), o[8 * u + 2 * t]);
        o[8 * u + 2 * t + 1] = fmaf(p, bf_hi(w[t]), o[8 * u + 2 * t + 1]);
      }
    }
  }
  if (q0 + r >= q_len) return;
  __nv_bfloat16* obase = out + ((size_t)(q_start + q0 + r) * NHEADS + head) * HEAD_DIM;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    __nv_bfloat16* dst = obase + ODIM(u);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<u32*>(dst + 2 * i) =
          cvt_bf16x2(o[8 * u + 2 * i + 1] * inv, o[8 * u + 2 * i] * inv);
  }
#undef ODIM
}

// ==================================================================================== launchers
void prep_kv(torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
             torch::Tensor seq_lens, torch::Tensor k_fp8, torch::Tensor v_fp8,
             torch::Tensor kscale,
             torch::Tensor vamax, int64_t max_pages) {
  const int n_req = (int)block_table.size(0);
  const int bt_stride = (int)block_table.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(312, n_req);
  atrex_ppu_prep_raw<<<grid, NTHREADS, 0, stream>>>(
      reinterpret_cast<const unsigned char*>(key_cache.data_ptr()),
      reinterpret_cast<const unsigned char*>(value_cache.data_ptr()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<unsigned char*>(k_fp8.data_ptr()),
      reinterpret_cast<unsigned char*>(v_fp8.data_ptr()), kscale.data_ptr<float>(),
      vamax.data_ptr<float>(), bt_stride, (int)max_pages);
}

// nsplit <= 1 launches atrex_ppu_attn<0>: 2D grid, no partial buffers touched, bf16 written straight from the
// accumulator, and NO combine launch at all.
template <bool GROUPED>
void launch_attn(torch::Tensor q, torch::Tensor k_fp8, torch::Tensor v_fp8, torch::Tensor kscale,
          torch::Tensor vamax, torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor out,
          torch::Tensor o_part, int64_t max_pages, int64_t nblk, int64_t nsplit) {
  const int n_req = (int)cu_q.size(0) - 1;
  const int n_tok = (int)out.size(0);
  const int launch_nblk = n_req == 1 ? (n_tok + MQ - 1) / MQ : (int)nblk;
  const int smem = NBUF * (KPAGE_BYTES + VPAGE_BYTES);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(atrex_ppu_attn<0, GROUPED>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(atrex_ppu_attn<1, GROUPED>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(atrex_ppu_attn<1, GROUPED, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    once = true;
  }
  const auto qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
  const auto kp = reinterpret_cast<const unsigned char*>(k_fp8.data_ptr());
  const auto vp = reinterpret_cast<const unsigned char*>(v_fp8.data_ptr());
  const auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (nsplit <= 1) {
    dim3 grid = GROUPED ? dim3((unsigned)launch_nblk * (NHEADS / 4)) : dim3((unsigned)launch_nblk, NHEADS / 4);
    atrex_ppu_attn<0, GROUPED><<<grid, NTHREADS, smem, stream>>>(
        qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), op, nullptr, n_req, (int)max_pages, n_tok, 1);
  } else {
    dim3 grid = GROUPED ? dim3((unsigned)launch_nblk * (NHEADS / 4), 1, (unsigned)nsplit) : dim3((unsigned)launch_nblk, NHEADS / 4, (unsigned)nsplit);
    if (nsplit == 2)
      atrex_ppu_attn<1, GROUPED, 2><<<grid, NTHREADS, smem, stream>>>(
          qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
          seq_lens.data_ptr<int>(), op, o_part.data_ptr<float>(), n_req, (int)max_pages, n_tok,
          (int)nsplit);
    else
      atrex_ppu_attn<1, GROUPED><<<grid, NTHREADS, smem, stream>>>(
          qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
          seq_lens.data_ptr<int>(), op, o_part.data_ptr<float>(), n_req, (int)max_pages, n_tok,
          (int)nsplit);
  }
}

void combine(torch::Tensor o_part, torch::Tensor out, int64_t nsplit) {
  const int nrow = (int)(out.size(0) * out.size(1));  // n_tok * NHEADS
  dim3 grid((unsigned)((nrow + CROWS - 1) / CROWS));
  switch (nsplit) {
    case 2:
      atrex_ppu_comb<2><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 3:
      atrex_ppu_comb<3><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 4:
      atrex_ppu_comb<4><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 5:
      atrex_ppu_comb<5><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 6:
      atrex_ppu_comb<6><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 7:
      atrex_ppu_comb<7><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    case 8:
      atrex_ppu_comb<8><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
      break;
    default:
      atrex_ppu_comb<0><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(o_part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nrow, (int)nsplit);
  }
}


const std::vector<int64_t>& launch_resources() {
  static const std::vector<int64_t> resources = []() {
    int device = 0;
    cudaDeviceProp properties;
    cudaFuncAttributes attributes;
    int active_blocks = 0;
    const int shared_bytes = NBUF * (KPAGE_BYTES + VPAGE_BYTES);
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess, "device properties failed");
    TORCH_CHECK(cudaFuncSetAttribute(atrex_ppu_attn<1, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) == cudaSuccess, "dynamic shared configuration failed");
    TORCH_CHECK(cudaFuncGetAttributes(&attributes, atrex_ppu_attn<1, true, 2>) == cudaSuccess, "kernel attributes failed");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, atrex_ppu_attn<1, true, 2>, NTHREADS, shared_bytes) == cudaSuccess, "occupancy query failed");
    return std::vector<int64_t>{active_blocks, properties.multiProcessorCount, attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes), shared_bytes};
  }();
  return resources;
}

std::vector<int64_t> launch_plan(int64_t query_length, int64_t kv_length) {
  const auto& resources = launch_resources();
  std::vector<int64_t> result{1};
  result.insert(result.end(), resources.begin(), resources.end());
  return result;
}

void attn(torch::Tensor q, torch::Tensor k_fp8, torch::Tensor v_fp8, torch::Tensor kscale,
          torch::Tensor vamax, torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor out,
          torch::Tensor o_part, int64_t max_pages, int64_t nblk, int64_t nsplit) {
  launch_attn<true>(q, k_fp8, v_fp8, kscale, vamax, cu_q, seq_lens, out, o_part, max_pages, nblk, nsplit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_plan", &launch_plan);
  m.def("prep_kv", &prep_kv);
  m.def("attn", &attn);
  m.def("combine", &combine);
}
