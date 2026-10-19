#include "prefill_common.cuh"

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
template <bool MULTI_HEAD = false>
__global__ __launch_bounds__(NTHREADS) void atrex_ppu_prep_raw(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    unsigned char* __restrict__ k_fp8, unsigned char* __restrict__ v_fp8,
    float* __restrict__ kscale, float* __restrict__ vamax,
    int bt_stride, int max_pages, int kv_heads) {
  const int head_count = MULTI_HEAD ? kv_heads : 1;
  const int req = blockIdx.y / head_count;
  const int kv_head = MULTI_HEAD ? blockIdx.y % head_count : 0;
  const int kv_len = seq_lens[req];
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int tid = threadIdx.x;
  __align__(16) __shared__ unsigned char Vs[PAGE * HEAD_DIM];

  for (int p = blockIdx.x; p < n_pages; p += gridDim.x) {
    const int pid = block_table[req * bt_stride + p];
    const unsigned char* kp = key_cache + ((size_t)pid * PAGE * head_count + kv_head) * HEAD_DIM;
    const unsigned char* vp = value_cache + ((size_t)pid * PAGE * head_count + kv_head) * HEAD_DIM;
    unsigned char* kdst = k_fp8 + ((size_t)blockIdx.y * max_pages + p) * KPAGE_BYTES;
    unsigned char* vdst = v_fp8 + ((size_t)blockIdx.y * max_pages + p) * VPAGE_BYTES;

    const int kv = tid >> 2, ka = tid & 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint2 code = *reinterpret_cast<const uint2*>(
          kp + kv * head_count * HEAD_DIM + ka * 8 + j * 32);
      const int dimension = 32 * j + 8 * ka;
      reinterpret_cast<u32*>(kdst)[k_ldmatrix_page_halfword(kv, dimension) / 2] = code.x;
      reinterpret_cast<u32*>(kdst)[k_ldmatrix_page_halfword(kv, dimension + 4) / 2] = code.y;
    }
    if ((p + 1) * PAGE <= kv_len) {
#pragma unroll
      for (int c = 0; c < (PAGE * HEAD_DIM) / (NTHREADS * 16); ++c) {
        const int e = (c * NTHREADS + tid) * 16;
        *reinterpret_cast<uint4*>(Vs + e) = *reinterpret_cast<const uint4*>(
            vp + (e / HEAD_DIM) * head_count * HEAD_DIM + e % HEAD_DIM);
      }
    } else {
      // Mask before PV: zero attention weights do not suppress NaN V padding.
#pragma unroll
      for (int c = 0; c < (PAGE * HEAD_DIM) / (NTHREADS * 16); ++c) {
        const int e = (c * NTHREADS + tid) * 16;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (p * PAGE + e / HEAD_DIM < kv_len) {
          value = *reinterpret_cast<const uint4*>(
              vp + (e / HEAD_DIM) * head_count * HEAD_DIM + e % HEAD_DIM);
        }
        *reinterpret_cast<uint4*>(Vs + e) = value;
      }
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
  const int kv_heads = (int)key_cache.size(2);
  dim3 grid(312, n_req * kv_heads);
  auto kernel = kv_heads == 1 ? atrex_ppu_prep_raw<false> : atrex_ppu_prep_raw<true>;
  kernel<<<grid, NTHREADS, 0, stream>>>(
      reinterpret_cast<const unsigned char*>(key_cache.data_ptr()),
      reinterpret_cast<const unsigned char*>(value_cache.data_ptr()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<unsigned char*>(k_fp8.data_ptr()),
      reinterpret_cast<unsigned char*>(v_fp8.data_ptr()), kscale.data_ptr<float>(),
      vamax.data_ptr<float>(), bt_stride, (int)max_pages, kv_heads);
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

void launch_attn_general(torch::Tensor q, torch::Tensor k_fp8, torch::Tensor v_fp8, torch::Tensor kscale,
          torch::Tensor vamax, torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor out,
          torch::Tensor o_part, int64_t max_pages, int64_t nblk, int64_t nsplit,
          int64_t kv_heads, torch::Tensor descales, double softmax_scale);

void attn(torch::Tensor q, torch::Tensor k_fp8, torch::Tensor v_fp8, torch::Tensor kscale,
          torch::Tensor vamax, torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor out,
          torch::Tensor o_part, int64_t max_pages, int64_t nblk, int64_t nsplit,
          int64_t kv_heads, torch::Tensor descales, double softmax_scale) {
  if (q.size(1) == NHEADS && kv_heads == 1 && descales.numel() == 0 && softmax_scale == 0.0625) {
    launch_attn<true>(q, k_fp8, v_fp8, kscale, vamax, cu_q, seq_lens, out, o_part,
        max_pages, nblk, nsplit);
  } else {
    launch_attn_general(q, k_fp8, v_fp8, kscale, vamax, cu_q, seq_lens, out, o_part,
        max_pages, nblk, nsplit, kv_heads, descales, softmax_scale);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_plan", &launch_plan);
  m.def("prep_kv", &prep_kv);
  m.def("attn", &attn);
  m.def("combine", &combine);
}
