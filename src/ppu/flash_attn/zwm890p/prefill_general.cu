#include "prefill_common.cuh"

template <int SPLIT, bool GROUPED = false, int QK_BUNDLE = 4>
__global__ __launch_bounds__(NTHREADS, 2) void atrex_ppu_attn_general(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ k_fp8,
    const unsigned char* __restrict__ v_fp8, const float* __restrict__ kscale,
    const float* __restrict__ vamax, const int* __restrict__ cu_q,
    const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out,
    float* __restrict__ o_part, int n_req, int max_pages, int n_tok, int nsplit,
    int query_heads, int kv_heads, const float* __restrict__ descales, float softmax_scale) {
  constexpr bool GENERAL = true;
  extern __shared__ unsigned char sm[];
  unsigned char* Ks = sm;
  unsigned char* Vs = sm + NBUF * KPAGE_BYTES;

  const int n_heads = GENERAL ? query_heads : NHEADS;
  int bx = GROUPED ? blockIdx.x / (n_heads / 4) : blockIdx.x;
  int req = -1, qbi = 0, nb_req = 0;
  for (int r = 0; r < n_req; ++r) {
    const int nb = ((cu_q[r + 1] - cu_q[r]) + MQ - 1) / MQ;
    if (bx < nb) { req = r; qbi = bx; nb_req = nb; break; }
    bx -= nb;
  }
  if (req < 0) return;

  const int q_start = cu_q[req], q_len = cu_q[req + 1] - q_start;
  const int kv_len = seq_lens[req];

  const int qb = nb_req - 1 - qbi;
  const int q0 = qb * MQ;
  const int base_pos = q0 + kv_len - q_len;
  const int hi_cta = min(base_pos + MQ - 1, kv_len - 1);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int unit = warp >> 1, wq = warp & 1;
  const int head = (GROUPED ? blockIdx.x % (n_heads / 4) : blockIdx.y) * 4 + unit;
  const int kv_head = GENERAL ? head / (query_heads / kv_heads) : 0;
  const int packed_request = GENERAL ? req * kv_heads + kv_head : req;
  const int scale_count = n_req * kv_heads;
  const float query_scale = GENERAL && descales ? descales[packed_request] : 1.f;
  const float key_scale = GENERAL && descales ? descales[scale_count + packed_request] : 1.f;
  const float value_scale = GENERAL && descales ? descales[2 * scale_count + packed_request] : 1.f;
  const float logit_scale = GENERAL ? softmax_scale * query_scale * key_scale * 1.4426950408889634f : 1.f;
  const int lq = lane >> 2, la = lane & 3;
  const int lp = lane >> 3, lb = (lane >> 2) & 1;

  const int bp_abs = base_pos + wq * MW;
  const int hi_abs = min(bp_abs + MW - 1, kv_len - 1);

  const int mlim_abs = bp_abs + 2 * lq - 4 * la;
  const int n_pages = (kv_len + PAGE - 1) / PAGE;
  const int last_page = min(n_pages - 1, hi_cta / PAGE);

  const int sl = SPLIT ? (int)blockIdx.z : 0;
  int p_lo = 0, p_hi = last_page;
  if (SPLIT) {
    const int pps = (n_pages + nsplit - 1) / nsplit;
    p_lo = sl * pps;
    p_hi = min(p_lo + pps - 1, last_page);
  }

  const int kv_lo = p_lo * PAGE;
  const int npg = p_hi - p_lo;
  const int bp = bp_abs - kv_lo;
  const int hi_w = hi_abs - kv_lo;
  const int mlim_base = mlim_abs - kv_lo;
  const int first_mask_page = bp >= 0 ? min(bp / PAGE, (kv_len - kv_lo) / PAGE) : -1;

  const unsigned char* qbase = q + (size_t)q_start * n_heads * HEAD_DIM + head * HEAD_DIM;
  const int rowa = min(q0 + wq * MW + 2 * lq, q_len - 1);
  const int rowb = min(q0 + wq * MW + 2 * lq + 1, q_len - 1);

  u32 qf[KSTEPS][4];
#pragma unroll
  for (int ks = 0; ks < KSTEPS; ++ks) {
    const int d0 = ks * 32 + la * 8;
    const uint2 a = *reinterpret_cast<const uint2*>(
        qbase + (size_t)rowa * n_heads * HEAD_DIM + d0);
    const uint2 b = *reinterpret_cast<const uint2*>(
        qbase + (size_t)rowb * n_heads * HEAD_DIM + d0);
    qf[ks][0] = a.x;
    qf[ks][1] = b.x;
    qf[ks][2] = a.y;
    qf[ks][3] = b.y;
  }

  float acc[NDT][8];
#pragma unroll
  for (int dt = 0; dt < NDT; ++dt)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[dt][i] = 0.f;

  float mq[2] = {-INFINITY, -INFINITY};
  float ll[2] = {0.f, 0.f};
#if LAGA
  u32 Aprev[4] = {0u, 0u, 0u, 0u};
#endif

  const unsigned char* kg = k_fp8 + ((size_t)packed_request * max_pages + p_lo) * KPAGE_BYTES;
  const unsigned char* vg = v_fp8 + ((size_t)packed_request * max_pages + p_lo) * VPAGE_BYTES;
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

#if SP2E

  if (npg >= 0) {
    STAGE_NC(0, 0);
    if (npg >= 1) STAGE_NC(1, 1);
    cp_commit();
  }
#elif SP2C

  if (npg >= 0) STAGE(0, 0);
  if (npg >= 1) STAGE(1, 1);
#elif SP2 || SP2B

  if (npg >= 0) {
    STAGE_NC(0, 0);
    if (npg >= 1) STAGE_NC(1, 1);
    cp_commit();
  }
#else
  if (npg >= 0) STAGE(0, 0);
#endif
#if PGSTEP2

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

      STAGE_NC((pp + 1) & 3, pp + 1);
      if (pp + 2 <= npg) STAGE_NC((pp + 2) & 3, pp + 2);
      cp_commit();
    }
#elif SP2C
    cp_wait<1>();
    __syncthreads();
    if (pp + 2 <= npg) STAGE((pp + 2) & 3, pp + 2);
#elif SP2 || SP2B

#if SP2B
    const bool _lead = true;
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
    const float aa = GENERAL ? 1.f : 0.0625f * 1.4426950408889634f;
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
    if constexpr (GENERAL) {
#pragma unroll
      for (int tile = 2; tile < 4; ++tile) {
#pragma unroll
        for (int element = 0; element < 8; ++element) page_scores[tile][element] *= logit_scale;
      }
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
    if constexpr (GENERAL) {
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
        for (int element = 0; element < 8; ++element) page_scores[tile][element] *= logit_scale;
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

  float ls[2];
#pragma unroll
  for (int pr = 0; pr < 2; ++pr) {
    float s = ll[pr];
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
    ls[pr] = s;
  }
  if (SPLIT) {

#if CF

    float inv[2];
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) inv[pr] = (ls[pr] > 0.f) ? (value_scale / ls[pr]) : 0.f;
    unsigned char* row0 =
        reinterpret_cast<unsigned char*>(o_part) +
        ((size_t)sl * n_tok * n_heads + (size_t)(q_start + q0) * n_heads + head) * PRECB;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      unsigned char* row = row0 + (size_t)rr * n_heads * PRECB;
      const bool nul = !(ls[pr] > 0.f);
      const float sc = inv[pr];
#pragma unroll
      for (int dt = 0; dt <= NDT; ++dt) {
        const int d = (dt < NDT) ? dt : 0;
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

    float* row0 = o_part +
                  ((size_t)sl * n_tok * n_heads + (size_t)(q_start + q0) * n_heads + head) * PREC;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      float* row = row0 + (size_t)rr * n_heads * PREC;
      const bool nul = !(ls[pr] > 0.f);
#pragma unroll
      for (int dt = 0; dt <= NDT; ++dt) {
        const int d = (dt < NDT) ? dt : 0;
        float4 v;
        v.x = (dt < NDT) ? ACCF(acc[d][2 * pr]) * value_scale : (nul ? -INFINITY : mq[pr]);
        v.y = (dt < NDT) ? ACCF(acc[d][2 * pr + 1]) * value_scale : (nul ? 0.f : ls[pr]);
        v.z = (dt < NDT) ? ACCF(acc[d][4 + 2 * pr]) * value_scale : 0.f;
        v.w = (dt < NDT) ? ACCF(acc[d][5 + 2 * pr]) * value_scale : 0.f;
        *reinterpret_cast<float4*>(row + dt * 16 + 4 * la) = v;
      }
    }
#endif
  } else {

    float inv[2];
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) inv[pr] = (ls[pr] > 0.f) ? (value_scale / ls[pr]) : 0.f;
    __nv_bfloat16* obase = out + (size_t)q_start * n_heads * HEAD_DIM + head * HEAD_DIM;
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
      const int rr = wq * MW + 2 * lq + pr;
      if (q0 + rr >= q_len) continue;
      __nv_bfloat16* row = obase + (size_t)(q0 + rr) * n_heads * HEAD_DIM;
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

void launch_attn_general(torch::Tensor q, torch::Tensor k_fp8, torch::Tensor v_fp8, torch::Tensor kscale,
          torch::Tensor vamax, torch::Tensor cu_q, torch::Tensor seq_lens, torch::Tensor out,
          torch::Tensor o_part, int64_t max_pages, int64_t nblk, int64_t nsplit,
          int64_t kv_heads, torch::Tensor descales, double softmax_scale) {
  constexpr bool GROUPED = true;
  const int n_req = (int)cu_q.size(0) - 1;
  const int n_tok = (int)out.size(0);
  const int query_heads = (int)q.size(1);
  const float* scales = descales.numel() ? descales.data_ptr<float>() : nullptr;
  const int launch_nblk = n_req == 1 ? (n_tok + MQ - 1) / MQ : (int)nblk;
  const int smem = NBUF * (KPAGE_BYTES + VPAGE_BYTES);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(atrex_ppu_attn_general<0, GROUPED, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(atrex_ppu_attn_general<1, GROUPED, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(atrex_ppu_attn_general<1, GROUPED, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    once = true;
  }
  const auto qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
  const auto kp = reinterpret_cast<const unsigned char*>(k_fp8.data_ptr());
  const auto vp = reinterpret_cast<const unsigned char*>(v_fp8.data_ptr());
  const auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (nsplit <= 1) {
    dim3 grid = GROUPED ? dim3((unsigned)launch_nblk * (query_heads / 4)) : dim3((unsigned)launch_nblk, query_heads / 4);
    atrex_ppu_attn_general<0, GROUPED, 4><<<grid, NTHREADS, smem, stream>>>(
        qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), op, nullptr, n_req, (int)max_pages, n_tok, 1,
        query_heads, (int)kv_heads, scales, (float)softmax_scale);
  } else {
    dim3 grid = GROUPED ? dim3((unsigned)launch_nblk * (query_heads / 4), 1, (unsigned)nsplit) : dim3((unsigned)launch_nblk, query_heads / 4, (unsigned)nsplit);
    if (nsplit == 2)
      atrex_ppu_attn_general<1, GROUPED, 2><<<grid, NTHREADS, smem, stream>>>(
          qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
          seq_lens.data_ptr<int>(), op, o_part.data_ptr<float>(), n_req, (int)max_pages, n_tok,
          (int)nsplit, query_heads, (int)kv_heads, scales, (float)softmax_scale);
    else
      atrex_ppu_attn_general<1, GROUPED, 4><<<grid, NTHREADS, smem, stream>>>(
          qp, kp, vp, kscale.data_ptr<float>(), vamax.data_ptr<float>(), cu_q.data_ptr<int>(),
          seq_lens.data_ptr<int>(), op, o_part.data_ptr<float>(), n_req, (int)max_pages, n_tok,
          (int)nsplit, query_heads, (int)kv_heads, scales, (float)softmax_scale);
  }
}
