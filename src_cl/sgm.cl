#define FEAT_DIM 64
#define USE_ATOMIC_ADD

void atomicAddShort(__global ushort *address, ushort val) {
    __global unsigned int *base_address = (__global unsigned int *) ((__global char *)address - ((size_t)address & 2));
    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
    unsigned int long_old = atomic_add(base_address, long_val);
}

void partial_reduce(const ushort val, const __private y, const __private l,
   const __private int M, const __private int L,  const __private int ps,
  __global ushort *out, __local ushort *sdata) {
    const int yid = get_local_id(0);
    const int tid = get_local_id(1);

    if (y < M) {
      const int yoff = yid*get_local_size(1);
      sdata[yoff + tid] = (l < L) ? val : USHRT_MAX;

      barrier(CLK_LOCAL_MEM_FENCE);

      for (unsigned int s = get_local_size(1)/2; s > 0; s >>= 1) {
          if (tid < s)
            sdata[yoff+tid] = min(sdata[yoff+tid], sdata[yoff+tid+s]);
          barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (tid == 0) out[y*get_num_groups(1) + get_group_id(1)] = sdata[yoff];
    }
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

ushort sgm_step(const __global float *refImg,
              const __global ushort *minvals,
              __global ushort *out,  __read_only image3d_t tmp_in,
              __private int M, __private int N, __private int L, __private int sL,
              __private int x, __private int y, __private int dx, __private int dy,
              __private int xyps,
              __private int inc, __private ushort P1, __private ushort P2) {
    const float D1 = fabs(refImg[M*x+y] - refImg[M*x+y-inc]);
    const ushort P2_ = (D1 < 0.02f) ? P2 : P2/3.0f;

    const ushort old_min = minvals[x*M+y-inc];
    const ushort P2cost = old_min + P2_;

    ushort prev = min(P2cost,
        (ushort)read_imageui(tmp_in, sampler, (int4)(xyps,dx,dy,0)).x);

    ushort prev2 =
      (min((ushort)read_imageui(tmp_in, sampler, (int4)(xyps,dx,dy-1,0)).x,
                      (ushort)read_imageui(tmp_in, sampler, (int4)(xyps,dx,dy+1,0)).x));

    ushort prev3 =
      (min((ushort)read_imageui(tmp_in, sampler, (int4)(xyps,dx+1,dy,0)).x,
                      (ushort)read_imageui(tmp_in, sampler, (int4)(xyps,dx-1,dy,0)).x));

    return min(prev, add_sat(P1, min(prev3, prev2))) - old_min;
}

__kernel void sgm_slice_horz(const __global uchar *cost,
    const __global float *refImg, const __global ushort *minvals,
    __private int M, __private int N, __private int L,
    __private int sL, __private int ps,
    __private int P1, __private int P2, __global ushort *out,
    __read_only image3d_t tmp_in, __write_only image3d_t tmp_out,
    __private int x, __private int inc,
    __global ushort *scratch, __local ushort *sdata) {

    const int y = get_global_id(0);
    const int l = get_global_id(1);

    if (y < M & l < L) {
      ushort outval = cost[(l*N + x)*M + y];

      const int dx = l % sL;
      const int dy = l / sL;

      if ((inc < 0) ? (x < N-1) : x > 0) {
        outval += sgm_step(refImg, minvals, out, tmp_in, M, N, L, sL,
                  x, y, dx, dy, y, inc, P1, P2);
      }

     write_imageui(tmp_out, (int4)(y, dx, dy, 0), outval);

      // Write result
#ifdef USE_ATOMIC_ADD
      atomicAddShort(&(out[(x*M + y)*L + l]), outval);
#else
      out[(x*M + y)*L + l] += outval;
#endif

      // Partial reduction for next pass
      partial_reduce(outval, y, l, M, L, ps, scratch, sdata);
    }
}

__kernel void sgm_slice_vert(const __global uchar *cost,
    const __global float *refImg, const __global ushort *minvals,
    __private int M, __private int N, __private int L,
    __private int sL, __private int ps,
    __private int P1, __private int P2, __global ushort *out,
    __read_only image3d_t tmp_in, __write_only image3d_t tmp_out,
    __private int y, __private int inc,
    __global ushort *scratch, __local ushort *sdata) {

    const int l = get_global_id(1);
    const int x = get_global_id(0);

    if (x < N & l < L) {
      ushort outval = cost[(l*N + x)*M + y];

      const int dx = l % sL;
      const int dy = l / sL;

      if ((inc < 0) ? (y < M-1) : y > 0) {
        outval += sgm_step(refImg, minvals, out, tmp_in, M, N, L, sL,
                  x, y, dx, dy, x, inc, P1, P2);
      }

      write_imageui(tmp_out, (int4)(x, dx, dy, 0), outval);

      // Write result
#ifdef USE_ATOMIC_ADD
      atomicAddShort(&(out[(x*M + y)*L + l]), outval);
#else
      out[(x*M + y)*L + l] += outval;
#endif

      // Partial reduction for next pass
      partial_reduce(outval, x, l, N, L, ps, scratch, sdata);
    }
}

// TODO(Rene): Parallel reduction in shared memory
ushort min_reduce_loc(const __global ushort *in, __private int L) {
  ushort mm = USHRT_MAX;

  #pragma unroll
  for (int l = 0; l < L; ++l)
    mm = min(in[l], mm);

  return mm;
}

// TODO(Rene): These are redundant ->fuse
__kernel void min_reduce_horz_fin(const __global ushort *in,
  const __private int M, const __private int N, const __private int L,
  const __private int x, __global ushort *out) {
  const int y = get_global_id(0);

  if (y < M) {
    out[x*M + y] = min_reduce_loc(in+y*L, L);
  }
}

__kernel void min_reduce_vert_fin(const __global ushort *in,
  const __private int M, const __private int N, const __private int L,
  const __private int y, __global ushort *out) {
  const int x = get_global_id(0);

  if (x < N) {
    out[x*M + y] = min_reduce_loc(in+x*L, L);
  }
}

__kernel void  match(__global float *feat1, __global float *feat2,
    __private int M, __private int N, __private int max_offset,
    __global uchar *unary1, __global uchar *unary2) {
  const int y = get_global_id(0);
  const int x = get_global_id(1);

  if ((x < N) & y < M) {

  const int slice_stride = M*N;

  const int label_stride = 2*max_offset + 1;
  const int base = y + M*x;

  float4 feat1_s[FEAT_DIM/4];

  // Cache left feature map
  #pragma unroll FEAT_DIM/4
  for (int l = 0; l < FEAT_DIM; l+=4) {
    // Weird, but seems to be slightly faster than direct adressing
    int addr = base + M*N*l;
    feat1_s[l >> 2] = (float4)(feat1[addr],
                               feat1[addr+=slice_stride],
                               feat1[addr+=slice_stride],
                               feat1[addr+=slice_stride]);
  }

  for (int j = -max_offset, cnt = 0; j <= max_offset; ++j) {
    const int x2 = x + j;
    const int lin_idx_x = (max_offset - j)*label_stride + max_offset;

    for (int i = -max_offset; i <= max_offset; ++i, ++cnt) {
        const int y2 = y + i;

        // TODO(Rene): We have thread divergence here, can we optimize this?
        if ((x2 < 0) | (x2 >= N) | (y2 < 0) | (y2 >= M)) {
          continue;
        }

        const int lin_idx2 = lin_idx_x - i;
        const int base2 = y2 + M*x2;

        // Compute dot product with prefetching
        float accum = 1.0f;

        #pragma unroll FEAT_DIM/4
        for (int l = 0; l < FEAT_DIM; l+=4) {
          int addr = base2 + M*N*l;
          float4 ff2 = (float4)(feat2[addr],
                                feat2[addr+=slice_stride],
                                feat2[addr+=slice_stride],
                                feat2[addr+=slice_stride]);

          accum -= dot(feat1_s[l >> 2], ff2);
        }

        accum *= 127.0f;

        unary1[base + slice_stride*cnt] = accum;
        unary2[base2 + slice_stride*lin_idx2] = accum;
    }
  }
  }
}

__kernel void recover_flow(const __global ushort *cvol,
    __private int M, __private int N,
    __private int L, __global short2 *flow,
    __private int n_range, __private int max_offset) {
  const int y = get_global_id(0);
  const int x = get_global_id(1);

  if ((y < M) & (x < N)) {
    ushort minval = USHRT_MAX;
    int minlabel = 0;

    int base = x*M + y;
    cvol += base*L;

    #pragma unroll
    for (int l = 0; l < L; ++l) {
      ushort cval = cvol[l];
      if (cval < minval) {
        minval = cval;
        minlabel = l;
      }
    }

    flow[base] = (short2)(minlabel/n_range, minlabel % n_range) - (short2)(max_offset);
  }
}


__kernel void fix_border(__global uchar *unary1, __global uchar *unary2,
    __private int M, __private int N, __private int L) {
  const int y = get_global_id(0);
  const int x = get_global_id(1);

  // Assume fixed window size for now
  // left border
  if (x < 4) {
    for (int l = 0; l < L; ++l) {
      unary1[y + M*(x + N*l)] = unary1[y + M*(7 - x + N*l)];
      unary2[y + M*(x + N*l)] = unary2[y + M*(7 - x + N*l)];
    }
  }

  // right border
  if ((x >= N-4) & (x < N)) {
    for (int l = 0; l < L; ++l) {
      unary1[y + M*(x + N*l)] = unary1[y + M*(2*N-9 - x + N*l)];
      unary2[y + M*(x + N*l)] = unary2[y + M*(2*N-9 - x + N*l)];
    }
  }

  // upper border
  if (y < 4) {
    for (int l = 0; l < L; ++l) {
      unary1[y + M*(x + N*l)] = unary1[7 - y + M*(x + N*l)];
      unary2[y + M*(x + N*l)] = unary2[7 - y + M*(x + N*l)];
    }
  }

  // lower border
  if ((y >= M-4) & (y < M)) {
    for (int l = 0; l < L; ++l) {
      unary1[y + M*(x + N*l)] = unary1[2*M - 9 - y + M*(x + N*l)];
      unary2[y + M*(x + N*l)] = unary2[2*M - 9 - y + M*(x + N*l)];
    }
  }
}
