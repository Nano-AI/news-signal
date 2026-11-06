// Fused fp16 BERT kernels (fp32 accumulation). Launched on the caller's stream; no allocation, no sync.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVE 64
#define LN_THREADS 256
#define LN_MAXV 4  // supports hidden <= LN_THREADS * LN_MAXV

__device__ __forceinline__ float wave_sum(float v) {
  for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
  return v;
}

// Block-wide sum; result broadcast to all threads. sh must hold LN_THREADS/WAVE floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
  int lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
  v = wave_sum(v);
  if (lane == 0) sh[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < LN_THREADS / WAVE; i++) t += sh[i];
  __syncthreads();
  return t;
}

__device__ __forceinline__ void ln_store(float (&v)[LN_MAXV], int H, float eps, const __half* g, const __half* b,
                                         __half* out, float* sh) {
  float s = 0.f;
  for (int k = 0; k < LN_MAXV; k++) { int i = threadIdx.x + k * LN_THREADS; if (i < H) s += v[k]; }
  float mean = block_sum(s, sh) / H;
  float q = 0.f;
  for (int k = 0; k < LN_MAXV; k++) { int i = threadIdx.x + k * LN_THREADS; if (i < H) { float d = v[k] - mean; q += d * d; } }
  float inv = rsqrtf(block_sum(q, sh) / H + eps);
  for (int k = 0; k < LN_MAXV; k++) {
    int i = threadIdx.x + k * LN_THREADS;
    if (i < H) out[i] = __float2half((v[k] - mean) * inv * __half2float(g[i]) + __half2float(b[i]));
  }
}

__global__ void emb_ln_k(const long* ids, const long* tt, const __half* word, const __half* pos, const __half* tok,
                         const __half* g, const __half* b, __half* out, int S, int H, float eps) {
  __shared__ float sh[LN_THREADS / WAVE];
  int row = blockIdx.x, p = row % S;
  const __half* w = word + ids[row] * H;
  const __half* t = tok + tt[row] * H;
  const __half* ps = pos + (long)p * H;
  float v[LN_MAXV];
  for (int k = 0; k < LN_MAXV; k++) {
    int i = threadIdx.x + k * LN_THREADS;
    // match torch fp16 rounding: (word + pos) + tok, each sum rounded to fp16
    v[k] = 0.f;
    if (i < H) {
      __half s1 = __float2half(__half2float(w[i]) + __half2float(ps[i]));
      v[k] = __half2float(__float2half(__half2float(s1) + __half2float(t[i])));
    }
  }
  ln_store(v, H, eps, g, b, out + (long)row * H, sh);
}

// out = LN(x + h); out may alias x.
__global__ void add_ln_k(const __half* x, const __half* h, const __half* g, const __half* b, __half* out, int H,
                         float eps) {
  __shared__ float sh[LN_THREADS / WAVE];
  long off = (long)blockIdx.x * H;
  float v[LN_MAXV];
  for (int k = 0; k < LN_MAXV; k++) {
    int i = threadIdx.x + k * LN_THREADS;
    v[k] = 0.f;
    if (i < H) v[k] = __half2float(__float2half(__half2float(x[off + i]) + __half2float(h[off + i])));
  }
  ln_store(v, H, eps, g, b, out + off, sh);
}

// One block (one wave) per (batch, head); thread i owns query i. D == 64, S <= 64.
// qkv: [B*S, 3*H] (q | k | v, each head-major); mask: [B,S] int64; out: [B*S, H].
__global__ void attn_k(const __half* qkv, const long* mask, __half* out, int S, int nh, float scale) {
  const int D = 64;
  __shared__ float Ks[64 * 64];
  __shared__ float Vs[64 * 64];
  __shared__ float bias[64];
  int bh = blockIdx.x, b = bh / nh, h = bh % nh, H = nh * D;
  long ld = 3L * H;
  const __half* base = qkv + (long)b * S * ld + h * D;
  for (int idx = threadIdx.x; idx < S * D; idx += blockDim.x) {
    int j = idx / D, d = idx % D;
    Ks[idx] = __half2float(base[j * ld + H + d]);
    Vs[idx] = __half2float(base[j * ld + 2 * H + d]);
  }
  if (threadIdx.x < S) bias[threadIdx.x] = mask[(long)b * S + threadIdx.x] ? 0.f : -1e4f;
  __syncthreads();
  int i = threadIdx.x;
  if (i >= S) return;
  float q[64], o[64];
  for (int d = 0; d < D; d++) { q[d] = __half2float(base[i * ld + d]) * scale; o[d] = 0.f; }
  float m = -INFINITY, l = 0.f;
  for (int j = 0; j < S; j++) {
    float s = bias[j];
    const float* kr = Ks + j * D;
    #pragma unroll
    for (int d = 0; d < D; d++) s += q[d] * kr[d];
    float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
    l = l * c + p;
    const float* vr = Vs + j * D;
    #pragma unroll
    for (int d = 0; d < D; d++) o[d] = o[d] * c + p * vr[d];
    m = mn;
  }
  float inv = 1.f / l;
  __half* op = out + ((long)b * S + i) * H + h * D;
  for (int d = 0; d < D; d++) op[d] = __float2half(o[d] * inv);
}

extern "C" {
int launch_emb_ln(const long* ids, const long* tt, const void* word, const void* pos, const void* tok, const void* g,
                   const void* b, void* out, int rows, int S, int H, float eps, hipStream_t st) {
  emb_ln_k<<<rows, LN_THREADS, 0, st>>>(ids, tt, (const __half*)word, (const __half*)pos, (const __half*)tok,
                                        (const __half*)g, (const __half*)b, (__half*)out, S, H, eps);
  return (int)hipGetLastError();
}
int launch_add_ln(const void* x, const void* h, const void* g, const void* b, void* out, int rows, int H, float eps,
                   hipStream_t st) {
  add_ln_k<<<rows, LN_THREADS, 0, st>>>((const __half*)x, (const __half*)h, (const __half*)g, (const __half*)b,
                                        (__half*)out, H, eps);
  return (int)hipGetLastError();
}
int launch_attn(const void* qkv, const long* mask, void* out, int B, int S, int nh, float scale, hipStream_t st) {
  attn_k<<<B * nh, 64, 0, st>>>((const __half*)qkv, mask, (__half*)out, S, nh, scale);
  return (int)hipGetLastError();
}
}
