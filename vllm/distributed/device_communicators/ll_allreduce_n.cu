// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Single-exchange all-reduce for N <= 8 ranks over P2P (LL protocol). Each 8-byte word
// carries 4 bytes of data and a 4-byte sequence flag. Every rank writes its words into
// its own slot of every peer's receive buffer, then polls its local buffer for the other
// N-1 slots: no barriers, no remote reads. Fixed grid, so word i always maps to the same
// block, whose call counter is the flag; two parity regions keep a peer that is at most
// one call behind from seeing its slot overwritten. The sum runs in rank order, so all
// ranks get bitwise identical results.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

constexpr int kMaxRanks = 8;
constexpr int kGrid = 16;
constexpr int kThreads = 512;

struct Peers {
  unsigned long long* recv[kMaxRanks];  // receive buffer of every rank (own one included)
};

// A peer that never arrives would make the others spin forever: after the timeout the
// waiting block reports what it waited for and traps, so the process fails visibly.
#ifndef LL_AR_TIMEOUT_NS
#define LL_AR_TIMEOUT_NS 60000000000ULL
#endif
__device__ int g_ll_timeout_reported = 0;  // one report per process, not one per waiting thread
struct SpinGuard {
  unsigned int n = 0;
  unsigned long long t0 = 0;
  __device__ __forceinline__ void tick(const char* kernel, int src, unsigned int seq) {
    if ((++n & 0xFFFu) != 0) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t0 == 0) {
      t0 = t;
    } else if (t - t0 > LL_AR_TIMEOUT_NS) {
      if (atomicExch(&g_ll_timeout_reported, 1) == 0)
        printf("%s timeout: block %d waited %llu ms for source %d, seq %u\n", kernel, blockIdx.x,
               (t - t0) / 1000000ULL, src, seq);
      __trap();
    }
  }
};

__global__ void __launch_bounds__(kThreads)
ll_allreduce_bf16(const __nv_bfloat162* __restrict__ in, __nv_bfloat162* __restrict__ out,
                  int nwords, Peers peers, int rank, int world, unsigned int* counter,
                  int region_words) {
  __shared__ unsigned int s_seq;
  if (threadIdx.x == 0) {
    unsigned int c = counter[blockIdx.x] + 1;
    counter[blockIdx.x] = c;
    s_seq = c;
  }
  __syncthreads();
  const unsigned int seq = s_seq;
  // Layout per rank buffer: [parity][src rank][word].
  const size_t par_off = (size_t)(seq & 1) * kMaxRanks * region_words;
  const size_t slot_off = par_off + (size_t)rank * region_words;
  const int stride = kGrid * kThreads;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < nwords; i += stride) {
    __nv_bfloat162 v = in[i];
    unsigned long long w = ((unsigned long long)seq << 32) | *reinterpret_cast<unsigned int*>(&v);
    for (int p = 1; p < world; ++p) {
      int peer = (rank + p) % world;
      asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(peers.recv[peer] + slot_off + i), "l"(w)
                   : "memory");
    }
  }
  unsigned long long* self = peers.recv[rank] + par_off;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < nwords; i += stride) {
    float2 acc = make_float2(0.f, 0.f);
    for (int src = 0; src < world; ++src) {
      __nv_bfloat162 v;
      if (src == rank) {
        v = in[i];
      } else {
        unsigned long long r;
        SpinGuard g;
        do {
          asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(r) : "l"(self + (size_t)src * region_words + i)
                       : "memory");
          g.tick("ll_allreduce", src, seq);
        } while ((unsigned int)(r >> 32) != seq);
        unsigned int d = (unsigned int)r;
        v = *reinterpret_cast<__nv_bfloat162*>(&d);
      }
      float2 f = __bfloat1622float2(v);
      acc.x += f.x;
      acc.y += f.y;
    }
    out[i] = __float22bfloat162_rn(acc);
  }
}

// Same protocol with 16-byte stores: {data0, seq, data1, seq} per v4.u32, as NCCL's LL
// does, which halves the PCIe write transactions. Each 8-byte half carries its own flag.
__global__ void __launch_bounds__(kThreads)
ll16_allreduce_bf16(const uint2* __restrict__ in, uint2* __restrict__ out, int npairs, Peers peers,
                    int rank, int world, unsigned int* counter, int region_words) {
  __shared__ unsigned int s_seq;
  if (threadIdx.x == 0) {
    unsigned int c = counter[blockIdx.x] + 1;
    counter[blockIdx.x] = c;
    s_seq = c;
  }
  __syncthreads();
  const unsigned int seq = s_seq;
  const size_t par_off = (size_t)(seq & 1) * kMaxRanks * region_words;
  const size_t slot_off = par_off + (size_t)rank * region_words;
  const int stride = kGrid * kThreads;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < npairs; i += stride) {
    uint2 v = in[i];
    for (int p = 1; p < world; ++p) {
      int peer = (rank + p) % world;
      unsigned long long* dst = peers.recv[peer] + slot_off + 2 * (size_t)i;
      asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(v.x), "r"(seq), "r"(v.y),
                   "r"(seq)
                   : "memory");
    }
  }
  unsigned long long* self = peers.recv[rank] + par_off;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < npairs; i += stride) {
    float2 acc0 = make_float2(0.f, 0.f), acc1 = make_float2(0.f, 0.f);
    for (int src = 0; src < world; ++src) {
      unsigned int d0, d1;
      if (src == rank) {
        uint2 v = in[i];
        d0 = v.x;
        d1 = v.y;
      } else {
        unsigned int f0, f1;
        const unsigned long long* p = self + (size_t)src * region_words + 2 * (size_t)i;
        SpinGuard g;
        do {
          asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(d0), "=r"(f0), "=r"(d1), "=r"(f1)
                       : "l"(p)
                       : "memory");
          g.tick("ll16_allreduce", src, seq);
        } while (f0 != seq || f1 != seq);
      }
      float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&d0));
      float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&d1));
      acc0.x += a.x;
      acc0.y += a.y;
      acc1.x += b.x;
      acc1.y += b.y;
    }
    __nv_bfloat162 r0 = __float22bfloat162_rn(acc0), r1 = __float22bfloat162_rn(acc1);
    out[i] = make_uint2(*reinterpret_cast<unsigned int*>(&r0), *reinterpret_cast<unsigned int*>(&r1));
  }
}

void ll16_allreduce(torch::Tensor in, torch::Tensor out, std::vector<int64_t> recv_ptrs, int64_t rank,
                    torch::Tensor counter, int64_t region_words) {
  TORCH_CHECK(in.scalar_type() == torch::kBFloat16 && in.numel() % 4 == 0);
  const int world = (int)recv_ptrs.size();
  TORCH_CHECK(world >= 2 && world <= kMaxRanks);
  const int npairs = in.numel() / 4;
  TORCH_CHECK(2 * npairs <= region_words);
  Peers peers;
  for (int r = 0; r < world; ++r) peers.recv[r] = reinterpret_cast<unsigned long long*>(recv_ptrs[r]);
  ll16_allreduce_bf16<<<kGrid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint2*>(in.data_ptr()), reinterpret_cast<uint2*>(out.data_ptr()), npairs, peers,
      (int)rank, world, reinterpret_cast<unsigned int*>(counter.data_ptr()), (int)region_words);
}

// Two-level LL all-reduce for GPUs spread over several CPU sockets. Per element, in one
// kernel: (1) members of a socket group exchange values over the fast intra-socket P2P path
// and all compute the group sum in member order; (2) only the group leaders exchange group
// sums across sockets, one stream per direction, and sum them in group order; (3) each
// leader sends the total back to its members. Each phase has its own buffer area.
struct HTopo {
  int nmem;              // members of this rank's socket group, sorted (this rank included)
  int mem[kMaxRanks];
  int nlead;             // group leaders, in group order (lowest rank of each group)
  int lead[kMaxRanks];
  int is_leader;
};
constexpr int kHierSlots = 2 * kMaxRanks + 1;  // phase 1 and 2: one slot per source; phase 3: one

__device__ __forceinline__ unsigned long long hier_slot(int par, int area_slot, int region_words) {
  return ((unsigned long long)par * kHierSlots + area_slot) * region_words;
}

__device__ __forceinline__ unsigned int ll_poll(const unsigned long long* p, unsigned int seq, int src) {
  unsigned long long r;
  SpinGuard g;
  do {
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(r) : "l"(p) : "memory");
    g.tick("hier_ll_allreduce", src, seq);
  } while ((unsigned int)(r >> 32) != seq);
  return (unsigned int)r;
}

__device__ __forceinline__ void ll_store(unsigned long long* p, unsigned int seq, unsigned int d) {
  unsigned long long w = ((unsigned long long)seq << 32) | d;
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(w) : "memory");
}

__global__ void __launch_bounds__(kThreads)
hier_ll_allreduce_bf16(const __nv_bfloat162* __restrict__ in, __nv_bfloat162* __restrict__ out, int nwords,
                       Peers peers, HTopo t, int rank, unsigned int* counter, int region_words) {
  __shared__ unsigned int s_seq;
  if (threadIdx.x == 0) {
    unsigned int c = counter[blockIdx.x] + 1;
    counter[blockIdx.x] = c;
    s_seq = c;
  }
  __syncthreads();
  const unsigned int seq = s_seq;
  const int par = seq & 1;
  const int stride = kGrid * kThreads;
  unsigned long long* self = peers.recv[rank];
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < nwords; i += stride) {
    __nv_bfloat162 v = in[i];
    unsigned int d = *reinterpret_cast<unsigned int*>(&v);
    for (int k = 0; k < t.nmem; ++k)
      if (t.mem[k] != rank) ll_store(peers.recv[t.mem[k]] + hier_slot(par, rank, region_words) + i, seq, d);
  }
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < nwords; i += stride) {
    float2 acc = make_float2(0.f, 0.f);
    for (int k = 0; k < t.nmem; ++k) {
      const int m = t.mem[k];
      unsigned int d;
      if (m == rank) {
        __nv_bfloat162 v = in[i];
        d = *reinterpret_cast<unsigned int*>(&v);
      } else {
        d = ll_poll(self + hier_slot(par, m, region_words) + i, seq, m);
      }
      float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&d));
      acc.x += f.x;
      acc.y += f.y;
    }
    __nv_bfloat162 gsum = __float22bfloat162_rn(acc);
    unsigned int gd = *reinterpret_cast<unsigned int*>(&gsum);
    if (t.is_leader) {
      for (int k = 0; k < t.nlead; ++k)
        if (t.lead[k] != rank)
          ll_store(peers.recv[t.lead[k]] + hier_slot(par, kMaxRanks + rank, region_words) + i, seq, gd);
      float2 tot = make_float2(0.f, 0.f);
      for (int k = 0; k < t.nlead; ++k) {
        const int l = t.lead[k];
        unsigned int s = (l == rank) ? gd : ll_poll(self + hier_slot(par, kMaxRanks + l, region_words) + i, seq, l);
        float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&s));
        tot.x += f.x;
        tot.y += f.y;
      }
      __nv_bfloat162 total = __float22bfloat162_rn(tot);
      unsigned int td = *reinterpret_cast<unsigned int*>(&total);
      for (int k = 0; k < t.nmem; ++k)
        if (t.mem[k] != rank) ll_store(peers.recv[t.mem[k]] + hier_slot(par, 2 * kMaxRanks, region_words) + i, seq, td);
      out[i] = total;
    } else {
      unsigned int td = ll_poll(self + hier_slot(par, 2 * kMaxRanks, region_words) + i, seq, t.mem[0]);
      out[i] = *reinterpret_cast<__nv_bfloat162*>(&td);
    }
  }
}

void hier_ll_allreduce(torch::Tensor in, torch::Tensor out, std::vector<int64_t> recv_ptrs, int64_t rank,
                       std::vector<int64_t> members, std::vector<int64_t> leaders, torch::Tensor counter,
                       int64_t region_words) {
  TORCH_CHECK(in.scalar_type() == torch::kBFloat16 && in.numel() % 2 == 0);
  const int world = (int)recv_ptrs.size();
  TORCH_CHECK(world >= 2 && world <= kMaxRanks && members.size() <= kMaxRanks && leaders.size() <= kMaxRanks);
  const int nwords = in.numel() / 2;
  TORCH_CHECK(nwords <= region_words);
  Peers peers;
  for (int r = 0; r < world; ++r) peers.recv[r] = reinterpret_cast<unsigned long long*>(recv_ptrs[r]);
  HTopo t;
  t.nmem = (int)members.size();
  t.nlead = (int)leaders.size();
  for (int k = 0; k < t.nmem; ++k) t.mem[k] = (int)members[k];
  for (int k = 0; k < t.nlead; ++k) t.lead[k] = (int)leaders[k];
  t.is_leader = 0;
  for (int k = 0; k < t.nlead; ++k) t.is_leader |= (t.lead[k] == rank);
  hier_ll_allreduce_bf16<<<kGrid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat162*>(in.data_ptr()), reinterpret_cast<__nv_bfloat162*>(out.data_ptr()),
      nwords, peers, t, (int)rank, reinterpret_cast<unsigned int*>(counter.data_ptr()), (int)region_words);
}

void ll_allreduce(torch::Tensor in, torch::Tensor out, std::vector<int64_t> recv_ptrs, int64_t rank,
                  torch::Tensor counter, int64_t region_words) {
  TORCH_CHECK(in.scalar_type() == torch::kBFloat16 && in.numel() % 2 == 0);
  const int world = (int)recv_ptrs.size();
  TORCH_CHECK(world >= 2 && world <= kMaxRanks);
  const int nwords = in.numel() / 2;
  TORCH_CHECK(nwords <= region_words);
  Peers peers;
  for (int r = 0; r < world; ++r) peers.recv[r] = reinterpret_cast<unsigned long long*>(recv_ptrs[r]);
  ll_allreduce_bf16<<<kGrid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat162*>(in.data_ptr()), reinterpret_cast<__nv_bfloat162*>(out.data_ptr()),
      nwords, peers, (int)rank, world, reinterpret_cast<unsigned int*>(counter.data_ptr()), (int)region_words);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ll_allreduce", &ll_allreduce);
  m.def("ll16_allreduce", &ll16_allreduce);
  m.def("hier_ll_allreduce", &hier_ll_allreduce);
}
