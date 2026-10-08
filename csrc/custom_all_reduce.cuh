#pragma once

#include "custom_collective_common.cuh"
#include <algorithm>

namespace vllm {

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

#ifndef USE_ROCM
// ---------------------------------------------------------------------------
// Fused GEMV + all-reduce for TP decode (M <= kFusedMaxM rows).
//
// y = x @ W^T summed over TP ranks, in one kernel per rank: each block
// computes a tile of partial rows, publishes it into its own IPC-registered
// buffer, signals the peers (one flag per block per rank, release/acquire),
// then waits for the peers' tiles and reduces them in a fixed rank order so
// every rank produces bitwise identical output. No separate all-reduce
// kernel, no stream-level barrier. Partial regions alternate by call parity
// so call n+1 never overwrites what a lagging peer may still read of call n
// (a peer can be at most one call behind: it must pass our flags of call n
// to finish call n, and it finishes call n before it starts n+1).
//
// Buffer layout per rank (reg_buffer of size reg_buffer_sz):
//   [parity 0 partials][parity 1 partials] ... [flags][call counters]
// flags:    FlagType[kFusedMaxBlocks][kMaxCustomCollectiveRanks], written by
//           peers (release), read by owner (acquire).
// counters: FlagType[kFusedMaxBlocks], owner only; parity and flag base.
// ---------------------------------------------------------------------------
constexpr int kFusedMaxM = 8;
constexpr int kFusedMaxBlocks = 256;
constexpr int kFusedThreads = 256;
constexpr int kFusedRowsPerIter = kFusedThreads / 32;  // one W row per warp
constexpr size_t kFusedFlagsBytes =
    sizeof(FlagType) * kFusedMaxBlocks * (kMaxCustomCollectiveRanks + 1);

template <typename T, int ngpus>
__global__ void __launch_bounds__(kFusedThreads, 1)
    fused_gemv_allreduce_kernel(RankData* _dp, const T* __restrict__ x,
                                const T* __restrict__ w,
                                const T* __restrict__ bias, T* __restrict__ out,
                                int M, int N, int K, int rank,
                                size_t partial_bytes, size_t flags_off) {
  using P = typename packed_t<T>::P;
  constexpr int VEC = P::size;
  auto dp = *_dp;
  char* self_base = reinterpret_cast<char*>(const_cast<void*>(dp.ptrs[rank]));
  FlagType* self_flags = reinterpret_cast<FlagType*>(self_base + flags_off);
  FlagType* counters = self_flags + kFusedMaxBlocks * kMaxCustomCollectiveRanks;

  __shared__ uint32_t s_call;
  if (threadIdx.x == 0) {
    uint32_t c = counters[blockIdx.x] + 1;
    counters[blockIdx.x] = c;
    s_call = c;
  }
  __syncthreads();
  const uint32_t call = s_call;
  const int parity = call & 1;
  const int tiles = N / kFusedRowsPerIter;
  const int tiles_per_block = (tiles + gridDim.x - 1) / gridDim.x;
  const uint32_t base = call * tiles_per_block;
  T* self_part = reinterpret_cast<T*>(self_base + parity * partial_bytes);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  int it = 0;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x, ++it) {
    const int row = tile * kFusedRowsPerIter + warp;
    float acc[kFusedMaxM];
  #pragma unroll
    for (int m = 0; m < kFusedMaxM; ++m) acc[m] = 0.f;
    const T* wrow = w + static_cast<size_t>(row) * K;
    // Keep kUnroll 16-byte weight loads in flight per lane: at M <= 8 the
    // GEMV is bound by memory latency, not by the FMAs.
    constexpr int kUnroll = 4;
    constexpr int kStep = 32 * VEC;
    int k = lane * VEC;
    for (; k + (kUnroll - 1) * kStep < K; k += kUnroll * kStep) {
      P wv[kUnroll];
  #pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        wv[u] = *reinterpret_cast<const P*>(wrow + k + u * kStep);
      for (int m = 0; m < M; ++m) {
        const T* xrow = x + static_cast<size_t>(m) * K + k;
  #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          P xv = *reinterpret_cast<const P*>(xrow + u * kStep);
  #pragma unroll
          for (int v = 0; v < VEC; ++v)
            acc[m] += upcast_s(wv[u].data[v]) * upcast_s(xv.data[v]);
        }
      }
    }
    for (; k < K; k += kStep) {
      P wv = *reinterpret_cast<const P*>(wrow + k);
      for (int m = 0; m < M; ++m) {
        P xv = *reinterpret_cast<const P*>(x + static_cast<size_t>(m) * K + k);
  #pragma unroll
        for (int v = 0; v < VEC; ++v)
          acc[m] += upcast_s(wv.data[v]) * upcast_s(xv.data[v]);
      }
    }
  #pragma unroll
    for (int m = 0; m < kFusedMaxM; ++m) {
  #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
    }
    if (lane == 0) {
      // Bias is folded into this rank's partial (the caller passes it on one
      // rank only, as RowParallelLinear does), so the sum carries it once.
      const float b = bias != nullptr ? upcast_s(bias[row]) : 0.f;
      for (int m = 0; m < M; ++m)
        self_part[static_cast<size_t>(m) * N + row] = downcast_s<T>(acc[m] + b);
    }
    // Publish this tile to the peers, then wait for theirs.
    __syncthreads();
    const uint32_t seq = base + it + 1;
    if (threadIdx.x < ngpus && threadIdx.x != rank) {
      FlagType* peer_flags = reinterpret_cast<FlagType*>(
          reinterpret_cast<char*>(const_cast<void*>(dp.ptrs[threadIdx.x])) +
          flags_off);
      st_flag_release(
          &peer_flags[blockIdx.x * kMaxCustomCollectiveRanks + rank], seq);
      while (
          ld_flag_acquire(&self_flags[blockIdx.x * kMaxCustomCollectiveRanks +
                                      threadIdx.x]) != seq);
    }
    __syncthreads();
    // Reduce the tile in fixed rank order (bitwise identical on all ranks).
    for (int idx = threadIdx.x; idx < kFusedRowsPerIter * M;
         idx += blockDim.x) {
      const int m = idx / kFusedRowsPerIter;
      const int r = tile * kFusedRowsPerIter + idx % kFusedRowsPerIter;
      float s = 0.f;
  #pragma unroll
      for (int g = 0; g < ngpus; ++g) {
        const T* part = reinterpret_cast<const T*>(
            reinterpret_cast<const char*>(dp.ptrs[g]) + parity * partial_bytes);
        s += upcast_s(part[static_cast<size_t>(m) * N + r]);
      }
      out[static_cast<size_t>(m) * N + r] = downcast_s<T>(s);
    }
  }
}
#endif  // !USE_ROCM

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  void allgather(cudaStream_t stream, void* input, void* output, int size_bytes,
                 int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_allgather(cudaStream_t stream, T* input, T* output,
                               void* local_buffer, void* multicast_buffer,
                               uint32_t* epochs, int size_bytes,
                               int stage_size_bytes);
  template <typename T>
  void reduce_scatter(cudaStream_t stream, T* input, T* output, int size,
                      int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_reduce_scatter(cudaStream_t stream, T* input, T* output,
                                    void* local_buffer, uint32_t* epochs,
                                    int size, int stage_size_bytes);
  template <typename T>
  void mnnvl_multimem_reduce_scatter(cudaStream_t stream,
                                     const T* multicast_input, T* output,
                                     Signal* local_signal,
                                     Signal* multicast_signal, int size,
                                     int block_limit);

  // Fused GEMV + all-reduce; see fused_gemv_allreduce_kernel. reg_buffer must
  // be the IPC-registered buffer of this rank (register_buffer).
  template <typename T>
  void fused_gemv_allreduce(cudaStream_t stream, const T* x, const T* w,
                            const T* bias, T* out, int M, int N, int K,
                            void* reg_buffer, size_t reg_buffer_sz) {
#ifdef USE_ROCM
    throw std::runtime_error("fused_gemv_allreduce is not supported on ROCm");
#else
    using P = typename packed_t<T>::P;
    if (M < 1 || M > kFusedMaxM)
      throw std::runtime_error("fused_gemv_allreduce: M must be in [1, " +
                               std::to_string(kFusedMaxM) + "]");
    if (N % kFusedRowsPerIter != 0 || K % P::size != 0)
      throw std::runtime_error(
          "fused_gemv_allreduce: N must be a multiple of " +
          std::to_string(kFusedRowsPerIter) + " and K of " +
          std::to_string(P::size));
    auto it = buffers_.find(reg_buffer);
    if (it == buffers_.end())
      throw std::runtime_error(
          "fused_gemv_allreduce: reg_buffer is not registered");
    RankData* ptrs = it->second;
    const size_t partial_bytes =
        ((static_cast<size_t>(M) * N * sizeof(T)) + 127) & ~size_t(127);
    if (reg_buffer_sz < kFusedFlagsBytes + 256)
      throw std::runtime_error("fused_gemv_allreduce: buffer too small");
    const size_t flags_off = (reg_buffer_sz - kFusedFlagsBytes) & ~size_t(127);
    if (2 * partial_bytes > flags_off)
      throw std::runtime_error(
          "fused_gemv_allreduce: M*N too large for the registered buffer");
    int dev = 0, sms = 0;
    CUDACHECK(cudaGetDevice(&dev));
    CUDACHECK(
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int tiles = N / kFusedRowsPerIter;
    // All blocks must be co-resident on every rank: a block only waits on
    // its peer block of the same index, so grid <= #SMs keeps it deadlock-free.
    const int blocks = std::min({tiles, kFusedMaxBlocks, sms});
  #define FUSED_KL(ngpus)                       \
    fused_gemv_allreduce_kernel<T, ngpus>       \
        <<<blocks, kFusedThreads, 0, stream>>>( \
            ptrs, x, w, bias, out, M, N, K, rank_, partial_bytes, flags_off)
    switch (world_size_) {
      case 2:
        FUSED_KL(2);
        break;
      case 4:
        FUSED_KL(4);
        break;
      case 8:
        FUSED_KL(8);
        break;
      default:
        throw std::runtime_error(
            "fused_gemv_allreduce supports world sizes 2, 4 and 8");
    }
  #undef FUSED_KL
#endif
  }

  // Zero the flag/counter region of this rank's buffer. All ranks must call
  // this (and barrier) before the first fused_gemv_allreduce.
  void fused_gemv_allreduce_reset(cudaStream_t stream, void* reg_buffer,
                                  size_t reg_buffer_sz) {
    if (reg_buffer_sz < kFusedFlagsBytes + 256)
      throw std::runtime_error("fused_gemv_allreduce_reset: buffer too small");
    const size_t flags_off = (reg_buffer_sz - kFusedFlagsBytes) & ~size_t(127);
    CUDACHECK(cudaMemsetAsync(static_cast<char*>(reg_buffer) + flags_off, 0,
                              reg_buffer_sz - flags_off, stream));
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 * add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 *                                                       half *, int, int, int);
 */
}  // namespace vllm
