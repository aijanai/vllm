# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023, E501
"""torchrun --nproc_per_node=4 benchmark_ll_allreduce.py
CUDA-graph latency on 4 PCIe GPUs at MiniMax-M3.1-Flash decode sizes (hidden 6144, bf16):
NCCL (what vLLM uses on >2 PCIe GPUs), vLLM's custom all-reduce forced on, and a
single-exchange LL all-reduce. Correctness of both custom paths is checked against NCCL."""

import ctypes
import importlib.util
import os

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

# Kernel source shipped next to vllm/distributed/device_communicators/ll_allreduce.py
HERE = os.path.dirname(
    importlib.util.find_spec("vllm.distributed.device_communicators").origin
)
HIDDEN = 6144
TOKENS = (1, 2, 4, 8, 16, 32)
REGION = 1 << 17  # words per (parity, source rank) slot; covers 32 tokens
MAX_RANKS = 8


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    bdir = os.environ.get("LL_BUILD", "/tmp/ll_ar_build")
    os.makedirs(bdir, exist_ok=True)
    if rank == 0:
        ext = load(
            "ll_allreduce_n",
            [os.path.join(HERE, "ll_allreduce_n.cu")],
            extra_cuda_cflags=["-O3"],
            build_directory=bdir,
        )
    dist.barrier()
    if rank != 0:
        ext = load(
            "ll_allreduce_n",
            [os.path.join(HERE, "ll_allreduce_n.cu")],
            extra_cuda_cflags=["-O3"],
            build_directory=bdir,
        )

    # vLLM refuses its custom all-reduce on >2 PCIe-only GPUs; pretend full connectivity.
    from vllm.platforms import current_platform

    current_platform.__class__.is_fully_connected = classmethod(lambda cls, ids: True)
    import vllm._custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import graph_capture

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, "env://", rank)
        initialize_model_parallel(tensor_model_parallel_size=world)
    ca = get_tp_group().device_communicator.ca_comm
    ca_ok = ca is not None and not ca.disabled
    if rank == 0:
        print(f"vLLM custom all-reduce forced on: {ca_ok}", flush=True)

    nbytes = 2 * MAX_RANKS * REGION * 8

    def ipc_buffer():
        # one receive buffer per protocol: sequence flags of one must never satisfy the other
        ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
        handles = [None] * world
        dist.all_gather_object(handles, handle)
        ctypes.CDLL("libcudart.so").cudaMemset(
            ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes)
        )
        return [
            ptr if r == rank else ops.open_mem_handle(handles[r]) for r in range(world)
        ]

    recv, recv16 = ipc_buffer(), ipc_buffer()
    counter = torch.zeros(16, dtype=torch.int32, device=dev)
    counter16 = torch.zeros(16, dtype=torch.int32, device=dev)

    from vllm.distributed.device_communicators.socket_topo import socket_groups

    groups, numa = socket_groups(dev)
    members = next(g for g in groups if rank in g)
    leaders = [g[0] for g in groups]
    if rank == 0:
        print(
            f"NUMA per rank {numa} -> socket groups {groups}, leaders {leaders}",
            flush=True,
        )
    hier_bytes = 2 * (2 * MAX_RANKS + 1) * REGION * 8
    hptr, hhandle = ops.allocate_shared_buffer_and_handle(hier_bytes)
    hh = [None] * world
    dist.all_gather_object(hh, hhandle)
    ctypes.CDLL("libcudart.so").cudaMemset(
        ctypes.c_void_p(hptr), 0, ctypes.c_size_t(hier_bytes)
    )
    recvh = [hptr if r == rank else ops.open_mem_handle(hh[r]) for r in range(world)]
    counterh = torch.zeros(16, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    dist.barrier()

    def ll(y, out):
        ext.ll_allreduce(y, out, recv, rank, counter, REGION)
        return out

    def ll16(y, out):
        ext.ll16_allreduce(y, out, recv16, rank, counter16, REGION)
        return out

    def hier(y, out):
        ext.hier_ll_allreduce(y, out, recvh, rank, members, leaders, counterh, REGION)
        return out

    for t in TOKENS:
        y = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev) * (rank + 1)
        out, out16, outh = torch.empty_like(y), torch.empty_like(y), torch.empty_like(y)
        for _ in range(5):
            ll(y, out)
            ll16(y, out16)
            hier(y, outh)
        ref = y.clone()
        dist.all_reduce(ref)
        ca_out = ca.custom_all_reduce(y) if ca_ok else ref
        torch.cuda.synchronize()
        g = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(g, out16)
        gh = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(gh, outh)
        if rank == 0:
            rel = lambda a: (
                (a.float() - ref.float()).norm() / ref.float().norm()
            ).item()  # noqa: E731
            print(
                f"tokens={t:2d} rel_err LL {rel(out):.1e} LL16 {rel(out16):.1e} HIER {rel(outh):.1e} "
                f"customAR {rel(ca_out):.1e} LL16==LL {torch.equal(out, out16)} identical across ranks "
                f"LL16 {all(torch.equal(g[0], x) for x in g)} HIER {all(torch.equal(gh[0], x) for x in gh)}",
                flush=True,
            )

    def graph_time(fn, reps=64, replays=40):
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        with graph_capture(device=dev) as gc:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=gc.stream):
                for _ in range(reps):
                    fn()
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000 / (reps * replays)

    if rank == 0:
        print(
            f"\n{'tokens':>6s} {'bytes':>7s} {'NCCL':>9s} {'vLLM CA':>9s} {'LL':>9s} {'LL16':>9s} {'HIER':>9s} "
            f"{'NCCL/best':>9s}",
            flush=True,
        )
    for t in TOKENS:
        y = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev)
        out, out16, outh = torch.empty_like(y), torch.empty_like(y), torch.empty_like(y)
        nc = y.clone()
        t_nccl = graph_time(lambda: dist.all_reduce(nc))
        t_ca = graph_time(lambda: ca.custom_all_reduce(y)) if ca_ok else float("nan")
        t_ll = graph_time(lambda: ll(y, out))
        t_ll16 = graph_time(lambda: ll16(y, out16))
        t_h = graph_time(lambda: hier(y, outh))
        if rank == 0:
            print(
                f"{t:6d} {2 * t * HIDDEN:7d} {t_nccl:7.1f}us {t_ca:7.1f}us {t_ll:7.1f}us {t_ll16:7.1f}us "
                f"{t_h:7.1f}us {t_nccl / min(t_ll, t_ll16, t_h):8.2f}x",
                flush=True,
            )
    dist.barrier()
    if rank == 0:
        print("DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
