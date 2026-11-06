# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023, E501
"""torchrun --nproc_per_node=4 stress_ll_allreduce.py [stress|timeout]
stress: many LL / LL16 / two-level all-reduces with random sizes, kernel order and rank skew,
eager and under CUDA graphs; each result is checked against NCCL and across ranks.
timeout: rank 3 skips one call; the others must trap with the timeout message, not hang."""

import ctypes
import importlib.util
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

# Kernel source shipped next to vllm/distributed/device_communicators/ll_allreduce.py
HERE = os.path.dirname(
    importlib.util.find_spec("vllm.distributed.device_communicators").origin
)
HIDDEN = 6144
REGION = 1 << 17
MAX_RANKS = 8
TOKENS = (1, 2, 3, 4, 5, 8, 13, 16, 31, 32)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stress"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    timeout_ns = int(float(os.environ.get("LL_TIMEOUT_S", "60")) * 1e9)
    bdir = os.environ.get("LL_BUILD", "/tmp/ll_ar_build") + f"_t{timeout_ns}"
    os.makedirs(bdir, exist_ok=True)
    build = lambda: load(
        "ll_allreduce_n",
        [os.path.join(HERE, "ll_allreduce_n.cu")],  # noqa: E731
        extra_cuda_cflags=["-O3", f"-DLL_AR_TIMEOUT_NS={timeout_ns}ULL"],
        build_directory=bdir,
    )
    if rank == 0:
        ext = build()
    dist.barrier()
    if rank != 0:
        ext = build()

    import vllm._custom_ops as ops
    from vllm.distributed.device_communicators.socket_topo import socket_groups

    def ipc_buffer(nbytes):
        ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
        handles = [None] * world
        dist.all_gather_object(handles, handle)
        ctypes.CDLL("libcudart.so").cudaMemset(
            ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes)
        )
        return [
            ptr if r == rank else ops.open_mem_handle(handles[r]) for r in range(world)
        ]

    flat = 2 * MAX_RANKS * REGION * 8
    recv8, recv16, recvh = (
        ipc_buffer(flat),
        ipc_buffer(flat),
        ipc_buffer(2 * (2 * MAX_RANKS + 1) * REGION * 8),
    )
    c8, c16, ch = (torch.zeros(16, dtype=torch.int32, device=dev) for _ in range(3))
    groups, numa = socket_groups(dev)
    members = next(g for g in groups if rank in g)
    leaders = [g[0] for g in groups]
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(
            f"NUMA {numa} groups {groups} leaders {leaders} timeout {timeout_ns / 1e9:.0f}s",
            flush=True,
        )

    kernels = {
        "ll": lambda x, o: ext.ll_allreduce(x, o, recv8, rank, c8, REGION),
        "ll16": lambda x, o: ext.ll16_allreduce(x, o, recv16, rank, c16, REGION),
        "hier": lambda x, o: ext.hier_ll_allreduce(
            x, o, recvh, rank, members, leaders, ch, REGION
        ),
    }

    if mode == "timeout":
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=dev)
        o = torch.empty_like(x)
        for name in ("hier", "ll"):
            kernels[name](x, o)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("warm calls ok; rank 3 now skips one two-level call", flush=True)
        if rank != 3:
            kernels["hier"](x, o)
            torch.cuda.synchronize()  # must raise: the kernel traps after the timeout
            print(f"rank {rank}: UNEXPECTED, call returned without rank 3", flush=True)
        else:
            import time

            time.sleep(float(os.environ.get("LL_TIMEOUT_S", "60")) * 3)
        return

    iters = int(os.environ.get("STRESS_ITERS", "20000"))
    rng = random.Random(
        1234
    )  # same sequence on every rank: sizes and kernel order agree
    skew = random.Random(rank * 7919 + 1)  # per-rank skew
    worst = 0.0
    for it in range(iters):
        t = rng.choice(TOKENS)
        name = rng.choice(("ll", "ll16", "hier"))
        torch.manual_seed(it * 31 + rank)
        x = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev)
        o = torch.empty_like(x)
        if skew.random() < 0.1:
            torch.cuda._sleep(
                skew.randint(1_000, 2_000_000)
            )  # up to ~1 ms behind the others
        kernels[name](x, o)
        ref = x.clone()
        dist.all_reduce(ref)
        err = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
        worst = max(worst, err)
        if err > 2e-2:
            print(f"rank {rank} it {it} {name} t={t}: rel err {err:.2e}", flush=True)
            raise SystemExit(1)
        if it % 200 == 0:
            g = [torch.empty_like(o) for _ in range(world)]
            dist.all_gather(g, o)
            assert all(torch.equal(g[0], y) for y in g), f"it {it} {name}: ranks differ"
        if rank == 0 and it % 5000 == 0:
            print(f"eager it {it}: worst rel err so far {worst:.2e}", flush=True)
    if rank == 0:
        print(f"eager: {iters} calls ok, worst rel err {worst:.2e}", flush=True)

    # CUDA graphs: one graph mixing kernels and sizes, replayed with random rank skew.
    seq = [(rng.choice(TOKENS), rng.choice(("ll", "ll16", "hier"))) for _ in range(48)]
    torch.manual_seed(99 + rank)
    xs = [torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev) for t, _ in seq]
    os_ = [torch.empty_like(x) for x in xs]
    refs = []
    for x in xs:
        r = x.clone()
        dist.all_reduce(r)
        refs.append(r)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (t, name), x, o in zip(seq, xs, os_):
            kernels[name](x, o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for (t, name), x, o in zip(seq, xs, os_):
            kernels[name](x, o)
    torch.cuda.synchronize()
    dist.barrier()
    replays = int(os.environ.get("STRESS_REPLAYS", "2000"))
    for rep in range(replays):
        if skew.random() < 0.2:
            torch.cuda._sleep(skew.randint(1_000, 2_000_000))
        g.replay()
        if rep % 250 == 0 or rep == replays - 1:
            torch.cuda.synchronize()
            for (t, name), o, r in zip(seq, os_, refs):
                err = ((o.float() - r.float()).norm() / r.float().norm()).item()
                assert err < 2e-2, f"replay {rep} {name} t={t}: rel err {err:.2e}"
            gath = [torch.empty_like(os_[-1]) for _ in range(world)]
            dist.all_gather(gath, os_[-1])
            assert all(torch.equal(gath[0], y) for y in gath), (
                f"replay {rep}: ranks differ"
            )
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"graphs: {replays} replays x {len(seq)} calls ok", flush=True)
        print("STRESS PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
