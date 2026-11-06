# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-exchange (LL) all-reduce over P2P for small bf16 messages.

Opt-in with VLLM_LL_AR_ENABLE=1. The path depends on the detected topology:
- GPUs on several CPU sockets: the two-level LL (intra-socket exchange, leaders across
  sockets, broadcast back) up to VLLM_HIER_AR_MAX_BYTES (default 384 KiB);
- GPUs on one socket: off, so the next backend runs (FlashInfer PCIe IPC wins there);
  VLLM_LL_AR_SINGLE_SOCKET=1 forces the flat 8/16-byte LL kernels for experiments.
Larger messages fall back to the next all-reduce backend. The kernel source is JIT-built
from VLLM_LL_AR_SRC.
"""

import ctypes
import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

from .socket_topo import socket_groups

logger = init_logger(__name__)

_MAX_RANKS = 8
_REGION_WORDS = 1 << 17  # words per (parity, slot): 384 KiB, 32 tokens of hidden 6144
_HIER_SLOTS = 2 * _MAX_RANKS + 1


class LLAllReduce:
    def __init__(self, cpu_group, device: torch.device, rank: int, world_size: int):
        from torch.utils.cpp_extension import load

        import vllm._custom_ops as ops

        self.rank = rank
        self.max8 = int(os.environ.get("VLLM_LL_AR_MAX_BYTES", "0"))
        self.max16 = int(os.environ.get("VLLM_LL16_AR_MAX_BYTES", "0"))
        self.maxh = int(os.environ.get("VLLM_HIER_AR_MAX_BYTES", "393216"))
        self.disabled = True

        # Measured: on one socket FlashInfer's PCIe IPC all-reduce beats the flat LL
        # kernels, across sockets the two-level kernel beats it. Stay off on one socket.
        groups, numa = socket_groups(device, cpu_group)
        self.multi_socket = len(groups) > 1
        if (
            not self.multi_socket
            and os.environ.get("VLLM_LL_AR_SINGLE_SOCKET", "0") != "1"
        ):
            logger.info(
                "LL all-reduce off: NUMA per rank %s is one socket; the next backend "
                "(FlashInfer PCIe IPC if enabled, else NCCL) handles all-reduce",
                numa,
            )
            return
        self.members = next(g for g in groups if rank in g)
        self.leaders = [g[0] for g in groups]

        # Prototype: JIT-built at startup; a real integration would compile it into _C.
        src = os.environ.get(
            "VLLM_LL_AR_SRC",
            os.path.join(os.path.dirname(__file__), "ll_allreduce_n.cu"),
        )
        bdir = os.environ.get("VLLM_LL_AR_BUILD", "/tmp/ll_ar_build")
        os.makedirs(bdir, exist_ok=True)
        # A rank that never arrives makes the kernel trap after this many seconds.
        timeout_ns = int(float(os.environ.get("VLLM_LL_AR_TIMEOUT_S", "60")) * 1e9)
        self.ext = load(
            "ll_allreduce_n",
            [src],
            build_directory=bdir,
            extra_cuda_cflags=["-O3", f"-DLL_AR_TIMEOUT_NS={timeout_ns}ULL"],
        )

        def ipc_buffer(nbytes):
            # one receive buffer per protocol: flags of one must never satisfy another
            ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
            handles = [None] * world_size
            dist.all_gather_object(handles, handle, group=cpu_group)
            ctypes.CDLL("libcudart.so").cudaMemset(
                ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes)
            )
            return [
                ptr if r == rank else ops.open_mem_handle(handles[r])
                for r in range(world_size)
            ]

        flat_bytes = 2 * _MAX_RANKS * _REGION_WORDS * 8
        with torch.cuda.device(device):
            # Only the buffers of the path this topology uses.
            self.recv8 = None if self.multi_socket else ipc_buffer(flat_bytes)
            self.recv16 = None if self.multi_socket else ipc_buffer(flat_bytes)
            self.recvh = (
                ipc_buffer(2 * _HIER_SLOTS * _REGION_WORDS * 8)
                if self.multi_socket
                else None
            )
            self.counter8 = torch.zeros(16, dtype=torch.int32, device=device)
            self.counter16 = torch.zeros(16, dtype=torch.int32, device=device)
            self.counterh = torch.zeros(16, dtype=torch.int32, device=device)
            torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        self.calls = 0
        self.disabled = False
        logger.info(
            "LL all-reduce enabled: NUMA per rank %s, socket groups %s (%s). "
            "8-byte up to %d B, %s up to %d B",
            numa,
            groups,
            "two-level" if self.multi_socket else "single socket",
            self.max8,
            "two-level" if self.multi_socket else "16-byte",
            self.maxh if self.multi_socket else self.max16,
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor | None:
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.numel() % 4 != 0:
            return None
        nbytes = x.numel() * 2
        if x.numel() // 2 > _REGION_WORDS:
            return None
        out = torch.empty_like(x)
        if self.multi_socket:
            # Measured on 1+3 GPUs over two sockets: two-level beats flat LL everywhere.
            if nbytes > self.maxh:
                return None
            self.ext.hier_ll_allreduce(
                x,
                out,
                self.recvh,
                self.rank,
                self.members,
                self.leaders,
                self.counterh,
                _REGION_WORDS,
            )
        elif nbytes <= self.max8:
            self.ext.ll_allreduce(
                x, out, self.recv8, self.rank, self.counter8, _REGION_WORDS
            )
        elif nbytes <= self.max16:
            self.ext.ll16_allreduce(
                x, out, self.recv16, self.rank, self.counter16, _REGION_WORDS
            )
        else:
            return None
        self.calls += 1
        if self.calls == 1:
            logger.info("LL all-reduce first call: %d bytes", nbytes)
        return out
