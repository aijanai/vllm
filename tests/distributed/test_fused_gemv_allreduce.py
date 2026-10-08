# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused row-parallel GEMV + all-reduce (custom all-reduce peers)."""

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

# (M, N, K_total): decode-sized rows, Flash-Next-like widths.
SHAPES = [(1, 2560, 2560), (4, 2560, 5120), (8, 4096, 2048), (3, 10240, 2560)]


def _reference(x_full, w_full, bias, dtype):
    y = x_full.float() @ w_full.float().T
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype)


@ray.remote(num_gpus=1, max_calls=1)
def _fused_gemv_allreduce_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    with monkeypatch.context():
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(rank)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        ca = group.device_communicator.ca_comm
        assert ca is not None and not ca.disabled, "custom all-reduce unavailable"

        for dtype in (torch.bfloat16, torch.float16):
            for m, n, k_total in SHAPES:
                k = k_total // tp_size
                # Identical full tensors on every rank, then slice like
                # RowParallelLinear does (weight [N, K] sharded on K).
                gen = torch.Generator(device=device).manual_seed(1234 + m + n)
                x_full = torch.randn(
                    m, k_total, dtype=dtype, device=device, generator=gen
                )
                w_full = (
                    torch.randn(n, k_total, dtype=dtype, device=device, generator=gen)
                    * 0.02
                )
                bias = torch.randn(n, dtype=dtype, device=device, generator=gen)
                x = x_full[:, rank * k : (rank + 1) * k].contiguous()
                w = w_full[:, rank * k : (rank + 1) * k].contiguous()
                bias_r = bias if rank == 0 else None  # as RowParallelLinear
                assert group.can_fuse_gemv_allreduce(x, w)

                ref = _reference(x_full, w_full, bias, dtype)
                # Repeat so both buffer parities and flag sequences are used.
                for _ in range(5):
                    out = group.fused_gemv_allreduce(x, w, bias_r)
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
                # Bitwise identical across ranks (fixed reduction order).
                gathered = group.all_gather(out, dim=0).view(tp_size, m, n)
                assert torch.equal(gathered[0], gathered[rank])

        # CUDA graph capture/replay with the registered buffers.
        m, n, k_total = SHAPES[1]
        k = k_total // tp_size
        dtype = torch.bfloat16
        gen = torch.Generator(device=device).manual_seed(99)
        x_full = torch.randn(m, k_total, dtype=dtype, device=device, generator=gen)
        w_full = (
            torch.randn(n, k_total, dtype=dtype, device=device, generator=gen) * 0.02
        )
        x = x_full[:, rank * k : (rank + 1) * k].contiguous()
        w = w_full[:, rank * k : (rank + 1) * k].contiguous()
        ref = _reference(x_full, w_full, None, dtype)
        with graph_capture(device=device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = group.fused_gemv_allreduce(x, w, None)
        for _ in range(3):
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        dist.barrier(group=group.device_group)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused GEMV all-reduce is CUDA only"
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_fused_gemv_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(
        monkeypatch, tp_size, pipeline_parallel_size, _fused_gemv_allreduce_worker
    )
