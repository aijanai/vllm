# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Socket grouping of a process group's ranks, from the NUMA node of each GPU."""

import torch
import torch.distributed as dist


def gpu_numa_node(device: torch.device) -> int:
    p = torch.cuda.get_device_properties(device)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            return int(f.read().strip())
    except OSError:
        return -1


def socket_groups(
    device: torch.device, group=None
) -> tuple[list[list[int]], list[int]]:
    """Returns (groups, numa per rank). Groups are lists of ranks sharing a NUMA
    node, ordered by their lowest rank. An unknown NUMA node (-1) puts every rank
    in one group."""
    world = dist.get_world_size(group)
    numa = [None] * world
    dist.all_gather_object(numa, gpu_numa_node(device), group=group)
    if any(n < 0 for n in numa):
        return [list(range(world))], numa
    by_node: dict[int, list[int]] = {}
    for r, n in enumerate(numa):
        by_node.setdefault(n, []).append(r)
    return sorted(by_node.values(), key=lambda g: g[0]), numa
