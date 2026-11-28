# Copyright (c) 2026 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120 Distributed Shared Memory (DSMEM) cluster split-K eligibility helpers."""

from __future__ import annotations

import os

import torch

from b12x._lib.intrinsics.arch import get_sm_version
from b12x._lib.utils.pdl import parse_pdl_env

SM120_DSMEM_CLUSTER_SIZES = (2, 4, 8)


def sm120_dsmem_split_eligible(num_splits: int) -> bool:
    """Return True when ``num_splits`` is a supported SM120 DSMEM cluster size."""
    return int(num_splits) in SM120_DSMEM_CLUSTER_SIZES


def sm120_dsmem_split_enabled(
    device: int | torch.device | str | None = None,
    *,
    num_splits: int,
    env_var: str | None = None,
    default: bool = False,
) -> bool:
    """Return whether opt-in SM120 DSMEM cluster split-K should be enabled.

    Requires ``num_splits in (2, 4, 8)`` and SM120+ hardware capability.
    Callers MUST fold the returned boolean into both the kernel constructor and
    its ``KernelCompileSpec`` / ``@program_cache`` key.
    """
    if not sm120_dsmem_split_eligible(num_splits):
        return False
    if isinstance(device, torch.device) and device.type != "cuda":
        return False
    if isinstance(device, str) and not device.startswith("cuda"):
        return False
    if get_sm_version(device) < 120:
        return False
    if env_var is not None:
        specific = parse_pdl_env(os.environ.get(env_var))
        if specific is not None:
            return specific
    global_override = parse_pdl_env(os.environ.get("B12X_SM120_DSMEM_SPLIT"))
    if global_override is not None:
        return global_override
    return bool(default)


__all__ = [
    "SM120_DSMEM_CLUSTER_SIZES",
    "sm120_dsmem_split_eligible",
    "sm120_dsmem_split_enabled",
]
