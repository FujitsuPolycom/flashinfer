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

"""SM120 Programmatic Dependent Launch (PDL) capability and toggle helpers."""

from __future__ import annotations

import os

import torch

from b12x._lib.intrinsics.arch import get_sm_version

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def parse_pdl_env(value: str | None) -> bool | None:
    """Parse an optional boolean environment variable for PDL toggles."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean PDL toggle value {value!r}; expected one of 0/1/true/false/yes/no/on/off"
    )


def sm120_consumer_pdl_enabled(
    device: int | torch.device | str | None = None,
    *,
    env_var: str | None = None,
    default: bool = True,
) -> bool:
    """Return whether consumer-side SM120 PDL should be enabled for ``device``.

    Gated on SM120+ hardware capability (``get_sm_version(device) >= 120``).
    When an op-specific ``env_var`` is set it takes precedence over the global
    ``B12X_SM120_CONSUMER_PDL`` toggle; when neither is set, ``default`` is
    used on SM120+ devices.  Callers MUST fold the returned boolean into both
    the kernel specialization and its ``KernelCompileSpec`` / ``@program_cache``
    key because b12x compile caches do not hash raw environment variables.
    """
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
    global_override = parse_pdl_env(os.environ.get("B12X_SM120_CONSUMER_PDL"))
    if global_override is not None:
        return global_override
    return bool(default)


__all__ = ["parse_pdl_env", "sm120_consumer_pdl_enabled"]
