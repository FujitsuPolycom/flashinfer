"""Shared policy for non-production PCIe collective launch geometry."""

from __future__ import annotations

import os


DIAGNOSTIC_GEOMETRY_ENV = "B12X_COMM_ALLOW_DIAGNOSTIC_GEOMETRY"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def require_diagnostic_geometry(
    family: str,
    **geometry: tuple[int, int],
) -> None:
    """Reject launch geometry outside reviewed defaults unless explicitly enabled.

    Each keyword value is ``(actual, reviewed_default)``.  Keeping the policy in
    one host-only helper gives every production communication path the same
    opt-in without adding work to captured CUDA graph replay.
    """

    changed = {
        name: (int(actual), int(reviewed_default))
        for name, (actual, reviewed_default) in geometry.items()
        if int(actual) != int(reviewed_default)
    }
    if not changed:
        return
    if os.getenv(DIAGNOSTIC_GEOMETRY_ENV, "").strip().lower() in _ENABLED_VALUES:
        return
    details = ", ".join(
        f"{name}={actual} (reviewed default {reviewed_default})"
        for name, (actual, reviewed_default) in changed.items()
    )
    raise ValueError(
        f"{family} non-default launch geometry is diagnostic-only: {details}; "
        f"set {DIAGNOSTIC_GEOMETRY_ENV}=1 to opt in"
    )


def validate_cluster_residency(launcher, blocks, threads, cluster_size):
    """Reject unsupported compiled cluster geometry without changing the captured grid."""
    import cuda.bindings.driver as cuda
    from b12x.preparation.types import CandidateResourceIneligibleError

    (raw,) = launcher.__b12x_dependencies__
    executable = raw._executable
    if len(executable.kernel_info) != 1:
        raise RuntimeError("clustered collective launcher must retain one CUDA kernel")
    (name,) = executable.kernel_info
    status, kernel = cuda.cuLibraryGetKernel(
        cuda.CUlibrary(int(executable.library)), name.encode()
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("cannot resolve clustered collective kernel")
    status, function = cuda.cuKernelGetFunction(kernel)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("cannot query clustered collective kernel")
    attribute = cuda.CUlaunchAttribute()
    attribute.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attribute.value.clusterDim.x = cluster_size
    attribute.value.clusterDim.y = attribute.value.clusterDim.z = 1
    config = cuda.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = blocks, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
    config.sharedMemBytes = 0
    config.attrs, config.numAttrs = [attribute], 1
    status, potential = cuda.cuOccupancyMaxPotentialClusterSize(function, config)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise CandidateResourceIneligibleError(
            "cannot support the prepared collective cluster geometry"
        )
    status, resident = cuda.cuOccupancyMaxActiveClusters(function, config)
    if (
        status != cuda.CUresult.CUDA_SUCCESS
        or potential < cluster_size
        or resident * cluster_size < blocks
    ):
        raise CandidateResourceIneligibleError(
            "collective cluster grid exceeds compiled kernel residency"
        )
