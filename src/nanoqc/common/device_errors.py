"""Tell a compute-resource failure apart from a scientific exclusion.

A GPU that runs out of memory, or a CUDA context that cannot be created, says
nothing about a candidate complex: it is a property of how many processes share
the device. Recording it as a preparation or target failure would put a
scheduling artifact into the study's denominator, so callers must raise
:class:`DeviceResourceError` and let the stage fail loudly instead.

Concurrency is therefore safe to raise only while every such error aborts the
stage rather than excluding a complex.
"""
from __future__ import annotations

import re

# Messages OpenMM/CUDA/Torch emit when a device cannot serve a context or
# allocation. Matched case-insensitively against the exception text.
RESOURCE_MESSAGE_PATTERN = re.compile(
    r"out of memory"
    r"|cudaerrormemoryallocation"
    r"|cuda_error_out_of_memory"
    r"|cudamalloc"
    r"|memory allocation (?:failed|error)"
    r"|failed to allocate"
    r"|no cuda[- ]capable device"
    r"|cudaerrordevicesunavailable"
    r"|all cuda-capable devices are busy"
    r"|cuda_error_no_device"
    r"|CUDA error: (?:invalid device ordinal|device-side assert|illegal memory access)"
    r"|CUDA_ERROR_ILLEGAL_ADDRESS"
    r"|no registered Platform called [\"']CUDA"
    r"|(?:CUBLAS|CUFFT)_STATUS_ALLOC_FAILED"
    r"|error (?:launching|initializing) (?:kernel|cuda)",
    re.IGNORECASE,
)


class DeviceResourceError(RuntimeError):
    """A device could not serve the computation; never a scientific exclusion."""

    category = "device_resource"

    def __init__(self, message: str, *, device: str = "", stage_hint: str = ""):
        super().__init__(message)
        self.device = device
        self.stage_hint = stage_hint


def is_resource_error(exc: BaseException) -> bool:
    """True when the exception describes a device resource limit, not the input."""
    for error in _chain(exc):
        if isinstance(error, (DeviceResourceError, MemoryError)):
            return True
        if type(error).__name__ in ("OutOfMemoryError", "CudaError", "BrokenProcessPool"):
            return True
        if RESOURCE_MESSAGE_PATTERN.search(str(error)):
            return True
    return False


def _chain(exc: BaseException, limit: int = 8):
    seen = []
    while exc is not None and len(seen) < limit and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def as_resource_error(exc: BaseException, *, device: str = "", stage_hint: str = "") -> DeviceResourceError:
    """Wrap a recognized resource failure, keeping the original as the cause."""
    error = DeviceResourceError(
        f"{type(exc).__name__}: {exc}"
        + (f" (CUDA device {device})" if device else "")
        + ". This is an execution resource/device failure, not a property of the complex: "
        "check available memory, devices and worker limits before retrying the stage.",
        device=device, stage_hint=stage_hint)
    error.__cause__ = exc
    return error


def raise_if_resource_error(exc: BaseException, *, device: str = "", stage_hint: str = "",
                            record_path=None) -> None:
    """Abort on infrastructure failures before any scientific failure accounting."""
    if not is_resource_error(exc):
        return
    if not device:
        import os
        device = os.environ.get("QP_OPENMM_DEVICE", "")
    error = exc if isinstance(exc, DeviceResourceError) else as_resource_error(
        exc, device=device, stage_hint=stage_hint)
    if record_path is not None:
        from nanoqc.common.repo_io import atomic_write_json_fsync
        atomic_write_json_fsync(record_path, dict(
            category=error.category, message=str(error),
            device=error.device or device, stage_hint=error.stage_hint or stage_hint,
            policy="stage_abort_not_sample_exclusion"))
    raise error
