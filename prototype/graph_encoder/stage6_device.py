"""Explicit, opt-in deterministic device controls for ADR-0014 only."""

from __future__ import annotations

import os
import platform
import socket

from .errors import GraphEncoderError


DEVICE_POLICY_VERSION = "GE1-STAGE6-DEVICE-POLICY-v1"
SUPPORTED_EXECUTION_DEVICES = ("cpu", "cuda:0")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def validate_execution_device(value):
    if value not in SUPPORTED_EXECUTION_DEVICES:
        raise GraphEncoderError(
            "invalid_stage6_execution_device", repr(value)
        )
    return value


def configure_stage6_runtime(torch_module, execution_device, *, seed):
    """Configure the selected device without permitting implicit fallback."""

    device = validate_execution_device(execution_device)
    torch = torch_module
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError("environment_mismatch", "PyTorch 1.11.0 required")
    if torch.get_num_threads() != 1:
        raise GraphEncoderError("environment_mismatch", "one CPU thread required")
    if device == "cuda:0":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible or "," in visible or visible.strip() in ("", "-1"):
            raise GraphEncoderError(
                "invalid_stage6_cuda_environment", "exactly one visible GPU required"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise GraphEncoderError(
                "stage6_cuda_unavailable", "cuda:0 must be available without fallback"
            )
        if torch.version.cuda is None:
            raise GraphEncoderError(
                "stage6_cuda_unavailable", "CUDA-enabled PyTorch build required"
            )
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
            raise GraphEncoderError(
                "invalid_stage6_cuda_environment",
                "CUBLAS_WORKSPACE_CONFIG must equal " + CUBLAS_WORKSPACE_CONFIG,
            )
        torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    import random
    random.seed(int(seed))
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device == "cuda:0":
        torch.cuda.manual_seed_all(int(seed))
    if not torch.are_deterministic_algorithms_enabled():
        raise GraphEncoderError(
            "invalid_stage6_cuda_environment", "deterministic algorithms are required"
        )
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise GraphEncoderError(
            "invalid_stage6_cuda_environment", "TF32 must remain disabled"
        )
    return runtime_identity(torch, device)


def runtime_identity(torch_module, execution_device):
    """Return finite, JSON-compatible CPU/CUDA runtime provenance."""

    device = validate_execution_device(execution_device)
    torch = torch_module
    cuda_available = bool(torch.cuda.is_available())
    gpu = None
    if device == "cuda:0":
        if not cuda_available or torch.cuda.device_count() != 1:
            raise GraphEncoderError("stage6_cuda_unavailable", "cuda:0 unavailable")
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "visible_device_identity": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    cudnn = torch.backends.cudnn.version()
    return {
        "device_policy_version": DEVICE_POLICY_VERSION,
        "execution_device": device,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cuda_available": cuda_available,
        "visible_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu": gpu,
        "cudnn_version": None if cudnn is None else int(cudnn),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cpu_threads": int(torch.get_num_threads()),
        "cpu_host": socket.gethostname(),
        "cpu_processor": platform.processor(),
        "platform": platform.platform(),
        "cuda_rng_preservation_required": device == "cuda:0",
    }


def timing_hardware_identity(runtime):
    """Extract the exact hardware/build identity a producer must match."""

    device = validate_execution_device(runtime.get("execution_device"))
    common = {
        "execution_device": device,
        "torch_version": runtime.get("torch_version"),
        "torch_cuda_build_version": runtime.get("torch_cuda_build_version"),
        "cpu_processor": runtime.get("cpu_processor"),
        "platform": runtime.get("platform"),
        "cpu_threads": runtime.get("cpu_threads"),
    }
    if device == "cuda:0":
        gpu = runtime.get("gpu") or {}
        common["gpu"] = {
            "name": gpu.get("name"),
            "compute_capability": gpu.get("compute_capability"),
            "total_memory_bytes": gpu.get("total_memory_bytes"),
        }
    else:
        common["gpu"] = None
    return common


def require_timing_hardware(runtime, expected):
    observed = timing_hardware_identity(runtime)
    if observed != expected:
        raise GraphEncoderError(
            "stage6_timing_hardware_mismatch", "producer hardware differs from timing"
        )
    return True


def move_tensor_tree(value, execution_device):
    """Move a nested tensor container without changing its public shape."""

    device = validate_execution_device(execution_device)
    if hasattr(value, "to") and hasattr(value, "device"):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tensor_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tensor_tree(item, device) for item in value]
    return value


def assert_tensor_tree_device(value, execution_device, label):
    device = validate_execution_device(execution_device)
    expected = device
    tensors = []

    def visit(item):
        if hasattr(item, "device") and hasattr(item, "dtype"):
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    if not tensors or any(str(item.device) != expected for item in tensors):
        raise GraphEncoderError(
            "stage6_mixed_device", "{} tensor placement differs".format(label)
        )
    return True
