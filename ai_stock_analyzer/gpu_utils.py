"""
GPU detection and usage utilities for AI Stock Analyzer.

This module provides functions to:
- Detect available GPU hardware (CUDA, MPS, or CPU)
- Report GPU memory usage
- Configure PyTorch device for model inference
- Display real-time GPU utilization
"""

import os
import sys
import platform


def get_device_info() -> dict:
    """
    Detect available compute device (GPU/CPU) and return detailed info.

    Returns
    -------
    dict
        Dictionary containing:
          - device       : torch.device object to use for inference
          - device_name  : Human-readable device name
          - device_type  : 'cuda', 'mps', or 'cpu'
          - gpu_available: True if a GPU is available
          - gpu_count    : Number of CUDA GPUs found
          - memory_info  : GPU memory stats (if applicable)
    """
    info = {
        "device": None,
        "device_name": "CPU",
        "device_type": "cpu",
        "gpu_available": False,
        "gpu_count": 0,
        "memory_info": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            gpu_index = 0
            info["device"] = torch.device("cuda", gpu_index)
            info["device_name"] = torch.cuda.get_device_name(gpu_index)
            info["device_type"] = "cuda"
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["memory_info"] = {
                "total_gb": torch.cuda.get_device_properties(gpu_index).total_memory / 1e9,
                "allocated_gb": torch.cuda.memory_allocated(gpu_index) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(gpu_index) / 1e9,
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon GPU
            info["device"] = torch.device("mps")
            info["device_name"] = "Apple Silicon GPU (MPS)"
            info["device_type"] = "mps"
            info["gpu_available"] = True
            info["gpu_count"] = 1
        else:
            info["device"] = torch.device("cpu")

    except ImportError:
        # PyTorch not installed – fall back to CPU-only descriptor
        info["device_type"] = "cpu"
        info["device_name"] = "CPU (PyTorch not installed)"

    return info


def print_device_report() -> dict:
    """
    Print a formatted GPU/device status report and return the device info dict.

    Returns
    -------
    dict
        Same structure as returned by ``get_device_info()``.
    """
    info = get_device_info()

    border = "=" * 60
    print(border)
    print("  GPU / COMPUTE DEVICE STATUS")
    print(border)
    print(f"  Platform      : {platform.system()} {platform.release()}")
    print(f"  Python        : {sys.version.split()[0]}")

    try:
        import torch
        print(f"  PyTorch       : {torch.__version__}")
    except ImportError:
        print("  PyTorch       : NOT INSTALLED")

    print(f"  Device Type   : {info['device_type'].upper()}")
    print(f"  Device Name   : {info['device_name']}")
    print(f"  GPU Available : {'YES ✓' if info['gpu_available'] else 'NO  ✗  (running on CPU)'}")

    if info["gpu_count"] > 1:
        print(f"  GPU Count     : {info['gpu_count']}")

    if info["memory_info"]:
        m = info["memory_info"]
        print(f"  VRAM Total    : {m['total_gb']:.2f} GB")
        print(f"  VRAM Used     : {m['allocated_gb']:.2f} GB")
        print(f"  VRAM Reserved : {m['reserved_gb']:.2f} GB")

    if not info["gpu_available"]:
        print()
        print("  NOTE: No GPU detected. Analysis will run on CPU.")
        print("  To enable GPU acceleration:")
        print("    1. Install CUDA-enabled PyTorch:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("    2. Verify with: python -c \"import torch; print(torch.cuda.is_available())\"")

    print(border)
    return info


def get_model_device():
    """
    Return the best available torch.device for model inference.

    Returns
    -------
    torch.device
    """
    info = get_device_info()
    if info["device"] is not None:
        return info["device"]

    try:
        import torch
        return torch.device("cpu")
    except ImportError:
        return None
