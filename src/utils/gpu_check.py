#!/usr/bin/env python3
"""
GPU validation utilities for Hugging Face Deep RL Course
Usage:
    from gpu_check import ensure_gpu, check_gpu_available
    ensure_gpu()  # Exits if GPU not available
    # or
    if not check_gpu_available():
        print("GPU not available, falling back to CPU")
"""

import sys
import subprocess
import platform
import torch

def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False, f"nvidia-smi failed: {result.stderr}"
        return True, result.stdout.splitlines()[0]
    except FileNotFoundError:
        return False, "nvidia-smi not found - NVIDIA driver may not be installed"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timeout"
    except Exception as e:
        return False, f"nvidia-smi error: {e}"

def check_pytorch_cuda():
    """Check if PyTorch can access CUDA"""
    if not torch.cuda.is_available():
        return False, f"CUDA not available in PyTorch (Platform: {platform.platform()})"

    try:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        return True, f"Found {device_count} GPU(s): {gpu_name} (CUDA {cuda_version})"
    except Exception as e:
        return False, f"Error accessing GPU: {e}"

def check_gpu_available(verbose=True):
    """
    Check if GPU is available and working
    Returns: bool - True if GPU is available
    """
    # Check NVIDIA driver
    driver_ok, driver_msg = check_nvidia_driver()
    if verbose:
        print(f"NVIDIA Driver: {'✓' if driver_ok else '✗'} {driver_msg}")

    # Check PyTorch CUDA
    pytorch_ok, pytorch_msg = check_pytorch_cuda()
    if verbose:
        print(f"PyTorch CUDA: {'✓' if pytorch_ok else '✗'} {pytorch_msg}")

    return driver_ok and pytorch_ok

def ensure_gpu():
    """
    Ensure GPU is available, exit with error if not
    Use this for training scripts that require GPU
    """
    print("Checking GPU availability...")

    if not check_gpu_available(verbose=True):
        sys.exit("\n❌ GPU validation failed. Exiting to prevent CPU fallback.")

    print("✅ GPU validation passed. Ready for training.\n")

def get_device():
    """
    Get the appropriate torch device
    Returns 'cuda' if available, 'cpu' otherwise
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # When run directly, just check and report
    print("=== GPU Check ===")
    available = check_gpu_available(verbose=True)
    print(f"\nGPU Available: {'Yes' if available else 'No'}")
    if not available:
        sys.exit(1)