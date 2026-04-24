"""
GPU Configuration and Multi-Device Utilities for JAX

This module provides utilities for:
- Detecting available GPUs
- Configuring JAX for multi-GPU usage
- Data parallelism helpers for NSGA-II
- GPU state management and reset
"""

import os
import jax
import jax.numpy as jnp
from jax import device_put, pmap
from typing import List, Dict, Any, Callable, Optional
import gc


def detect_gpus() -> Dict[str, Any]:
    """
    Detect available GPUs and return detailed information.

    Returns:
        Dictionary containing:
        - num_gpus: Number of available GPUs
        - devices: List of JAX device objects
        - device_info: List of device descriptions
        - total_memory_gb: Total VRAM across all GPUs (approximate)
    """
    devices = jax.devices("gpu")

    info = {
        "num_gpus": len(devices),
        "devices": devices,
        "device_info": [str(d) for d in devices],
        "backend": jax.default_backend()
    }

    # Try to estimate memory (RTX 4090 = 24GB each)
    # This is approximate - actual detection requires nvidia-smi parsing
    if len(devices) > 0:
        # Rough estimate based on typical configurations
        info["estimated_memory_per_gpu_gb"] = 24 if "4090" in str(devices[0]) else 8
        info["total_estimated_memory_gb"] = info["estimated_memory_per_gpu_gb"] * len(devices)

    return info


def configure_jax_multi_gpu(memory_fraction: float = 0.9, preallocate: bool = False,
                            cache_dir: Optional[str] = None):
    """
    Configure JAX for optimal multi-GPU usage.

    Args:
        memory_fraction: Fraction of GPU memory to preallocate (0.0-1.0)
                        0.9 = use 90% of available VRAM
        preallocate: Whether to preallocate GPU memory. Should be False for
                    multi-process parallelism to avoid OOM in worker processes.
        cache_dir: Path for persistent XLA compilation cache. When provided,
                  compiled XLA programs are saved to disk, eliminating ~30-60s
                  recompilation on restart. Default None disables persistent cache.
    """
    # For multi-process (multi-GPU), disable preallocation to allow workers to allocate memory
    if preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # CRITICAL: Still set MEM_FRACTION to limit memory growth even without preallocation
        # This prevents XLA from consuming all available memory
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    # Use async GPU allocator to prevent fragmentation (recommended by TensorFlow/JAX)
    # This uses cudaMallocAsync instead of BFC allocator for better memory management
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Enable GPU persistence mode (if not already enabled via nvidia-smi)
    # This prevents CUDA context unloading during idle periods
    os.environ["CUDA_VISIBLE_DEVICES_ORDER"] = "PCI_BUS_ID"

    # Disable JIT cache size limits for large populations
    os.environ["JAX_COMPILATION_CACHE_SIZE"] = "10000"

    # Persistent compilation cache: saves compiled XLA programs to disk
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("[OK] JAX configured for multi-GPU usage")
    print(f"  Memory fraction: {memory_fraction}")
    print(f"  Preallocate: {'enabled' if preallocate else 'disabled'}")
    if cache_dir is not None:
        print(f"  Persistent cache: {cache_dir}")


def reset_gpu_state_comprehensive():
    """
    Comprehensive GPU state reset to prevent CUDA_ERROR_LAUNCH_FAILED.

    This should be called:
    - Between experiments
    - After OOM errors
    - Periodically during long runs (every N genomes)
    """
    # Clear all JAX caches
    jax.clear_caches()

    # Force complete backend reset (releases GPU context)
    # WARNING: This recompiles all JIT functions on next use
    try:
        jax.clear_backends()
    except Exception as e:
        print(f"  [WARN] Backend clear failed: {e}")

    # Force Python garbage collection
    gc.collect()

    print("[OK] GPU state reset complete")


def reset_gpu_state_lite():
    """
    Lightweight GPU cleanup (doesn't reset backend).

    Use this during experiments (e.g., after each genome evaluation)
    to prevent memory buildup without recompilation overhead.
    """
    jax.clear_caches()
    gc.collect()


def split_population_across_gpus(population: List[Any], devices: List[Any]) -> List[List[Any]]:
    """
    Split NSGA-II population across multiple GPUs for data parallelism.

    Args:
        population: List of genomes to evaluate
        devices: List of JAX GPU devices

    Returns:
        List of sublists, one per GPU

    Example:
        population = [g1, g2, g3, g4, g5, g6]  # 6 genomes
        devices = [gpu0, gpu1]                  # 2 GPUs
        → [[g1, g2, g3], [g4, g5, g6]]         # 3 per GPU
    """
    num_gpus = len(devices)
    pop_size = len(population)

    # Calculate chunk size (round up to handle uneven splits)
    chunk_size = (pop_size + num_gpus - 1) // num_gpus

    # Split population into chunks
    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, pop_size)
        if start_idx < pop_size:
            chunks.append(population[start_idx:end_idx])

    return chunks


def create_parallel_evaluator(eval_fn: Callable, devices: List[Any]) -> Callable:
    """
    Create a parallelized version of an evaluation function using pmap.

    Args:
        eval_fn: Function to parallelize (must be JAX-compatible)
        devices: List of GPU devices to use

    Returns:
        Parallelized function that runs across all GPUs

    Note:
        The input function must be pure (no side effects) and JAX-compatible.
        For NSGA-II, this works best for the inner GOLEM optimization loop.
    """
    return pmap(eval_fn, devices=devices)


def get_optimal_batch_size(num_features: int, num_samples: int, num_gpus: int,
                          memory_per_gpu_gb: float = 24.0) -> int:
    """
    Calculate optimal batch size based on dataset and GPU memory.

    Args:
        num_features: Number of features (adjacency matrix will be num_features×num_features)
        num_samples: Number of training samples
        num_gpus: Number of available GPUs
        memory_per_gpu_gb: VRAM per GPU in GB (24 for RTX 4090)

    Returns:
        Recommended batch size for GOLEM optimization

    Formula:
        - Adjacency matrix: num_features × num_features × 4 bytes (float32)
        - Gradient accumulation: ~4× matrix size
        - Safety margin: Use only 60% of available memory for batches
    """
    # Size of adjacency matrix in GB
    matrix_size_gb = (num_features * num_features * 4) / (1024**3)

    # GOLEM needs ~4× matrix size for forward/backward/optimizer states
    golem_memory_per_iter_gb = matrix_size_gb * 4

    # Total available memory across all GPUs (use 60% for safety)
    available_memory_gb = memory_per_gpu_gb * num_gpus * 0.6

    # How many samples can we process in parallel?
    max_batch_size = int(available_memory_gb / golem_memory_per_iter_gb)

    # Don't exceed number of samples
    optimal_batch_size = min(max_batch_size, num_samples)

    # Minimum batch size = 16 (for stability)
    optimal_batch_size = max(16, optimal_batch_size)

    return optimal_batch_size


def print_gpu_info():
    """Print detailed GPU configuration information."""
    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)

    info = detect_gpus()

    print(f"Backend: {info['backend']}")
    print(f"Number of GPUs: {info['num_gpus']}")
    print()

    if info['num_gpus'] > 0:
        print("Detected devices:")
        for i, device_str in enumerate(info['device_info']):
            print(f"  GPU {i}: {device_str}")

        if "estimated_memory_per_gpu_gb" in info:
            print()
            print(f"Estimated VRAM per GPU: {info['estimated_memory_per_gpu_gb']} GB")
            print(f"Total estimated VRAM: {info['total_estimated_memory_gb']} GB")
    else:
        print("[WARN] No GPUs detected! Running on CPU.")

    print("=" * 80)
    print()

    return info


if __name__ == "__main__":
    # Test GPU detection
    print_gpu_info()

    # Test configuration
    configure_jax_multi_gpu(memory_fraction=0.9)

    # Test population splitting
    print("\nTesting population splitting:")
    population = list(range(10))  # Mock population of 10 genomes
    devices = jax.devices("gpu")

    if len(devices) > 0:
        chunks = split_population_across_gpus(population, devices)
        print(f"  Population size: {len(population)}")
        print(f"  Number of GPUs: {len(devices)}")
        print(f"  Chunks: {chunks}")

        # Test batch size calculation
        print("\nOptimal batch sizes for different datasets:")
        configs = [
            ("Heart Disease", 14, 270),
            ("TEP (30 features)", 30, 4000),
            ("TEP (52 features)", 52, 10000),
        ]

        for name, n_features, n_samples in configs:
            batch_size = get_optimal_batch_size(
                n_features, n_samples, len(devices),
                memory_per_gpu_gb=24.0
            )
            print(f"  {name:25} (n={n_features:2}): batch_size={batch_size:5}")
    else:
        print("  No GPUs available for testing")
