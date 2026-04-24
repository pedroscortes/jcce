#!/usr/bin/env python3
"""Test DiffAN on GPU and CPU to find working configuration."""

import sys

import numpy as np

sys.path.insert(0, "/tmp/DiffAN")

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Check if DiffAN has a device parameter
import inspect

from diffan.diffan import DiffAN

sig = inspect.signature(DiffAN.__init__)
print(f"\nDiffAN.__init__ params: {list(sig.parameters.keys())}")

sig_fit = inspect.signature(DiffAN.fit)
print(f"DiffAN.fit params: {list(sig_fit.parameters.keys())}")

# Small test data
np.random.seed(42)
X = np.random.randn(200, 5).astype(np.float32)

# Test 1: CPU tensor
print("\n--- Test 1: CPU float32 tensor ---")
try:
    model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=100)
    X_cpu = torch.tensor(X, device="cpu")
    A, order = model.fit(X_cpu)
    print(f"  SUCCESS: {int(A.sum())} edges")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: GPU float32 tensor
if torch.cuda.is_available():
    print("\n--- Test 2: CUDA float32 tensor ---")
    try:
        model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=100)
        X_gpu = torch.tensor(X, device="cuda:0")
        A, order = model.fit(X_gpu)
        print(f"  SUCCESS: {int(A.sum())} edges")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 3: GPU tensor, move model to GPU
    print("\n--- Test 3: CUDA tensor + model.to(cuda) ---")
    try:
        model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=100)
        # Try to move internal model to GPU
        if hasattr(model, "model"):
            model.model = model.model.to("cuda:0")
            print("  Moved model.model to cuda:0")
        if hasattr(model, "net"):
            model.net = model.net.to("cuda:0")
            print("  Moved model.net to cuda:0")
        X_gpu = torch.tensor(X, device="cuda:0")
        A, order = model.fit(X_gpu)
        print(f"  SUCCESS: {int(A.sum())} edges")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 4: numpy float64 (original DiffAN expected input)
    print("\n--- Test 4: numpy float64 (no tensor) ---")
    try:
        model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=100)
        A, order = model.fit(X.astype(np.float64))
        print(f"  SUCCESS: {int(A.sum())} edges")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 5: numpy float32
    print("\n--- Test 5: numpy float32 (no tensor) ---")
    try:
        model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=100)
        A, order = model.fit(X)
        print(f"  SUCCESS: {int(A.sum())} edges")
    except Exception as e:
        print(f"  FAILED: {e}")

# Test 6: More epochs + larger data (the real use case)
print("\n--- Test 6: numpy float64, 3000 epochs, n=500 ---")
X_big = np.random.randn(500, 5).astype(np.float64)
try:
    model = DiffAN(n_nodes=5, masking=True, residue=True, epochs=3000)
    A, order = model.fit(X_big)
    print(f"  SUCCESS: {int(A.sum())} edges, order={order}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 7: residue=False (Lasso fallback, no R dependency)
print("\n--- Test 7: numpy float64, 3000 epochs, residue=False ---")
try:
    model = DiffAN(n_nodes=5, masking=True, residue=False, epochs=3000)
    A, order = model.fit(X_big)
    print(f"  SUCCESS: {int(A.sum())} edges, order={order}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 8: CUDA_VISIBLE_DEVICES="" to force CPU for everything
print("\n--- Test 8: Check if DiffAN works with real causal data ---")
# Generate data with known structure: X0 -> X1 -> X2
W = np.array([[0, 0.8, 0], [0, 0, 0.6], [0, 0, 0]])
X_causal = np.random.randn(1000, 3)
for i in range(3):
    X_causal[:, i] += X_causal @ W[:, i]
try:
    model = DiffAN(n_nodes=3, masking=True, residue=False, epochs=3000)
    A, order = model.fit(X_causal)
    print(f"  SUCCESS: {int(A.sum())} edges, order={order}")
    print(f"  A = {A}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone.")
