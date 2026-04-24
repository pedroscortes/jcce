"""
Synthetic Non-Linear Structural Causal Model (SCM) Generator.

Generates data with non-linear causal relationships for testing v3.0 processors.

Supported non-linearity types:
- Polynomial: X_j = Σ W_ij × X_i² + ε
- Sigmoidal: X_j = tanh(Σ W_ij × X_i) + ε
- Multiplicative: X_j = (Π X_i^W_ij) + ε
- Mixed: Combination of above

Used for:
1. Validating that v3.0 processors can learn non-linear relationships
2. Comparing linear (v2.0) vs non-linear (v3.0) methods
3. Understanding when processors help vs hurt
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Optional, Literal


def generate_dag_erdos_renyi(
    n_vars: int,
    edge_prob: float = 0.3,
    key: random.PRNGKey = None
) -> jnp.ndarray:
    """
    Generate random DAG using Erdős-Rényi model.

    Args:
        n_vars: Number of variables
        edge_prob: Probability of edge between any two variables
        key: JAX random key

    Returns:
        A: (n_vars, n_vars) adjacency matrix (strict upper triangular)
            A[i, j] != 0 means edge i → j (i < j to ensure DAG)
    """
    if key is None:
        key = random.PRNGKey(42)

    # Generate random weights
    key, weight_key = random.split(key)
    W = random.uniform(weight_key, (n_vars, n_vars), minval=0.5, maxval=2.0)

    # Generate random edges
    key, edge_key = random.split(key)
    edges = random.bernoulli(edge_key, p=edge_prob, shape=(n_vars, n_vars))

    # Enforce DAG: Keep only upper triangular (i < j)
    A = jnp.triu(W * edges, k=1)

    return A


def generate_dag_chain(n_vars: int) -> jnp.ndarray:
    """
    Generate chain DAG: X₁ → X₂ → X₃ → ... → X_n

    Args:
        n_vars: Number of variables

    Returns:
        A: (n_vars, n_vars) adjacency matrix (chain structure)
    """
    A = jnp.zeros((n_vars, n_vars))
    for i in range(n_vars - 1):
        A = A.at[i, i+1].set(1.0)
    return A


def generate_dag_fork(n_vars: int) -> jnp.ndarray:
    """
    Generate fork DAG: X₁ → {X₂, X₃, ..., X_n}

    Root causes all other variables.

    Args:
        n_vars: Number of variables

    Returns:
        A: (n_vars, n_vars) adjacency matrix (fork structure)
    """
    A = jnp.zeros((n_vars, n_vars))
    A = A.at[0, 1:].set(1.0)
    return A


def generate_nonlinear_scm_data(
    A: jnp.ndarray,
    n_samples: int,
    nonlinearity: Literal['polynomial', 'sigmoidal', 'multiplicative', 'mixed'] = 'sigmoidal',
    noise_scale: float = 0.5,
    key: random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate data from non-linear SCM given DAG structure.

    Forward model: X = f(A, noise)

    Args:
        A: (n_vars, n_vars) adjacency matrix (DAG)
        n_samples: Number of samples to generate
        nonlinearity: Type of non-linear relationship
        noise_scale: Standard deviation of noise
        key: JAX random key

    Returns:
        X: (n_samples, n_vars) generated data
        A: (n_vars, n_vars) ground truth DAG (same as input)
    """
    if key is None:
        key = random.PRNGKey(42)

    n_vars = A.shape[0]

    # Generate noise
    key, noise_key = random.split(key)
    noise = random.normal(noise_key, (n_samples, n_vars)) * noise_scale

    # Initialize X
    X = jnp.zeros((n_samples, n_vars))

    # Forward pass: Generate data in topological order
    # Since A is upper triangular, we can go left to right
    for j in range(n_vars):
        # Get parents of variable j
        parents = jnp.where(jnp.abs(A[:, j]) > 1e-6)[0]

        if len(parents) == 0:
            # Root variable: Just noise
            X = X.at[:, j].set(noise[:, j])
        else:
            # Non-linear function of parents
            X_parents = X[:, parents]  # (n_samples, n_parents)
            W_parents = A[parents, j]   # (n_parents,) weights

            if nonlinearity == 'polynomial':
                # X_j = Σ W_i × X_i² + ε
                X_j = jnp.sum(W_parents * (X_parents ** 2), axis=1) + noise[:, j]

            elif nonlinearity == 'sigmoidal':
                # X_j = tanh(Σ W_i × X_i) + ε
                linear_combination = jnp.sum(W_parents * X_parents, axis=1)
                X_j = jnp.tanh(linear_combination) + noise[:, j]

            elif nonlinearity == 'multiplicative':
                # X_j = (Π X_i^W_i) + ε
                # Use log-exp trick for numerical stability
                log_product = jnp.sum(W_parents * jnp.log(jnp.abs(X_parents) + 1e-6), axis=1)
                X_j = jnp.exp(log_product) + noise[:, j]

            elif nonlinearity == 'mixed':
                # Mix of polynomial and sigmoidal
                linear_combination = jnp.sum(W_parents * X_parents, axis=1)
                polynomial_part = jnp.sum(W_parents * (X_parents ** 2), axis=1)
                X_j = 0.5 * jnp.tanh(linear_combination) + 0.5 * polynomial_part + noise[:, j]

            else:
                raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

            X = X.at[:, j].set(X_j)

    return X, A


def generate_nonlinear_classification_data(
    n_samples: int = 500,
    n_vars: int = 5,
    graph_type: Literal['chain', 'fork', 'erdos_renyi'] = 'chain',
    nonlinearity: Literal['polynomial', 'sigmoidal', 'multiplicative', 'mixed'] = 'sigmoidal',
    noise_scale: float = 0.5,
    key: random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate non-linear SCM data with binary classification target.

    Args:
        n_samples: Number of samples
        n_vars: Number of feature variables (target will be n_vars+1)
        graph_type: DAG structure type
        nonlinearity: Type of non-linear relationships
        noise_scale: Noise level
        key: JAX random key

    Returns:
        X: (n_samples, n_vars) feature data
        Y: (n_samples,) binary classification target
        A_true: (n_vars, n_vars) ground truth DAG (features only)
    """
    if key is None:
        key = random.PRNGKey(42)

    # Generate DAG structure (n_vars + 1 variables, last is target)
    n_total = n_vars + 1

    if graph_type == 'chain':
        A_full = generate_dag_chain(n_total)
    elif graph_type == 'fork':
        A_full = generate_dag_fork(n_total)
    elif graph_type == 'erdos_renyi':
        key, dag_key = random.split(key)
        A_full = generate_dag_erdos_renyi(n_total, edge_prob=0.3, key=dag_key)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Ensure target (last variable) has at least 2 parents
    if jnp.sum(jnp.abs(A_full[:, -1]) > 0) < 2:
        # Add edges from first 2 variables to target
        A_full = A_full.at[0, -1].set(1.0)
        A_full = A_full.at[1, -1].set(1.0)

    # Generate continuous data
    key, data_key = random.split(key)
    X_full, _ = generate_nonlinear_scm_data(
        A_full, n_samples, nonlinearity, noise_scale, data_key
    )

    # Split into features and target
    X = X_full[:, :-1]  # Features
    Y_continuous = X_full[:, -1]  # Continuous target

    # Binarize target
    Y = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.int32)

    # Extract feature subgraph (exclude target)
    A_true = A_full[:-1, :-1]

    return X, Y, A_true


# ============================================================================
# Quick Test Functions
# ============================================================================

def test_nonlinear_scm():
    """Quick test of non-linear SCM generator."""
    print("Testing Non-Linear SCM Generator")
    print("="*60)

    # Generate chain DAG
    n_vars = 5
    A = generate_dag_chain(n_vars)
    print(f"\nDAG structure: chain ({n_vars} variables)")
    print(f"Edges: {int(jnp.sum(jnp.abs(A) > 0))}")

    # Test different non-linearities
    for nonlinearity in ['polynomial', 'sigmoidal', 'multiplicative', 'mixed']:
        print(f"\nNonlinearity: {nonlinearity}")
        X, _ = generate_nonlinear_scm_data(A, n_samples=100, nonlinearity=nonlinearity)
        print(f"  Data shape: {X.shape}")
        print(f"  Mean: {jnp.mean(X):.3f}, Std: {jnp.std(X):.3f}")

    # Test classification data
    print(f"\nClassification Data:")
    X, Y, A_true = generate_nonlinear_classification_data(
        n_samples=200, n_vars=4, graph_type='chain', nonlinearity='sigmoidal'
    )
    print(f"  X: {X.shape}")
    print(f"  Y: {Y.shape}, classes: {jnp.unique(Y)}")
    print(f"  Y distribution: {jnp.bincount(Y)}")
    print(f"  Ground truth DAG edges: {int(jnp.sum(jnp.abs(A_true) > 0))}")

    print("\n" + "="*60)
    print("[OK] All tests passed!")


if __name__ == '__main__':
    test_nonlinear_scm()
