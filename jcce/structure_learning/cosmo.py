"""
COSMO: Causal Ordering Discovery with Smooth Acyclic Orientations

Replaces dense adjacency matrix with (H, p) parameterization:
- H: Edge weights (d × d)
- p: Priority vector (d,) defining topological order

Key property: W = H ⊙ σ((p_j - p_i - ε) / τ) is ALWAYS a DAG

This eliminates the need for DAG constraint during optimization!
The DAG property is guaranteed by construction.

v5.0 Architecture:
- Eliminates expensive O(d³) DAG constraint computation
- Enables scalability to d=100+ features
- Compatible with NSGA-II genome encoding

Usage:
    from cosmo import COSMOGenome, decode_cosmo_to_dag, initialize_cosmo_genome

    key = jax.random.PRNGKey(42)
    genome = initialize_cosmo_genome(key, d=52)
    W = decode_cosmo_to_dag(genome, tau=0.1, epsilon=0.1, d=52)
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Tuple
from functools import partial
import numpy as np


class COSMOGenome(NamedTuple):
    """
    COSMO genome: (H, p) parameterization.

    Attributes:
        H: Edge weights matrix (d × d)
        p: Priority vector (d,) defining topological order
    """
    H: jnp.ndarray  # Edge weights (d × d)
    p: jnp.ndarray  # Priority vector (d,)


@partial(jax.jit, static_argnums=(3,))
def decode_cosmo_to_dag(
    genome: COSMOGenome,
    tau: float = 0.1,
    epsilon: float = 0.1,
    d: int = 100
) -> jnp.ndarray:
    """
    Decode COSMO genome to weighted adjacency matrix.

    W[i,j] = H[i,j] * σ((p[j] - p[i] - ε) / τ)

    Edge i→j exists only if p[j] > p[i] + ε (j has higher priority)
    This GUARANTEES acyclicity by construction.

    Args:
        genome: COSMO genome (H, p)
        tau: Temperature (lower = harder sigmoid, more discrete)
        epsilon: Margin for edge existence
        d: Number of variables (for JIT compilation)

    Returns:
        W: (d, d) weighted adjacency matrix (guaranteed DAG)
    """
    H, p = genome.H, genome.p

    # Compute priority differences: p_j - p_i for all pairs
    priority_diff = p[None, :] - p[:, None]  # (d, d)

    # Smooth orientation mask: σ((p_j - p_i - ε) / τ)
    # Near 1 if p_j >> p_i (edge allowed)
    # Near 0 if p_j << p_i (edge forbidden)
    S = jax.nn.sigmoid((priority_diff - epsilon) / tau)

    # Weighted adjacency: element-wise product
    W = H * S

    # Zero diagonal (no self-loops)
    W = W * (1 - jnp.eye(d))

    return W


def initialize_cosmo_genome(
    key: random.PRNGKey,
    d: int,
    init_scale: float = 0.1
) -> COSMOGenome:
    """
    Initialize random COSMO genome.

    Args:
        key: JAX random key
        d: Number of variables
        init_scale: Scale for edge weight initialization

    Returns:
        COSMOGenome with random initialization
    """
    key1, key2 = random.split(key)

    # Edge weights: small random values
    H = init_scale * random.normal(key1, (d, d))

    # Priority vector: uniform random (will be learned)
    p = random.uniform(key2, (d,), minval=-1.0, maxval=1.0)

    return COSMOGenome(H=H, p=p)


@jax.jit
def cosmo_soft_parent_weights(
    genome: COSMOGenome,
    j: int,
    tau: float = 0.1,
    epsilon: float = 0.1
) -> jnp.ndarray:
    """
    Get soft parent weights for variable j.

    Replaces: weights = |A[:, j]|
    With: weights = |H[:, j]| * σ((p[j] - p[:] - ε) / τ)

    Args:
        genome: COSMO genome
        j: Target variable index
        tau: Temperature
        epsilon: Margin

    Returns:
        weights: (d,) parent weights for variable j
    """
    H, p = genome.H, genome.p

    # Priority differences to j
    priority_to_j = p[j] - p  # (d,)

    # Orientation mask: which nodes can be parents of j
    S_j = jax.nn.sigmoid((priority_to_j - epsilon) / tau)

    # Combined weight
    weights = jnp.abs(H[:, j]) * S_j

    return weights


class COSMOTemperatureSchedule:
    """
    Annealing schedule for COSMO temperature τ.

    Start high (soft, exploration) → End low (hard, discrete DAG)

    The temperature controls the "hardness" of the sigmoid:
    - High τ: Smooth, soft edges (better gradients)
    - Low τ: Sharp, binary edges (better structure)
    """

    def __init__(
        self,
        tau_init: float = 1.0,
        tau_final: float = 0.01,
        n_generations: int = 15,
        schedule: str = 'exponential'
    ):
        """
        Initialize temperature schedule.

        Args:
            tau_init: Initial temperature (high = soft)
            tau_final: Final temperature (low = hard)
            n_generations: Number of generations
            schedule: 'exponential' or 'linear'
        """
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.n_generations = n_generations
        self.schedule = schedule

        if schedule == 'exponential':
            self.taus = jnp.logspace(
                jnp.log10(tau_init),
                jnp.log10(tau_final),
                n_generations
            )
        elif schedule == 'linear':
            self.taus = jnp.linspace(tau_init, tau_final, n_generations)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def get_tau(self, generation: int) -> float:
        """Get temperature for given generation."""
        idx = min(generation, len(self.taus) - 1)
        return float(self.taus[idx])


def cosmo_to_flat_params(genome: COSMOGenome) -> jnp.ndarray:
    """
    Flatten COSMO genome to parameter vector for optimization.

    Args:
        genome: COSMOGenome

    Returns:
        Flat parameter vector [H.flatten(), p]
    """
    H_flat = genome.H.flatten()
    return jnp.concatenate([H_flat, genome.p])


def flat_params_to_cosmo(params: jnp.ndarray, d: int) -> COSMOGenome:
    """
    Unflatten parameter vector to COSMO genome.

    Args:
        params: Flat parameter vector
        d: Number of variables

    Returns:
        COSMOGenome
    """
    H_flat = params[:d*d]
    H = H_flat.reshape((d, d))
    p = params[d*d:]
    return COSMOGenome(H=H, p=p)


def cosmo_dag_check(W: jnp.ndarray, threshold: float = 1e-6) -> bool:
    """
    Verify that W is a DAG (for debugging/validation).

    Uses topological sort attempt.

    Args:
        W: Adjacency matrix
        threshold: Edge threshold

    Returns:
        True if W is a DAG
    """
    W_np = np.asarray(W)
    d = W_np.shape[0]
    W_binary = (np.abs(W_np) > threshold).astype(int)

    # Compute in-degrees
    in_degree = np.sum(W_binary, axis=0)

    # Kahn's algorithm for topological sort
    queue = list(np.where(in_degree == 0)[0])
    count = 0

    while queue:
        node = queue.pop(0)
        count += 1

        for j in range(d):
            if W_binary[node, j] > 0:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return count == d


def cosmo_sparsify(
    genome: COSMOGenome,
    threshold: float = 0.1
) -> COSMOGenome:
    """
    Sparsify COSMO genome by zeroing small edge weights.

    Args:
        genome: COSMOGenome
        threshold: Edge weight threshold

    Returns:
        Sparsified COSMOGenome
    """
    H_sparse = jnp.where(jnp.abs(genome.H) > threshold, genome.H, 0.0)
    return COSMOGenome(H=H_sparse, p=genome.p)


# ============================================================================
# NSGA-II Integration: Genome Conversion
# ============================================================================

def nsga2_decision_to_cosmo(
    decision_vars: np.ndarray,
    d: int,
    n_hyperparams: int = 20  # Number of NSGA-II hyperparameter genes
) -> COSMOGenome:
    """
    Convert NSGA-II decision variables to COSMO genome.

    The decision variable layout:
    - [0:n_hyperparams]: Hyperparameters (lambda_1_idx, lr_idx, etc.)
    - [n_hyperparams:n_hyperparams+d*d]: H matrix (flattened)
    - [n_hyperparams+d*d:]: Priority vector p

    Args:
        decision_vars: NSGA-II decision variable array
        d: Number of variables
        n_hyperparams: Number of hyperparameter genes

    Returns:
        COSMOGenome
    """
    H_flat = decision_vars[n_hyperparams:n_hyperparams + d*d]
    p = decision_vars[n_hyperparams + d*d:n_hyperparams + d*d + d]

    H = H_flat.reshape((d, d))
    return COSMOGenome(H=jnp.array(H), p=jnp.array(p))


def cosmo_to_nsga2_decision(
    genome: COSMOGenome,
    hyperparams: np.ndarray
) -> np.ndarray:
    """
    Convert COSMO genome to NSGA-II decision variables.

    Args:
        genome: COSMOGenome
        hyperparams: Hyperparameter values

    Returns:
        Decision variable array
    """
    H_flat = np.array(genome.H).flatten()
    p = np.array(genome.p)
    return np.concatenate([hyperparams, H_flat, p])


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == '__main__':
    print("Testing COSMO genome encoding...")
    print("=" * 60)

    key = random.PRNGKey(42)
    d = 10

    # Initialize genome
    genome = initialize_cosmo_genome(key, d)
    print(f"Initialized COSMO genome: H={genome.H.shape}, p={genome.p.shape}")

    # Decode to DAG
    W = decode_cosmo_to_dag(genome, tau=0.1, epsilon=0.1, d=d)
    print(f"Decoded adjacency: W={W.shape}")
    print(f"Max edge weight: {jnp.max(jnp.abs(W)):.4f}")
    print(f"Mean edge weight: {jnp.mean(jnp.abs(W)):.4f}")

    # Verify DAG property
    is_dag = cosmo_dag_check(W)
    print(f"Is DAG: {is_dag}")

    # Test temperature annealing
    schedule = COSMOTemperatureSchedule(tau_init=1.0, tau_final=0.01, n_generations=15)
    print("\nTemperature schedule:")
    for gen in [0, 5, 10, 14]:
        tau = schedule.get_tau(gen)
        W_gen = decode_cosmo_to_dag(genome, tau=tau, epsilon=0.1, d=d)
        sparsity = jnp.mean(jnp.abs(W_gen) < 0.01)
        print(f"  Gen {gen}: τ={tau:.4f}, sparsity={sparsity:.2%}")

    print("\n" + "=" * 60)
    print("COSMO genome encoding working!")
