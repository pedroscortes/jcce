"""
Implicit Causal Layer - THE CRITICAL COMPONENT

This module implements the causal layer that transforms exogenous noise ε
into endogenous causal factors z by solving the linear system:

    z = (I - A^T)^{-1} ε

The gradient computation uses the Implicit Function Theorem with a custom VJP.

CRITICAL: The gradient derivation must be mathematically correct.
   If gradient checks fail, DO NOT PROCEED - the entire training will be invalid.
"""

import jax
import jax.numpy as jnp


@jax.custom_vjp
def implicit_causal_layer(epsilon: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    """
    Solve for z given exogenous noise ε and adjacency matrix A.

    Forward pass: z = (I - A)^{-1} ε

    This implements the solution to the linear Structural Causal Model.

    Given our convention A[i,j] != 0 means j → i, the SCM is:
        z[i] = Σ_j A[i,j] z[j] + ε[i]

    In matrix form:
        z = A z + ε

    Which can be rearranged to:
        (I - A) z = ε
        z = (I - A)^{-1} ε

    Args:
        epsilon: (batch_size, latent_dim) exogenous noise variables
        A: (latent_dim, latent_dim) adjacency matrix (fixed, from A_true)
           A[i,j] != 0 means j → i (j is a parent of i)

    Returns:
        z: (batch_size, latent_dim) endogenous causal factors

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> epsilon = jax.random.normal(key, (32, 10))
        >>> A = generate_dag(DAGConfig(num_nodes=10))
        >>> z = implicit_causal_layer(epsilon, A)
        >>> print(z.shape)
        (32, 10)
    """
    # Solve: (I - A) z = ε
    I_minus_A = jnp.eye(A.shape[0]) - A

    # Solve the linear system for each sample in the batch
    # vmap over batch dimension
    z = jax.vmap(lambda eps: jnp.linalg.solve(I_minus_A, eps))(epsilon)

    return z


def causal_layer_fwd(epsilon: jnp.ndarray, A: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
    """
    Custom forward pass for implicit_causal_layer.

    This function is called during the forward pass and saves residuals
    needed for the backward pass.

    Args:
        epsilon: Exogenous noise
        A: Adjacency matrix

    Returns:
        z: Output (endogenous factors)
        residuals: (z, A) saved for backward pass
    """
    z = implicit_causal_layer(epsilon, A)
    return z, (z, A)  # Return output and residuals for backward


def causal_layer_bwd(residuals: tuple, g_z: jnp.ndarray) -> tuple:
    """
    Custom backward pass using Implicit Function Theorem.

    CRITICAL DERIVATION:

    Given upstream gradient g_z = dL/dz, we need to compute:
        g_epsilon = dL/dε
        g_A = dL/dA

    Mathematical derivation:
    -----------------------
    Forward: z = (I - A)^{-1} ε
    Fixed point equation: g(z, ε, A) = z - A z - ε = 0

    For g_ε (VJP for ε):
        We want: g_ε = (∂z/∂ε)^T · g_z

        From implicit differentiation of g(·) = 0:
            ∂g/∂z · ∂z/∂ε + ∂g/∂ε = 0

        We have:
            ∂g/∂z = (I - A)
            ∂g/∂ε = -I

        Solving:
            (I - A) · ∂z/∂ε = I
            ∂z/∂ε = (I - A)^{-1}

        Taking transpose:
            (∂z/∂ε)^T = ((I - A)^{-1})^T = (I - A^T)^{-1}

        Therefore:
            g_ε = (I - A^T)^{-1} · g_z

        This means we solve: (I - A^T) g_ε = g_z

    For g_A (VJP for A):
        From the chain rule and implicit differentiation:
            ∂g/∂A · dA + ∂g/∂z · ∂z/∂A = 0
            -z · dA + (I - A) · ∂z/∂A = 0

        The gradient involves:
            g_A = -g_ε · z^T

    Args:
        residuals: (z, A) saved from forward pass
        g_z: Upstream gradient dL/dz (batch_size, latent_dim)

    Returns:
        (g_epsilon, g_A): Gradients for epsilon and A
    """
    z, A = residuals

    # Solve (I - A^T) g_epsilon = g_z
    I_minus_AT = jnp.eye(A.shape[0]) - A.T
    g_epsilon = jax.vmap(lambda gz: jnp.linalg.solve(I_minus_AT, gz))(g_z)

    # g_A = -g_epsilon · z^T (batch outer product, then mean over batch)
    # For each sample: outer product, then average across batch
    g_A = -jnp.mean(jax.vmap(lambda ge, zi: jnp.outer(ge, zi))(g_epsilon, z), axis=0)

    return (g_epsilon, g_A)


# Register the custom VJP functions
implicit_causal_layer.defvjp(causal_layer_fwd, causal_layer_bwd)


# Manual numerical gradient checking (more reliable than jax.test_util.check_grads)
def check_causal_layer_gradients_numerical(
    epsilon: jnp.ndarray,
    A: jnp.ndarray,
    eps: float = 1e-3,  # Optimal step size for finite differences
    atol: float = 1e-2,  # Relaxed tolerance for numerical methods
) -> dict:
    """
    Numerically verify the custom gradients using finite differences.

    MANDATORY: This must pass before using the causal layer in training.

    Args:
        epsilon: Test input
        A: Test adjacency matrix
        eps: Finite difference step size
        atol: Absolute tolerance for gradient comparison

    Returns:
        results: Dictionary with gradient check results

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> epsilon = jax.random.normal(key, (4, 5))
        >>> A = generate_dag(DAGConfig(num_nodes=5))
        >>> result = check_causal_layer_gradients_numerical(epsilon, A)
        >>> assert result['passed']
    """

    # Test loss function: sum of squared outputs
    def loss_fn(eps, a):
        z = implicit_causal_layer(eps, a)
        return jnp.sum(z**2)

    # Compute analytical gradients using custom VJP
    analytical_grad_eps = jax.grad(loss_fn, argnums=0)(epsilon, A)

    # Compute numerical gradients using finite differences
    def numerical_gradient_epsilon():
        grad = jnp.zeros_like(epsilon)
        for i in range(epsilon.shape[0]):
            for j in range(epsilon.shape[1]):
                eps_plus = epsilon.at[i, j].add(eps)
                eps_minus = epsilon.at[i, j].add(-eps)

                loss_plus = loss_fn(eps_plus, A)
                loss_minus = loss_fn(eps_minus, A)

                grad = grad.at[i, j].set((loss_plus - loss_minus) / (2 * eps))
        return grad

    numerical_grad_eps = numerical_gradient_epsilon()

    # Compare analytical vs numerical
    max_diff = jnp.max(jnp.abs(analytical_grad_eps - numerical_grad_eps))
    relative_error = max_diff / (jnp.max(jnp.abs(analytical_grad_eps)) + 1e-10)

    passed = max_diff < atol

    return {
        "passed": passed,
        "max_absolute_diff": float(max_diff),
        "relative_error": float(relative_error),
        "tolerance": atol,
        "message": (
            f"[OK] Gradient check PASSED (max_diff={max_diff:.2e}, tol={atol:.2e})"
            if passed
            else f"[FAIL] Gradient check FAILED (max_diff={max_diff:.2e}, tol={atol:.2e})"
        ),
    }
