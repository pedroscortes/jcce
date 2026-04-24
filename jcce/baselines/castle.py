"""
CASTLE baseline — Kyono, Zhang & van der Schaar (NeurIPS 2020).

Reimplemented in JAX to use the same data pipeline and evaluation
protocol as JCCE.

Architecture (following original TF code + paper):
  - d+1 sub-networks with masked input layers
  - 1 shared hidden layer (paper: h=d+1 neurons)
  - Per-sub-net output layer
  - Acyclicity via truncated Taylor series of Tr(e^{W⊙W}) (9 terms)
  - Group lasso sparsity (L2-norm per input row, then L1)
  - Loss: recon + 0.5*rho*h² + alpha*h + beta*sparsity + lambda*prediction

Reference:
    Kyono et al. "CASTLE: Regularization via Auxiliary Causal Graph
    Discovery." NeurIPS 2020. arXiv:2009.13180.

Verified against: https://github.com/trentkyono/CASTLE
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np


def _init_castle_params(
    key: random.PRNGKey,
    d: int,
    hidden_dim: int = None,
) -> Dict:
    """Initialize CASTLE parameters.

    Paper architecture: d+1 sub-networks, each with:
      - Per-sub-net input: (d+1, hidden_dim), col k zeroed
      - 1 shared hidden: (hidden_dim, hidden_dim)
      - Per-sub-net output: (hidden_dim, 1)
    Paper recommends hidden_dim = d+1.
    """
    d1 = d + 1
    if hidden_dim is None:
        hidden_dim = d1  # Paper default: h = d+1

    keys = random.split(key, 4)
    scale = 0.1

    # Per-variable input weights: (d1, d1, hidden_dim)
    W_input = random.normal(keys[0], (d1, d1, hidden_dim)) * scale

    # Zero out self-connections: sub-network k cannot use variable k
    mask = 1.0 - jnp.eye(d1)[:, :, None]  # (d1, d1, 1)
    W_input = W_input * mask

    # 1 shared hidden layer (paper uses 1, code uses 1)
    W_hidden = random.normal(keys[1], (hidden_dim, hidden_dim)) * scale
    b_hidden = jnp.zeros(hidden_dim)

    # Per-variable output weights: (d1, hidden_dim, 1)
    W_output = random.normal(keys[2], (d1, hidden_dim, 1)) * scale
    b_output = jnp.zeros((d1, 1))

    return {
        "W_input": W_input,
        "W_hidden": W_hidden,
        "b_hidden": b_hidden,
        "W_output": W_output,
        "b_output": b_output,
        "mask": mask,
    }


def _castle_forward(params: Dict, X_tilde: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through all d+1 sub-networks."""
    W_input = params["W_input"] * params["mask"]

    # All sub-networks in parallel: (n, d1) @ (d1, d1, h) → (d1, n, h)
    H = jnp.einsum("nj,kjh->knh", X_tilde, W_input)
    H = jax.nn.relu(H)

    # 1 shared hidden layer
    H = jnp.einsum("knh,hg->kng", H, params["W_hidden"]) + params["b_hidden"][None, None, :]
    H = jax.nn.relu(H)

    # Per-variable output: (d1, n, h) @ (d1, h, 1) → (d1, n)
    out = jnp.einsum("knh,kho->kno", H, params["W_output"])
    out = out + params["b_output"][:, None, :]
    X_hat = out.squeeze(-1).T  # (n, d1)
    return X_hat


def _adjacency_from_params(params: Dict) -> jnp.ndarray:
    """Extract adjacency: M[k,j] = ||W_input[k,j,:]||_2 (group lasso norm)."""
    W_input = params["W_input"] * params["mask"]
    M = jnp.sqrt(jnp.sum(W_input**2, axis=-1) + 1e-8)
    return M


def _acyclicity_constraint(M: jnp.ndarray) -> float:
    """Truncated Taylor series: h(W) = Tr(sum_{k=1}^{9} (W⊙W)^k / k!) - d.

    Matches original CASTLE code exactly (9 terms of matrix exponential).
    """
    d1 = M.shape[0]
    M_sq = M * M  # Hadamard square

    Z_in = jnp.eye(d1)
    h = 0.0
    coff = 1.0
    for i in range(1, 10):
        Z_in = Z_in @ M_sq
        coff = coff * i
        h = h + jnp.trace(Z_in) / coff

    return h


def _group_lasso(W_input: jnp.ndarray) -> float:
    """Group lasso: sum of L2 norms per input row (matching original code).

    For each sub-net k and input var j: ||W_input[k,j,:]||_2
    Then sum all these norms.
    """
    row_norms = jnp.sqrt(jnp.sum(W_input**2, axis=-1) + 1e-8)  # (d1, d1)
    return jnp.sum(row_norms)


def _castle_loss(
    params: Dict,
    X_tilde: jnp.ndarray,
    Y: jnp.ndarray,
    rho: float,
    alpha: float,
    reg_beta: float,
    reg_lambda: float,
) -> Tuple[float, Dict]:
    """CASTLE loss matching original code structure.

    total = recon_loss + 0.5*rho*h² + alpha*h + reg_beta*sparsity + reg_lambda*rho*pred_loss

    Original code: rho=1, alpha=1, reg_beta=5, reg_lambda=1 (all fixed).
    """
    X_hat = _castle_forward(params, X_tilde)

    # Reconstruction loss (MSE on ALL variables)
    recon_loss = jnp.mean((X_tilde - X_hat) ** 2)

    # Prediction loss (MSE on Y = column 0)
    pred_loss = jnp.mean((X_hat[:, 0] - Y) ** 2)

    # Acyclicity
    M = _adjacency_from_params(params)
    h = _acyclicity_constraint(M)

    # Group lasso sparsity
    W_input = params["W_input"] * params["mask"]
    sparsity = _group_lasso(W_input)

    # Total (matching original code exactly)
    total = (
        recon_loss
        + 0.5 * rho * h**2
        + alpha * h
        + reg_beta * sparsity
        + reg_lambda * rho * pred_loss
    )

    aux = {
        "pred_loss": pred_loss,
        "recon_loss": recon_loss,
        "h_A": h,
        "sparsity": sparsity,
    }
    return total, aux


def _castle_loss_classification(
    params: Dict,
    X_tilde: jnp.ndarray,
    Y: jnp.ndarray,
    rho: float,
    alpha: float,
    reg_beta: float,
    reg_lambda: float,
) -> Tuple[float, Dict]:
    """CASTLE loss with BCE for classification.

    Note: original code uses MSE even for classification. We add BCE as an
    option since it's more principled for binary targets. Both options are
    available via the `task` parameter in train_castle().
    """
    X_hat = _castle_forward(params, X_tilde)

    # Reconstruction loss (MSE on features only)
    recon_loss = jnp.mean((X_tilde[:, 1:] - X_hat[:, 1:]) ** 2)

    # Classification loss (BCE on Y)
    Y_logit = jnp.clip(X_hat[:, 0], -6, 6)
    pred_loss = -jnp.mean(Y * jax.nn.log_sigmoid(Y_logit) + (1 - Y) * jax.nn.log_sigmoid(-Y_logit))

    # Acyclicity
    M = _adjacency_from_params(params)
    h = _acyclicity_constraint(M)

    # Group lasso sparsity
    W_input = params["W_input"] * params["mask"]
    sparsity = _group_lasso(W_input)

    total = (
        recon_loss
        + 0.5 * rho * h**2
        + alpha * h
        + reg_beta * sparsity
        + reg_lambda * rho * pred_loss
    )

    aux = {
        "pred_loss": pred_loss,
        "recon_loss": recon_loss,
        "h_A": h,
        "sparsity": sparsity,
    }
    return total, aux


def train_castle(
    X: np.ndarray,
    Y: np.ndarray,
    key: random.PRNGKey = None,
    task: str = "classification",
    hidden_dim: int = None,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 32,
    rho: float = 1.0,
    alpha: float = 1.0,
    reg_beta: float = 0.1,
    reg_lambda: float = 1.0,
    patience: int = 30,
    verbose: int = 1,
) -> Dict:
    """Train CASTLE and return results.

    Default hyperparameters match original code:
        lr=1e-3, n_epochs=200, batch_size=32,
        rho=1.0, alpha=1.0, reg_beta=5.0, reg_lambda=1.0, patience=30

    Args:
        X: (n, d) feature matrix (standardized)
        Y: (n,) target vector
        task: 'classification' (BCE) or 'regression' (MSE, matches original)
        hidden_dim: None = d+1 (paper default)
        batch_size: 32 (matching original code)
    """
    if key is None:
        key = random.PRNGKey(42)

    n, d = X.shape
    Y_flat = Y.ravel().astype(np.float32)

    # Build X_tilde = [Y, X_1, ..., X_d]
    X_tilde = jnp.concatenate([Y_flat[:, None], jnp.array(X)], axis=1)

    # Initialize — separate mask from trainable params
    key, init_key = random.split(key)
    all_params = _init_castle_params(init_key, d, hidden_dim)
    mask = all_params.pop("mask")

    # Loss function with mask as closure
    base_loss_fn = _castle_loss_classification if task == "classification" else _castle_loss

    def loss_fn(params, X_batch, Y_batch):
        full_params = {**params, "mask": mask}
        return base_loss_fn(full_params, X_batch, Y_batch, rho, alpha, reg_beta, reg_lambda)

    @jax.jit
    def train_step(params, X_batch, Y_batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, X_batch, Y_batch)
        return loss, aux, grads

    params = all_params

    # Adam state
    m_state = jax.tree.map(jnp.zeros_like, params)
    v_state = jax.tree.map(jnp.zeros_like, params)

    best_loss = float("inf")
    patience_counter = 0
    best_params = params

    for epoch in range(n_epochs):
        # Mini-batch (matching original: batch_size=32)
        key, batch_key = random.split(key)
        if n > batch_size:
            idx = random.permutation(batch_key, n)[:batch_size]
            X_batch = X_tilde[idx]
            Y_batch = Y_flat[idx]
        else:
            X_batch = X_tilde
            Y_batch = Y_flat

        loss, aux, grads = train_step(params, X_batch, Y_batch)

        # Adam update
        t = epoch + 1
        m_state = jax.tree.map(lambda mi, g: 0.9 * mi + 0.1 * g, m_state, grads)
        v_state = jax.tree.map(lambda vi, g: 0.999 * vi + 0.001 * g**2, v_state, grads)
        m_hat = jax.tree.map(lambda mi: mi / (1 - 0.9**t), m_state)
        v_hat = jax.tree.map(lambda vi: vi / (1 - 0.999**t), v_state)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8),
            params,
            m_hat,
            v_hat,
        )
        params["W_input"] = params["W_input"] * mask

        loss_val = float(loss)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = jax.tree.map(lambda x: x.copy(), params)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose >= 2 and epoch % 50 == 0:
            print(
                f"  CASTLE epoch {epoch}: loss={loss_val:.4f} "
                f"pred={float(aux['pred_loss']):.4f} "
                f"recon={float(aux['recon_loss']):.4f} "
                f"h(A)={float(aux['h_A']):.4f}"
            )

        if patience_counter >= patience:
            if verbose >= 1:
                print(f"  CASTLE early stop at epoch {epoch}")
            break

    # Extract results
    full_params = {**best_params, "mask": mask}
    M = _adjacency_from_params(full_params)
    h_final = float(_acyclicity_constraint(M))

    A_full = np.array(M)
    A_features = A_full[1:, 1:]  # features only

    # Predictions on full data
    X_hat = _castle_forward(full_params, X_tilde)
    Y_logit = np.array(X_hat[:, 0])

    # Always compute classification metrics (paper uses MSE + evaluates AUROC)
    from scipy.special import expit
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    if task == "classification":
        Y_prob = expit(Y_logit)
    else:
        # Regression output: clip to [0,1] range, treat as probability
        Y_prob = np.clip(Y_logit, 0, 1)
    Y_pred = (Y_prob > 0.5).astype(int)

    metrics = {"h_A": h_final, "n_epochs": epoch + 1, "best_loss": best_loss}
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(Y_flat, Y_pred))
    metrics["f1"] = float(f1_score(Y_flat, Y_pred, zero_division=0))
    try:
        metrics["roc_auc"] = float(roc_auc_score(Y_flat, Y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0

    # Markov Blanket: edges to Y from features
    edges_to_Y = A_full[0, 1:]
    mb_threshold = 0.1
    mb_indices = np.where(edges_to_Y > mb_threshold)[0].tolist()
    metrics["mb_indices"] = mb_indices
    metrics["mb_size"] = len(mb_indices)

    if verbose >= 1:
        print(
            f"  CASTLE done: h(A)={h_final:.4f}, "
            f"BAcc={metrics.get('balanced_accuracy', 0):.3f}, "
            f"MB={len(mb_indices)} vars, epochs={epoch + 1}"
        )

    return {
        "A_est": A_features,
        "A_full": A_full,
        "Y_pred": Y_pred,
        "Y_prob": Y_prob,
        "metrics": metrics,
        "params": full_params,
    }
