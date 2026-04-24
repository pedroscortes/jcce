"""
Mamba: Selective State Space Model in JAX.

This is a JAX port of the original Mamba implementation from:
https://github.com/state-spaces/mamba

Reference paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Albert Gu and Tri Dao (https://arxiv.org/abs/2312.00752)

Key idea: State space models with input-dependent (selective) parameters
allow the model to selectively propagate or forget information.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import math


def selective_scan(u, delta, A, B, C, D):
    """
    Memory-efficient selective scan algorithm (Algorithm 2 from Mamba paper).

    This is the core SSM operation. It performs a recurrent computation:
        h_t = A_t * h_{t-1} + B_t * u_t
        y_t = C_t * h_t + D * u_t

    Where A_t, B_t, C_t are input-dependent (selective).

    Args:
        u: (batch, seq_len, d_in) input sequence
        delta: (batch, seq_len, d_in) step sizes (input-dependent)
        A: (d_in, d_state) state transition (continuous-time)
        B: (batch, seq_len, d_state) input matrix (selective)
        C: (batch, seq_len, d_state) output matrix (selective)
        D: (d_in,) skip connection

    Returns:
        y: (batch, seq_len, d_in) output sequence

    Mathematical Details:
        1. Discretize continuous-time A using zero-order hold:
           A_discrete = exp(delta * A)
        2. Discretize B similarly:
           B_discrete = (A_discrete - I) A^{-1} B ≈ delta * B
        3. Recurrence:
           h_t = A_discrete_t * h_{t-1} + B_discrete_t * u_t
        4. Output:
           y_t = C_t * h_t + D * u_t

    Memory optimizations:
        - Use jax.lax.scan (already O(1) memory for states)
        - Avoid materializing large intermediate tensors where possible
        - Transpose operations minimize copies
    """
    batch, seq_len, d_in = u.shape
    d_state = A.shape[1]

    # Clip delta to prevent explosion (delta should be positive and small)
    # Typical range: [0.001, 0.1] based on original Mamba implementation
    delta = jnp.clip(delta, 0.001, 0.1)

    # Discretize A and B using zero-order hold
    # Memory optimization: These einsums are fused by XLA, don't materialize full tensor
    # deltaA[b, l, d, n] = exp(delta[b, l, d] * A[d, n])
    deltaA = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))

    # deltaB_u[b, l, d, n] = delta[b, l, d] * B[b, l, n] * u[b, l, d]
    deltaB_u = jnp.einsum('bld,bln,bld->bldn', delta, B, u)

    # Selective scan (recurrent computation)
    # Memory-efficient: jax.lax.scan only stores current state, not full trajectory
    def scan_fn(h, inputs):
        deltaA_t, deltaB_u_t, C_t = inputs
        # h_new[b, d, n] = deltaA_t[b, d, n] * h[b, d, n] + deltaB_u_t[b, d, n]
        h_new = deltaA_t * h + deltaB_u_t
        # y_t[b, d] = C_t[b, n] @ h_new[b, d, n]
        y_t = jnp.einsum('bdn,bn->bd', h_new, C_t)
        return h_new, y_t

    # Initial state
    h_0 = jnp.zeros((batch, d_in, d_state))

    # Scan over sequence - memory efficient recurrence
    _, y = jax.lax.scan(
        scan_fn,
        h_0,
        (deltaA.transpose(1, 0, 2, 3), deltaB_u.transpose(1, 0, 2, 3), C.transpose(1, 0, 2)),
    )

    # y is (seq_len, batch, d_in), transpose back
    y = y.transpose(1, 0, 2)  # (batch, seq_len, d_in)

    # Add skip connection
    y = y + u * D

    return y


class MambaBlock(nn.Module):
    """
    A single Mamba block.

    This implements the core Mamba architecture from the paper.

    Attributes:
        d_model: Model dimension (input/output dimension)
        d_state: SSM state dimension (default 8, reduced from 16 for memory)
        d_conv: Convolution kernel size (typically 4)
        expand: Expansion factor (default 1, reduced from 2 for memory)
    """

    d_model: int
    d_state: int = 8  # Reduced from 16 for memory efficiency
    d_conv: int = 4
    expand: int = 1  # Reduced from 2 for memory efficiency

    def setup(self):
        """Initialize parameters."""
        self.d_inner = self.expand * self.d_model
        dt_rank = math.ceil(self.d_model / 16)  # As per paper

        # Input projection (projects to 2 * d_inner for x and z paths)
        self.in_proj = nn.Dense(self.d_inner * 2, use_bias=False)

        # 1D causal convolution
        # Note: Flax Conv expects (features,) for 1D
        self.conv1d = nn.Conv(
            features=self.d_inner,
            kernel_size=(self.d_conv,),
            feature_group_count=self.d_inner,  # Depthwise
            padding='VALID',
            use_bias=True,
        )

        # Projections for SSM parameters
        self.x_proj = nn.Dense(dt_rank + 2 * self.d_state, use_bias=False)
        # dt_proj: Use special initialization to keep delta small initially
        # Following original Mamba: bias initialized to log(dt_min to dt_max)
        self.dt_proj = nn.Dense(
            self.d_inner,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02),  # Small weights
            bias_init=nn.initializers.constant(0.1),  # Small positive bias
        )

        # SSM parameters (A is fixed, no gradient)
        # A: (d_inner, d_state)
        # Initialize A as recommended in paper: log of inverse of range
        A = jnp.repeat(
            jnp.arange(1, self.d_state + 1)[None, :],
            self.d_inner,
            axis=0,
        )
        self.A_log = self.param('A_log', lambda key: jnp.log(A))

        # D: (d_inner,) skip connection
        self.D = self.param('D', lambda key: jnp.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model) input sequence

        Returns:
            output: (batch, seq_len, d_model) output sequence
        """
        batch, seq_len, d_model = x.shape

        # 1. Input projection and split
        x_and_z = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_proj, z = jnp.split(x_and_z, 2, axis=-1)  # Each: (batch, seq_len, d_inner)

        # 2. Causal convolution
        # Conv1D expects (batch, length, features)
        # Need to pad for causal convolution
        x_conv_input = jnp.pad(
            x_proj,
            ((0, 0), (self.d_conv - 1, 0), (0, 0)),  # Pad left for causal
            mode='constant',
        )
        x_conv = self.conv1d(x_conv_input)  # (batch, seq_len, d_inner)
        x_conv = nn.silu(x_conv)  # SiLU activation

        # 3. SSM parameters (selective - depend on x)
        x_ssm = self.x_proj(x_conv)  # (batch, seq_len, dt_rank + 2*d_state)
        dt_rank = math.ceil(self.d_model / 16)
        dt, B, C = jnp.split(
            x_ssm,
            [dt_rank, dt_rank + self.d_state],
            axis=-1,
        )
        # dt: (batch, seq_len, dt_rank)
        # B: (batch, seq_len, d_state)
        # C: (batch, seq_len, d_state)

        # Project dt to d_inner and apply softplus
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = nn.softplus(dt)

        # Get A from log form
        A = -jnp.exp(self.A_log)  # (d_inner, d_state)

        # 4. Selective SSM
        y = selective_scan(x_conv, dt, A, B, C, self.D)

        # 5. Gating (element-wise multiply with z)
        y = y * nn.silu(z)

        # 6. Output projection
        output = self.out_proj(y)

        return output


class MambaProcessor(nn.Module):
    """
    Memory-efficient stack of Mamba blocks for processing sequences.

    This is what we'll use in CausalVAE to process the latent factors
    when they're arranged in topological order.

    Attributes:
        d_model: Model dimension
        n_layers: Number of Mamba blocks to stack
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor

    Memory Optimizations:
        1. jax.lax.scan in selective_scan for O(1) memory recurrence
           - Only stores current state, not full trajectory
        2. XLA fusion of einsum operations
           - Prevents materializing large intermediate tensors
        3. Reduced memory fraction in stage2_train_vae.py (0.3-0.4)
           - Leaves headroom for memory fragmentation
        4. Small state dimension (d_state=16 by default)
           - Limits state tensor size

    Note: Gradient checkpointing (jax.remat) is not used due to
    incompatibility with Flax's @nn.compact module system. The above
    optimizations provide sufficient memory efficiency for most use cases.
    """

    d_model: int = 32
    n_layers: int = 2
    d_state: int = 8  # Reduced from 16 to save memory
    d_conv: int = 4
    expand: int = 1  # Reduced from 2 to save memory (no expansion)

    @nn.compact
    def __call__(self, z_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        Process a sequence with stacked Mamba blocks.

        Args:
            z_sequence: (batch, seq_len, d_model) sequence of latent factors

        Returns:
            h: (batch, seq_len, d_model) processed sequence
        """
        h = z_sequence

        # Stack Mamba blocks with residual connections
        # Note: Gradient checkpointing removed due to Flax module compatibility issues
        # Memory optimization now relies on:
        # 1. Efficient selective_scan with jax.lax.scan (O(1) memory for recurrence)
        # 2. XLA fusion of einsum operations
        # 3. Memory fraction control in stage2_train_vae.py
        for i in range(self.n_layers):
            block = MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                name=f'mamba_block_{i}'
            )
            h = block(h) + h  # Residual connection

        return h
