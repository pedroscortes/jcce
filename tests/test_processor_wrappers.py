"""
Test processor wrappers for MA-Full.
"""

import jax
import jax.numpy as jnp
from jax import random

from jcce.models.processor_wrappers import (
    create_processor,
    get_processor_output_dim,
    MambaProcessorWrapper,
    TransformerProcessor,
    LSTMProcessor,
    GRUProcessor,
)


def test_mamba_processor():
    """Test Mamba processor creation and forward pass."""
    config = {
        'processor_type': 'mamba',
        'd_model': 64,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'n_layers': 2,
    }

    processor = create_processor(config)
    output_dim = get_processor_output_dim(config)

    assert output_dim == 64, f"Expected output_dim=64, got {output_dim}"

    # Test forward pass
    key = random.PRNGKey(0)
    batch_size, n_vars = 8, 5
    z = random.normal(key, (batch_size, n_vars))
    A = jnp.eye(n_vars)  # Simple identity graph

    # Initialize
    variables = processor.init(key, z, A)

    # Forward pass
    h = processor.apply(variables, z, A)

    assert h.shape == (batch_size, n_vars, output_dim), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim}), got {h.shape}"

    print(f"[OK] Mamba processor: input {z.shape} -> output {h.shape}")


def test_transformer_processor():
    """Test Transformer processor creation and forward pass."""
    config = {
        'processor_type': 'transformer',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
    }

    processor = create_processor(config)
    output_dim = get_processor_output_dim(config)

    assert output_dim == 128, f"Expected output_dim=128, got {output_dim}"

    # Test forward pass
    key = random.PRNGKey(0)
    batch_size, n_vars = 8, 5
    z = random.normal(key, (batch_size, n_vars))
    A = jnp.eye(n_vars)

    # Initialize
    variables = processor.init(key, z, A, training=False)

    # Forward pass
    h = processor.apply(variables, z, A, training=False)

    assert h.shape == (batch_size, n_vars, output_dim), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim}), got {h.shape}"

    print(f"[OK] Transformer processor: input {z.shape} -> output {h.shape}")


def test_lstm_processor():
    """Test LSTM processor creation and forward pass."""
    # Unidirectional
    config = {
        'processor_type': 'lstm',
        'hidden_size': 64,
        'n_layers': 2,
        'bidirectional': False,
        'dropout': 0.1,
    }

    processor = create_processor(config)
    output_dim = get_processor_output_dim(config)

    assert output_dim == 64, f"Expected output_dim=64, got {output_dim}"

    # Test forward pass
    key = random.PRNGKey(0)
    batch_size, n_vars = 8, 5
    z = random.normal(key, (batch_size, n_vars))
    A = jnp.eye(n_vars)

    # Initialize
    variables = processor.init(key, z, A, training=False)

    # Forward pass
    h = processor.apply(variables, z, A, training=False)

    assert h.shape == (batch_size, n_vars, output_dim), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim}), got {h.shape}"

    print(f"[OK] LSTM processor (unidirectional): input {z.shape} -> output {h.shape}")

    # Bidirectional
    config['bidirectional'] = True
    processor_bidir = create_processor(config)
    output_dim_bidir = get_processor_output_dim(config)

    assert output_dim_bidir == 128, f"Expected output_dim=128 (bidirectional), got {output_dim_bidir}"

    variables_bidir = processor_bidir.init(key, z, A, training=False)
    h_bidir = processor_bidir.apply(variables_bidir, z, A, training=False)

    assert h_bidir.shape == (batch_size, n_vars, output_dim_bidir), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim_bidir}), got {h_bidir.shape}"

    print(f"[OK] LSTM processor (bidirectional): input {z.shape} -> output {h_bidir.shape}")


def test_gru_processor():
    """Test GRU processor creation and forward pass."""
    # Unidirectional
    config = {
        'processor_type': 'gru',
        'hidden_size': 64,
        'n_layers': 2,
        'bidirectional': False,
        'dropout': 0.1,
    }

    processor = create_processor(config)
    output_dim = get_processor_output_dim(config)

    assert output_dim == 64, f"Expected output_dim=64, got {output_dim}"

    # Test forward pass
    key = random.PRNGKey(0)
    batch_size, n_vars = 8, 5
    z = random.normal(key, (batch_size, n_vars))
    A = jnp.eye(n_vars)

    # Initialize
    variables = processor.init(key, z, A, training=False)

    # Forward pass
    h = processor.apply(variables, z, A, training=False)

    assert h.shape == (batch_size, n_vars, output_dim), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim}), got {h.shape}"

    print(f"[OK] GRU processor (unidirectional): input {z.shape} -> output {h.shape}")

    # Bidirectional
    config['bidirectional'] = True
    processor_bidir = create_processor(config)
    output_dim_bidir = get_processor_output_dim(config)

    assert output_dim_bidir == 128, f"Expected output_dim=128 (bidirectional), got {output_dim_bidir}"

    variables_bidir = processor_bidir.init(key, z, A, training=False)
    h_bidir = processor_bidir.apply(variables_bidir, z, A, training=False)

    assert h_bidir.shape == (batch_size, n_vars, output_dim_bidir), \
        f"Expected shape ({batch_size}, {n_vars}, {output_dim_bidir}), got {h_bidir.shape}"

    print(f"[OK] GRU processor (bidirectional): input {z.shape} -> output {h_bidir.shape}")


def test_all_processors_same_interface():
    """Test that all processors follow the same interface."""
    batch_size, n_vars = 8, 5
    key = random.PRNGKey(42)
    z = random.normal(key, (batch_size, n_vars))
    A = jnp.eye(n_vars)

    processors = [
        {'processor_type': 'mamba', 'd_model': 64, 'd_state': 16, 'd_conv': 4, 'expand': 2, 'n_layers': 2},
        {'processor_type': 'transformer', 'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'd_ff': 128},
        {'processor_type': 'lstm', 'hidden_size': 64, 'n_layers': 2, 'bidirectional': False},
        {'processor_type': 'gru', 'hidden_size': 64, 'n_layers': 2, 'bidirectional': False},
    ]

    for config in processors:
        processor = create_processor(config)
        output_dim = get_processor_output_dim(config)

        # Initialize
        variables = processor.init(key, z, A)

        # Forward pass
        h = processor.apply(variables, z, A)

        # Check shape
        assert h.shape == (batch_size, n_vars, output_dim), \
            f"{config['processor_type']}: Expected shape ({batch_size}, {n_vars}, {output_dim}), got {h.shape}"

    print(f"[OK] All processors follow same interface")


if __name__ == '__main__':
    print("="*60)
    print("Testing Processor Wrappers")
    print("="*60 + "\n")

    test_mamba_processor()
    test_transformer_processor()
    test_lstm_processor()
    test_gru_processor()
    test_all_processors_same_interface()

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
