"""
Training script for the Safety Value Function (V-Network).

Trains V(x) to estimate the safety value of states using
temporal difference learning or supervised learning from demonstrations.
"""
import jax
import jax.numpy as jnp
import optax
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SafetyNetworks, v_forward


def create_optimizer(learning_rate: float = 1e-3):
    """Create Adam optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )


def v_loss_fn(params: dict, states: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error loss for V-network training.
    
    Args:
        params: Network parameters
        states: Batch of states (batch_size, state_dim)
        targets: Target safety values (batch_size, 1)
    
    Returns:
        Scalar MSE loss
    """
    predictions = jax.vmap(lambda s: v_forward(params, s))(states)
    return jnp.mean((predictions - targets) ** 2)


@jax.jit
def train_step(
    params: dict,
    opt_state,
    optimizer,
    states: jnp.ndarray,
    targets: jnp.ndarray
) -> Tuple[dict, any, jnp.ndarray]:
    """
    Single training step with gradient update.
    
    Args:
        params: Current network parameters
        opt_state: Optimizer state
        optimizer: Optax optimizer
        states: Batch of training states
        targets: Batch of target values
    
    Returns:
        Tuple of (updated_params, updated_opt_state, loss)
    """
    loss, grads = jax.value_and_grad(v_loss_fn)(params, states, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def generate_synthetic_data(
    key: jax.random.PRNGKey,
    n_samples: int = 1000,
    state_dim: int = 133
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic training data for demonstration.
    
    In practice, this would be replaced with real trajectory data
    from expert demonstrations or simulation rollouts.
    
    Args:
        key: JAX random key
        n_samples: Number of samples to generate
        state_dim: Dimension of state space
    
    Returns:
        Tuple of (states, safety_values)
    """
    k1, k2 = jax.random.split(key)
    
    # Generate random states
    states = jax.random.normal(k1, (n_samples, state_dim))
    
    # Synthetic safety values based on state norm
    # Higher norm = less safe (simple heuristic for demo)
    state_norms = jnp.linalg.norm(states, axis=1, keepdims=True)
    safety_values = 1.0 / (1.0 + 0.1 * state_norms)  # Bounded (0, 1]
    
    return states, safety_values


def train_v_network(
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42
):
    """
    Main training loop for V-network.
    
    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seed: Random seed
    """
    print("=" * 50)
    print("Training Safety Value Function (V-Network)")
    print("=" * 50)
    
    # Initialize
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    
    # Create networks and initialize parameters
    networks = SafetyNetworks()
    params = networks.init_params(seed)
    
    # Create optimizer
    optimizer = create_optimizer(learning_rate)
    opt_state = optimizer.init(params)
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    states, targets = generate_synthetic_data(k1, n_samples=10000)
    n_samples = states.shape[0]
    n_batches = n_samples // batch_size
    
    print(f"Training samples: {n_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {n_batches}")
    print(f"Epochs: {n_epochs}")
    print()
    
    # Training loop
    for epoch in range(n_epochs):
        # Shuffle data each epoch
        k2, shuffle_key = jax.random.split(k2)
        perm = jax.random.permutation(shuffle_key, n_samples)
        states_shuffled = states[perm]
        targets_shuffled = targets[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_states = states_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            
            params, opt_state, loss = train_step(
                params, opt_state, optimizer, batch_states, batch_targets
            )
            epoch_loss += loss
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    
    # Evaluate on a few test samples
    print("\nEvaluation on test samples:")
    test_states = jax.random.normal(jax.random.PRNGKey(999), (5, 133))
    test_preds = jax.vmap(lambda s: v_forward(params, s))(test_states)
    
    for i, (state, pred) in enumerate(zip(test_states, test_preds)):
        state_norm = jnp.linalg.norm(state)
        print(f"  Sample {i+1}: ||state|| = {state_norm:.3f}, V(state) = {pred[0]:.4f}")
    
    return params


if __name__ == "__main__":
    trained_params = train_v_network()
