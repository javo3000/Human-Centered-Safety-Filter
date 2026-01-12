"""
Utility functions for JaxSafeFilter.
Math helpers for neural network operations.
"""
import jax.numpy as jnp


def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)


def normalize_action(action, action_min=-1.0, action_max=1.0):
    """
    Normalize action to be within specified bounds.
    
    Args:
        action: Input action tensor
        action_min: Minimum action value
        action_max: Maximum action value
    
    Returns:
        Clipped action within bounds
    """
    return jnp.clip(action, action_min, action_max)


def lerp(a, b, t):
    """
    Linear interpolation between a and b.
    
    Args:
        a: Start value
        b: End value  
        t: Interpolation factor (0 to 1)
    
    Returns:
        Interpolated value: (1-t)*a + t*b
    """
    return (1.0 - t) * a + t * b


def compute_action_distance(u1, u2):
    """
    Compute the Euclidean distance between two actions.
    
    Args:
        u1: First action vector
        u2: Second action vector
    
    Returns:
        Euclidean distance (L2 norm of difference)
    """
    return jnp.linalg.norm(u1 - u2)
