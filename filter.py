"""
Human-Centered Safety Filter (HCSF) Optimization.

Implements the 2000-candidate line search algorithm for finding
the minimally-intervening safe action.
"""
import jax
import jax.numpy as jnp
from functools import partial

from src.model import v_forward, q_forward


@partial(jax.jit, static_argnums=())
def hcsf_optimization(
    params: dict,
    state: jnp.ndarray,
    u_human: jnp.ndarray,
    u_fallback: jnp.ndarray,
    gamma: float = 0.7,
    n_candidates: int = 2000
) -> tuple:
    """
    Human-Centered Safety Filter optimization.
    
    Finds u* that minimizes ||u - u_human||² subject to Q(x, u) >= gamma * V(x).
    
    Uses a line search along the segment from u_human to u_fallback,
    finding the first (smallest alpha) action that satisfies safety.
    
    Args:
        params: Network parameters dictionary with 'v' and 'q' keys
        state: Current state vector of shape (state_dim,)
        u_human: Human's desired action of shape (action_dim,)
        u_fallback: Safe fallback action of shape (action_dim,)
        gamma: Safety margin factor (0 < gamma <= 1)
        n_candidates: Number of candidates in line search
    
    Returns:
        tuple: (u_safe, alpha_used)
            - u_safe: The safe action closest to human intent
            - alpha_used: The interpolation factor used (0 = fully human, 1 = fully fallback)
    """
    # 1. Evaluate current state safety value
    v_val = v_forward(params, state)
    safety_threshold = gamma * v_val
    
    # 2. Create alpha values from 0.0 to 1.0
    # Alpha = 0 means u_human, Alpha = 1 means u_fallback
    alphas = jnp.linspace(0.0, 1.0, n_candidates)
    
    # 3. Generate candidate actions via linear interpolation (LERP)
    # u_candidate = (1 - alpha) * u_human + alpha * u_fallback
    def get_candidate(alpha):
        return (1.0 - alpha) * u_human + alpha * u_fallback
    
    u_candidates = jax.vmap(get_candidate)(alphas)  # Shape: (n_candidates, action_dim)
    
    # 4. Batch evaluate all candidates through Q-network
    # Need to broadcast state for vmap
    def eval_q(u):
        return q_forward(params, state, u)
    
    q_vals = jax.vmap(eval_q)(u_candidates)  # Shape: (n_candidates, 1)
    q_vals = q_vals.squeeze(-1)  # Shape: (n_candidates,)
    
    # 5. Find the first candidate that satisfies safety constraint
    # Q(x, u) >= gamma * V(x)
    safe_mask = q_vals >= safety_threshold.squeeze()
    
    # Use argmax to find the first True (1) in the mask
    # If no candidate is safe, argmax returns 0, so we fall back to u_human
    # But we should actually fall back to u_fallback (alpha=1) if nothing is safe
    
    # Create a weighted version: we want the FIRST safe action (minimum alpha)
    # If safe_mask is all False, we default to the last candidate (u_fallback)
    any_safe = jnp.any(safe_mask)
    
    # Trick: use argmax on safe_mask to find first True
    # If all False, argmax returns 0, but we want n_candidates-1 (u_fallback)
    first_safe_idx = jnp.argmax(safe_mask)
    best_idx = jnp.where(any_safe, first_safe_idx, n_candidates - 1)
    
    u_safe = u_candidates[best_idx]
    alpha_used = alphas[best_idx]
    
    return u_safe, alpha_used


@jax.jit
def compute_intervention_cost(u_human: jnp.ndarray, u_safe: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the squared L2 distance between human input and safe action.
    
    This represents the "cost" of safety intervention - smaller is better.
    
    Args:
        u_human: Original human action
        u_safe: Safety-filtered action
    
    Returns:
        Squared L2 distance (intervention magnitude)
    """
    return jnp.sum((u_human - u_safe) ** 2)


@jax.jit  
def check_safety_constraint(
    params: dict,
    state: jnp.ndarray,
    action: jnp.ndarray,
    gamma: float = 0.7
) -> bool:
    """
    Check if an action satisfies the safety constraint.
    
    Args:
        params: Network parameters
        state: Current state
        action: Action to check
        gamma: Safety margin
    
    Returns:
        True if Q(x, u) >= gamma * V(x), False otherwise
    """
    v_val = v_forward(params, state)
    q_val = q_forward(params, state, action)
    return q_val >= gamma * v_val
