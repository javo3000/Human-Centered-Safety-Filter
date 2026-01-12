"""
Neural Network architectures for Human-Centered Safety Filter.

Contains V-Network (State Value) and Q-Network (State-Action Value) 
for safety estimation in autonomous systems.
"""
import jax
import jax.numpy as jnp
from jax import nn


class SafetyNetworks:
    """
    Safety value networks for HCSF optimization.
    
    V-Network: Maps state -> safety value (how safe is this state?)
    Q-Network: Maps (state, action) -> safety value (how safe is taking this action?)
    """
    
    def __init__(self, state_dim: int = 133, action_dim: int = 3, hidden_dim: int = 256):
        """
        Initialize network dimensions.
        
        Args:
            state_dim: Dimension of state vector (default 133 for motorsport)
            action_dim: Dimension of action vector (default 3: steering, throttle, brake)
            hidden_dim: Hidden layer dimension
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def init_params(self, seed: int = 42) -> dict:
        """
        Initialize network parameters with Xavier-like initialization.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing 'v' and 'q' network parameters
        """
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Xavier-like scaling for better gradient flow
        v_scale_1 = jnp.sqrt(2.0 / self.state_dim)
        v_scale_2 = jnp.sqrt(2.0 / self.hidden_dim)
        q_scale_1 = jnp.sqrt(2.0 / (self.state_dim + self.action_dim))
        q_scale_2 = jnp.sqrt(2.0 / self.hidden_dim)
        
        # V-Network (State -> 1)
        params_v = {
            'w1': jax.random.normal(k1, (self.state_dim, self.hidden_dim)) * v_scale_1,
            'b1': jnp.zeros(self.hidden_dim),
            'w2': jax.random.normal(k2, (self.hidden_dim, 1)) * v_scale_2,
            'b2': jnp.zeros(1)
        }
        
        # Q-Network (State + Action -> 1)
        # Input dimension: state_dim + action_dim = 133 + 3 = 136
        params_q = {
            'w1': jax.random.normal(k3, (self.state_dim + self.action_dim, self.hidden_dim)) * q_scale_1,
            'b1': jnp.zeros(self.hidden_dim),
            'w2': jax.random.normal(k4, (self.hidden_dim, 1)) * q_scale_2,
            'b2': jnp.zeros(1)
        }
        
        return {'v': params_v, 'q': params_q}


# =============================================================================
# Pure Functions for JIT Compilation
# These are static methods that work with JAX's functional paradigm
# =============================================================================

def v_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass through V-Network.
    
    Args:
        params: Full parameter dictionary (must contain 'v' key)
        x: State tensor of shape (state_dim,) or (batch, state_dim)
    
    Returns:
        Safety value of shape (1,) or (batch, 1)
    """
    v_params = params['v']
    z1 = jnp.dot(x, v_params['w1']) + v_params['b1']
    a1 = nn.relu(z1)
    return jnp.dot(a1, v_params['w2']) + v_params['b2']


def q_forward(params: dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass through Q-Network.
    
    Args:
        params: Full parameter dictionary (must contain 'q' key)
        x: State tensor of shape (state_dim,) or (batch, state_dim)
        u: Action tensor of shape (action_dim,) or (batch, action_dim)
    
    Returns:
        State-action safety value of shape (1,) or (batch, 1)
    """
    q_params = params['q']
    # Concatenate state and action along last axis
    xu = jnp.concatenate([x, u], axis=-1)
    z1 = jnp.dot(xu, q_params['w1']) + q_params['b1']
    a1 = nn.relu(z1)
    return jnp.dot(a1, q_params['w2']) + q_params['b2']
