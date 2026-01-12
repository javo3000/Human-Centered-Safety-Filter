"""
JaxSafeFilter - Human-Centered Safety Filter

End-to-end execution demonstrating the HCSF algorithm:
State -> Safe Action filtering with minimal intervention.
"""
import jax.numpy as jnp
from src.model import SafetyNetworks, v_forward, q_forward
from src.filter import hcsf_optimization, compute_intervention_cost, check_safety_constraint


def run_filter_demo():
    """Demonstrate the HCSF algorithm on mock data."""
    
    print("=" * 60)
    print("JaxSafeFilter - Human-Centered Safety Filter Demo")
    print("=" * 60)
    
    # Initialize networks and parameters
    print("\n1. Initializing Safety Networks...")
    nn_builder = SafetyNetworks(state_dim=133, action_dim=3, hidden_dim=256)
    params = nn_builder.init_params(seed=42)
    print(f"   V-Network: {nn_builder.state_dim} -> {nn_builder.hidden_dim} -> 1")
    print(f"   Q-Network: {nn_builder.state_dim + nn_builder.action_dim} -> {nn_builder.hidden_dim} -> 1")
    
    # Create mock driving scenario
    print("\n2. Setting up driving scenario...")
    
    # 133D state vector (e.g., car telemetry, track position, etc.)
    current_state = jnp.zeros(133)
    
    # Human wants to turn hard right with full throttle (potentially unsafe)
    human_input = jnp.array([0.5, 1.0, 0.0])  # [steering, throttle, brake]
    
    # Safe fallback: straighten wheel, release throttle, apply brakes
    fallback_input = jnp.array([0.0, 0.0, 1.0])
    
    print(f"   State: 133D vector (zeros for demo)")
    print(f"   Human Input:    [Steering={human_input[0]:.2f}, Throttle={human_input[1]:.2f}, Brake={human_input[2]:.2f}]")
    print(f"   Fallback Input: [Steering={fallback_input[0]:.2f}, Throttle={fallback_input[1]:.2f}, Brake={fallback_input[2]:.2f}]")
    
    # Check initial safety values
    print("\n3. Safety Analysis...")
    v_current = v_forward(params, current_state)
    q_human = q_forward(params, current_state, human_input)
    q_fallback = q_forward(params, current_state, fallback_input)
    
    print(f"   V(state) = {v_current[0]:.6f} (State safety value)")
    print(f"   Q(state, human_action) = {q_human[0]:.6f}")
    print(f"   Q(state, fallback_action) = {q_fallback[0]:.6f}")
    
    gamma = 0.7
    threshold = gamma * v_current[0]
    print(f"   Safety threshold (γ·V) = {threshold:.6f}")
    
    human_safe = check_safety_constraint(params, current_state, human_input, gamma)
    print(f"   Human action safe? {bool(human_safe)}")
    
    # Execute HCSF optimization
    print("\n4. Running HCSF Optimization...")
    print(f"   Searching 2000 candidates along u_human → u_fallback...")
    
    u_safe, alpha_used = hcsf_optimization(
        params, 
        current_state, 
        human_input, 
        fallback_input,
        gamma=gamma,
        n_candidates=2000
    )
    
    # Compute intervention metrics
    intervention_cost = compute_intervention_cost(human_input, u_safe)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n   Human Input:     {human_input}")
    print(f"   Safe Output:     {u_safe}")
    print(f"   Fallback Input:  {fallback_input}")
    print(f"\n   Alpha (Intervention Level): {float(alpha_used):.4f}")
    print(f"     → 0.0 = No intervention (pure human)")
    print(f"     → 1.0 = Full override (pure fallback)")
    print(f"\n   Intervention Cost (L2²): {float(intervention_cost):.6f}")
    
    # Verify final action safety
    q_safe = q_forward(params, current_state, u_safe)
    print(f"\n   Q(state, safe_action) = {q_safe[0]:.6f}")
    print(f"   Constraint: Q >= γ·V? {float(q_safe[0]) >= float(threshold)}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def run_scenario_comparison():
    """Compare filter behavior across different scenarios."""
    
    print("\n" + "=" * 60)
    print("Scenario Comparison: Varying Human Aggression")
    print("=" * 60)
    
    nn_builder = SafetyNetworks()
    params = nn_builder.init_params(seed=42)
    
    state = jnp.zeros(133)
    fallback = jnp.array([0.0, 0.0, 1.0])
    
    scenarios = [
        ("Conservative", jnp.array([0.1, 0.3, 0.2])),
        ("Moderate", jnp.array([0.3, 0.6, 0.0])),
        ("Aggressive", jnp.array([0.5, 1.0, 0.0])),
        ("Very Aggressive", jnp.array([0.8, 1.0, 0.0])),
    ]
    
    print(f"\n{'Scenario':<18} {'Human Input':<24} {'Alpha':<10} {'Safe Output'}")
    print("-" * 80)
    
    for name, human_input in scenarios:
        u_safe, alpha = hcsf_optimization(params, state, human_input, fallback)
        print(f"{name:<18} {str(human_input.tolist()):<24} {float(alpha):.4f}     {u_safe.tolist()}")


if __name__ == "__main__":
    run_filter_demo()
    run_scenario_comparison()
