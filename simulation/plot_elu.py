import numpy as np
import matplotlib.pyplot as plt
import jax.nn as jnn
import jax.numpy as jnp

from ant_simulation import (
    PHEROMONE_MAX_TIMESTEP, MAX_PHEROMONE_STRENGTH, PHEROMONE_ELU_TRANSITION_FRAC, PHEROMONE_ELU_STEEPNESS,
    DT,
)

alpha = 1.0 # Standard ELU alpha

# --- Calculation ---
T_max = PHEROMONE_MAX_TIMESTEP * DT
k = jnp.maximum(1e-6, PHEROMONE_ELU_STEEPNESS)
x_offset = jnp.clip(PHEROMONE_ELU_TRANSITION_FRAC, 1e-6, 1.0 - 1e-6)

# Calculate A and B
denom_A = k * (1.0 - x_offset) - (jnp.exp(-k * x_offset) - 1.0)
A = MAX_PHEROMONE_STRENGTH / (denom_A + 1e-9)
B = -A * (jnp.exp(-k * x_offset) - 1.0)

# Generate time points
times = np.linspace(0, T_max, 500)

# Calculate strength at each time point using the same logic as in update_step
x_elu_vals = k * (times / (T_max + 1e-9) - x_offset)
elu_vals = jnn.elu(x_elu_vals, alpha=alpha)
strength_raw_vals = A * elu_vals + B
individual_strengths = np.clip(strength_raw_vals, 0.0, MAX_PHEROMONE_STRENGTH)

# Transition time
transition_time = x_offset * T_max

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(times, individual_strengths, label='Individual Pheromone Strength (ELU-based)')
plt.axvline(transition_time, color='r', linestyle='--', label=f'Transition Point (t={transition_time:.1f})')
plt.title('Individual Pheromone Strength Over Time')
plt.xlabel('Time (simulation units)')
plt.ylabel('Individual Pheromone Strength')
plt.ylim(bottom=-0.05, top=MAX_PHEROMONE_STRENGTH + 0.05)
plt.grid(True)
plt.legend()
plt.show()
