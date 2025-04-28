import os
import jax
import jax.nn
import jax.numpy as jnp
import jax.random as random
from functools import partial # <<< NEW >>> For cleaner jitting

try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('..') # Or provide a default path

# --- Simulation Parameters ---
NUM_ANTS = 64
ARENA_RADIUS = 50.0  # mm
ANT_LENGTH = 2.5     # mm
ANT_WIDTH = ANT_LENGTH / 2.0
ANT_RADIUS = ANT_LENGTH / 2.0
K_PUSH = 0.1

DT = 0.1

# --- Pheromone Parameters ---
USE_GRID_PHEROMONES = False # Flag to switch between grid and direct detection
PHEROMONE_RADIUS = ANT_LENGTH * 2.0 # Range for direct detection OR grid sampling radius
DISCRETE_PHEROMONE = True   # Applies only if USE_GRID_PHEROMONES = False
PHEROMONE_THRESHOLD = 2.5 # Signal strength threshold for 50% stop probability
PHEROMONE_STEEPNESS = 4.0 # Sigmoid steepness
MAX_PHEROMONE_STRENGTH = 0.9 # Max individual pheromone strength/deposition rate
PHEROMONE_MAX_TIMESTEP = 250 # Time (in state) to reach max strength/deposition
PHEROMONE_ELU_TRANSITION_FRAC = 0.4
PHEROMONE_ELU_STEEPNESS = 5.0

# --- Grid Pheromone Parameters (Only used if USE_GRID_PHEROMONES = True) ---
GRID_RESOLUTION = 100 # Number of cells along each axis of the pheromone grid
GRID_CELL_SIZE = 2.0 * ARENA_RADIUS / GRID_RESOLUTION # Size of each grid cell
PHEROMONE_GRID_RADIUS_CELLS = int(jnp.ceil(PHEROMONE_RADIUS / GRID_CELL_SIZE)) # Radius in grid cells for sampling
PHEROMONE_DECAY_RATE = 0.99 # Multiplicative decay factor per DT
PHEROMONE_DEPOSITION_RATE = 1.0 # Scaling factor for deposition amount per DT

# --- Wall Interaction Parameters ---
WALL_ZONE_WIDTH = ANT_LENGTH * 1.5
WALL_AVOID_STRENGTH = 0.9

# --- State constants ---
STATE_RESTING = 0
STATE_MOVING_BURST = 1
STATE_ARRESTED = 2

# --- State durations and burst speed parameters ---
MEAN_REST_DURATION = 3.0
STD_REST_DURATION = 2.5
MEAN_BURST_DURATION = 7.0
STD_BURST_DURATION = 2.5
MEAN_BURST_SPEED = 6.0
STD_BURST_SPEED = 1.0
MEAN_ARREST_DURATION = 5.5
STD_ARREST_DURATION = 3.0
MIN_STATE_DURATION = 0.2
TURN_RATE_STD = 1.4
ARREST2BURST_GRACE_PERIOD = 2.0

# --- Helper Functions ---

def wrap_angle(angle):
    """Helper function to wrap angles to [-pi, pi]"""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

def draw_durations(key, mean, std, num_samples):
    """Draws durations from Normal dist, clipped at MIN_STATE_DURATION."""
    durations = mean + random.normal(key, (num_samples,)) * std
    return jnp.maximum(MIN_STATE_DURATION, durations)

# Function to calculate individual pheromone strength based on time in state
@jax.jit
def calculate_individual_pheromone_strength(time_in_state):
    """Calculates the pheromone strength/deposition rate for each ant based on its time_in_state."""
    T_max = PHEROMONE_MAX_TIMESTEP
    k = PHEROMONE_ELU_STEEPNESS
    x_offset = PHEROMONE_ELU_TRANSITION_FRAC
    alpha = 1.0 # Standard ELU alpha

    # Ensure parameters are valid
    k = jnp.maximum(1e-6, k)
    x_offset = jnp.clip(x_offset, 1e-6, 1.0 - 1e-6)

    # Calculate scaling constants A and B
    denom_A = k * (1.0 - x_offset) - (jnp.exp(-k * x_offset) - 1.0)
    A = MAX_PHEROMONE_STRENGTH / (denom_A + 1e-9)
    B = -A * (jnp.exp(-k * x_offset) - 1.0)

    # Calculate the input to ELU based on individual time_in_state
    t_clamped = jnp.clip(time_in_state, 0.0, T_max)
    x_elu = k * (t_clamped / (T_max + 1e-9) - x_offset)

    # Calculate the ELU-based strength for each ant
    elu_val = jax.nn.elu(x_elu, alpha=alpha)
    strength_raw = A * elu_val + B

    # Clip final strength
    individual_strength = jnp.clip(strength_raw, 0.0, MAX_PHEROMONE_STRENGTH)
    return individual_strength

# Grid Helper Functions
@partial(jax.jit, static_argnames=('grid_resolution'))
def pos_to_grid_idx(position, arena_radius, grid_resolution):
    """Converts world coordinates (x, y) to grid indices (row, col)."""
    # Scale positions from [-arena_radius, arena_radius] to [0, grid_resolution]
    # Add epsilon to avoid index -1 for position exactly at -arena_radius
    idx = (position + arena_radius) / (2 * arena_radius) * grid_resolution + 1e-6
    # Clip to ensure indices are within [0, grid_resolution - 1]
    idx_clipped = jnp.clip(idx, 0, grid_resolution - 1)
    return idx_clipped.astype(jnp.int32)

@partial(jax.jit, static_argnames=('radius_cells', 'grid_resolution'))
def sample_grid_radius(ant_pos_idx, pheromone_map, radius_cells, grid_resolution):
    """Samples the pheromone map in a square region around the ant's grid index."""
    # ant_pos_idx: (2,) array [row, col]
    # radius_cells: Static integer radius
    # grid_resolution: Static integer grid size

    # Pad the map with zeros to handle boundary cases easily.
    # Padding width is 'radius_cells' on all sides.
    padded_map = jnp.pad(pheromone_map, pad_width=radius_cells, mode='constant', constant_values=0)

    # The size of the square slice is fixed (static)
    slice_size = 2 * radius_cells + 1

    # Calculate the start indices in the *padded* map.
    # The original ant index (row, col) corresponds to (row + radius_cells, col + radius_cells)
    # in the padded map. We want to start slicing 'radius_cells' before this center.
    row_center, col_center = ant_pos_idx[0], ant_pos_idx[1]

    padded_start_row = row_center # Indexing starts from 0, so row_center is row_center cells from the top padding
    padded_start_col = col_center # Similarly for columns

    # Perform the dynamic slice on the padded map with a *static* slice size
    # start_indices are dynamic, but slice_sizes must be static for JIT.
    sampled_region = jax.lax.dynamic_slice(
        padded_map,
        (padded_start_row, padded_start_col), # Top-left corner of slice in padded map
        (slice_size, slice_size) # Static size
    )
    total_pheromone = jnp.sum(sampled_region)

    return total_pheromone

# mapped version for all ants
vmapped_sample_grid_radius = jax.vmap(sample_grid_radius, in_axes=(0, None, None, None))


# --- Initialization ---

def initialise_state(key, num_ants, arena_radius, grid_resolution): # Added num_ants, grid_res
    """Initialises the state of all ants including the pheromone map."""
    key, pos_key, angle_key, state_key, duration_key = random.split(key, 5)

    # Initialise ant positions, angles, states
    radius = jnp.sqrt(random.uniform(pos_key, (num_ants,), minval=0, maxval=arena_radius**2))
    theta = random.uniform(pos_key, (num_ants,), minval=0, maxval=2 * jnp.pi)
    positions = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1)
    angles = random.uniform(angle_key, (num_ants,), minval=0, maxval=2 * jnp.pi)
    initial_states = random.bernoulli(state_key, 0.5, (num_ants,)).astype(jnp.int32) # Start as RESTING or MOVING
    time_in_state = jnp.zeros(num_ants)

    # Initialise state durations
    key_r, key_b = random.split(duration_key)
    rest_durations = draw_durations(key_r, MEAN_REST_DURATION, STD_REST_DURATION, num_ants)
    burst_durations = draw_durations(key_b, MEAN_BURST_DURATION, STD_BURST_DURATION, num_ants)
    initial_durations = jnp.where(initial_states == STATE_RESTING, rest_durations, burst_durations)

    # Initialise pheromone map
    pheromone_map = jnp.zeros((grid_resolution, grid_resolution), dtype=jnp.float32)

    state = {
        'position': positions,
        'angle': angles,
        'speed': jnp.zeros(num_ants),
        'behavioural_state': initial_states,
        'time_in_state': time_in_state,
        'current_state_duration': initial_durations,
        'previous_behavioural_state': initial_states,
        'pheromone_map': pheromone_map
    }
    return state

# --- Core Update Logic ---

# Make parameters static for jitting, pass USE_GRID_PHEROMONES as arg
@partial(jax.jit, static_argnames=("num_ants", "arena_radius", "grid_resolution", "use_grid_pheromones"))
def update_step(state, key, dt, num_ants, arena_radius, grid_resolution, use_grid_pheromones):
    """Performs one vectorised update step."""

    positions = state['position']
    behavioural_state = state['behavioural_state']
    time_in_state = state['time_in_state']
    previous_behavioural_state = state['previous_behavioural_state']
    current_state_duration = state['current_state_duration']
    angles = state['angle']
    pheromone_map = state['pheromone_map']

    # --- 0. Split Keys ---
    key, key_dur_rest, key_dur_burst, key_dur_arrest,\
        key_speed, key_turn, key_pheromone_stop = random.split(key, 7)

    # --- 1. Pheromone Update and Signal Calculation ---

    # Identify emitting ants (Resting OR Arrested)
    is_emitter = (behavioural_state == STATE_RESTING) | (behavioural_state == STATE_ARRESTED)

    # Calculate individual pheromone strength based on time in state <<< REFACTORED >>>
    # This strength is used for deposition (grid) or direct signal (direct)
    individual_strength = calculate_individual_pheromone_strength(time_in_state)

    # --- Pheromone Grid Update (Decay and Deposition) --- <<< NEW Section >>>
    # Apply decay to the entire map
    decayed_pheromone_map = pheromone_map * PHEROMONE_DECAY_RATE

    # Calculate deposition amount for emitters
    deposition_amount = individual_strength * is_emitter * PHEROMONE_DEPOSITION_RATE * dt

    # Get grid indices for all ants (needed for deposition and maybe grid sampling)
    grid_indices = pos_to_grid_idx(positions, arena_radius, grid_resolution) # Shape: (num_ants, 2)

    # Add deposition to the grid using scatter add (.at[indices].add(values))
    # Need to handle potential multiple ants depositing in the same cell correctly.
    # .at provides atomic updates.
    # Indices need to be tuples for .at: (rows, cols)
    rows, cols = grid_indices[:, 0], grid_indices[:, 1]
    updated_pheromone_map = decayed_pheromone_map.at[rows, cols].add(deposition_amount)

    # --- Calculate Signal Strength (Conditional: Grid or Direct) ---

    def calculate_signal_strength_grid(state_local, pheromone_map_local):
        # Sample the grid around each ant
        # grid_indices already calculated
        detected_pheromone = vmapped_sample_grid_radius(
            grid_indices, pheromone_map_local, PHEROMONE_GRID_RADIUS_CELLS, grid_resolution
        )
        # The summed value from the grid is the signal strength
        return detected_pheromone

    def calculate_signal_strength_direct(state_local, _pheromone_map_local): # Ignore map if direct
        # --- This is the original direct detection logic, adapted slightly ---
        pos_local = state_local['position']
        b_state_local = state_local['behavioural_state']
        t_in_state_local = state_local['time_in_state'] # Needed for individual strength

        # Identify emitters
        is_emitter_local = (b_state_local == STATE_RESTING) | (b_state_local == STATE_ARRESTED)

        # Calculate pairwise distances
        delta_x_pair = pos_local[:, 0, None] - pos_local[None, :, 0]
        delta_y_pair = pos_local[:, 1, None] - pos_local[None, :, 1]
        distances_sq_pair = delta_x_pair**2 + delta_y_pair**2 + 1e-9
        distances_pair = jnp.sqrt(distances_sq_pair)

        # Determine which ants (j) are within pheromone radius of each ant (i)
        within_radius = (distances_pair < PHEROMONE_RADIUS)

        # Pheromone signal matrix: True if j is emitter and within radius of i
        pheromone_signal_matrix = jnp.where(within_radius, is_emitter_local[None, :], False)

        # Exclude self-influence
        mask_no_self = ~jnp.eye(num_ants, dtype=bool)
        pheromone_signal_matrix = jnp.where(mask_no_self, pheromone_signal_matrix, False)

        # Get individual strength (already calculated based on time_in_state)
        individual_strength_local = calculate_individual_pheromone_strength(t_in_state_local)

        # --- Calculate Total Signal Strength based on DISCRETE_PHEROMONE flag ---
        if DISCRETE_PHEROMONE:
            # Discrete: Count emitting neighbours * their individual strength
            # We need the strength of the *emitter* (j), not the receiver (i)
            # Broadcast individual strength of emitters across rows
            emitter_strength_broadcast = individual_strength_local[None, :]
            # Sum the strength of emitting neighbours for each ant i
            strength = jnp.sum(jnp.where(pheromone_signal_matrix, emitter_strength_broadcast, 0.0), axis=1)
        else:
            # Continuous: Sum of (emitter strength * distance falloff)
            clipped_distances = jnp.minimum(distances_pair, PHEROMONE_RADIUS)
            distance_scale = (PHEROMONE_RADIUS - clipped_distances) / (PHEROMONE_RADIUS + 1e-9)
            # Broadcast individual strength of emitters across rows
            emitter_strength_broadcast = individual_strength_local[None, :]
            # Calculate contribution of each neighbour j to ant i
            individual_contributions = emitter_strength_broadcast * distance_scale * pheromone_signal_matrix.astype(jnp.float32)
            strength = jnp.sum(individual_contributions, axis=1)

        return strength

    # Use jax.lax.cond to select the signal strength calculation method
    signal_strength = jax.lax.cond(
        use_grid_pheromones,
        calculate_signal_strength_grid,
        calculate_signal_strength_direct,
        state, # operand passed to both branches
        updated_pheromone_map # operand passed to both branches (ignored by direct)
    )

    # --- Calculate Probability of Stopping (Same logic as before) ---
    exponent = -PHEROMONE_STEEPNESS * (signal_strength - PHEROMONE_THRESHOLD)
    clipped_exponent = jnp.clip(exponent, -20.0, 20.0)
    prob_stop_pheromone = 1.0 / (1.0 + jnp.exp(clipped_exponent))

    # Roll the dice for stopping
    rand_stop = random.uniform(key_pheromone_stop, (num_ants,))

    # Determine if pheromone arrest applies (Same logic as before)
    is_moving = (behavioural_state == STATE_MOVING_BURST)
    just_escaped_arrest = (previous_behavioural_state == STATE_ARRESTED)
    in_grace_period = just_escaped_arrest & (time_in_state <= ARREST2BURST_GRACE_PERIOD)
    pheromone_check_passed = (rand_stop < prob_stop_pheromone)
    stops_due_to_pheromone = is_moving & (~in_grace_period) & pheromone_check_passed

    # --- 2. State Transition Logic (Duration-based + Pheromone Override) ---
    # (This logic remains largely the same, just uses the calculated stops_due_to_pheromone)

    next_time_in_state_if_no_stop = time_in_state + dt
    duration_expired = (next_time_in_state_if_no_stop >= current_state_duration)

    # Draw new durations
    new_rest_durations = draw_durations(key_dur_rest, MEAN_REST_DURATION, STD_REST_DURATION, num_ants)
    new_burst_durations = draw_durations(key_dur_burst, MEAN_BURST_DURATION, STD_BURST_DURATION, num_ants)
    new_arrest_durations = draw_durations(key_dur_arrest, MEAN_ARREST_DURATION, STD_ARREST_DURATION, num_ants)

    # Determine Final State, Duration, and Time-in-State
    next_state_val = behavioural_state
    next_duration = current_state_duration
    next_time = next_time_in_state_if_no_stop

    # Case 1: Moving -> ARRESTED (Pheromone)
    next_state_val = jnp.where(stops_due_to_pheromone, STATE_ARRESTED, next_state_val)
    next_duration = jnp.where(stops_due_to_pheromone, new_arrest_durations, next_duration)
    next_time = jnp.where(stops_due_to_pheromone, 0.0, next_time)

    # Case 2: Moving -> RESTING (Duration)
    moving_duration_expired = (behavioural_state == STATE_MOVING_BURST) & duration_expired & (~stops_due_to_pheromone)
    next_state_val = jnp.where(moving_duration_expired, STATE_RESTING, next_state_val)
    next_duration = jnp.where(moving_duration_expired, new_rest_durations, next_duration)
    next_time = jnp.where(moving_duration_expired, 0.0, next_time)

    # Case 3: Resting -> MOVING (Duration)
    resting_duration_expired = (behavioural_state == STATE_RESTING) & duration_expired
    next_state_val = jnp.where(resting_duration_expired, STATE_MOVING_BURST, next_state_val)
    next_duration = jnp.where(resting_duration_expired, new_burst_durations, next_duration)
    next_time = jnp.where(resting_duration_expired, 0.0, next_time)

    # Case 4: Arrested -> MOVING (Duration)
    # Note: If an ant is arrested and its duration expires, it tries to move.
    # It might immediately get arrested again in the *next* step if pheromone levels are high
    # and it's outside the grace period, which seems reasonable.
    arrested_duration_expired = (behavioural_state == STATE_ARRESTED) & duration_expired
    next_state_val = jnp.where(arrested_duration_expired, STATE_MOVING_BURST, next_state_val)
    next_duration = jnp.where(arrested_duration_expired, new_burst_durations, next_duration)
    next_time = jnp.where(arrested_duration_expired, 0.0, next_time)

    # Final Assignment
    final_behavioural_state = next_state_val
    final_current_state_duration = next_duration
    final_time_in_state = next_time
    # Store the state *before* transitions for the next step's grace period check
    final_previous_behavioural_state = behavioural_state


    # --- 3. State-Dependent Behaviour (Speed and Base Turning) ---
    # (Remains the same - depends on final_behavioural_state)
    burst_speeds = jnp.maximum(0.0, MEAN_BURST_SPEED + random.normal(key_speed, (num_ants,)) * STD_BURST_SPEED)
    current_speed = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_speeds, 0.0)
    burst_turn_noise = random.normal(key_turn, (num_ants,)) * TURN_RATE_STD * dt
    base_turn = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_turn_noise, 0.0)


    # --- 4. Wall Avoidance Turning ---
    # (Remains the same - depends on final_behavioural_state and position)
    dist_from_center = jnp.linalg.norm(positions, axis=1)
    in_wall_zone = (dist_from_center > (arena_radius - WALL_ZONE_WIDTH))
    apply_wall_turn = in_wall_zone & (final_behavioural_state == STATE_MOVING_BURST)

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    safe_x = jnp.where(dist_from_center < 1e-6, 1.0, pos_x)
    safe_y = jnp.where(dist_from_center < 1e-6, 0.0, pos_y)
    radial_angle = jnp.arctan2(safe_y, safe_x)
    desired_angle_to_center = wrap_angle(radial_angle + jnp.pi)
    angle_error_to_center = wrap_angle(angles - desired_angle_to_center)
    wall_avoid_turn_magnitude = -jnp.sign(angle_error_to_center) * WALL_AVOID_STRENGTH * dt
    total_turn = base_turn + jnp.where(apply_wall_turn, wall_avoid_turn_magnitude, 0.0)
    new_angles = wrap_angle(angles + total_turn)


    # --- 5. Movement, Collision, Boundary ---
    # (Remains the same)
    vx = current_speed * jnp.cos(new_angles)
    vy = current_speed * jnp.sin(new_angles)
    velocity = jnp.stack([vx, vy], axis=-1)
    potential_new_positions = positions + velocity * dt

    # Collision Resolution
    delta_x_coll = potential_new_positions[:, 0, None] - potential_new_positions[None, :, 0]
    delta_y_coll = potential_new_positions[:, 1, None] - potential_new_positions[None, :, 1]
    distances_sq_coll = delta_x_coll**2 + delta_y_coll**2
    distances_coll = jnp.sqrt(distances_sq_coll + 1e-9)
    distances_coll = jnp.where(jnp.eye(num_ants, dtype=bool), jnp.inf, distances_coll)
    collision_radius = 2 * ANT_RADIUS
    is_overlapping = distances_coll < collision_radius
    overlap_depth = jnp.maximum(0, collision_radius - distances_coll)
    inv_distance = 1.0 / (distances_coll + 1e-9)
    push_dir_x = delta_x_coll * inv_distance
    push_dir_y = delta_y_coll * inv_distance
    push_magnitude = overlap_depth * K_PUSH
    total_push_x = jnp.sum(jnp.where(is_overlapping, push_dir_x * push_magnitude, 0.0), axis=1)
    total_push_y = jnp.sum(jnp.where(is_overlapping, push_dir_y * push_magnitude, 0.0), axis=1)
    pushed_positions_x = potential_new_positions[:, 0] + total_push_x
    pushed_positions_y = potential_new_positions[:, 1] + total_push_y
    new_positions_after_collision = jnp.stack([pushed_positions_x, pushed_positions_y], axis=-1)

    # Boundary Conditions
    dist_from_center_new = jnp.linalg.norm(new_positions_after_collision, axis=1)
    is_outside = dist_from_center_new > arena_radius
    dist_from_center_safe = jnp.maximum(1e-9, dist_from_center_new)
    normalized_pos = new_positions_after_collision / dist_from_center_safe[:, None]
    clamped_positions = jnp.where(is_outside[:, None],
                                  normalized_pos * arena_radius,
                                  new_positions_after_collision)

    # --- 6. Assemble Next State ---
    next_state = {
        'position': clamped_positions,
        'angle': new_angles,
        'speed': current_speed,
        'behavioural_state': final_behavioural_state,
        'time_in_state': final_time_in_state,
        'current_state_duration': final_current_state_duration,
        'previous_behavioural_state': final_previous_behavioural_state,
        'pheromone_map': updated_pheromone_map # <<< MODIFIED >>> Store the updated map
    }
    return next_state