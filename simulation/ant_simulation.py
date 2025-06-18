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


# --- State constants ---
STATE_RESTING = 0
STATE_MOVING_BURST = 1
STATE_ARRESTED = 2

def wrap_angle(angle):
    """Helper function to wrap angles to [-pi, pi]"""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

def draw_durations(key, mean, std, num_samples, min_state_duration):
    """Draws durations from Normal dist, clipped at min_state_duration."""
    durations = mean + random.normal(key, (num_samples,)) * std
    return jnp.maximum(min_state_duration, durations)

@partial(jax.jit, static_argnames=('grid_resolution',))
def efficient_pheromone_deposition(positions, deposition_amounts, arena_radius, grid_resolution):
    """
    Efficient pheromone deposition using one-hot encoding matrices.
    Much faster than scatter-add operations, especially when multiple ants
    occupy the same grid cells.
    """
    num_ants = positions.shape[0]
    
    # Convert positions to grid indices
    grid_indices = pos_to_grid_idx(positions, arena_radius, grid_resolution)
    rows, cols = grid_indices[:, 0], grid_indices[:, 1]
    
    # Create one-hot matrices for each ant's position
    # This avoids the scatter-add bottleneck entirely
    row_one_hot = jax.nn.one_hot(rows, grid_resolution, dtype=jnp.float32)  # (num_ants, grid_resolution)
    col_one_hot = jax.nn.one_hot(cols, grid_resolution, dtype=jnp.float32)  # (num_ants, grid_resolution)
    
    # Outer product to get position matrices for each ant
    position_matrices = row_one_hot[:, :, None] * col_one_hot[:, None, :]  # (num_ants, grid_resolution, grid_resolution)
    
    # Weight by deposition amounts and sum across all ants
    weighted_deposits = position_matrices * deposition_amounts[:, None, None]
    total_deposition = jnp.sum(weighted_deposits, axis=0)  # (grid_resolution, grid_resolution)
    
    return total_deposition

@partial(jax.jit, static_argnames=('grid_resolution', 'radius_cells'))
def efficient_pheromone_detection_batch(positions, pheromone_map, arena_radius, grid_resolution, radius_cells):
    """
    Batch detection using convolution-like operations.
    Much more efficient than individual dynamic slices.
    """
    num_ants = positions.shape[0]
    
    # Convert positions to grid indices
    grid_indices = pos_to_grid_idx(positions, arena_radius, grid_resolution)
    
    # Create detection masks for all ants at once
    circular_mask = create_circular_mask(2 * radius_cells + 1, radius_cells)
    
    # Create one-hot position matrices for all ants
    rows, cols = grid_indices[:, 0], grid_indices[:, 1]
    row_one_hot = jax.nn.one_hot(rows, grid_resolution, dtype=jnp.float32)
    col_one_hot = jax.nn.one_hot(cols, grid_resolution, dtype=jnp.float32)
    ant_position_matrices = row_one_hot[:, :, None] * col_one_hot[:, None, :]  # (num_ants, grid_resolution, grid_resolution)
    
    # Pad the pheromone map once
    padded_map = jnp.pad(pheromone_map, pad_width=radius_cells, mode='constant', constant_values=0)
    
    # Use correlation to efficiently compute all detections at once
    # This is essentially a vectorised convolution operation
    detections = []
    for i in range(num_ants):
        # Get the ant's position in the padded map
        row_center, col_center = grid_indices[i, 0], grid_indices[i, 1]
        
        # Extract the local region
        local_region = jax.lax.dynamic_slice(
            padded_map,
            (row_center, col_center),
            (2 * radius_cells + 1, 2 * radius_cells + 1)
        )
        
        # Apply circular mask and sum
        masked_region = local_region * circular_mask
        detection = jnp.sum(masked_region)
        detections.append(detection)
    
    return jnp.array(detections)

def update_pheromone_map_efficient(current_map, positions, time_in_state, is_emitter, 
                                  pheromone_max_timestep, max_pheromone_strength, 
                                  deposition_rate, dt, arena_radius, grid_resolution, decay_rate):
    """Drop-in replacement for pheromone map updates - handles everything internally"""
    # Apply decay
    decayed_map = current_map * decay_rate
    
    # Calculate individual strength for deposition (same logic as before)
    individual_strength = calculate_individual_pheromone_strength(
        time_in_state, pheromone_max_timestep, max_pheromone_strength
    )
    
    # Calculate final deposition amounts
    deposition_amounts = individual_strength * is_emitter * deposition_rate * dt
    
    # Add new deposits efficiently (avoiding scatter-add)
    deposition_matrix = efficient_pheromone_deposition(
        positions, deposition_amounts, arena_radius, grid_resolution
    )
    
    return decayed_map + deposition_matrix


# Function to calculate individual pheromone strength based on time in state
@jax.jit
def calculate_individual_pheromone_strength(
    time_in_state,
    pheromone_max_timestep,
    max_pheromone_strength
):
    """Calculates the pheromone strength/deposition rate for each ant based on its time_in_state."""
    T_max = pheromone_max_timestep

    # Ensure T_max is positive to avoid division by zero or negative rates
    T_max_safe = jnp.maximum(1e-9, T_max)

    # Calculate normalised time: ranges from 0 to 1 as time_in_state goes from 0 to T_max_safe.
    # Time is clamped to T_max_safe, so normalised_time won't exceed 1.
    normalised_time = jnp.clip(time_in_state, 0.0, T_max_safe) / T_max_safe
    
    # Linearly scale to max_pheromone_strength
    strength_raw = normalised_time * max_pheromone_strength
    
    # Final clip to ensure it's within [0, max_pheromone_strength].
    # This also handles cases where time_in_state might be negative, though typically it shouldn't be.
    individual_strength = jnp.clip(strength_raw, 0.0, max_pheromone_strength)
    
    return individual_strength

# --- Grid Helper Functions ---

@partial(jax.jit, static_argnames=('size',))
def create_circular_mask(size, radius_cells):
    """Creates a circular mask of a given size and radius."""
    center = (size - 1) / 2.0
    x = jnp.arange(size) - center
    y = jnp.arange(size) - center
    xx, yy = jnp.meshgrid(x, y)
    distances_sq = xx**2 + yy**2
    mask = distances_sq <= radius_cells**2
    return mask.astype(jnp.float32)

@partial(jax.jit, static_argnames=('grid_resolution'))
def pos_to_grid_idx(position, arena_radius, grid_resolution):
    """Converts world coordinates (x, y) to grid indices (row, col)."""
    # Scale positions from [-arena_radius, arena_radius] to [0, grid_resolution]
    # Add epsilon to avoid index -1 for position exactly at -arena_radius
    idx = (position + arena_radius) / (2 * arena_radius) * grid_resolution + 1e-6
    # Clip to ensure indices are within [0, grid_resolution - 1]
    idx_clipped = jnp.clip(idx, 0, grid_resolution - 1)
    return idx_clipped.astype(jnp.int32)

@partial(jax.jit, static_argnames=('radius_cells'))
def sample_grid_radius(ant_pos_idx, pheromone_map, radius_cells):
    """Samples the pheromone map in a square region around the ant's grid index."""
    # ant_pos_idx: (2,) array [row, col] representing the ant's current cell
    # radius_cells: Static integer radius
    # grid_resolution: Static integer grid size

    # Pad the map with zeros to handle boundary cases easily.
    # Padding width is 'radius_cells' on all sides.
    padded_map = jnp.pad(pheromone_map, pad_width=radius_cells, mode='constant', constant_values=0)

    # The size of the square slice is fixed (static)
    slice_size = 2 * radius_cells + 1

    # Create a circular mask for this slice size
    circular_mask = create_circular_mask(slice_size, radius_cells)

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
    
    # Apply the circular mask to the sampled region
    masked_region = sampled_region * circular_mask
    total_pheromone = jnp.sum(masked_region)

    return total_pheromone

# mapped version for all ants
vmapped_sample_grid_radius = jax.vmap(sample_grid_radius, in_axes=(0, None, None))

def initialise_state(key, params):
    """Initialises the state of all ants including the pheromone map."""
    num_ants = params['num_ants'] # Example accessing param
    arena_radius = params['arena_radius']
    grid_resolution = params['pheromones']['grid_resolution']
    mean_rest_duration = params['mean_rest_duration']
    std_rest_duration = params['std_rest_duration']
    mean_burst_duration = params['mean_burst_duration']
    std_burst_duration = params['std_burst_duration']
    min_state_duration = params['min_state_duration']
    
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
    rest_durations = draw_durations(key_r, mean_rest_duration, std_rest_duration, num_ants, min_state_duration)
    burst_durations = draw_durations(key_b, mean_burst_duration, std_burst_duration, num_ants, min_state_duration)
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

def update_step(state, key, t, params):
    """Performs one vectorised update step."""

    num_ants = params['num_ants']
    arena_radius = params['arena_radius']
    ant_length = params['ant_length']
    grid_resolution = params['pheromones']['grid_resolution']
    use_grid_pheromones = params['pheromones']['use_grid_pheromones']
    dt = params['dt']
    k_push = params['k_push']
    
    # General pheromone parameters 
    max_pheromone_strength = params['pheromones']['max_pheromone_strength']

    # State parameters
    arrest2burst_grace_period = params['arrest2burst_grace_period']
    mean_rest_duration = params['mean_rest_duration']
    std_rest_duration = params['std_rest_duration']
    mean_burst_duration = params['mean_burst_duration']
    std_burst_duration = params['std_burst_duration']
    mean_arrest_duration = params['mean_arrest_duration']
    std_arrest_duration = params['std_arrest_duration']
    mean_burst_speed = params['mean_burst_speed']
    std_burst_speed = params['std_burst_speed']
    min_state_duration = params['min_state_duration']
    turn_rate_std = params['turn_rate_std']
    wall_avoid_strength = params['wall_avoid_strength']
    wall_zone_width = params['wall_zone_width']

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

    grid_resolution = params['pheromones']['grid_resolution']

    # Calculate strength differently based on mode
    if use_grid_pheromones:
        # EFFICIENT GRID MODE - everything handled in one call
        decay_rate = params['pheromones']['pheromone_decay_rate']
        deposition_rate = params['pheromones']['pheromone_deposition_rate']
        
        updated_pheromone_map = update_pheromone_map_efficient(
            pheromone_map,  # Use the map from state directly
            positions, 
            time_in_state,
            is_emitter,
            params['pheromones']['pheromone_max_timestep'],
            max_pheromone_strength,
            deposition_rate,
            dt,
            arena_radius, 
            grid_resolution, 
            decay_rate
        )
        # Placeholder for direct mode compatibility
        global_time_strength = 0.0
    else:
        # DIRECT MODE - no grid updates needed
        updated_pheromone_map = pheromone_map  # Map doesn't change in direct mode
        
        # Calculate global time-based strength for direct detection
        T_max_t = params['pheromones']['pheromone_max_timestep']
        k_t = params['pheromones']['pheromone_elu_steepness']
        x_offset_t = params['pheromones']['pheromone_elu_transition_frac']
        alpha_t = 1.0
        k_t = jnp.maximum(1e-6, k_t)
        x_offset_t = jnp.clip(x_offset_t, 1e-6, 1.0 - 1e-6)
        denom_A_t = k_t * (1.0 - x_offset_t) - (jnp.exp(-k_t * x_offset_t) - 1.0)
        A_t = max_pheromone_strength / (denom_A_t + 1e-9)
        B_t = -A_t * (jnp.exp(-k_t * x_offset_t) - 1.0)
        t_clamped_t = jnp.clip(t, 0.0, T_max_t)
        x_elu_t = k_t * (t_clamped_t / (T_max_t + 1e-9) - x_offset_t)
        elu_val_t = jax.nn.elu(x_elu_t, alpha=alpha_t)
        strength_raw_t = A_t * elu_val_t + B_t
        global_time_strength = jnp.clip(strength_raw_t, 0.0, max_pheromone_strength)

    # Pass global_strength calculated based on 't'
    def calculate_signal_strength_direct(state_local, global_strength):
        # --- This is the original direct detection logic, adapted slightly ---
        pos_local = state_local['position']
        b_state_local = state_local['behavioural_state']
        pheromone_radius_direct = params['pheromones']['pheromone_radius']

        # Identify emitters
        is_emitter_local = (b_state_local == STATE_RESTING) | (b_state_local == STATE_ARRESTED)

        # Calculate pairwise distances
        delta_x_pair = pos_local[:, 0, None] - pos_local[None, :, 0]
        delta_y_pair = pos_local[:, 1, None] - pos_local[None, :, 1]
        distances_sq_pair = delta_x_pair**2 + delta_y_pair**2 + 1e-9
        distances_pair = jnp.sqrt(distances_sq_pair)

        # Determine which ants (j) are within pheromone radius of each ant (i)
        within_radius = (distances_pair < pheromone_radius_direct)

        # Pheromone signal matrix: True if j is emitter and within radius of i
        pheromone_signal_matrix = jnp.where(within_radius, is_emitter_local[None, :], False)

        # Exclude self-influence
        mask_no_self = ~jnp.eye(num_ants, dtype=bool)
        pheromone_signal_matrix = jnp.where(mask_no_self, pheromone_signal_matrix, False)

        # Use the pre-calculated global_strength based on 't'

        # --- Calculate Total Signal Strength based on discrete_pheromone flag ---
        if params['pheromones']['discrete_pheromone']:
            # Discrete: Count emitting neighbours * the single global_strength value
            num_emitting_neighbours = jnp.sum(pheromone_signal_matrix, axis=1)
            strength = num_emitting_neighbours.astype(jnp.float32) * global_strength
        else:
            # Continuous: Sum of (emitter strength * distance falloff)
            clipped_distances = jnp.minimum(distances_pair, pheromone_radius_direct)
            distance_scale = (pheromone_radius_direct - clipped_distances) / (pheromone_radius_direct + 1e-9)
            # Broadcast the single global_strength value
            emitter_strength_broadcast = jnp.full((1, num_ants), global_strength)
            # Calculate contribution of each neighbour j to ant i
            individual_contributions = emitter_strength_broadcast * distance_scale * pheromone_signal_matrix.astype(jnp.float32)
            strength = jnp.sum(individual_contributions, axis=1)

        return strength

    # Use jax.lax.cond to select the signal strength calculation method
    signal_strength = jax.lax.cond(
        use_grid_pheromones,
        # True branch: Call the actual function directly
        lambda state_op, map_op, strength_op: efficient_pheromone_detection_batch(
            state_op['position'], map_op, arena_radius, grid_resolution, 
            params['pheromones']['grid_radius_cells']
        ),
        # False branch: Direct calculation (unchanged)
        lambda state_op, map_op, strength_op: calculate_signal_strength_direct(state_op, strength_op),
        # Operands:
        state, updated_pheromone_map, global_time_strength
    )

    # --- Calculate Probability of Stopping (Same logic as before) ---
    pheromone_steepness = params['pheromones']['pheromone_steepness']
    pheromone_threshold = params['pheromones']['pheromone_threshold']
    exponent = -pheromone_steepness * (signal_strength - pheromone_threshold)
    clipped_exponent = jnp.clip(exponent, -20.0, 20.0)
    prob_stop_pheromone = 1.0 / (1.0 + jnp.exp(clipped_exponent))

    # Roll the dice for stopping
    rand_stop = random.uniform(key_pheromone_stop, (num_ants,))

    # Determine if pheromone arrest applies (Same logic as before)
    is_moving = (behavioural_state == STATE_MOVING_BURST)
    just_escaped_arrest = (previous_behavioural_state == STATE_ARRESTED)
    in_grace_period = just_escaped_arrest & (time_in_state <= arrest2burst_grace_period)
    pheromone_check_passed = (rand_stop < prob_stop_pheromone)
    stops_due_to_pheromone = is_moving & (~in_grace_period) & pheromone_check_passed

    # --- 2. State Transition Logic (Duration-based + Pheromone Override) ---
    # (This logic remains largely the same, just uses the calculated stops_due_to_pheromone)

    next_time_in_state_if_no_stop = time_in_state + dt
    duration_expired = (next_time_in_state_if_no_stop >= current_state_duration)

    # Draw new durations
    new_rest_durations = draw_durations(key_dur_rest, mean_rest_duration, std_rest_duration, num_ants, min_state_duration)
    new_burst_durations = draw_durations(key_dur_burst, mean_burst_duration, std_burst_duration, num_ants, min_state_duration)
    new_arrest_durations = draw_durations(key_dur_arrest, mean_arrest_duration, std_arrest_duration, num_ants, min_state_duration)

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
    burst_speeds = jnp.maximum(0.0, mean_burst_speed + random.normal(key_speed, (num_ants,)) * std_burst_speed)
    current_speed = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_speeds, 0.0)
    burst_turn_noise = random.normal(key_turn, (num_ants,)) * turn_rate_std * dt
    base_turn = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_turn_noise, 0.0)


    # --- 4. Wall Avoidance Turning ---
    # (Remains the same - depends on final_behavioural_state and position)
    dist_from_center = jnp.linalg.norm(positions, axis=1)
    in_wall_zone = (dist_from_center > (arena_radius - wall_zone_width))
    apply_wall_turn = in_wall_zone & (final_behavioural_state == STATE_MOVING_BURST)

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    safe_x = jnp.where(dist_from_center < 1e-6, 1.0, pos_x)
    safe_y = jnp.where(dist_from_center < 1e-6, 0.0, pos_y)
    radial_angle = jnp.arctan2(safe_y, safe_x)
    desired_angle_to_center = wrap_angle(radial_angle + jnp.pi)
    angle_error_to_center = wrap_angle(angles - desired_angle_to_center)
    wall_avoid_turn_magnitude = -jnp.sign(angle_error_to_center) * wall_avoid_strength * dt
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
    collision_radius = ant_length
    is_overlapping = distances_coll < collision_radius
    overlap_depth = jnp.maximum(0, collision_radius - distances_coll)
    inv_distance = 1.0 / (distances_coll + 1e-9)
    push_dir_x = delta_x_coll * inv_distance
    push_dir_y = delta_y_coll * inv_distance
    push_magnitude = overlap_depth * k_push
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
        'pheromone_map': updated_pheromone_map
    }
    return next_state