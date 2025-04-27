import os

import jax
import jax.nn
import jax.numpy as jnp
import jax.random as random
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('..') # Or provide a default path

# --- Simulation Parameters ---
NUM_ANTS = 64
ARENA_RADIUS = 50.0  # mm
ANT_LENGTH = 2.5     # mm (for visualization, but needed for ANT_RADIUS/WALL_ZONE)
ANT_WIDTH = ANT_LENGTH / 2.0 # For triangle base width (viz)
ANT_RADIUS = ANT_LENGTH / 2.0 # Radius for collision detection
K_PUSH = 0.1                  # How strongly ants push each other apart (0 to 1)

DT = 0.1  # Simulation time step (arbitrary units)

# Parameters for Arrestant Pheromone
PHEROMONE_RADIUS = ANT_LENGTH * 2.0 # How far the 'signal' reaches
DISCRETE_PHEROMONE = True
PHEROMONE_THRESHOLD = 2.5 # Signal strength (~num neighbours) for 50% stop probability (must be > 1.0)
PHEROMONE_STEEPNESS = 4.0 # Controls how sharp the transition is around the threshold
MAX_PHEROMONE_STRENGTH = 0.9  # Max contribution of a single resting ant (scales the signal)
# Calculate PHEROMONE_MIDPOINT_TIME based on PHEROMONE_MAX_TIMESTEP
PHEROMONE_MAX_TIMESTEP = 250
PHEROMONE_ELU_TRANSITION_FRAC = 0.4 # (0 to 1) Fraction of T_max where growth becomes linear. TUNE ME!
PHEROMONE_ELU_STEEPNESS = 5.0       # (> 0) Controls the initial exponential rise steepness. TUNE ME!

# Wall Interaction Parameters
WALL_ZONE_WIDTH = ANT_LENGTH * 1.5 # How far from the wall the turning response starts
WALL_AVOID_STRENGTH = 0.9          # How strongly ants turn towards center (rad/sec)

# State constants
STATE_RESTING = 0
STATE_MOVING_BURST = 1
STATE_ARRESTED = 2

# State durations and burst speed parameters
MEAN_REST_DURATION = 3.0   # Average time (in sim time units) to rest
STD_REST_DURATION = 2.5
MEAN_BURST_DURATION = 7.0  # Average time (in sim time units) for a movement burst
STD_BURST_DURATION = 2.5
MEAN_BURST_SPEED = 6.0     # Average speed during a burst (units per dt)
STD_BURST_SPEED = 1.0
MEAN_ARREST_DURATION = 5.5 # Shorter duration for pheromone-induced stop? TUNE ME!
STD_ARREST_DURATION = 3.0  # TUNE ME!
MIN_STATE_DURATION = 0.2   # Minimum duration for any state bout (in sim time units)# Turning rate during bursts
TURN_RATE_STD = 1.4
ARREST2BURST_GRACE_PERIOD = 2.0 # Sim time units ant must be MOVING before pheromone arrest applies again.

# --- Initialization ---

def draw_durations(key, mean, std):
    """Draws durations from Normal dist, clipped at MIN_STATE_DURATION."""
    # Use NUM_ANTS defined above
    durations = mean + random.normal(key, (NUM_ANTS,)) * std
    return jnp.maximum(MIN_STATE_DURATION, durations)

def initialise_state(key, arena_radius):
    """Initialises the state of all ants with state machine variables."""
    # Uses NUM_ANTS defined above
    key, pos_key, angle_key, state_key, duration_key = random.split(key, 5)
    radius = jnp.sqrt(random.uniform(pos_key, (NUM_ANTS,), minval=0, maxval=arena_radius**2))
    theta = random.uniform(pos_key, (NUM_ANTS,), minval=0, maxval=2 * jnp.pi)
    positions = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1)
    angles = random.uniform(angle_key, (NUM_ANTS,), minval=0, maxval=2 * jnp.pi)
    initial_states = random.bernoulli(state_key, 0.5, (NUM_ANTS,)).astype(jnp.int32)
    time_in_state = jnp.zeros(NUM_ANTS)
    key_r, key_b = random.split(duration_key)
    rest_durations = draw_durations(key_r, MEAN_REST_DURATION, STD_REST_DURATION)
    burst_durations = draw_durations(key_b, MEAN_BURST_DURATION, STD_BURST_DURATION)
    initial_durations = jnp.where(initial_states == STATE_RESTING, rest_durations, burst_durations)
    state = {
        'position': positions, 'angle': angles, 'speed': jnp.zeros(NUM_ANTS),
        'behavioural_state': initial_states, 'time_in_state': time_in_state,
        'current_state_duration': initial_durations,
        'previous_behavioural_state': initial_states
    }
    return state

def wrap_angle(angle):
    """Helper function to wrap angles to [-pi, pi]"""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

# --- Core Update Logic ---

@jax.jit
def update_step(state, key, t, dt):
    """Performs one vectorized update step using state machine, pheromones, and wall avoidance."""
    num_ants = state['position'].shape[0] # Use state size directly
    positions = state['position']
    behavioural_state = state['behavioural_state']
    time_in_state = state['time_in_state']
    previous_behavioural_state = state['previous_behavioural_state']
    current_state_duration = state['current_state_duration']
    angles = state['angle'] # Ant's current heading angle

    # --- 0. Split Keys ---
    # Need one more key for the pheromone stop roll
    key, key_dur_rest, key_dur_burst, key_dur_arrest,\
        key_speed, key_turn, key_pheromone_stop = random.split(key, 7)

    # --- 1. Pheromone Influence Calculation ---

    # Identify which ants are emitting pheromones (Resting OR Arrested)
    is_emitter = (behavioural_state == STATE_RESTING) | (behavioural_state == STATE_ARRESTED) # <<< MODIFIED >>>

    # Calculate pairwise distances
    delta_x_pair = positions[:, 0, None] - positions[None, :, 0]
    delta_y_pair = positions[:, 1, None] - positions[None, :, 1]
    # Add small epsilon to avoid sqrt(0) if ants are exactly on top (unlikely after collision)
    distances_sq_pair = delta_x_pair**2 + delta_y_pair**2 + 1e-9
    distances_pair = jnp.sqrt(distances_sq_pair)

    # Determine which ants (j) are within pheromone radius of each ant (i)
    within_radius = (distances_pair < PHEROMONE_RADIUS)

    # Create a matrix where pheromone_signal_matrix[i, j] is True if:
    #   - ant j is an emitter (resting or arrested) AND
    #   - ant j is within PHEROMONE_RADIUS of ant i
    # Broadcasting is_emitter to compare against each column (j)
    pheromone_signal_matrix = jnp.where(within_radius, is_emitter[None, :], False)

    # --- Calculate time-dependent individual pheromone strength using ELU ---
    T_max = PHEROMONE_MAX_TIMESTEP # Total time over which strength grows
    k = PHEROMONE_ELU_STEEPNESS        # Steepness factor (k)
    x_offset = PHEROMONE_ELU_TRANSITION_FRAC # Transition fraction (x_offset)
    alpha = 1.0 # Standard ELU alpha

    # Ensure parameters are valid (avoid division by zero or log(neg))
    k = jnp.maximum(1e-6, k) # k must be positive
    x_offset = jnp.clip(x_offset, 1e-6, 1.0 - 1e-6) # x_offset must be in (0, 1)

    # Calculate scaling constants A and B to ensure f(0)=0 and f(T_max)=MAX
    # Denominator for A: k*(1-x_offset) - (exp(-k*x_offset) - 1)
    denom_A = k * (1.0 - x_offset) - (jnp.exp(-k * x_offset) - 1.0)
    # Add epsilon to avoid division by zero if k is tiny or numerically unstable
    A = MAX_PHEROMONE_STRENGTH / (denom_A + 1e-9)
    # B = -A * (exp(-k*x_offset) - 1)
    B = -A * (jnp.exp(-k * x_offset) - 1.0)

    # Calculate the input to ELU based on current time t
    # Clamp t to be within [0, T_max] for stability in calculation
    t_clamped = jnp.clip(t, 0.0, T_max)
    # Add epsilon to T_max denominator to avoid division by zero if T_max is 0
    x_elu = k * (t_clamped / (T_max + 1e-9) - x_offset)

    # Calculate the ELU-based strength
    elu_val = jax.nn.elu(x_elu, alpha=alpha)
    strength_raw = A * elu_val + B

    # Clip final strength to ensure it's within [0, MAX]
    current_individual_strength = jnp.clip(strength_raw, 0.0, MAX_PHEROMONE_STRENGTH)

    # --- Calculate probability of stopping using a sigmoid based on signal strength ---
    # Exclude self-influence (set diagonal to False for neighbour checks)
    mask_no_self = ~jnp.eye(num_ants, dtype=bool)
    pheromone_signal_matrix = jnp.where(mask_no_self, pheromone_signal_matrix, False)

    # --- Calculate Total Signal Strength based on DISCRETE_PHEROMONE flag ---
    def calculate_signal_strength(is_discrete):
        if is_discrete:
            # Discrete: Count emitting neighbours * current individual strength
            num_resting_neighbours = jnp.sum(pheromone_signal_matrix, axis=1)
            strength = num_resting_neighbours.astype(jnp.float32) * current_individual_strength
        else:
            # Continuous: Sum of (individual strength * distance falloff) for each neighbour
            # Calculate distance scaling factor (linear falloff from 1 at dist=0 to 0 at dist=PHEROMONE_RADIUS)
            # Clip distance to avoid negative scaling or issues at radius edge
            clipped_distances = jnp.minimum(distances_pair, PHEROMONE_RADIUS)
            # Add epsilon to denominator for safety if PHEROMONE_RADIUS is tiny
            distance_scale = (PHEROMONE_RADIUS - clipped_distances) / (PHEROMONE_RADIUS + 1e-9)
            # Calculate contribution of each neighbour j to ant i (masked by signal matrix)
            individual_contributions = current_individual_strength * distance_scale * pheromone_signal_matrix.astype(jnp.float32)
            # Total signal strength is the sum of contributions for each ant i
            strength = jnp.sum(individual_contributions, axis=1)
        return strength

    signal_strength = calculate_signal_strength(DISCRETE_PHEROMONE) # <<< MODIFIED >>> Calculate based on flag
    exponent = -PHEROMONE_STEEPNESS * (signal_strength - PHEROMONE_THRESHOLD)
    # Clip exponent to avoid potential overflow/underflow in jnp.exp
    clipped_exponent = jnp.clip(exponent, -20.0, 20.0)
    prob_stop_pheromone = 1.0 / (1.0 + jnp.exp(clipped_exponent))

    # Roll the dice for stopping due to pheromone for each ant
    rand_stop = random.uniform(key_pheromone_stop, (num_ants,))

    # --- Determine if pheromone arrest applies ---
    # Condition 1: Is the ant currently moving?
    is_moving = (behavioural_state == STATE_MOVING_BURST)
    # Condition 2: Did the ant *just* escape arrest? Check if previous state was ARRESTED.
    just_escaped_arrest = (previous_behavioural_state == STATE_ARRESTED)
    # Condition 3: Has the grace period *not* passed yet for those that just escaped?
    in_grace_period = just_escaped_arrest & (time_in_state <= ARREST2BURST_GRACE_PERIOD) # Use time_in_state since it resets on transition
    # Condition 4: Does the random roll trigger the stop based on pheromone probability?
    pheromone_check_passed = (rand_stop < prob_stop_pheromone)

    # Final check: Arrest happens if moving, NOT in grace period, AND pheromone check passes
    stops_due_to_pheromone = is_moving & (~in_grace_period) & pheromone_check_passed


    # --- 2. State Transition Logic (Duration-based + Pheromone Override) ---

    # Calculate potential next state based on duration expiry *first*
    next_time_in_state_if_no_stop = time_in_state + dt
    duration_expired = (next_time_in_state_if_no_stop >= current_state_duration)

    # Draw new durations for *all* ants (simpler than conditional drawing)
    # We'll select the correct one based on the *final* next state
    new_rest_durations = draw_durations(key_dur_rest, MEAN_REST_DURATION, STD_REST_DURATION)
    new_burst_durations = draw_durations(key_dur_burst, MEAN_BURST_DURATION, STD_BURST_DURATION)

    new_arrest_durations = draw_durations(key_dur_arrest, MEAN_ARREST_DURATION, STD_ARREST_DURATION)

    # --- Determine Final State, Duration, and Time-in-State ---
    # This logic becomes more complex due to the third state and the arrest check

    # Initialize next state variables assuming no transition happens
    next_state_val = behavioural_state
    next_duration = current_state_duration
    next_time = next_time_in_state_if_no_stop

    # --- Transitions OUT OF MOVING ---
    # Case 1: Moving ant stops due to pheromone -> ARRESTED
    next_state_val = jnp.where(stops_due_to_pheromone, STATE_ARRESTED, next_state_val)
    next_duration = jnp.where(stops_due_to_pheromone, new_arrest_durations, next_duration)
    next_time = jnp.where(stops_due_to_pheromone, 0.0, next_time) # Reset timer

    # Case 2: Moving ant's duration expires (and wasn't stopped by pheromone) -> RESTING (naturally)
    moving_duration_expired = (behavioural_state == STATE_MOVING_BURST) & duration_expired & (~stops_due_to_pheromone)
    next_state_val = jnp.where(moving_duration_expired, STATE_RESTING, next_state_val)
    next_duration = jnp.where(moving_duration_expired, new_rest_durations, next_duration)
    next_time = jnp.where(moving_duration_expired, 0.0, next_time) # Reset timer

    # --- Transitions OUT OF RESTING ---
    # Case 3: Resting ant's duration expires -> MOVING
    resting_duration_expired = (behavioural_state == STATE_RESTING) & duration_expired
    next_state_val = jnp.where(resting_duration_expired, STATE_MOVING_BURST, next_state_val)
    next_duration = jnp.where(resting_duration_expired, new_burst_durations, next_duration)
    next_time = jnp.where(resting_duration_expired, 0.0, next_time) # Reset timer

    # Case 4: Arrested ant's duration expires AND it's NOT kept arrested by pheromones -> MOVING
    arrested_duration_expired = (behavioural_state == STATE_ARRESTED) & duration_expired
    next_state_val = jnp.where(arrested_duration_expired, STATE_MOVING_BURST, next_state_val)
    next_duration = jnp.where(arrested_duration_expired, new_burst_durations, next_duration)
    next_time = jnp.where(arrested_duration_expired, 0.0, next_time) # Reset timer

    # --- Final Assignment ---
    final_behavioural_state = next_state_val
    final_current_state_duration = next_duration
    final_time_in_state = next_time
    final_previous_behavioural_state = behavioural_state


    # --- 3. State-Dependent Behaviour (Speed and Base Turning) ---
    # Speed and turning depend on the *final* state after considering pheromones
    burst_speeds = jnp.maximum(0.0, MEAN_BURST_SPEED + random.normal(key_speed, (num_ants,)) * STD_BURST_SPEED)
    # Set speed to 0 if RESTING or ARRESTED
    current_speed = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_speeds, 0.0) # (logic is same, check ensures ARRESTED included)

    # Set base turn to 0 if RESTING or ARRESTED
    burst_turn_noise = random.normal(key_turn, (num_ants,)) * TURN_RATE_STD * dt
    base_turn = jnp.where(final_behavioural_state == STATE_MOVING_BURST, burst_turn_noise, 0.0)


    # --- 4. Wall Avoidance / Following Turning ---
    # This calculation depends on the ant being *potentially* moving and near the wall
    # Uses the _final_ state to determine if wall turning should apply
    dist_from_center = jnp.linalg.norm(positions, axis=1)
    in_wall_zone = (dist_from_center > (ARENA_RADIUS - WALL_ZONE_WIDTH))
    # Apply wall turn only if the ant _ended up_ in the MOVING state
    apply_wall_turn = in_wall_zone & (final_behavioural_state == STATE_MOVING_BURST)

    # Calculate radial_angle safely
    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    # Avoid arctan2(0,0) - use a default angle if at center
    safe_x = jnp.where(dist_from_center < 1e-6, 1.0, pos_x)
    safe_y = jnp.where(dist_from_center < 1e-6, 0.0, pos_y)
    radial_angle = jnp.arctan2(safe_y, safe_x)

    # Calculate desired angle towards the center (0,0)
    # This is 180 degrees (pi radians) opposite to the radial angle
    desired_angle_to_center = wrap_angle(radial_angle + jnp.pi)

    # Calculate angle error relative to the center direction
    angle_error_to_center = wrap_angle(angles - desired_angle_to_center)

    # Calculate wall *avoidance* turn magnitude
    # Turn opposite to the sign of the error to correct towards the center angle
    # Strength is controlled by WALL_AVOID_STRENGTH
    wall_avoid_turn_magnitude = -jnp.sign(angle_error_to_center) * WALL_AVOID_STRENGTH * dt

    # Combine base random turn and conditional wall avoidance turn
    total_turn = base_turn + jnp.where(apply_wall_turn, wall_avoid_turn_magnitude, 0.0)

    # Update angles using the combined turn
    new_angles = wrap_angle(angles + total_turn)


    # --- 5. Movement, Collision, Boundary ---
    # Movement uses the current_speed calculated based on the final state
    vx = current_speed * jnp.cos(new_angles)
    vy = current_speed * jnp.sin(new_angles)
    velocity = jnp.stack([vx, vy], axis=-1)
    potential_new_positions = positions + velocity * dt

    # Collision Resolution (Slightly adjusted from original - using num_ants)
    delta_x_coll = potential_new_positions[:, 0, None] - potential_new_positions[None, :, 0]
    delta_y_coll = potential_new_positions[:, 1, None] - potential_new_positions[None, :, 1]
    distances_sq_coll = delta_x_coll**2 + delta_y_coll**2
    distances_coll = jnp.sqrt(distances_sq_coll + 1e-9) # Epsilon for safety
    # Prevent self-collision check and division by zero if distances_coll is ~0
    distances_coll = jnp.where(jnp.eye(num_ants, dtype=bool), jnp.inf, distances_coll)

    collision_radius = 2 * ANT_RADIUS
    is_overlapping = distances_coll < collision_radius
    overlap_depth = jnp.maximum(0, collision_radius - distances_coll)

    # Avoid division by zero if distance is exactly zero (though unlikely with epsilon)
    inv_distance = 1.0 / (distances_coll + 1e-9)
    # Push direction is normalized delta vector
    push_dir_x = delta_x_coll * inv_distance
    push_dir_y = delta_y_coll * inv_distance

    # Push magnitude proportional to overlap depth and K_PUSH
    push_magnitude = overlap_depth * K_PUSH

    # Calculate total push vector by summing contributions from all overlapping neighbours
    # Need to handle the case where an ant isn't overlapping with anyone (push = 0)
    total_push_x = jnp.sum(jnp.where(is_overlapping, push_dir_x * push_magnitude, 0.0), axis=1)
    total_push_y = jnp.sum(jnp.where(is_overlapping, push_dir_y * push_magnitude, 0.0), axis=1)

    # Apply the push correction
    pushed_positions_x = potential_new_positions[:, 0] + total_push_x
    pushed_positions_y = potential_new_positions[:, 1] + total_push_y
    new_positions_after_collision = jnp.stack([pushed_positions_x, pushed_positions_y], axis=-1)

    # Boundary Conditions (Clamping applied *after* collision resolution)
    dist_from_center_new = jnp.linalg.norm(new_positions_after_collision, axis=1)
    is_outside = dist_from_center_new > ARENA_RADIUS
    # Avoid division by zero if ant is exactly at the center after collision/push
    dist_from_center_safe = jnp.maximum(1e-9, dist_from_center_new)
    # Normalize position vector and scale to ARENA_RADIUS if outside
    normalized_pos = new_positions_after_collision / dist_from_center_safe[:, None]
    clamped_positions = jnp.where(is_outside[:, None],
                                  normalized_pos * ARENA_RADIUS,
                                  new_positions_after_collision)

    # --- 6. Assemble Next State ---
    # Use the final_* variables determined by duration and pheromone logic
    next_state = {
        'position': clamped_positions,
        'angle': new_angles,
        'speed': current_speed, # Speed reflects the final state
        'behavioural_state': final_behavioural_state,
        'time_in_state': final_time_in_state,
        'current_state_duration': final_current_state_duration,
        'previous_behavioural_state': final_previous_behavioural_state
    }
    return next_state
