import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import math
import tqdm
# Need partial for static arguments if we chose that route, but we'll use globals
# from functools import partial


# --- Simulation Parameters ---
NUM_ANTS = 64
ARENA_RADIUS = 50.0  # mm
ANT_LENGTH = 2.5     # mm (for visualization)
ANT_WIDTH = ANT_LENGTH / 2.0 # For triangle base width
ANT_RADIUS = ANT_LENGTH / 2.0 # Radius for collision detection
K_PUSH = 0.6                  # How strongly ants push each other apart (0 to 1)

# Animation Parameters
SIMULATION_STEPS = 500
FRAME_INTERVAL = 10   # Reduced for smoother animation attempt
# Arena will be centered at (0,0)
WINDOW_PADDING = 2.0
WINDOW_SIZE = 2 * (ARENA_RADIUS + WINDOW_PADDING)

DT = 0.1  # Simulation time step (arbitrary units)

# <<< PHEROMONE >>> Parameters for Arrestant Pheromone
PHEROMONE_RADIUS = ANT_LENGTH * 3.0 # How far the 'signal' reaches (tune me!)
# <<< PHEROMONE >>> Sigmoid function parameters for stopping probability based on signal strength
PHEROMONE_THRESHOLD = 1.5 # Signal strength (~num neighbours) for 50% stop probability (must be > 1.0) - TUNE ME!
PHEROMONE_STEEPNESS = 4.0 # Controls how sharp the transition is around the threshold - TUNE ME!
MAX_PHEROMONE_STRENGTH = 1.0  # Max contribution of a single resting ant (scales the signal)
PHEROMONE_GROWTH_RATE = 0.02 # How fast pheromone strength increases (related to 1/time units) - TUNE ME!
PHEROMONE_MIDPOINT_TIME = SIMULATION_STEPS / 2 # Time at which strength reaches 50% max - TUNE ME!


# <<< Wall Interaction Parameters >>>
WALL_ZONE_WIDTH = ANT_LENGTH * 1.5 # How far from the wall the turning response starts
WALL_AVOID_STRENGTH = 1.2          # How strongly ants turn towards center (rad/sec) - TUNE ME!


# State constants
STATE_RESTING = 0
STATE_MOVING_BURST = 1

# State durations and burst speed parameters
MEAN_REST_DURATION = 2.0   # Average time (in sim time units) to rest
STD_REST_DURATION = 0.5
MEAN_BURST_DURATION = 4.0  # Average time (in sim time units) for a movement burst
STD_BURST_DURATION = 1.0
MEAN_BURST_SPEED = 6.0     # Average speed during a burst (units per dt)
STD_BURST_SPEED = 1.0
MIN_STATE_DURATION = 0.1   # Minimum duration for any state bout (in sim time units)

# Turning rate during bursts
TURN_RATE_STD = 1.2

# --- Global state for animation artists ---
ant_patches = []
time_text_artist = None

# --- Animation Saving Flag ---
SAVE_ANIMATION = True  # Set to True to save animation as GIF, False to only display

# --- Initialization ---

# <<< Uses global NUM_ANTS >>>
def draw_durations(key, mean, std):
    """Draws durations from Normal dist, clipped at MIN_STATE_DURATION."""
    durations = mean + random.normal(key, (NUM_ANTS,)) * std
    return jnp.maximum(MIN_STATE_DURATION, durations)

# <<< Uses global NUM_ANTS >>>
def initialize_state(key, arena_radius):
    """Initializes the state of all ants with state machine variables."""
    key, pos_key, angle_key, state_key, duration_key = random.split(key, 5)

    # Initial positions and angles
    radius = jnp.sqrt(random.uniform(pos_key, (NUM_ANTS,), minval=0, maxval=arena_radius**2))
    theta = random.uniform(pos_key, (NUM_ANTS,), minval=0, maxval=2 * jnp.pi)
    positions = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1)
    angles = random.uniform(angle_key, (NUM_ANTS,), minval=0, maxval=2 * jnp.pi)

    # Initial behavioral state
    initial_states = random.bernoulli(state_key, 0.5, (NUM_ANTS,)).astype(jnp.int32)

    # Initial time in state
    time_in_state = jnp.zeros(NUM_ANTS)

    # Initial target durations
    key_r, key_b = random.split(duration_key)
    rest_durations = draw_durations(key_r, MEAN_REST_DURATION, STD_REST_DURATION)
    burst_durations = draw_durations(key_b, MEAN_BURST_DURATION, STD_BURST_DURATION)
    initial_durations = jnp.where(initial_states == STATE_RESTING, rest_durations, burst_durations)

    state = {
        'position': positions,
        'angle': angles,
        'speed': jnp.zeros(NUM_ANTS), # Start stationary regardless of state
        'behavioral_state': initial_states,
        'time_in_state': time_in_state,
        'current_state_duration': initial_durations
    }
    return state

# <<< Added wrap_angle helper function >>>
def wrap_angle(angle):
    """Helper function to wrap angles to [-pi, pi]"""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

# --- Core Update Logic ---

@jax.jit
def update_step(state, key, t, dt):
    """Performs one vectorized update step using state machine, pheromones, and wall avoidance."""
    num_ants = state['position'].shape[0] # Use state size directly
    positions = state['position']
    behavioral_state = state['behavioral_state']
    time_in_state = state['time_in_state']
    current_state_duration = state['current_state_duration']
    angles = state['angle'] # Ant's current heading angle

    # --- 0. Split Keys ---
    # Need one more key for the pheromone stop roll
    key, key_dur_rest, key_dur_burst, key_speed, key_turn, key_collision, key_pheromone_stop = random.split(key, 7)

    # --- 1. Pheromone Influence Calculation --- <<< PHEROMONE >>> NEW BLOCK

    # Identify which ants are resting (potential pheromone emitters)
    is_resting_emitter = (behavioral_state == STATE_RESTING)

    # Calculate pairwise distances (could reuse collision calculation later, but do here for clarity)
    delta_x_pair = positions[:, 0, None] - positions[None, :, 0]
    delta_y_pair = positions[:, 1, None] - positions[None, :, 1]
    # Add small epsilon to avoid sqrt(0) if ants are exactly on top (unlikely after collision)
    distances_sq_pair = delta_x_pair**2 + delta_y_pair**2 + 1e-9
    distances_pair = jnp.sqrt(distances_sq_pair)

    # Determine which ants (j) are within pheromone radius of each ant (i)
    within_radius = (distances_pair < PHEROMONE_RADIUS)

    # Create a matrix where pheromone_signal_matrix[i, j] is True if:
    #   - ant j is resting AND
    #   - ant j is within PHEROMONE_RADIUS of ant i
    # Broadcasting is_resting_emitter to compare against each column (j)
    pheromone_signal_matrix = jnp.where(within_radius, is_resting_emitter[None, :], False)

    # Exclude self-influence (set diagonal to False)
    # An ant cannot detect its own pheromone if it were resting
    mask_no_self = ~jnp.eye(num_ants, dtype=bool)
    pheromone_signal_matrix = jnp.where(mask_no_self, pheromone_signal_matrix, False)

    # Count emitting neighbours for each ant (sum across columns for each row i)
    num_resting_neighbours = jnp.sum(pheromone_signal_matrix, axis=1)

    # --- Calculate time-dependent individual pheromone strength ---
    # Use a sigmoid growth function for individual ant strength over time 't'
    strength_exponent = -PHEROMONE_GROWTH_RATE * (t - PHEROMONE_MIDPOINT_TIME)
    # Clip exponent for stability
    clipped_strength_exponent = jnp.clip(strength_exponent, -20.0, 20.0) # Avoid exp overflow
    current_individual_strength = MAX_PHEROMONE_STRENGTH / (1.0 + jnp.exp(clipped_strength_exponent))

    # --- Calculate probability of stopping using a sigmoid based on signal strength ---
    # Total signal strength = num_neighbours * time_dependent_strength
    signal_strength = num_resting_neighbours.astype(jnp.float32) * current_individual_strength
    # Calculate probability using a sigmoid function centered around the threshold
    exponent = -PHEROMONE_STEEPNESS * (signal_strength - PHEROMONE_THRESHOLD)
    # Clip exponent to avoid potential overflow/underflow in jnp.exp
    clipped_exponent = jnp.clip(exponent, -20.0, 20.0)
    prob_stop_pheromone = 1.0 / (1.0 + jnp.exp(clipped_exponent))

    # Roll the dice for stopping due to pheromone for each ant
    rand_stop = random.uniform(key_pheromone_stop, (num_ants,))

    # Determine which *moving* ants actually stop due to pheromone this step
    stops_due_to_pheromone = (behavioral_state == STATE_MOVING_BURST) & \
                             (rand_stop < prob_stop_pheromone)

    # --- 2. State Transition Logic (Duration-based + Pheromone Override) ---

    # Calculate potential next state based on duration expiry *first*
    next_time_in_state_if_no_stop = time_in_state + dt
    duration_expired = (next_time_in_state_if_no_stop >= current_state_duration)

    # Potential next state if duration expires
    potential_next_state = jnp.where(duration_expired, 1 - behavioral_state, behavioral_state)

    # Draw new durations for *all* ants (simpler than conditional drawing)
    # We'll select the correct one based on the *final* next state
    new_rest_durations = draw_durations(key_dur_rest, MEAN_REST_DURATION, STD_REST_DURATION)
    new_burst_durations = draw_durations(key_dur_burst, MEAN_BURST_DURATION, STD_BURST_DURATION)

    # Determine the appropriate duration for the *potential* next state
    duration_if_expired = jnp.where(potential_next_state == STATE_RESTING, new_rest_durations, new_burst_durations)
    potential_next_duration = jnp.where(duration_expired, duration_if_expired, current_state_duration)
    potential_next_time_in_state = jnp.where(duration_expired, 0.0, next_time_in_state_if_no_stop)


    # --- 2b. Apply Pheromone Override <<< PHEROMONE >>>
    # If an ant stops due to pheromone:
    # - Its state becomes STATE_RESTING.
    # - Its time_in_state resets to 0.0.
    # - Its current_state_duration becomes a *newly drawn* REST duration.

    # Final state is STATE_RESTING if stopped by pheromone, otherwise it's the potential_next_state
    final_behavioral_state = jnp.where(stops_due_to_pheromone, STATE_RESTING, potential_next_state)

    # Final duration: if stopped by pheromone, use a new rest duration; otherwise use potential_next_duration
    final_current_state_duration = jnp.where(stops_due_to_pheromone, new_rest_durations, potential_next_duration)

    # Final time in state: if stopped by pheromone, reset to 0; otherwise use potential_next_time_in_state
    final_time_in_state = jnp.where(stops_due_to_pheromone, 0.0, potential_next_time_in_state)


    # --- 3. State-Dependent Behavior (Speed and Base Turning) ---
    # Speed and turning depend on the *final* state after considering pheromones
    burst_speeds = jnp.maximum(0.0, MEAN_BURST_SPEED + random.normal(key_speed, (num_ants,)) * STD_BURST_SPEED)
    current_speed = jnp.where(final_behavioral_state == STATE_MOVING_BURST, burst_speeds, 0.0)

    burst_turn_noise = random.normal(key_turn, (num_ants,)) * TURN_RATE_STD * dt
    base_turn = jnp.where(final_behavioral_state == STATE_MOVING_BURST, burst_turn_noise, 0.0)


    # --- 4. Wall Avoidance / Following Turning ---
    # This calculation depends on the ant being *potentially* moving and near the wall
    # Uses the *final* state to determine if wall turning should apply
    dist_from_center = jnp.linalg.norm(positions, axis=1)
    in_wall_zone = (dist_from_center > (ARENA_RADIUS - WALL_ZONE_WIDTH))
    # Apply wall turn only if the ant *ended up* in the moving state
    apply_wall_turn = in_wall_zone & (final_behavioral_state == STATE_MOVING_BURST)

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
        'behavioral_state': final_behavioral_state,
        'time_in_state': final_time_in_state,
        'current_state_duration': final_current_state_duration
    }
    return next_state

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
# Adjust margins - values closer to 0/1 fill more space
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

def setup_visualization():
    """Sets up the static parts of the plot and initializes patches and text."""
    global ant_patches, time_text_artist # Use global references
    ant_patches = [] # Reset list

    ax.set_xlim(-WINDOW_SIZE / 2, WINDOW_SIZE / 2)
    ax.set_ylim(-WINDOW_SIZE / 2, WINDOW_SIZE / 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.1)

    # Draw Arena Boundary (Filled)
    arena_circle = Circle((0, 0), float(ARENA_RADIUS),
                          facecolor='lavender', edgecolor='darkgrey', fill=True, zorder=0, linewidth=1)
    ax.add_patch(arena_circle)

    # Initialize Time Step Text
    time_text_artist = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                                 color='black', verticalalignment='top', zorder=5)

    # Create initial patch objects for ants
    for i in range(NUM_ANTS):
        initial_vertices = [[ANT_LENGTH/2, 0], [-ANT_LENGTH/2, ANT_WIDTH/2], [-ANT_LENGTH/2, -ANT_WIDTH/2]]
        ant_poly = Polygon(initial_vertices, closed=True, color='black', zorder=1) # Start black
        ax.add_patch(ant_poly)
        ant_patches.append(ant_poly)

# --- Animation Function ---
# Initialize PRNG Key and State *before* setup
key = random.PRNGKey(0)
key, init_key = random.split(key)
current_state = initialize_state(init_key, ARENA_RADIUS)

def update_animation(frame):
    """Updates the animation frame."""
    global current_state, key, time_text_artist, ant_patches # Ensure access
    key, step_key = random.split(key)

    # Calculate current simulation time to pass to update_step
    current_sim_time = frame * DT
    # --- Update ant states ---
    current_state = update_step(current_state, step_key, current_sim_time, DT)


    # --- Update visualization ---
    positions = current_state['position']
    angles = current_state['angle']
    behavioral_states = current_state['behavioral_state'] # Get current state

    updated_artists = []
    for i in range(NUM_ANTS):
        pos = positions[i]
        angle = angles[i]

        # Calculate triangle vertices
        # Use numpy math functions here as matplotlib expects standard floats
        tip_x = pos[0] + (ANT_LENGTH / 2) * math.cos(angle)
        tip_y = pos[1] + (ANT_LENGTH / 2) * math.sin(angle)
        base_center_x = pos[0] - (ANT_LENGTH / 2) * math.cos(angle)
        base_center_y = pos[1] - (ANT_LENGTH / 2) * math.sin(angle)
        dx_base = (ANT_WIDTH / 2) * math.sin(angle)
        dy_base = -(ANT_WIDTH / 2) * math.cos(angle)
        base1_x = base_center_x + dx_base
        base1_y = base_center_y + dy_base
        base2_x = base_center_x - dx_base
        base2_y = base_center_y - dy_base
        # Convert JAX array vertices to list of lists for Polygon
        vertices = [[float(tip_x), float(tip_y)],
                    [float(base1_x), float(base1_y)],
                    [float(base2_x), float(base2_y)]]

        # Update existing patch vertices
        ant_patches[i].set_xy(vertices)

        # <<< Update color based on state >>>
        state_is_resting = behavioral_states[i] == STATE_RESTING
        # Example: Red for resting, Black for moving
        color = 'red' if state_is_resting else 'black'
        ant_patches[i].set_color(color)

        updated_artists.append(ant_patches[i])

    # Update time step text
    time_text_artist.set_text(f't = {(frame + 1) * DT:.1f}') # Show simulation time
    updated_artists.append(time_text_artist)

    return updated_artists

# --- Run Simulation ---
setup_visualization() # Call AFTER initializing state
ani = animation.FuncAnimation(fig, update_animation, frames=int(SIMULATION_STEPS/DT),
                              interval=FRAME_INTERVAL, blit=True, repeat=False)

if SAVE_ANIMATION:
    # Save as GIF using Pillow writer
    ani.save('visualisation/ant_simulation.gif', writer='pillow', fps=int(1000/FRAME_INTERVAL))
    print('Animation saved as ant_simulation.gif')

plt.show()
