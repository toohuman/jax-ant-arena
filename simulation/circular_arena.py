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

PHEROMONE_RADIUS = ANT_LENGTH * 3.0 # How far the 'signal' reaches (tune me!)
P_STOP_PER_NEIGHBOUR = 0.15         # Probability increase factor per resting neighbour (tune me!)
                                   # Effectively: chance a single neighbour causes a stop per dt step.

# <<< Wall Following Parameters >>>
WALL_ZONE_WIDTH = ANT_LENGTH * 1.5 # How far from the wall the turning response starts
WALL_TURN_STRENGTH = 1.8          # How strongly ants turn to align (rad/sec) - TUNE ME!

# Arena will be centered at (0,0)
WINDOW_PADDING = 2.0
WINDOW_SIZE = 2 * (ARENA_RADIUS + WINDOW_PADDING)

DT = 0.1  # Simulation time step (arbitrary units)

# State constants
STATE_RESTING = 0
STATE_MOVING_BURST = 1

# State durations and burst speed parameters
MEAN_REST_DURATION = 3.0   # Average time (in sim time units) to rest
STD_REST_DURATION = 0.5
MEAN_BURST_DURATION = 4.0  # Average time (in sim time units) for a movement burst
STD_BURST_DURATION = 1.0
MEAN_BURST_SPEED = 6.0     # Average speed during a burst (units per dt)
STD_BURST_SPEED = 1.0
MIN_STATE_DURATION = 0.1   # Minimum duration for any state bout (in sim time units)

# Turning rate during bursts
TURN_RATE_STD = 1.2

# Animation Parameters
SIMULATION_STEPS = 500
FRAME_INTERVAL = 25   # Reduced for smoother animation attempt

# --- Global state for animation artists ---
ant_patches = []
time_text_artist = None

# --- Initialization ---

def draw_durations(key, mean, std): # Removed num_ants argument
    """Draws durations from Normal dist, clipped at MIN_STATE_DURATION."""
    # Use the global NUM_ANTS constant for the shape
    durations = mean + random.normal(key, (NUM_ANTS,)) * std
    # Ensure duration is scaled correctly relative to DT if means/stds are in seconds
    # Assuming means/stds are in simulation time units for now. If they are seconds, divide by DT here.
    # Example if params are seconds: return jnp.maximum(MIN_STATE_DURATION/DT, durations / DT)
    # Clipping should also be in sim time units or steps. Let's assume MIN_STATE_DURATION is sim time units.
    return jnp.maximum(MIN_STATE_DURATION, durations)

def initialize_state(key, arena_radius): # Removed num_ants arg, uses global
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
    # <<< FIX for JIT: Call draw_durations without num_ants >>>
    rest_durations = draw_durations(key_r, MEAN_REST_DURATION, STD_REST_DURATION)
    burst_durations = draw_durations(key_b, MEAN_BURST_DURATION, STD_BURST_DURATION)
    initial_durations = jnp.where(initial_states == STATE_RESTING, rest_durations, burst_durations)

    state = {
        'position': positions,
        'angle': angles,
        'speed': jnp.zeros(NUM_ANTS),
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
def update_step(state, key, dt):
    """Performs one vectorized update step using state machine and wall avoidance."""
    num_ants = state['position'].shape[0]
    positions = state['position']
    behavioral_state = state['behavioral_state']
    time_in_state = state['time_in_state']
    current_state_duration = state['current_state_duration']
    angles = state['angle'] # Ant's current heading angle

    # --- 0. Split Keys ---
    # Need one more key for the pheromone stop roll
    key, key_dur_rest, key_dur_burst, key_speed, key_turn, key_collision, key_pheromone_stop = random.split(key, 7)

    # --- 1. Pheromone Influence Calculation --- <<< NEW BLOCK >>>

    # Identify which ants are resting (emitting pheromone)
    is_resting_emitter = (behavioral_state == STATE_RESTING)

    # Calculate pairwise distances (as used in collision)
    delta_x_pair = positions[:, 0, None] - positions[None, :, 0]
    delta_y_pair = positions[:, 1, None] - positions[None, :, 1]
    distances_sq_pair = delta_x_pair**2 + delta_y_pair**2
    distances_pair = jnp.sqrt(distances_sq_pair)

    # Determine which ants are within pheromone radius of each other
    within_radius = (distances_pair < PHEROMONE_RADIUS)

    # Create a matrix where True indicates ant j is resting and within radius of ant i
    # Start with the within_radius mask
    # Then mask based on whether ant j is resting (apply to columns)
    pheromone_signal_matrix = jnp.where(within_radius, is_resting_emitter[None, :], False)

    # Exclude self-influence (set diagonal to False)
    n = pheromone_signal_matrix.shape[0]
    mask = ~jnp.eye(n, dtype=bool)
    pheromone_signal_matrix = jnp.where(mask, pheromone_signal_matrix, False)
    # pheromone_signal_matrix = jnp.fill_diagonal(pheromone_signal_matrix, False)

    # Count emitting neighbours for each ant (sum across columns for each row i)
    num_resting_neighbours = jnp.sum(pheromone_signal_matrix, axis=1)

    # Calculate probability of stopping due to pheromone for *moving* ants
    # Using model: Prob = 1 - (1 - P_indiv)^N --> chance that *at least one* neighbour causes stop
    # Avoid potential floating point issue if P_STOP_PER_NEIGHBOUR is exactly 0 or 1
    prob_stop_pheromone = 1.0 - (1.0 - P_STOP_PER_NEIGHBOUR)**num_resting_neighbours

    # Roll the dice for stopping due to pheromone
    rand_stop = random.uniform(key_pheromone_stop, (num_ants,))
    stops_due_to_pheromone = (behavioral_state == STATE_MOVING_BURST) & (rand_stop < prob_stop_pheromone)


    # --- 2. State Transition Logic ---
    next_time_in_state = time_in_state + dt
    should_transition = (next_time_in_state >= current_state_duration)
    next_behavioral_state = jnp.where(should_transition, 1 - behavioral_state, behavioral_state)
    
    new_rest_durations = draw_durations(key_dur_rest, MEAN_REST_DURATION, STD_REST_DURATION)
    new_burst_durations = draw_durations(key_dur_burst, MEAN_BURST_DURATION, STD_BURST_DURATION)
    duration_for_next_state = jnp.where(next_behavioral_state == STATE_RESTING, new_rest_durations, new_burst_durations)
    next_current_state_duration = jnp.where(should_transition, duration_for_next_state, current_state_duration)
    final_time_in_state = jnp.where(should_transition, 0.0, next_time_in_state)


    # --- 3. State-Dependent Behavior (Speed and Base Turning) ---
    burst_speeds = jnp.maximum(0.0, MEAN_BURST_SPEED + random.normal(key_speed, (NUM_ANTS,)) * STD_BURST_SPEED)
    current_speed = jnp.where(next_behavioral_state == STATE_MOVING_BURST, burst_speeds, 0.0)
    burst_turn_noise = random.normal(key_turn, (NUM_ANTS,)) * TURN_RATE_STD * dt
    base_turn = jnp.where(next_behavioral_state == STATE_MOVING_BURST, burst_turn_noise, 0.0)


    # --- 4. Wall Avoidance / Following Turning --- <<< INSERTED BLOCK >>>
    dist_from_center = jnp.linalg.norm(positions, axis=1)
    in_wall_zone = (dist_from_center > (ARENA_RADIUS - WALL_ZONE_WIDTH))
    apply_wall_turn = in_wall_zone & (next_behavioral_state == STATE_MOVING_BURST)

    # Calculate radial_angle safely
    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    safe_x = jnp.where(dist_from_center < 1e-6, 1.0, pos_x)
    safe_y = jnp.where(dist_from_center < 1e-6, 0.0, pos_y)
    radial_angle = jnp.arctan2(safe_y, safe_x)

    # Calculate tangent angle (counter-clockwise) and angle error
    tangent_angle_ccw = wrap_angle(radial_angle + jnp.pi / 2.0)
    angle_error = wrap_angle(angles - tangent_angle_ccw)

    # Calculate wall turn magnitude (fixed rate * dt to correct alignment)
    # Note: WALL_TURN_STRENGTH is like angular speed (rad / sim time unit)
    wall_turn = -jnp.sign(angle_error) * WALL_TURN_STRENGTH * dt

    # Combine base turn and conditional wall turn
    total_turn = base_turn + jnp.where(apply_wall_turn, wall_turn, 0.0)
    # <<< END INSERTED BLOCK >>>

    # Update angles using the combined turn
    new_angles = wrap_angle(angles + total_turn) # <<< Use total_turn >>>


    # --- 5. Movement, Collision, Boundary ---
    vx = current_speed * jnp.cos(new_angles)
    vy = current_speed * jnp.sin(new_angles)
    velocity = jnp.stack([vx, vy], axis=-1)
    potential_new_positions = positions + velocity * dt

    # Collision Resolution (Uses num_ants for jnp.eye - Okay)
    delta_x = potential_new_positions[:, 0, None] - potential_new_positions[None, :, 0]
    delta_y = potential_new_positions[:, 1, None] - potential_new_positions[None, :, 1]
    distances_sq = delta_x**2 + delta_y**2
    distances = jnp.sqrt(distances_sq)
    distances = jnp.where(jnp.eye(num_ants, dtype=bool), jnp.inf, distances)
    collision_radius = 2 * ANT_RADIUS
    is_overlapping = distances < collision_radius
    overlap_depth = jnp.maximum(0, collision_radius - distances)
    inv_distance = 1.0 / (distances + 1e-9)
    push_dir_x = delta_x * inv_distance
    push_dir_y = delta_y * inv_distance
    push_magnitude = overlap_depth * K_PUSH
    total_push_x = jnp.sum(jnp.where(is_overlapping, push_dir_x * push_magnitude, 0.0), axis=1)
    total_push_y = jnp.sum(jnp.where(is_overlapping, push_dir_y * push_magnitude, 0.0), axis=1)
    pushed_positions_x = potential_new_positions[:, 0] + total_push_x
    pushed_positions_y = potential_new_positions[:, 1] + total_push_y
    new_positions = jnp.stack([pushed_positions_x, pushed_positions_y], axis=-1)

    # Boundary Conditions (Clamping as fallback)
    dist_from_center_new = jnp.linalg.norm(new_positions, axis=1) # Recalculate dist for new pos
    is_outside = dist_from_center_new > ARENA_RADIUS
    dist_from_center_safe = jnp.maximum(1e-9, dist_from_center_new)
    normalized_pos = new_positions / dist_from_center_safe[:, None]
    clamped_positions = jnp.where(is_outside[:, None],
                                  normalized_pos * ARENA_RADIUS,
                                  new_positions)

    # --- 5. Assemble Next State ---
    next_state = {
        'position': clamped_positions,
        'angle': new_angles,
        'speed': current_speed,
        'behavioral_state': next_behavioral_state,
        'time_in_state': final_time_in_state,
        'current_state_duration': next_current_state_duration
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

    # <<< Set xlim/ylim based on ARENA_RADIUS for tight fit >>>
    ax.set_xlim(-WINDOW_SIZE / 2, WINDOW_SIZE / 2)
    ax.set_ylim(-WINDOW_SIZE / 2, WINDOW_SIZE / 2)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_title('') # Remove title setting here
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.1)

    # <<< Hide axes for clean look >>>
    # ax.axis('off')

    # Draw Arena Boundary (Filled)
    arena_circle = Circle((0, 0), float(ARENA_RADIUS),
                          facecolor='lavender', edgecolor='none', fill=True, zorder=0)
    ax.add_patch(arena_circle)

    # Initialize Time Step Text
    time_text_artist = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                                 color='black', verticalalignment='top', zorder=5)

    # Create initial patch objects for ants
    # Need initial state to place them correctly if desired, otherwise start at 0,0
    # Let's initialize state *before* calling setup_visualization
    for i in range(NUM_ANTS):
        # Initial placeholder vertices (small triangle at origin pointing right)
        initial_vertices = [[ANT_LENGTH/2, 0], [-ANT_LENGTH/2, ANT_WIDTH/2], [-ANT_LENGTH/2, -ANT_WIDTH/2]]
        ant_poly = Polygon(initial_vertices, closed=True, color='darkblue', zorder=1)
        ax.add_patch(ant_poly)
        ant_patches.append(ant_poly)

# --- Animation Function ---
# Initialize PRNG Key and State *before* setup
key = random.PRNGKey(0)
key, init_key = random.split(key)
# <<< Call initialize_state using global NUM_ANTS >>>
current_state = initialize_state(init_key, ARENA_RADIUS)

def update_animation(frame):
    """Updates the animation frame."""
    global current_state, key, time_text_artist, ant_patches # Ensure access
    key, step_key = random.split(key)

    # --- Update ant states ---
    current_state = update_step(current_state, step_key, DT)

    # --- Update visualization ---
    positions = current_state['position']
    angles = current_state['angle']
    behavioral_states = current_state['behavioral_state'] # Get current state

    updated_artists = []
    for i in range(NUM_ANTS):
        pos = positions[i]
        angle = angles[i]

        # Calculate triangle vertices
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
        vertices = jnp.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]])

        # Update existing patch vertices
        ant_patches[i].set_xy(vertices)

        # <<< Update color based on state >>> Optional, uncomment to use
        state_is_resting = behavioral_states[i] == STATE_RESTING
        color = 'red' if state_is_resting else 'black' # Example: red/blue
        ant_patches[i].set_color(color)

        updated_artists.append(ant_patches[i])

    # Update time step text
    time_text_artist.set_text(f't = {frame + 1}')
    updated_artists.append(time_text_artist)

    return updated_artists

# --- Run Simulation ---
# <<< Call setup_visualization AFTER initializing state >>>
setup_visualization()
ani = animation.FuncAnimation(fig, update_animation, frames=SIMULATION_STEPS,
                              interval=FRAME_INTERVAL, blit=True, repeat=False)
plt.show()