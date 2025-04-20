# visualise_sim.py
# Handles the visualization of the ant simulation using Matplotlib.
# Imports core logic from ant_simulation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import math
import jax.numpy as jnp # Often useful for state manipulation
import jax.random as random

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Import core simulation components ---
from ant_simulation import (
    # Parameters needed for setup/visualization
    NUM_ANTS, ARENA_RADIUS, ANT_LENGTH, ANT_WIDTH, DT, MAX_TIMESTEPS,
    # Constants needed
    STATE_RESTING,
    # Core functions
    initialise_state, update_step, wrap_angle # wrap_angle might be needed if viz does calculations
)

# --- Visualization Parameters ---
FRAME_INTERVAL = 10   # Animation frame interval (ms)
WINDOW_PADDING = 2.0
WINDOW_SIZE = 2 * (ARENA_RADIUS + WINDOW_PADDING)

# --- Global state for animation artists & simulation ---
# (Using globals here for simplicity with FuncAnimation)
fig, ax = plt.subplots(figsize=(8, 8))
ant_patches = []
time_text_artist = None
key = random.PRNGKey(0) # Simulation random key
key, init_key = random.split(key)
current_state = initialise_state(init_key, ARENA_RADIUS) # Initial simulation state

# --- Visualization Setup ---
def setup_visualization():
    """Sets up the static parts of the plot and initializes patches and text."""
    global ant_patches, time_text_artist # Use global references

    # Reset lists if setup is called again
    ant_patches = []
    # Clear previous axes content if necessary (good practice if re-running setup)
    ax.cla()

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
        # Initial placeholder vertices
        initial_vertices = [[ANT_LENGTH/2, 0], [-ANT_LENGTH/2, ANT_WIDTH/2], [-ANT_LENGTH/2, -ANT_WIDTH/2]]
        ant_poly = Polygon(initial_vertices, closed=True, color='black', zorder=1) # Start black
        ax.add_patch(ant_poly)
        ant_patches.append(ant_poly)

# --- Animation Function ---
def update_animation(frame):
    """Updates the animation frame."""
    global current_state, key, time_text_artist, ant_patches # Ensure access to globals

    # Get a new key for this step
    key, step_key = random.split(key)

    # Calculate current simulation time to pass to update_step
    # Use frame number directly if PHEROMONE_MIDPOINT_TIME was based on steps
    # If PHEROMONE_MIDPOINT_TIME uses DT, use frame * DT
    current_sim_time = frame * DT
    # --- Update ant states by calling the imported core function ---
    current_state = update_step(current_state, step_key, current_sim_time, DT)

    # --- Update visualization ---
    positions = current_state['position']
    angles = current_state['angle']
    behavioral_states = current_state['behavioral_state'] # Get current state

    updated_artists = []
    for i in range(NUM_ANTS):
        pos = positions[i]
        angle = angles[i]

        # Calculate triangle vertices using standard math for plotting
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

        # Update color based on state
        state_is_resting = behavioral_states[i] == STATE_RESTING
        color = 'red' if state_is_resting else 'black'
        ant_patches[i].set_color(color)

        updated_artists.append(ant_patches[i])

    # Update time step text - display simulation time
    time_text_artist.set_text(f't = {current_sim_time:.1f}')
    updated_artists.append(time_text_artist)

    # Add the axes background to the list of updated artists if blitting.
    # This might be needed if clearing the axes (`ax.cla()`) but usually
    # handled automatically by blit=True if patches cover previous frame.
    # If flickering occurs, might need ax.draw_artist(ax.patch) first
    # and return ax.patch in list. For now, assume patches cover ok.

    return updated_artists # Return list of modified artists for blitting

# --- Run Simulation Visualization ---
if __name__ == "__main__": # Standard practice for executable scripts
    print("Setting up visualization...")
    setup_visualization() # Call AFTER initializing state globally
    print("Starting animation...")
    # Note: The number of frames is determined by MAX_TIMESTEPS imported from core.
    # Consider if MAX_TIMESTEPS should be defined locally in visualise_sim.py
    # or if it should always match the core definition. Let's use the imported one.
    num_frames = MAX_TIMESTEPS # Total steps match core definition for now

    ani = animation.FuncAnimation(fig, update_animation, frames=num_frames,
                                  interval=FRAME_INTERVAL, blit=True, repeat=False)
    plt.show()
    print("Animation finished.")