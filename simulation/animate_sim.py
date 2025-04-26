# animate_sim.py
# Handles the visualisation of the ant simulation using Matplotlib.
# Imports core logic from ant_simulation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import math
import jax.numpy as jnp # Often useful for state manipulation
import jax.random as random
import numpy as np
from tqdm import tqdm # Progress bar for simulation loop

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Import core simulation components ---
from ant_simulation import (
    NUM_ANTS, ARENA_RADIUS, ANT_LENGTH, ANT_WIDTH, DT, STATE_RESTING, STATE_ARRESTED, STATE_MOVING_BURST,
    # Core functions
    initialise_state, update_step
)

# --- Simulation & Video Parameters ---
MAX_SIM_TIME = 500.0 # Max simulation time to record (e.g., 500 seconds)
SAVE_ANIMATION = True # Assuming we always save when running this script version
VIDEO_FPS = 30 # Desired output video frame rate
CONTENT_SPEED_FACTOR = 16.0 # Simulation speed shown in video (e.g., 4x real-time)
VIDEO_FILENAME = f"ant_simulation_{int(CONTENT_SPEED_FACTOR)}x_{VIDEO_FPS}fps.mp4" # Descriptive filename

# --- Visualisation Parameters ---
WINDOW_PADDING = 2.0
WINDOW_SIZE = 2 * (ARENA_RADIUS + WINDOW_PADDING)


# --- Variables for storing simulation history ---
fig, ax = plt.subplots(figsize=(8, 8))
ant_patches = []
time_text_artist = None
simulation_history = [] # List to store relevant state data for each step
total_sim_steps = 0

# --- Visualisation Setup ---
def setup_visualisation():
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

# --- Function to update plot from stored history ---
def update_frame_from_history(frame_index):
    """Updates plot artists based on pre-computed simulation history."""
    global simulation_history, time_text_artist, ant_patches, total_sim_steps

    # Calculate the simulation step index corresponding to this video frame
    # Video time progresses linearly: frame_index / VIDEO_FPS
    # Simulation time progresses faster: video_time * CONTENT_SPEED_FACTOR
    # Corresponding simulation step index: sim_time / DT
    target_video_time = frame_index / VIDEO_FPS
    target_sim_time = target_video_time * CONTENT_SPEED_FACTOR
    # Use round() to pick the nearest simulation step, clamp index to bounds
    sim_step_index = min(total_sim_steps - 1, int(round(target_sim_time / DT)))

    # Retrieve the state data for that simulation step
    state_data = simulation_history[sim_step_index]
    positions = state_data['position'] # Should be numpy arrays now
    angles = state_data['angle']
    behavioural_states = state_data['behavioural_state']
    actual_sim_time = state_data['sim_time'] # Get the actual time for accurate display
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
        state = behavioural_states[i]
        if state == STATE_RESTING:
            color = 'red'
        elif state == STATE_ARRESTED:
            color = 'black'
        else: # STATE_MOVING_BURST
            color = 'blue'
        ant_patches[i].set_color(color) # Use the determined color string

        updated_artists.append(ant_patches[i])

    # Update time step text - display simulation time
    time_text_artist.set_text(f't = {actual_sim_time:.1f}')
    updated_artists.append(time_text_artist)

    return updated_artists # Return list of modified artists for blitting


# --- Run Simulation Visualisation ---
if __name__ == "__main__": # Standard practice for executable scripts
    # Pre-compute simulation history
    print("Pre-computing simulation...")
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    current_state = initialise_state(init_key, ARENA_RADIUS)
    # Calculate total steps based on MAX_SIM_TIME
    total_sim_steps = int(MAX_SIM_TIME / DT)
    simulation_history = [] # Clear/initialize history storage

    for step in tqdm(range(total_sim_steps), desc="Simulating steps", unit="step"):
        current_sim_time = step * DT
        # Store relevant state data (convert JAX arrays to NumPy for safety/ease)
        state_to_store = {
            'position': np.array(current_state['position']),
            'angle': np.array(current_state['angle']),
            'behavioural_state': np.array(current_state['behavioural_state']),
            'sim_time': current_sim_time # Store the precise time for this state
        }
        simulation_history.append(state_to_store)

        # Update state for the *next* step
        key, step_key = random.split(key)
        current_state = update_step(current_state, step_key, current_sim_time, DT)

    print(f"Simulation pre-computation finished. Stored {len(simulation_history)} states.")

    print("Setting up visualisation...")
    setup_visualisation() # Creates the plot elements and populates ant_patches

    # --- Create animation using history ---
    print("Generating animation frames...")
    # Calculate total video frames needed based on desired duration and FPS
    total_video_duration_sec = MAX_SIM_TIME / CONTENT_SPEED_FACTOR
    total_video_frames = int(round(total_video_duration_sec * VIDEO_FPS))
    print(f"  Will generate {total_video_frames} frames for video...")

    # Use the new update function and calculated frame count
    # interval doesn't matter much for saving, blit=True is good for performance
    ani = animation.FuncAnimation(fig, update_frame_from_history, frames=total_video_frames,
                                  blit=True, repeat=False)

    if SAVE_ANIMATION:
        save_dir = os.path.join(PROJECT_ROOT, "visualisation")
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        save_path = os.path.join(save_dir, VIDEO_FILENAME)
        print(f"Saving animation to {save_path} at {VIDEO_FPS} FPS...")

        # Save using the desired VIDEO_FPS. Adjust dpi for resolution/quality.
        ani.save(save_path, writer="ffmpeg", fps=VIDEO_FPS, dpi=150)
        print("Animation saved.")
    else:
        print("Displaying animation (display speed may not match real-time)...")
        plt.show() # Show plot window if not saving

    print("Done.")