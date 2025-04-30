# visualise_sim.py
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random
import hydra
from omegaconf import DictConfig
from functools import partial
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen, QTransform

# --- Import from your JAX simulation file ---
# Make sure ant_simulation.py is in the same directory or your PYTHONPATH
try:
    import ant_simulation as antsim
except ImportError:
    print("Error: Could not import ant_simulation.py.")
    print("Please ensure it's in the same directory or accessible in your PYTHONPATH.")
    sys.exit(1)


MAX_SIMULATION_TIME = 1000   # None = run indefinitely


# --- Visualisation Parameters ---
WINDOW_SIZE = 800
ARENA_PADDING = 20 # Pixel padding around the arena
VISUAL_ANT_MULTIPLIER = 1.0
TIMER_INTERVAL_MS = 16 # Target ~60 FPS for visualisation updates
# Define the speed levels (multipliers relative to real-time)
# 1x means 1 sim second per real second
SPEED_LEVELS = (1, 4, 8, 16, 64, 128)
GRID_MAX_PHEROMONE_FOR_COLOUR = 5.0
VISUAL_DIRECT_PHEROMONE_ALPHA = 20

# --- Colours ---
COLOUR_BACKGROUND = QColor(245, 245, 245)
COLOUR_ARENA = QColor(230, 220, 240)
COLOUR_MOVING = QColor(0, 0, 200)     # Blue
COLOUR_RESTING = QColor(200, 50, 50)    # Red
COLOUR_ARRESTED = QColor(0, 0, 0)       # Black
COLOUR_PHEROMONE_BASE = QColor(150, 50, 200) # Base colour for grid cells
COLOUR_DIRECT_PHEROMONE = QColor(150, 50, 200, VISUAL_DIRECT_PHEROMONE_ALPHA)

class AntSimulationVisualiser(QWidget):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("JAX Ant Simulation Visualiser")
        self.setGeometry(100, 100, WINDOW_SIZE, WINDOW_SIZE + 50) # Extra space for controls

        # --- JAX Simulation State ---
        self.key = random.PRNGKey(0)
        self.key, subkey = random.split(self.key)
        # Initialise state using the updated signature from antsim
        # We get parameters directly from the imported module for simplicity here
        # In a parameterised setup, these would come from a loaded config
        self.sim_state = antsim.initialise_state(
            subkey,
            config["parameters"]["ants"]["num_ants"],
            config["parameters"]["ants"]["arena_radius"],
            config["parameters"]["ants"]["grid_resolution"]
        )
        self.sim_time = 0.0
        self.dt = antsim.DT # Simulation time step size

        # --- Real-Time Synchronization ---
        self.simulation_time_debt = 0.0
        self.last_update_time = time.perf_counter()
        self.is_running = True
        
        # --- Visualization Control ---
        self.current_speed_level_index = 0 # Start at 1x speed
        self.speedup_factor = SPEED_LEVELS[self.current_speed_level_index]

        # Use partial to fix the static arguments for the JIT compilation
        self.update_fn = partial(
            antsim.update_step,
            dt=self.dt,
            num_ants=antsim.NUM_ANTS,
            arena_radius=antsim.ARENA_RADIUS,
            grid_resolution=antsim.GRID_RESOLUTION,
            use_grid_pheromones=antsim.USE_GRID_PHEROMONES
        )
        self.jitted_update_fn = jax.jit(self.update_fn)

        # --- UI Elements ---
        self.speed_button = QPushButton(f"Speed: x{self.speedup_factor}", self)
        self.speed_button.setStyleSheet("color: black;")
        self.speed_button.clicked.connect(self.cycle_speed)

        self.status_label = QLabel("Status: Initializing...", self)
        self.status_label.setStyleSheet("color: black;")
        self.update_status_label() # Initialise label text correctly

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.speed_button)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch(1) # Pushes controls to the left

        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1) # Visualiser area will expand
        main_layout.addLayout(control_layout) # Controls at the bottom

        # --- Drawing Parameters ---
        self.center_x = 0
        self.center_y = 0
        self.arena_pixel_radius = 1
        self.scale_factor = 1.0
        self.grid_cell_pixel_size = 1.0

        # --- Timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(TIMER_INTERVAL_MS)

    def update_status_label(self):
        """ Updates the status label. """
        status_text = "Running" if self.is_running else "Finished"
        pheromone_mode = "Grid" if antsim.USE_GRID_PHEROMONES else "Direct"
        self.status_label.setText(f"Status: {status_text} ({pheromone_mode}) | Sim Time: {self.sim_time:.2f}")
        self.speed_button.setText(f"Speed: x{self.speedup_factor}")

    def resizeEvent(self, event):
        self.calculate_drawing_parameters()
        super().resizeEvent(event)

    def calculate_drawing_parameters(self):
        """ Calculates center, scaling, and grid cell size. """
        draw_height = self.height() - 50
        draw_width = self.width()
        drawable_size = min(draw_width, draw_height)

        self.center_x = draw_width / 2
        self.center_y = draw_height / 2
        self.arena_pixel_radius = (drawable_size / 2) - ARENA_PADDING

        if antsim.ARENA_RADIUS > 1e-6:
            self.scale_factor = self.arena_pixel_radius / antsim.ARENA_RADIUS
        else:
            self.scale_factor = 1.0

        # Calculate grid cell size in pixels
        if antsim.GRID_RESOLUTION > 0:
            # Total width of arena in pixels / number of cells
            self.grid_cell_pixel_size = (self.arena_pixel_radius * 2) / antsim.GRID_RESOLUTION
        else:
            self.grid_cell_pixel_size = 1.0

    def cycle_speed(self):
        """ Cycles through the defined speed levels relative to real-time. """
        self.current_speed_level_index = (self.current_speed_level_index + 1) % len(SPEED_LEVELS)
        self.speedup_factor = SPEED_LEVELS[self.current_speed_level_index]
        self.simulation_time_debt = 0.0
        self.update_status_label()

    def update_simulation(self):
        """
        Advances the simulation based on real time elapsed and current speed factor,
        then schedules a repaint.
        """
        if not self.is_running:
            return # Don't do anything if simulation is stopped
        
        if MAX_SIMULATION_TIME and self.sim_time >= MAX_SIMULATION_TIME:
            if self.is_running:
                self.is_running = False
                self.timer.stop()
                self.speed_button.setEnabled(False)
                self.update_status_label() # Show final status
                print(f"Simulation finished at time {self.sim_time:.2f}")
                self.update()
            return

        current_time = time.perf_counter()
        real_dt = current_time - self.last_update_time
        self.last_update_time = current_time

        sim_time_to_advance = (real_dt * self.speedup_factor) + self.simulation_time_debt
        steps_to_run = max(0, int(sim_time_to_advance / self.dt))
        sim_time_advanced_this_frame = steps_to_run * self.dt
        self.simulation_time_debt = sim_time_to_advance - sim_time_advanced_this_frame

        if steps_to_run > 0:
            try:
                current_state = self.sim_state
                for _ in range(steps_to_run):
                    step_time = self.sim_time + (_ * self.dt)
                    self.key, subkey = random.split(self.key)
                    current_state = self.jitted_update_fn(current_state, subkey, step_time)

                self.sim_state = current_state # Update state after all steps
                self.sim_time += sim_time_advanced_this_frame # Update total time

            except Exception as e:
                print(f"Error during JAX simulation step: {e}")
                import traceback
                traceback.print_exc() # Print full traceback
                self.timer.stop()
                self.status_label.setText(f"Status: Error! Check console.")
                self.speed_button.setEnabled(False)
                self.is_running = False # Stop simulation on error
                return

        self.update()
        self.update_status_label()

    def paintEvent(self, event):
        """ Draws the simulation state. """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.arena_pixel_radius <= 1:
            self.calculate_drawing_parameters()

        # --- Draw Background and Arena ---
        painter.setBrush(COLOUR_BACKGROUND)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        painter.setBrush(COLOUR_ARENA)
        arena_rect = QRectF(self.center_x - self.arena_pixel_radius,
                           self.center_y - self.arena_pixel_radius,
                           self.arena_pixel_radius * 2,
                           self.arena_pixel_radius * 2)
        painter.drawEllipse(arena_rect)

        # --- Get State Data ---
        try:
            # Get data onto CPU for drawing
            positions = jax.device_get(self.sim_state['position'])
            angles = jax.device_get(self.sim_state['angle'])
            states = jax.device_get(self.sim_state['behavioural_state'])
            if antsim.USE_GRID_PHEROMONES:
                pheromone_map = jax.device_get(self.sim_state['pheromone_map'])
            else:
                pheromone_map = None # Not needed if not using grid

        except Exception as e:
             print(f"Error getting data from JAX state: {e}")
             painter.setPen(QColor("red"))
             painter.drawText(20, 20, "Error accessing JAX state.")
             return

        # --- Draw Pheromone Grid (if enabled) ---
        if antsim.USE_GRID_PHEROMONES and pheromone_map is not None and self.grid_cell_pixel_size > 0.1:
            painter.setPen(Qt.NoPen)
            max_val = jnp.maximum(GRID_MAX_PHEROMONE_FOR_COLOUR, 1e-6)

            for r in range(antsim.GRID_RESOLUTION):
                for c in range(antsim.GRID_RESOLUTION):
                    value = pheromone_map[r, c]
                    if value > 1e-4:
                        screen_x = (self.center_x - self.arena_pixel_radius) + c * self.grid_cell_pixel_size
                        screen_y = (self.center_y + self.arena_pixel_radius) - (r + 1) * self.grid_cell_pixel_size

                        intensity_fraction = jnp.clip(value / max_val, 0.0, 1.0)
                        alpha = int(intensity_fraction * 255)

                        if alpha > 0:
                            cell_colour = QColor(COLOUR_PHEROMONE_BASE)
                            cell_colour.setAlpha(alpha)
                            painter.setBrush(QBrush(cell_colour))
                            painter.drawRect(QRectF(screen_x, screen_y,
                                                    self.grid_cell_pixel_size,
                                                    self.grid_cell_pixel_size))

        # --- Draw Pheromone Radii (if Direct mode is enabled) --- <<< MODIFIED Section >>>
        elif not antsim.USE_GRID_PHEROMONES:
            # Calculate the pheromone radius in screen pixels
            pheromone_pixel_radius = antsim.PHEROMONE_RADIUS * self.scale_factor

            # Set brush and pen for drawing radii
            painter.setBrush(QBrush(COLOUR_DIRECT_PHEROMONE)) # Use the dedicated colour
            painter.setPen(Qt.NoPen) # No outline for the pheromone radius

            for i in range(antsim.NUM_ANTS):
                state = states[i]
                # Check if the ant is an emitter (Resting or Arrested)
                is_emitter = (state == antsim.STATE_RESTING) or (state == antsim.STATE_ARRESTED)

                if is_emitter and pheromone_pixel_radius > 0.1: # Only draw if emitter and radius is visible
                    sim_x, sim_y = positions[i]
                    screen_x = self.center_x + sim_x * self.scale_factor
                    screen_y = self.center_y - sim_y * self.scale_factor # Invert Y

                    # Draw the pheromone radius circle
                    pheromone_rect = QRectF(screen_x - pheromone_pixel_radius,
                                           screen_y - pheromone_pixel_radius,
                                           pheromone_pixel_radius * 2,
                                           pheromone_pixel_radius * 2)
                    painter.drawEllipse(pheromone_rect)


        # --- Draw Ants ---
        for i in range(antsim.NUM_ANTS):
            sim_x, sim_y = positions[i]
            angle_rad = angles[i]
            state = states[i]

            screen_x = self.center_x + sim_x * self.scale_factor
            screen_y = self.center_y - sim_y * self.scale_factor # Invert Y for screen

            # Calculate Ant Pixel Size
            sim_length = antsim.ANT_LENGTH
            sim_width = antsim.ANT_WIDTH
            pixel_length = sim_length * self.scale_factor * VISUAL_ANT_MULTIPLIER
            pixel_width = sim_width * self.scale_factor * VISUAL_ANT_MULTIPLIER
            min_pixel_size = 2.0
            pixel_length = max(min_pixel_size, pixel_length)
            pixel_width = max(min_pixel_size * (sim_width / sim_length), pixel_width)

            # Define Ant Polygon
            ant_poly = QPolygonF([
                QPointF(pixel_length / 2.0, 0),
                QPointF(-pixel_length / 2.0, -pixel_width / 2.0),
                QPointF(-pixel_length / 2.0, pixel_width / 2.0)
            ])

            # Assign Colour
            if state == antsim.STATE_RESTING:
                colour = COLOUR_RESTING
            elif state == antsim.STATE_ARRESTED:
                colour = COLOUR_ARRESTED
            else: # Moving
                colour = COLOUR_MOVING
            painter.setBrush(QBrush(colour))
            painter.setPen(Qt.NoPen)

            # Draw Ant
            painter.save()
            painter.translate(screen_x, screen_y)
            painter.rotate(-jnp.rad2deg(angle_rad)) # Rotate based on angle
            painter.drawPolygon(ant_poly)
            painter.restore()

        painter.end()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration used:")
    print(OmegaConf.to_yaml(cfg))

    # --- Prepare Parameters for Simulation ---
    # Convert OmegaConf to a regular dictionary or a custom object if preferred
    # This avoids potential issues with JAX tracing OmegaConf objects.
    params = OmegaConf.to_container(cfg, resolve=True) # Resolve interpolations if any

    # Calculate derived parameters and add them to the params dict
    # It's often cleaner to do this here than relying on Hydra interpolations
    # for things involving complex logic or multiple base parameters.
    params['ant_width'] = params['ant_length'] / 2.0
    params['ant_radius'] = params['ant_length'] / 2.0
    if params['grid_resolution'] > 0:
         params['grid_cell_size'] = 2.0 * params['arena_radius'] / params['grid_resolution']
         params['pheromone_radius'] = params['ant_length'] * params['pheromone_radius_multiplier']
         params['pheromone_grid_radius_cells'] = int(jnp.ceil(params['pheromone_radius'] / params['grid_cell_size']))
    else:
         params['grid_cell_size'] = 0.0
         params['pheromone_grid_radius_cells'] = 0
         params['pheromone_radius'] = params['ant_length'] * params['pheromone_radius_multiplier'] # Still needed for direct mode

    params['wall_zone_width'] = params['ant_length'] * params['wall_zone_width_multiplier']
    
    # --- JAX GPU/TPU Configuration (Optional but Recommended) ---
    try:
        # If you have a GPU/TPU, uncomment these lines:
        #jax.config.update('jax_platform_name', 'gpu') # or 'tpu'
        #print("Using platform:", jax.lib.xla_bridge.get_backend().platform)
        pass # Keep using default (CPU) if not specified
    except Exception as e:
        print(f"Could not configure JAX platform: {e}")

    app = QApplication(sys.argv)
    visualiser = AntSimulationVisualiser()
    visualiser.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
