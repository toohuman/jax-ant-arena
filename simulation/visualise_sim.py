# visualise_sim.py
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random
import hydra
from omegaconf import DictConfig, OmegaConf
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
    def __init__(self, params: dict, cfg: DictConfig):
        super().__init__()
        self.setWindowTitle("JAX Ant Simulation Visualiser")
        self.setGeometry(100, 100, cfg.visualisation.window_size, cfg.visualisation.window_size + 50) # Extra space for controls

        # Store params and config if needed later
        self.params = params
        self.cfg = cfg

        # --- Create the JITted update function ---
        # 1. Create a version of update_step with self.params "baked in"
        #    using partial. This binds self.params to the 'params' argument.
        update_fn = partial(antsim.update_step, params=self.params)

        # 2. JIT *this* new function. JAX will trace the values inside
        #    self.params when it compiles this specific version.
        self.jitted_update_fn = jax.jit(update_fn)

        # --- JAX Simulation State ---
        self.key = random.PRNGKey(0)
        self.key, subkey = random.split(self.key)
        # Initialise state using the updated signature from antsim
        # We get parameters directly from the imported module for simplicity here
        # In a parameterised setup, these would come from a loaded config
        self.sim_state = antsim.initialise_state(subkey, self.params)
        self.sim_time = 0.0
        self.dt = self.params['dt'] # Simulation time step size

        # --- Real-Time Synchronization ---
        self.simulation_time_debt = 0.0
        self.last_update_time = time.perf_counter()
        self.is_running = True
        
        # --- Visualization Control ---
        self.current_speed_level_index = 0 # Start at 1x speed
        self.speedup_factor = SPEED_LEVELS[self.current_speed_level_index]

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
        self.max_simulation_time = cfg.visualisation.max_simulation_time

        # --- Timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(cfg.visualisation.timer_interval_ms)

    def update_status_label(self):
        """ Updates the status label. """
        status_text = "Running" if self.is_running else "Finished"
        pheromone_mode = "Grid" if self.params['pheromones']['use_grid_pheromones'] else "Direct" # Use params
        self.status_label.setText(f"Status: {status_text} ({pheromone_mode}) | Sim Time: {self.sim_time:.2f}")
        self.speed_button.setText(f"Speed: x{self.speedup_factor}")

    def resizeEvent(self, event):
        self.calculate_drawing_parameters()
        super().resizeEvent(event)

    def calculate_drawing_parameters(self):
        """ Calculates center, scaling, and grid cell size using params and cfg. """
        draw_height = self.height() - 50
        draw_width = self.width()
        drawable_size = min(draw_width, draw_height)

        self.center_x = draw_width / 2
        self.center_y = draw_height / 2
        self.arena_pixel_radius = (drawable_size / 2) - self.cfg.visualisation.arena_padding # Use cfg

        arena_radius_sim = self.params['arena_radius'] # Use params
        if arena_radius_sim > 1e-6:
            self.scale_factor = self.arena_pixel_radius / arena_radius_sim
        else:
            self.scale_factor = 1.0

        grid_resolution = self.params['pheromones']['grid_resolution'] # Use params
        if grid_resolution > 0:
            self.grid_cell_pixel_size = (self.arena_pixel_radius * 2) / grid_resolution
        else:
            self.grid_cell_pixel_size = 1.0

    def cycle_speed(self):
        """ Cycles through the defined speed levels. """
        self.current_speed_level_index = (self.current_speed_level_index + 1) % len(SPEED_LEVELS)
        self.speedup_factor = SPEED_LEVELS[self.current_speed_level_index]
        self.simulation_time_debt = 0.0 # Reset debt when changing speed
        self.update_status_label()

    def update_simulation(self):
        """
        Advances the simulation based on real time elapsed and current speed factor,
        then schedules a repaint.
        """
        if not self.is_running:
            return # Don't do anything if simulation is stopped
        
        if self.max_simulation_time and self.sim_time >= self.max_simulation_time:
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
                # NOTE: Pass dynamic args (state, key, t) to the pre-compiled function
                # The 'params' argument is already baked in via partial/JIT decorator
                for step_idx in range(steps_to_run):
                    step_time = self.sim_time + (step_idx * self.dt)
                    self.key, subkey = random.split(self.key)
                    # Pass only dynamic arguments here:
                    current_state = self.jitted_update_fn(current_state, subkey, step_time)

                self.sim_state = current_state
                self.sim_time += sim_time_advanced_this_frame

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
            if self.params['pheromones']['use_grid_pheromones']:
                pheromone_map = jax.device_get(self.sim_state['pheromone_map'])
            else:
                pheromone_map = None # Not needed if not using grid

        except Exception as e:
             print(f"Error getting data from JAX state: {e}")
             painter.setPen(QColor("red"))
             painter.drawText(20, 20, "Error accessing JAX state.")
             return

        # --- Draw Pheromone Grid (if enabled) ---
        use_grid = self.params['pheromones']['use_grid_pheromones'] # Use params
        grid_res = self.params['pheromones']['grid_resolution']   # Use params
        if use_grid and pheromone_map is not None and self.grid_cell_pixel_size > 0.1:
            painter.setPen(Qt.NoPen)
            max_val = jnp.maximum(self.cfg.visualisation.grid_max_pheromone_for_colour, 1e-6) # Use cfg

            for r in range(grid_res):
                for c in range(grid_res):
                    value = pheromone_map[r, c]
                    if value > 1e-4:
                        # Corrected Y coordinate for grid drawing (bottom-up for sim, top-down for screen)
                        # r is sim_x_grid_index, c is sim_y_grid_index
                        # screen_x should depend on r (sim_x_idx)
                        # screen_y should depend on c (sim_y_idx), and be inverted for screen coordinates
                        screen_x = (self.center_x - self.arena_pixel_radius) + r * self.grid_cell_pixel_size
                        screen_y = (self.center_y - self.arena_pixel_radius) + (grid_res - 1 - c) * self.grid_cell_pixel_size

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
        elif not use_grid:
            # Calculate the pheromone radius in screen pixels
            pheromone_pixel_radius = self.params['pheromones']['pheromone_radius'] * self.scale_factor
            num_ants = self.params['num_ants']
            # Set brush and pen for drawing radii
            painter.setBrush(QBrush(COLOUR_DIRECT_PHEROMONE)) # Use the dedicated colour
            painter.setPen(Qt.NoPen) # No outline for the pheromone radius

            for i in range(num_ants):
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
        num_ants = self.params['num_ants'] # Use params
        ant_length = self.params['ant_length'] # Use params
        ant_width = self.params['ant_width'] # Use params (calculated)
        vis_multiplier = self.cfg.visualisation.visual_ant_multiplier # Use cfg

        for i in range(num_ants):
            sim_x, sim_y = positions[i]
            angle_rad = angles[i]
            state = states[i]

            screen_x = self.center_x + sim_x * self.scale_factor
            screen_y = self.center_y - sim_y * self.scale_factor # Invert Y for screen

            # Calculate Ant Pixel Size
            pixel_length = ant_length * self.scale_factor * vis_multiplier
            pixel_width = ant_width * self.scale_factor * vis_multiplier
            min_pixel_size = 2.0
            pixel_length = max(min_pixel_size, pixel_length)
            pixel_width = max(min_pixel_size * (ant_width / ant_length), pixel_width)

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


@hydra.main(config_path="../conf", config_name="config", version_base=None)
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
    if params['pheromones']['grid_resolution'] > 0:
        params['grid_cell_size'] = 2.0 * params['arena_radius'] / params['pheromones']['grid_resolution']
        params['pheromones']['pheromone_radius'] = params['ant_length'] * params['pheromones']['pheromone_radius_multiplier']
        params['pheromones']['grid_radius_cells'] = int(jnp.ceil(params['pheromones']['pheromone_radius'] / params['grid_cell_size']))
    else:
        params['grid_cell_size'] = 0.0
        params['pheromones']['grid_radius_cells'] = 0
        params['pheromones']['pheromone_radius'] = params['ant_length'] * params['pheromones']['pheromone_radius_multiplier']

    params['wall_zone_width'] = params['ant_length'] * params['wall_zone_width_multiplier']
    
    # --- JAX GPU/TPU Configuration (Optional but Recommended) ---
    try:
        # If you have a GPU/TPU, uncomment these lines:
        #jax.config.update('jax_platform_name', 'gpu') # or 'tpu'
        #print("Using platform:", jax.lib.xla_bridge.get_backend().platform)
        pass # Keep using default (CPU) if not specified
    except Exception as e:
        print(f"Could not configure JAX platform: {e}")

    app = QApplication.instance() # Check if already exists
    if app is None:
         app = QApplication(sys.argv)

    visualiser = AntSimulationVisualiser(params, cfg)
    visualiser.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
