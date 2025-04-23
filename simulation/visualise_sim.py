# visualise_sim.py
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random
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


MAX_SIMULATION_TIME = 500   # None = run indefinitely


# --- Visualisation Parameters ---
WINDOW_SIZE = 800
ARENA_PADDING = 20 # Pixel padding around the arena
VISUAL_ANT_MULTIPLIER = 1.0
TIMER_INTERVAL_MS = 16 # Target ~60 FPS for visualisation updates
# Define the speed levels (multipliers relative to real-time)
# 1x means 1 sim second per real second
SPEED_LEVELS = (1, 4, 8, 16, 64, 128)

# --- Colours ---
COLOUR_BACKGROUND = QColor(245, 245, 245)
COLOUR_ARENA = QColor(230, 220, 240)
COLOUR_MOVING = QColor(0, 0, 200)     # Blue
COLOUR_RESTING = QColor(200, 50, 50)    # Red
COLOUR_ARRESTED = QColor(0, 0, 0)       # Black

class AntSimulationVisualiser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JAX Ant Simulation Visualiser")
        self.setGeometry(100, 100, WINDOW_SIZE, WINDOW_SIZE + 50) # Extra space for controls

        # --- JAX Simulation State ---
        self.key = random.PRNGKey(0)
        self.key, subkey = random.split(self.key)
        self.sim_state = antsim.initialise_state(subkey, antsim.ARENA_RADIUS)
        self.sim_time = 0.0 # Tracks the current time within the simulation
        self.dt = antsim.DT # Simulation time step size

        # --- Real-Time Synchronization ---
        # Stores simulation time we owe but couldn't run due to discrete steps
        self.simulation_time_debt = 0.0
        # Stores the real time the last update occurred
        self.last_update_time = time.perf_counter()
        self.is_running = True
        
        # --- Visualization Control ---
        self.current_speed_level_index = 0 # Start at 1x speed
        # speedup_factor is now the multiplier for real-time
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

        # --- Timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(TIMER_INTERVAL_MS) # Timer dictates visualisation refresh rate

    def update_status_label(self):
        """ Updates the status label with current speed and sim time. """
        self.status_label.setText(f"Status: Running | Sim Time: {self.sim_time:.2f}")
        self.speed_button.setText(f"Speed: x{self.speedup_factor}")

    def resizeEvent(self, event):
        self.calculate_drawing_parameters()
        super().resizeEvent(event)

    def calculate_drawing_parameters(self):
        """ Calculates center and scaling based on current window size """
        # Use the available space minus the control layout height (approx)
        draw_height = self.height() - 50
        draw_width = self.width()
        drawable_size = min(draw_width, draw_height)

        self.center_x = draw_width / 2
        self.center_y = draw_height / 2 # Top area for drawing
        self.arena_pixel_radius = (drawable_size / 2) - ARENA_PADDING

        # Prevent division by zero if arena radius is zero
        if antsim.ARENA_RADIUS > 1e-6:
            self.scale_factor = self.arena_pixel_radius / antsim.ARENA_RADIUS
        else:
            self.scale_factor = 1.0

    def cycle_speed(self):
        """ Cycles through the defined speed levels relative to real-time. """
        self.current_speed_level_index = (self.current_speed_level_index + 1) % len(SPEED_LEVELS)
        self.speedup_factor = SPEED_LEVELS[self.current_speed_level_index]
        # Reset debt when changing speed to avoid sudden jumps
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
            if self.is_running: # Only do this once
                self.is_running = False
                self.timer.stop()
                self.speed_button.setEnabled(False)
                self.status_label.setText(f"Status: Finished | Sim Time: {self.sim_time:.2f}")
                print(f"Simulation finished at time {self.sim_time:.2f}")
                self.update() # Ensure final frame is painted
            return

        # -------------------------------------------------------
            
        # 1. Calculate real time elapsed since last update
        current_time = time.perf_counter()
        real_dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 2. Determine simulation time to advance in this frame
        # Target sim time = real time elapsed * speed multiplier + any leftover debt
        sim_time_to_advance = (real_dt * self.speedup_factor) + self.simulation_time_debt

        # 3. Calculate how many whole simulation steps (dt) fit into this
        # Use max(0,...) in case real_dt is negative or tiny due to timing issues
        steps_to_run = max(0, int(sim_time_to_advance / self.dt))

        # 4. Calculate simulation time actually covered by these whole steps
        sim_time_advanced_this_frame = steps_to_run * self.dt

        # 5. Update the simulation time debt for the next frame
        # debt = (what we should have run) - (what we actually ran)
        self.simulation_time_debt = sim_time_to_advance - sim_time_advanced_this_frame

        # 6. Run the simulation steps
        if steps_to_run > 0:
            # Ensure state is on CPU if needed (uncomment if using GPU/TPU heavily)
            # self.sim_state = jax.device_get(self.sim_state)

            # Store the simulation time at the start of this batch of steps
            step_time = self.sim_time
            try:
                for _ in range(steps_to_run):
                    self.key, subkey = random.split(self.key)
                    # Pass the correct simulation time for this specific step
                    self.sim_state = antsim.update_step(self.sim_state, subkey, step_time, self.dt)
                    # Increment the time for the *next* step
                    step_time += self.dt

                # Update the global simulation time tracker *after* the loop
                self.sim_time = step_time

            except Exception as e:
                print(f"Error during JAX simulation step: {e}")
                self.timer.stop()
                self.status_label.setText(f"Status: Error! Check console.")
                self.speed_button.setEnabled(False)
                return

        # 7. Trigger a repaint regardless of whether steps were run (keeps UI responsive)
        self.update()
        self.update_status_label()

    def paintEvent(self, event):
        """ Draws the simulation state (same as before) """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.arena_pixel_radius <= 1:
            self.calculate_drawing_parameters()

        painter.setBrush(COLOUR_BACKGROUND)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        painter.setBrush(COLOUR_ARENA)
        arena_rect = QRectF(self.center_x - self.arena_pixel_radius,
                           self.center_y - self.arena_pixel_radius,
                           self.arena_pixel_radius * 2,
                           self.arena_pixel_radius * 2)
        painter.drawEllipse(arena_rect)

        try:
            positions = jax.device_get(self.sim_state['position'])
            angles = jax.device_get(self.sim_state['angle'])
            states = jax.device_get(self.sim_state['behavioural_state'])
        except Exception as e:
             print(f"Error getting data from JAX state: {e}")
             painter.setPen(QColor("red"))
             painter.drawText(20, 20, "Error accessing JAX state.")
             return

        for i in range(antsim.NUM_ANTS):
            sim_x, sim_y = positions[i]
            angle_rad = angles[i]
            state = states[i]

            screen_x = self.center_x + sim_x * self.scale_factor
            screen_y = self.center_y - sim_y * self.scale_factor

            # --- Calculate Ant Pixel Size for this Frame ---
            # Base size in simulation units
            sim_length = antsim.ANT_LENGTH
            # ant_simulation.py defines width = length / 2.0
            sim_width = antsim.ANT_WIDTH # Or calculate as antsim.ANT_LENGTH / 2.0

            # Convert to pixels using scale_factor and apply visual multiplier
            # Use self.scale_factor which IS proportional to arena size
            pixel_length = sim_length * self.scale_factor * VISUAL_ANT_MULTIPLIER
            pixel_width = sim_width * self.scale_factor * VISUAL_ANT_MULTIPLIER

            # Ensure minimum size to prevent tiny/invisible ants if window is very small
            min_pixel_size = 2.0 # Minimum length in pixels
            pixel_length = max(min_pixel_size, pixel_length)
            pixel_width = max(min_pixel_size * (sim_width / sim_length), pixel_width) # Maintain aspect ratio

            # --- Define Ant Polygon Dynamically (using calculated pixel values) ---
            ant_poly = QPolygonF([
                QPointF(pixel_length / 2.0, 0),
                QPointF(-pixel_length / 2.0, -pixel_width / 2.0),
                QPointF(-pixel_length / 2.0, pixel_width / 2.0)
            ])

            # --- Assign Colour Based on State --- <<< MODIFIED LOGIC >>>
            if state == antsim.STATE_RESTING:
                colour = COLOUR_RESTING
            elif state == antsim.STATE_ARRESTED:
                colour = COLOUR_ARRESTED
            else: # Default to Moving
                colour = COLOUR_MOVING
            painter.setBrush(QBrush(colour))
            painter.setPen(Qt.NoPen)

            painter.save()
            painter.translate(screen_x, screen_y)
            painter.rotate(-jnp.rad2deg(angle_rad))
            painter.drawPolygon(ant_poly)
            painter.restore()

        painter.end()


if __name__ == "__main__":
    # --- JAX GPU/TPU Configuration (Optional but Recommended) ---
    # try:
    #     # If you have a GPU/TPU, uncomment these lines:
    #     #jax.config.update('jax_platform_name', 'gpu') # or 'tpu'
    #     #print("Using platform:", jax.lib.xla_bridge.get_backend().platform)
    #     pass # Keep using default (CPU) if not specified
    # except Exception as e:
    #     print(f"Could not configure JAX platform: {e}")

    app = QApplication(sys.argv)
    visualiser = AntSimulationVisualiser()
    visualiser.show()
    sys.exit(app.exec_())