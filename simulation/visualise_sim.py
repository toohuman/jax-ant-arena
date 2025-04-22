# visualise_sim.py
import sys
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

# --- Visualization Parameters ---
WINDOW_SIZE = 800
ANT_DRAW_LENGTH = antsim.ANT_LENGTH * 2.5 # Make ants more visible
ANT_DRAW_WIDTH = antsim.ANT_WIDTH * 2.5
ARENA_PADDING = 20 # Pixel padding around the arena
TIMER_INTERVAL_MS = 16 # Target ~60 FPS for visualization updates
DEFAULT_SPEEDUP_FACTOR = 16

# --- Colors ---
COLOR_BACKGROUND = QColor(204, 204, 204) # #cccccc
COLOR_ARENA = QColor(106, 127, 163)      # #6a7fa3
COLOR_MOVING = QColor(34, 34, 34)        # #222222
COLOR_RESTING = QColor(200, 50, 50)      # Reddish

class AntSimulationVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JAX Ant Simulation Visualizer (PyQt5)")
        self.setGeometry(100, 100, WINDOW_SIZE, WINDOW_SIZE + 50) # Extra space for controls

        # --- JAX Simulation State ---
        self.key = random.PRNGKey(0) # Initial JAX random key
        self.key, subkey = random.split(self.key)
        # Use NUM_ANTS from the imported module
        self.sim_state = antsim.initialise_state(subkey, antsim.ARENA_RADIUS)
        self.sim_time = 0.0
        self.dt = antsim.DT # Get timestep from simulation module

        # --- UI Elements ---
        self.speedup_button = QPushButton(f"Speed Up (x{DEFAULT_SPEEDUP_FACTOR})", self)
        self.speedup_button.setCheckable(True)
        self.speedup_button.clicked.connect(self.toggle_speedup)

        self.status_label = QLabel("Status: Running (x1)", self)

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.speedup_button)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch(1) # Pushes controls to the left

        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1) # Visualizer area will expand
        main_layout.addLayout(control_layout) # Controls at the bottom

        # --- Visualization Control ---
        self.speedup_active = False
        self.speedup_factor = DEFAULT_SPEEDUP_FACTOR
        self.update_status_label()

        # Center and scaling parameters (will be updated in resizeEvent)
        self.center_x = 0
        self.center_y = 0
        self.arena_pixel_radius = 1 # Placeholder
        self.scale_factor = 1.0    # Placeholder (sim units to pixels)

        # --- Timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(TIMER_INTERVAL_MS)

    def update_status_label(self):
        speed = f"x{self.speedup_factor}" if self.speedup_active else "x1"
        self.status_label.setText(f"Status: Running ({speed}) | Sim Time: {self.sim_time:.1f}")

    def resizeEvent(self, event):
        """ Handles window resizing """
        # Keep controls proportional if needed, recalculate drawing area
        self.calculate_drawing_parameters()
        super().resizeEvent(event) # Call base class implementation

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

    def toggle_speedup(self):
        """ Toggles the simulation speedup """
        self.speedup_active = self.speedup_button.isChecked()
        if self.speedup_active:
            self.speedup_button.setText(f"Normal Speed (x1)")
        else:
            self.speedup_button.setText(f"Speed Up (x{self.speedup_factor})")
        self.update_status_label() # Update label immediately

    def update_simulation(self):
        """ Performs simulation steps and schedules a repaint """
        steps_to_run = self.speedup_factor if self.speedup_active else 1

        # Ensure state is on CPU for visualization if it was moved to GPU/TPU
        # self.sim_state = jax.device_get(self.sim_state) # Usually needed if using GPU/TPU

        for _ in range(steps_to_run):
            self.key, subkey = random.split(self.key)
            try:
                # Call the JAX update function from the imported module
                self.sim_state = antsim.update_step(self.sim_state, subkey, self.sim_time, self.dt)
                self.sim_time += self.dt
            except Exception as e:
                print(f"Error during JAX simulation step: {e}")
                self.timer.stop() # Stop simulation on error
                self.status_label.setText(f"Status: Error! Check console.")
                return # Stop processing

        # After running steps, trigger a repaint
        self.update() # Calls paintEvent()
        self.update_status_label() # Update time display

    def paintEvent(self, event):
        """ Draws the simulation state """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate parameters if not done yet (e.g., first paint)
        if self.arena_pixel_radius <= 1:
            self.calculate_drawing_parameters()

        # 1. Draw Background
        painter.setBrush(COLOR_BACKGROUND)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect()) # Fill the whole widget area

        # 2. Draw Arena (Drawing area)
        painter.setBrush(COLOR_ARENA)
        arena_rect = QRectF(self.center_x - self.arena_pixel_radius,
                           self.center_y - self.arena_pixel_radius,
                           self.arena_pixel_radius * 2,
                           self.arena_pixel_radius * 2)
        painter.drawEllipse(arena_rect)

        # 3. Draw Ants
        # Get data from JAX state (ensure it's accessible, might need device_get if on GPU/TPU)
        try:
            positions = jax.device_get(self.sim_state['position'])
            angles = jax.device_get(self.sim_state['angle'])
            states = jax.device_get(self.sim_state['behavioral_state'])
        except Exception as e:
             print(f"Error getting data from JAX state: {e}")
             # Draw error message?
             painter.setPen(QColor("red"))
             painter.drawText(20, 20, "Error accessing JAX state.")
             return


        # Define the standard ant shape (triangle pointing right)
        # Coordinates relative to the ant's center (0,0)
        ant_poly = QPolygonF([
            QPointF(ANT_DRAW_LENGTH / 2, 0),
            QPointF(-ANT_DRAW_LENGTH / 2, -ANT_DRAW_WIDTH / 2),
            QPointF(-ANT_DRAW_LENGTH / 2, ANT_DRAW_WIDTH / 2)
        ])

        for i in range(antsim.NUM_ANTS):
            sim_x, sim_y = positions[i]
            angle_rad = angles[i]
            state = states[i]

            # --- Coordinate Transformation ---
            # Map simulation coordinates (origin at center, y-up)
            # to Qt coordinates (origin top-left, y-down)
            screen_x = self.center_x + sim_x * self.scale_factor
            screen_y = self.center_y - sim_y * self.scale_factor # Invert Y-axis

            # --- Set Color ---
            color = COLOR_RESTING if state == antsim.STATE_RESTING else COLOR_MOVING
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen) # No outline for ants

            # --- Draw Ant with Transformation ---
            painter.save() # Save current painter state (transformations, etc.)

            # 1. Translate to the ant's position on screen
            painter.translate(screen_x, screen_y)
            # 2. Rotate around the ant's center. Qt rotation is clockwise in degrees.
            # JAX angle is likely counter-clockwise in radians.
            painter.rotate(-jnp.rad2deg(angle_rad))

            # 3. Draw the standard ant polygon at the translated/rotated origin
            painter.drawPolygon(ant_poly)

            painter.restore() # Restore painter state for the next ant

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
    visualizer = AntSimulationVisualizer()
    visualizer.show()
    sys.exit(app.exec_())