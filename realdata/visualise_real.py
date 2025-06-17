import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen
import math

# --- Visualisation Parameters ---
WINDOW_SIZE = 800
ARENA_PADDING = 20  # Pixel padding around the arena
VISUAL_ANT_MULTIPLIER = 1.0
TIMER_INTERVAL_MS = 16  # ~60 FPS

COLOUR_BACKGROUND = QColor(245, 245, 245)
COLOUR_ARENA = QColor(230, 220, 240)
COLOUR_MOVING = QColor(0, 0, 200)     # Blue

class RealAntVisualiser(QWidget):
    def __init__(self, data, ant_length=20.0, arena_radius=500.0):
        super().__init__()
        self.setWindowTitle("Ant Real Data Visualiser")
        self.setGeometry(100, 100, WINDOW_SIZE, WINDOW_SIZE)
        self.data = data
        self.num_ants = data.shape[1] // 2
        self.ant_length = ant_length
        self.ant_width = ant_length / 2.0
        self.arena_radius = arena_radius
        self.current_frame = 0
        self.max_frame = data.shape[0]
        self.scale_factor = (WINDOW_SIZE / 2 - ARENA_PADDING) / self.arena_radius

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(TIMER_INTERVAL_MS)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.max_frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.setBrush(COLOUR_BACKGROUND)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # Draw arena
        center_x = WINDOW_SIZE / 2
        center_y = WINDOW_SIZE / 2
        arena_pixel_radius = self.arena_radius * self.scale_factor
        painter.setBrush(COLOUR_ARENA)
        painter.drawEllipse(QRectF(center_x - arena_pixel_radius, center_y - arena_pixel_radius,
                                   arena_pixel_radius * 2, arena_pixel_radius * 2))

        # Draw ants
        frame = self.data.iloc[self.current_frame]
        for i in range(self.num_ants):
            x = frame[i * 2]
            y = frame[i * 2 + 1]
            # Compute angle if possible (from diff)
            if self.current_frame > 0:
                prev_x = self.data.iloc[self.current_frame - 1, i * 2]
                prev_y = self.data.iloc[self.current_frame - 1, i * 2 + 1]
                dx = x - prev_x
                dy = y - prev_y
                angle = math.atan2(dy, dx)
            else:
                angle = 0.0

            # Convert to screen coordinates
            screen_x = center_x + x * self.scale_factor
            screen_y = center_y - y * self.scale_factor  # Invert y for screen

            # Ant polygon
            pixel_length = self.ant_length * self.scale_factor * VISUAL_ANT_MULTIPLIER
            pixel_width = self.ant_width * self.scale_factor * VISUAL_ANT_MULTIPLIER
            min_pixel_size = 2.0
            pixel_length = max(min_pixel_size, pixel_length)
            pixel_width = max(min_pixel_size * (self.ant_width / self.ant_length), pixel_width)

            ant_poly = QPolygonF([
                QPointF(pixel_length / 2.0, 0),
                QPointF(-pixel_length / 2.0, -pixel_width / 2.0),
                QPointF(-pixel_length / 2.0, pixel_width / 2.0)
            ])

            painter.setBrush(QBrush(COLOUR_MOVING))
            painter.setPen(Qt.NoPen)
            painter.save()
            painter.translate(screen_x, screen_y)
            painter.rotate(-math.degrees(angle))
            painter.drawPolygon(ant_poly)
            painter.restore()

        painter.end()

def main():
    # Example: load a maruyamadata .pkl file (edit path as needed)
    data = pd.read_pickle("ant-info/maruyama_data/KC048_10cm_5h_20230610_1h-2h.pkl")
    # Optionally, scale/center data here if needed to match simulation arena

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    visualiser = RealAntVisualiser(data)
    visualiser.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
