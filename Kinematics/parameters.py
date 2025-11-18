import numpy as np

# ================================
# Simulation / Screen Parameters
# ================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# ================================
# Road Parameters
# ================================
ROAD_WIDTH = 200
ROAD_WIDTH_M = 7.0  # meters

LANE_LINE_WIDTH = 5
LANE_DASH_LENGTH = 40
LANE_GAP = 20

PIXELS_PER_METER = ROAD_WIDTH / ROAD_WIDTH_M
METERS_PER_PIXEL = ROAD_WIDTH_M / ROAD_WIDTH

# ================================
# Vehicle Parameters (Defaults)
# ================================
CAR_WIDTH = 40
CAR_LENGTH = 60
CAR_WHEELBASE = 40
CAR_SPEED = 5.0  # initial ego speed (pixels/frame)
TURNING_RADIUS = 30.0

# ================================
# Utility Classes
# ================================
class Vehicle:
    def __init__(self, length, width, wheel_base=None, turning_radius=None):
        self.length = length
        self.width = width
        self.wheel_base = wheel_base
        self.turning_radius = turning_radius

# ================================
# Utility Functions
# ================================
def round_node(node):
    """Round continuous (x, y, phi) to (int, int, rounded heading)."""
    return (round(node[0]), round(node[1]), round(node[2], 2))

car = Vehicle()