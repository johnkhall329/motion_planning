import pygame
import math
import sys
import random
from planner import get_motion_step

# -----------------------
# Simulation Parameters
# -----------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
ROAD_WIDTH = 200
CAR_WIDTH = 40
CAR_LENGTH = 60
CAR_SPEED = 5.0  # initial ego speed (pixels/frame)
LANE_LINE_WIDTH = 5
LANE_DASH_LENGTH = 40
LANE_GAP = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# -----------------------
# Car Class (screen coords)
# -----------------------
class Car:
    def __init__(self, x, y, speed=0.0, color=BLUE):
        self.x = x
        self.y = float(y)   # pixel coordinate on screen (float for smooth updates)
        self.speed = float(speed)  # absolute forward speed in "world" terms
        self.heading = 0.0  # radians, 0 = straight up the screen
        self.color = color

    def update(self, U, dt=1.0):
        # Other car moves down the screen at (ego_speed - other.speed)
        # Positive -> moves down, Negative -> moves up
        theta, a = U  # acceleration, steering angle
        x_dot = self.speed * -math.sin(self.heading)
        y_dot = self.speed * math.cos(self.heading)
        v_dot = a  # acceleration
        phi_dot = self.speed * math.tan(theta) / CAR_LENGTH  # simple bicycle model
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.speed += v_dot * dt
        self.heading += phi_dot * dt

    def draw(self, screen):
        car_rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_LENGTH)
        car_rect.center = (int(self.x), int(self.y))
        pygame.draw.rect(screen, self.color, car_rect)


# -----------------------
# Draw Lane Lines
# -----------------------
def draw_lane_lines(screen, offset):
    center_x = SCREEN_WIDTH // 2
    seg = LANE_DASH_LENGTH + LANE_GAP
    # keep offset reasonably small
    offset = offset % seg
    # draw a band larger than the screen so dashes wrap cleanly
    start_y = -seg * 3
    end_y = SCREEN_HEIGHT + seg * 3
    for y in range(start_y, end_y, seg):
        dash_y = y + offset
        pygame.draw.rect(
            screen,
            YELLOW,
            pygame.Rect(center_x - LANE_LINE_WIDTH // 2, int(dash_y), LANE_LINE_WIDTH, LANE_DASH_LENGTH)
        )


# -----------------------
# Main Simulation
# -----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Relative Motion: ego vs traffic (screen coords)")
    clock = pygame.time.Clock()

    # lane x positions
    right_lane_x = SCREEN_WIDTH // 2 + ROAD_WIDTH // 4
    left_lane_x = SCREEN_WIDTH // 2 - ROAD_WIDTH // 4


    # Ego car (fixed on screen, as a Car object)
    ego_car = Car(right_lane_x, int(SCREEN_HEIGHT * 0.75), speed=CAR_SPEED, color=BLUE)

    # Single traffic car (screen y starts up the screen)
    other_car = Car(right_lane_x, SCREEN_HEIGHT * 0.25, speed=4.0, color=RED)

    lane_offset = 0.0  # offset for lane dashes (in screen pixels)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds per frame (not strictly needed but kept for future)
        screen.fill(GRAY)

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        U = get_motion_step() # U = control input: [steering angle, acceleration]
        ego_car.update(U, dt)  # If you want to update ego_car with controls

        # --- Input: change ego speed with up/down ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            ego_car.speed += 0.1
        if keys[pygame.K_DOWN]:
            ego_car.speed = max(0.0, ego_car.speed - 0.1)

        # --- Road scrolls down at ego_car.speed (pixel units per frame) ---
        lane_offset += ego_car.speed

        # --- Update traffic car in screen coords ---
        relative_speed = ego_car.speed - other_car.speed
        other_car.y += relative_speed

        # --- Draw road and lane dashes (they move down at ego_car.speed) ---
        road_rect = pygame.Rect((SCREEN_WIDTH - ROAD_WIDTH) // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, BLACK, road_rect)
        draw_lane_lines(screen, lane_offset)

        # --- Draw the single traffic car ---
        other_car.draw(screen)

        # --- Draw ego car fixed on screen ---
        ego_car.draw(screen)

        # HUD (small readout)
        font = pygame.font.SysFont(None, 20)
        txt = font.render(f"ego_speed: {ego_car.speed:.2f}   other_speed: {other_car.speed:.2f}", True, WHITE)
        screen.blit(txt, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()