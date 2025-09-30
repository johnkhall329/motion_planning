import pygame
import math
import sys
import random
from planner import get_motion_step
import cv2
import numpy as np

# -----------------------
# Simulation Parameters
# -----------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
ROAD_WIDTH = 200
ROAD_WIDTH_M = 7.0 # meters
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

PIXELS_PER_METER = ROAD_WIDTH / ROAD_WIDTH_M
METERS_PER_PIXEL = ROAD_WIDTH_M / ROAD_WIDTH

def ppt_to_mph(ppt: float, dt) -> float:
    '''
    Convert pixels per tick to miles per hour
    '''
    return ppt * METERS_PER_PIXEL / dt * 2.23694  # 1 m/s = 2.23694 mph

def ppt_to_kmh(ppt: float, dt) -> float:
    '''
    Convert pixels per tick to kilometers per hour
    '''
    return ppt * METERS_PER_PIXEL / dt * 3.6  # 1 m/s = 3.6 km/h

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
        # U = [steering angle (theta), acceleration (a)]
        theta, a = U
        v_dot = a  # acceleration
        self.speed += v_dot * dt
        phi_dot = self.speed * math.tan(theta) / CAR_LENGTH  # simple bicycle model
        self.heading += phi_dot * dt
        self.x_dot = self.speed * -math.sin(self.heading)
        self.y_dot = self.speed * math.cos(self.heading)
        # self.x += self.x_dot * dt
        # self.y += self.y_dot * dt
        # print(f"x: {self.x}, heading: {math.degrees(self.heading):.2f}, speed: {self.speed:.2f}")
        # print(f"x_dot: {self.x_dot:.2f}, y_dot: {self.y_dot:.2f}")
        return [self.x_dot * dt, self.y_dot * dt, phi_dot * dt]

    def draw(self, screen):
        car_surf = pygame.Surface((CAR_WIDTH, CAR_LENGTH), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.color, (0, 0, CAR_WIDTH, CAR_LENGTH))
        rotated_surf = pygame.transform.rotate(car_surf, math.degrees(self.heading))
        rotated_rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surf, rotated_rect)


# -----------------------
# Draw Lane Lines
# -----------------------
def draw_lane_lines(screen, offset_y, offset_x):
    center_x = SCREEN_WIDTH // 2 + int(offset_x)
    seg = LANE_DASH_LENGTH + LANE_GAP
    offset_y = offset_y % seg
    start_y = -seg * 3
    end_y = SCREEN_HEIGHT + seg * 3
    for y in range(start_y, end_y, seg):
        dash_y = y + offset_y
        pygame.draw.rect(
            screen,
            YELLOW,
            pygame.Rect(center_x - LANE_LINE_WIDTH // 2, int(dash_y),
                        LANE_LINE_WIDTH, LANE_DASH_LENGTH)
        )
    pygame.draw.rect(
        screen,
        WHITE,
        pygame.Rect(center_x - ROAD_WIDTH // 2,0,LANE_LINE_WIDTH, SCREEN_HEIGHT)
    )
    pygame.draw.rect(
        screen,
        WHITE,
        pygame.Rect(center_x + ROAD_WIDTH // 2 - LANE_LINE_WIDTH,0,LANE_LINE_WIDTH, SCREEN_HEIGHT)
    )

def get_bin_road(screen, scale, ego:Car):
    road_img = cv2.cvtColor(cv2.transpose(pygame.surfarray.array3d(screen)), cv2.COLOR_RGB2BGR)
    cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
    lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))
    
    cars_resize = cv2.resize(cars, (SCREEN_WIDTH//scale,SCREEN_HEIGHT//scale))
    lines_resize = cv2.resize(lines, (SCREEN_WIDTH//scale,SCREEN_HEIGHT//scale))
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cars_dilute = cv2.dilate(cars_resize,dilute_kernel,iterations=5)
    lines_dilute = cv2.dilate(lines_resize ,dilute_kernel,iterations=2)
    combined = cv2.bitwise_or(cars_dilute, lines_dilute)
    
    # eroded = cv2.erode(combined,erode_kernel)
    # dilated = cv2.dilate(eroded,dilute_kernel,iterations=3)
    blurred = cv2.GaussianBlur(combined, (45,45),0)
    road = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    center = (int(ego.y)//scale, int(ego.x)//scale)
    size = (CAR_WIDTH//scale, CAR_LENGTH//scale)
    box_points = cv2.boxPoints(cv2.RotatedRect(center,size, math.degrees(-ego.heading)))
    box_points = box_points.astype(np.int32)
    cv2.drawContours(road, [box_points], 0, RED, -1)
    return road
    



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

    lane_offset_x = 0.0
    lane_offset_y = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds per frame
        screen.fill(GRAY)

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        U = get_motion_step()  # [steering angle, acceleration]
        ego_car.update(U, dt)

        # Traffic car: straight, constant speed
        if not hasattr(other_car, 'x_dot'):
            other_car.x_dot = 0.0
        if not hasattr(other_car, 'y_dot'):
            other_car.y_dot = other_car.speed
        other_car.heading = 0.0
        other_car.x_dot = other_car.speed * -math.sin(other_car.heading)
        other_car.y_dot = other_car.speed * math.cos(other_car.heading)

        # --- Manual controls (optional override) ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            ego_car.speed += 0.1
        if keys[pygame.K_DOWN]:
            ego_car.speed = max(0.0, ego_car.speed - 0.1)
        if keys[pygame.K_LEFT]:
            ego_car.heading += math.radians(0.5)
        if keys[pygame.K_RIGHT]:
            ego_car.heading -= math.radians(0.5)

        # --- Road scrolls with ego motion ---
        lane_offset_y += ego_car.y_dot
        lane_offset_x -= ego_car.x_dot

        # --- Update traffic car in screen coords (relative to ego) ---
        dy = ego_car.y_dot - other_car.y_dot
        dx = ego_car.x_dot - other_car.x_dot
        other_car.y += dy
        other_car.x -= dx

        # --- Draw road and lane dashes (they move with ego motion) ---
        road_rect = pygame.Rect((SCREEN_WIDTH - ROAD_WIDTH) // 2 + int(lane_offset_x),
                                0, ROAD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, BLACK, road_rect)
        draw_lane_lines(screen, lane_offset_y, lane_offset_x)

        # --- Draw traffic car ---
        other_car.draw(screen)
        road_bin = get_bin_road(screen, 2, ego_car)

        # --- Draw ego car fixed on screen ---
        ego_car.draw(screen)

        cv2.imshow('binary road', road_bin)

        # HUD
        font = pygame.font.SysFont(None, 20)
        txt = font.render(f"ego_speed: {(ego_car.speed):.2f}   other_speed: {(other_car.speed):.2f}", True, WHITE)
        screen.blit(txt, (10, 10))

        pygame.display.flip()
        cv2.waitKey(1)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
