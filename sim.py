import pygame
import math
import sys
import random
import cv2
import numpy as np
import os
import time

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

from parameters import *
from PathPlanning.planner import MotionPlanner

FIXED_DT = 0.02   # ✅ CONTROL TIMESTEP (50 Hz)


# -----------------------
# Car Class (screen coords)
# -----------------------
class Car:
    def __init__(self, x, y, speed=0.0, color=BLUE):
        self.x = x
        self.y = float(y)
        self.speed = float(speed)
        self.heading = 0.0
        self.color = color
        self.x_dot = 0.0
        self.y_dot = 0.0

    def update(self, U, dt):
        theta, a = U

        self.speed += a * dt
        phi_dot = self.speed * math.tan(theta) / CAR_LENGTH
        self.heading += phi_dot * dt

        self.x_dot = self.speed * -math.sin(self.heading)
        self.y_dot = self.speed * math.cos(self.heading)

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
            pygame.Rect(center_x - LANE_LINE_WIDTH // 2,
                        int(dash_y),
                        LANE_LINE_WIDTH,
                        LANE_DASH_LENGTH)
        )

    pygame.draw.rect(
        screen, WHITE,
        pygame.Rect(center_x - ROAD_WIDTH // 2, 0,
                    LANE_LINE_WIDTH, SCREEN_HEIGHT)
    )

    pygame.draw.rect(
        screen, WHITE,
        pygame.Rect(center_x + ROAD_WIDTH // 2 - LANE_LINE_WIDTH, 0,
                    LANE_LINE_WIDTH, SCREEN_HEIGHT)
    )


# -----------------------
# Main Simulation
# -----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Relative Motion: ego vs traffic")
    clock = pygame.time.Clock()

    # lane x positions
    right_lane_x = SCREEN_WIDTH // 2 + ROAD_WIDTH // 4

    ego_car = Car(right_lane_x, int(SCREEN_HEIGHT * 0.75),
                  speed=CAR_SPEED, color=BLUE)

    other_car = Car(right_lane_x, SCREEN_HEIGHT * 0.25,
                    speed=4.0, color=RED)

    lane_offset_x = 0.0
    lane_offset_y = 0.0

    overtaking = False
    planner_state = 'start'
    overtaking_phase = 0

    planner = MotionPlanner(5.0, 0.0)

    accumulator = 0.0
    last_time = time.time()

    running = True
    while running:

        # -------------------------
        # REAL TIME TRACKING
        # -------------------------
        now = time.time()
        frame_dt = now - last_time
        last_time = now
        accumulator += frame_dt

        # -------------------------
        # EVENTS
        # -------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # -------------------------
        # APPLY FINISHED PLANS
        # -------------------------
        if planner.pending_traj is not None:
            planner.load_controls_from_traj(planner.pending_traj)
            overtaking_phase = planner.pending_phase
            planner.pending_traj = None
            overtaking = True

            if overtaking_phase == 1:
                planner_state = 'executing1'
            elif overtaking_phase == 2:
                planner_state = 'executing2'
            else:
                planner_state = 'executing'

        # -------------------------
        # FIXED-TIMESTEP CONTROL
        # -------------------------
        while accumulator >= FIXED_DT:

            if not overtaking:
                U = planner.maintain(
                    ego_car, other_car, lane_offset_x, FIXED_DT)
            else:
                U = planner.overtake_step()
                if U == (-1, -1):
                    overtaking = False

                    if overtaking_phase == 1:
                        planner_state = 'passing'
                        planner.idle_speed = 5.0
                        planner.in_left_lane = True

                    elif overtaking_phase == 2:
                        planner_state = 'passed'
                        planner.idle_speed = 4.0
                        planner.in_left_lane = False

                    U = (0, 0)

            ego_car.update(U, FIXED_DT)

            # traffic car update (relative world)
            other_car.heading = 0.0
            other_car.x_dot = other_car.speed * -math.sin(other_car.heading)
            other_car.y_dot = other_car.speed * math.cos(other_car.heading)

            lane_offset_y += ego_car.y_dot
            lane_offset_x -= ego_car.x_dot

            dy = ego_car.y_dot - other_car.y_dot
            dx = ego_car.x_dot - other_car.x_dot
            other_car.y += dy
            other_car.x -= dx

            accumulator -= FIXED_DT

        # -------------------------
        # MANUAL INPUT
        # -------------------------
        if keys[pygame.K_UP]:
            ego_car.speed += 0.1
        if keys[pygame.K_DOWN]:
            ego_car.speed = max(0.0, ego_car.speed - 0.1)
        if keys[pygame.K_LEFT]:
            ego_car.heading += math.radians(0.5)
        if keys[pygame.K_RIGHT]:
            ego_car.heading -= math.radians(0.5)

        # -------------------------
        # PHASE TRIGGERS
        # -------------------------
        if keys[pygame.K_a]:
            if planner_state == 'start':
                planner.prep_path_async(
                    screen,
                    center_y=other_car.y,
                    ego_speed=ego_car.speed,
                    phase=1
                )
                planner_state = 'planning1'

        if keys[pygame.K_d]:
            if planner_state == 'passing':
                planner.prep_path_async(
                    screen,
                    center_y=other_car.y,
                    ego_speed=ego_car.speed,
                    phase=2
                )
                planner_state = 'planning2'

        # -------------------------
        # DRAW
        # -------------------------
        screen.fill(GRAY)

        road_rect = pygame.Rect(
            (SCREEN_WIDTH - ROAD_WIDTH) // 2 + int(lane_offset_x),
            0,
            ROAD_WIDTH,
            SCREEN_HEIGHT
        )
        pygame.draw.rect(screen, BLACK, road_rect)
        draw_lane_lines(screen, lane_offset_y, lane_offset_x)

        other_car.draw(screen)
        ego_car.draw(screen)

        font = pygame.font.SysFont(None, 20)
        txt = font.render(
            f"ego_speed: {ego_car.speed:.2f}   other_speed: {other_car.speed:.2f}",
            True, WHITE
        )
        screen.blit(txt, (10, 10))

        if planner.planning_in_progress:
            plan_txt = font.render("PLANNING...", True, YELLOW)
            screen.blit(plan_txt, (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
