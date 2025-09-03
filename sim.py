import pygame
import sys
import math

# -----------------------
# Simulation Parameters
# -----------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
ROAD_WIDTH = 200
CAR_WIDTH = 40
CAR_LENGTH = 60
CAR_SPEED = 5  # initial speed of main car
ROAD_LENGTH = 100000  # very long road
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
# Car Class
# -----------------------
class Car:
    def __init__(self, x, y, speed=CAR_SPEED, color=BLUE):
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color

    def update(self):
        # Move car upwards along y-axis
        self.y -= self.speed

    def draw(self, screen, camera_offset_y):
        car_rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_LENGTH)
        car_rect.center = (self.x, self.y - camera_offset_y)
        pygame.draw.rect(screen, self.color, car_rect)

# -----------------------
# Draw Lane Lines
# -----------------------
def draw_lane_lines(screen, camera_offset_y):
    center_x = SCREEN_WIDTH // 2
    num_dashes = SCREEN_HEIGHT // (LANE_DASH_LENGTH + LANE_GAP) + 2
    for i in range(num_dashes):
        dash_y = i * (LANE_DASH_LENGTH + LANE_GAP) - (camera_offset_y % (LANE_DASH_LENGTH + LANE_GAP))
        pygame.draw.rect(
            screen,
            YELLOW,
            pygame.Rect(center_x - LANE_LINE_WIDTH // 2, dash_y, LANE_LINE_WIDTH, LANE_DASH_LENGTH)
        )

# -----------------------
# Main Simulation
# -----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Road Simulation")
    clock = pygame.time.Clock()

    # Cars stay on the right half of the road
    right_lane_x = SCREEN_WIDTH // 2 + ROAD_WIDTH // 4

    # Initialize cars
    main_car = Car(right_lane_x, ROAD_LENGTH - 500, speed=CAR_SPEED, color=BLUE)
    other_car = Car(right_lane_x, ROAD_LENGTH - 1000, speed=4, color=RED)

    running = True
    camera_offset_y = 0

    while running:
        screen.fill(GRAY)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard input to adjust speed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            main_car.speed += 0.1
        if keys[pygame.K_DOWN]:
            main_car.speed = max(0, main_car.speed - 0.1)

        # Update cars
        main_car.update()
        other_car.update()

        # Update camera to follow main car
        camera_offset_y = main_car.y - SCREEN_HEIGHT * 0.75

        # Draw road
        road_rect = pygame.Rect(
            (SCREEN_WIDTH - ROAD_WIDTH) // 2,
            -camera_offset_y,
            ROAD_WIDTH,
            ROAD_LENGTH
        )
        pygame.draw.rect(screen, BLACK, road_rect)

        # Draw lane line
        draw_lane_lines(screen, camera_offset_y)

        # Draw cars
        other_car.draw(screen, camera_offset_y)
        main_car.draw(screen, camera_offset_y)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
