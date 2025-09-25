import pygame
import math
import sys

# -----------------------
# Parameters
# -----------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
CAR_WIDTH = 40
CAR_LENGTH = 60

# Colors
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

GOAL_STATE = (200, 150, 0.0, math.radians(90))  # x, y, speed, heading

# -----------------------
# Car Class
# -----------------------
class Car:
    def __init__(self, x, y, speed=0.0, color=BLUE):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.heading = 0.0  # radians, 0 = up the screen
        self.color = color

    def state_update(self, U, dt=1.0, perform_update=True):
        # U = [steering angle (theta), acceleration (a)]
        theta, a = U
        v_dot = a  # acceleration
        speed = self.speed + v_dot * dt
        phi_dot = speed * math.tan(theta) / CAR_LENGTH  # simple bicycle model
        heading = self.heading + phi_dot * dt
        x_dot = speed * -math.sin(self.heading)
        y_dot = -speed * math.cos(self.heading)
        x = self.x + x_dot * dt
        y = self.y + y_dot * dt

        # print(f"x: {self.x}, heading: {math.degrees(self.heading):.2f}, speed: {self.speed:.2f}")
        # print(f"x_dot: {self.x_dot:.2f}, y_dot: {self.y_dot:.2f}")

        if perform_update:
            self.speed = speed
            self.heading = heading
            self.x_dot = x_dot
            self.y_dot = y_dot
            self.x = x
            self.y = y

        return (x, y, speed, heading)

    def get_state(self):
        return (self.x, self.y, self.speed, self.heading)

    def draw(self, screen):
        car_surf = pygame.Surface((CAR_WIDTH, CAR_LENGTH), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.color, (0, 0, CAR_WIDTH, CAR_LENGTH))  # rectangle points DOWN by default, so add 180° so heading=0 points UP
        rotated = pygame.transform.rotate(car_surf, math.degrees(self.heading) + 180)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect)

# -----------------------
# Main
# -----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simple Car - corrected")
    clock = pygame.time.Clock()
    car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.75, speed=0.0)
    goal_car = Car(200, 150, speed=0.0, color=GREEN)
    goal_car.heading = math.radians(90)
    font = pygame.font.SysFont(None, 20)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        screen.fill(GRAY)

        print("State:", car.get_state())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # determine optimal U at this time step based on
        # 1. L2 norm current state vs goal state
        # 2. cost with steering and acceleration (prefer less)
        # 3. Euclidean distance so far

        U_opt = [0.0, 0.0]  # default placeholder to no steering, no acceleration
        car.state_update(U_opt, dt)

        car.draw(screen)
        goal_car.draw(screen)

        txt = font.render(f"speed: {car.speed:.1f} x: {car.x:.1f} y: {car.y:.1f}", True, WHITE)
        screen.blit(txt, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
