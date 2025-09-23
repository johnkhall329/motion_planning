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

    def update(self, steering, acceleration, dt):
        # integrate speed
        self.speed += acceleration * dt

        # simple damping so it doesn't runaway
        self.speed *= 0.995

        # bicycle-ish heading update (depends on speed)
        self.heading += (self.speed / CAR_LENGTH) * math.tan(steering) * dt

        # world update (Pygame y increases downward)
        dx = self.speed * -math.sin(self.heading) * dt
        dy = -self.speed * math.cos(self.heading) * dt   # <-- note the negative sign
        self.x += dx
        self.y += dy

    def draw(self, screen):
        car_surf = pygame.Surface((CAR_WIDTH, CAR_LENGTH), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.color, (0, 0, CAR_WIDTH, CAR_LENGTH))
        # rectangle points DOWN by default, so add 180° so heading=0 points UP
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

    font = pygame.font.SysFont(None, 20)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        screen.fill(GRAY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        steering = 0.0
        acceleration = 0.0

        if keys[pygame.K_UP]:
            acceleration = 200.0   # pixels/s^2 forward
        if keys[pygame.K_DOWN]:
            acceleration = -200.0  # brake / reverse
        if keys[pygame.K_LEFT]:
            steering = math.radians(30)
        if keys[pygame.K_RIGHT]:
            steering = -math.radians(30)

        car.update(steering, acceleration, dt)
        car.draw(screen)

        txt = font.render(f"speed: {car.speed:.1f}  x: {car.x:.1f}  y: {car.y:.1f}", True, WHITE)
        screen.blit(txt, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
