import numpy as np
import math
import cv2
from queue import PriorityQueue
import time

from Kinematics.parameters import round_node, Vehicle
from PathPlanning.unconstrained import Unconstrained
from PathPlanning.dubins import plan_dubins_path as dubins


class HybridAStar:
    def __init__(
        self,
        vehicle: Vehicle,
        resolution=10,
        heading_delta=np.pi / 20,
        turn_cost=10,
        show_arrows=True
    ):
        self.vehicle = vehicle
        self.RESOLUTION = resolution
        self.D_HEADING = heading_delta
        self.TURN_COST = turn_cost
        self.SHOW_ARROWS = show_arrows

        # Fallback turning radius if the vehicle didn't specify one
        self.TURNING_RADIUS = (
            vehicle.turning_radius
            if vehicle.turning_radius is not None
            else resolution / heading_delta
        )

    # =========================================================
    # MAP PROCESSING
    # =========================================================
    def get_bin_road(self, road_img):
        cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
        lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        cars_dilated = cv2.dilate(cars, dilate_kernel, iterations=5)
        lines_dilated = cv2.dilate(lines, dilate_kernel, iterations=2)
        combined = cv2.bitwise_or(cars_dilated, lines_dilated)

        blurred = cv2.GaussianBlur(combined, (45,45), 0)
        return cars, blurred

    # =========================================================
    # DISCRETIZATION
    # =========================================================
    def discretize(self, node):
        """Convert continuous state → grid cell index."""
        x = round(node[0] / self.RESOLUTION)
        y = round(node[1] / self.RESOLUTION)

        phi = node[2]
        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2 * np.pi

        phi_i = round(phi / self.D_HEADING)
        return (x, y, phi_i)

    # =========================================================
    # NEIGHBOR GENERATION (NO BACKWARD MOTION)
    # =========================================================
    def find_neighbors(self, node):
        x, y, phi = node

        # Straight
        dx_s = self.RESOLUTION * -math.sin(phi)
        dy_s = self.RESOLUTION * -math.cos(phi)
        straight = (x + dx_s, y + dy_s, phi)

        # Turning
        d = self.RESOLUTION  # same as original behavior
        dx_l = d * -math.sin(phi + self.D_HEADING / 2)
        dy_l = d * -math.cos(phi + self.D_HEADING / 2)
        left = (x + dx_l, y + dy_l, phi + self.D_HEADING)

        dx_r = d * -math.sin(phi - self.D_HEADING / 2)
        dy_r = d * -math.cos(phi - self.D_HEADING / 2)
        right = (x + dx_r, y + dy_r, phi - self.D_HEADING)

        return (
            (left, self.TURN_COST),
            (straight, 0),
            (right, self.TURN_COST)
        )

    # =========================================================
    # HEURISTIC (Dubins + 2D A*)
    # =========================================================
    def get_heuristic(self, curr_state, curr_disc, goal, twodastar):
        path = dubins(
            curr_state[0],
            curr_state[1],
            curr_state[2] - np.pi / 2,
            goal[0],
            goal[1],
            goal[2] - np.pi / 2,
            1 / self.TURNING_RADIUS
        )[4]
        h1 = sum(path)

        _, h2 = twodastar.get_unconstrained_path(
            (curr_disc[1], curr_disc[0]), self.RESOLUTION
        )

        return max(h1, h2)

    # =========================================================
    # COLLISION CHECK (same logic as function version)
    # =========================================================
    def check_collision(self, objects, came_from, node):
        path_img = np.zeros_like(objects)
        path = []

        curr = node
        while came_from.get(round_node(curr)) is not None:
            path.append(curr)
            cx = curr[0] + math.cos(-np.pi/2 - curr[2]) * (self.vehicle.wheel_base / 2)
            cy = curr[1] + math.sin(-np.pi/2 - curr[2]) * (self.vehicle.wheel_base / 2)

            car_rect = cv2.RotatedRect(
                (cx, cy),
                (self.vehicle.length, self.vehicle.width),
                np.rad2deg(np.pi/2 - curr[2])
            )
            pts = car_rect.points().astype(np.int32).reshape((-1, 1, 2))
            cv2.fillConvexPoly(path_img, pts, (255, 255, 255))

            curr = came_from[round_node(curr)]

        mask = cv2.bitwise_and(objects, path_img)
        return np.any(mask), path

    # =========================================================
    # MAIN SEARCH
    # =========================================================
    def plan(self, start_loc, goal_loc, screen):
        car_img, diluted_img = self.get_bin_road(screen)

        frontier = PriorityQueue()
        frontier.put((0, start_loc))

        came_from = {round_node(start_loc): None}
        cost_so_far = {self.discretize(start_loc): 0}

        goal_disc = self.discretize(goal_loc)
        twodastar = Unconstrained((goal_disc[1], goal_disc[0]), diluted_img)

        collision_itr = 0
        min_h = screen.shape[0] / 2
        collision_check_ratio = 3

        while not frontier.empty():
            _, curr_node = frontier.get()
            curr_disc = self.discretize(curr_node)

            # Reached goal discretely
            if curr_disc == goal_disc:
                collided, path = self.check_collision(car_img, came_from, curr_node)
                if not collided:
                    return path
                else:
                    continue

            # Periodic collision check (same as original)
            if collision_itr >= min_h / collision_check_ratio:
                collided, _ = self.check_collision(car_img, came_from, curr_node)
                collision_itr = 0
                if collided:
                    cost_so_far[curr_disc] = 1e9
                    continue

            # Expand neighbors
            for next_node, turn_cost in self.find_neighbors(curr_node):
                next_disc = self.discretize(next_node)
                new_cost = cost_so_far[curr_disc] + turn_cost + self.RESOLUTION
                prev_cost = cost_so_far.get(next_disc)

                if prev_cost is None or new_cost < prev_cost:
                    cost_so_far[next_disc] = new_cost
                    h = self.get_heuristic(curr_node, curr_disc, goal_loc, twodastar)
                    min_h = min(h, min_h)
                    priority = new_cost + h

                    frontier.put((priority, next_node))
                    came_from[round_node(next_node)] = curr_node

            collision_itr += 1

        raise ValueError("Hybrid A* failed to find a path")

if __name__ == '__main__':
    # Load map + screen (unchanged)
    map = cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE)
    screen = cv2.imread('screen.jpg', cv2.IMREAD_COLOR)

    start = (450.0, 450.0, 0.0)
    center = (350.0, 240.0, 0.0)
    goal = (450.0, 25.0, 0.0)

    t = time.time()

    # Car mask (unchanged)
    cars = cv2.inRange(screen, np.array([0, 0, 200]), np.array([50, 50, 255]))

    TWO_PHASES = False

    # Create planner instance
    planner = HybridAStar()

    # 2-phase mode (unchanged)
    if TWO_PHASES:
        phase1 = planner.plan(start, center, screen)
        phase2 = planner.plan(center, goal, screen)
        path = phase1 + phase2

    # 1-phase mode (unchanged)
    else:
        path = planner.plan(start, goal, screen)

    print("Time taken:", time.time() - t)

    # Path visualization parameters
    ARROW_LENGTH = 5

    # Save the path (unchanged)
    filepath = 'hybrid_astar_path.npy'
    np.save(filepath, np.array(path))
    print(f"Saved path to {filepath}")

    # Copy screen image for drawing
    color_map = screen.copy()

    for loc in path:
        x, y, phi = loc
        center_pt = (int(x), int(y))

        # Draw center point
        cv2.circle(color_map, center_pt, 3, (255, 0, 0), -1)

        if planner.SHOW_ARROWS:
            dx = -ARROW_LENGTH * math.sin(phi)
            dy = -ARROW_LENGTH * math.cos(phi)
            tip = (int(x + dx), int(y + dy))

            cv2.arrowedLine(
                color_map,
                center_pt,
                tip,
                (0, 0, 255),
                2,
                tipLength=0.4
            )

    cv2.imshow('Progress', color_map)
    while True:
        cv2.waitKey(1)
