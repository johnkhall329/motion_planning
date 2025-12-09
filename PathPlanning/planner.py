import cv2
import numpy as np
import sys
import os
from enum import Enum
import pygame
import time
import threading

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameters import *
from PathPlanning.hybrida_star import hybrid_a_star_path, dubins, Unconstrained
from PathPlanning.p_rrt_star import P_RRTStar
import PathPlanning.control as control
from PathPlanning.trajectory import smooth_and_resample, parameterize_path_trapezoid, sanitize_rrt_path


class CarState(Enum):
    IDLE = 'idle'
    EXECUTING = 'ex'


class MotionPlanner():
    def __init__(self, idle_speed, idle_heading=0.0, spacing=250.0, left_lane_speed=5.0):
        self.state = CarState.IDLE
        self.idle_speed = idle_speed
        self.idle_heading = idle_heading
        self.spacing = spacing
        # Parameterized left-lane (holding / final) speed
        self.left_lane_speed = left_lane_speed

        self.planner = "RRT" # A* or RRT
        
        self.K_x = 0.025
        self.K_xd = 0.1
        self.K_heading = 0.025
        self.K_heading_i = 5.0
        self.I_heading = 0.0
        
        self.K_v = 0.25
        self.K_v_rel = 0.1
        self.K_v_rel_i = 0.001
        self.K_dist = 0.005
        self.K_dist_d = 0.000001
        
        self.I_v_rel = 0.0

        # Control buffers
        self.U_buffer = None
        self.t_buffer = None
        self.U_index = 0

        # Async planner state
        self.planning_thread = None
        self.planning_in_progress = False
        self.pending_traj = None
        self.pending_phase = 0

        # Lane mode
        self.in_left_lane = False

    # ------------------------
    # MAINTAIN CONTROLLER
    # ------------------------
    def maintain(self, ego, nonego, lane_offset, dt):

        # LEFT LANE MODE
        if self.in_left_lane:
            if abs(ego.heading - self.idle_heading) <= 1e-2:
                self.I_heading = ego.heading - self.idle_heading
            else:
                self.I_heading += (ego.heading - self.idle_heading) * dt

            theta = (
                -self.K_x * (lane_offset - 100)
                - self.K_heading * (ego.heading - self.idle_heading)
                + self.K_xd * ego.x_dot
                - self.K_heading_i * self.I_heading
            )

            target_speed_left = self.left_lane_speed
            a = self.K_v * (target_speed_left - ego.speed)

            theta = np.clip(theta, -10, 10)
            a = np.clip(a, -0.5, 0.5)
            return (theta, a)

        # NORMAL FOLLOW MODE
        if abs(ego.heading - self.idle_heading) <= 1e-2:
            self.I_heading = ego.heading - self.idle_heading
        else:
            self.I_heading += (ego.heading - self.idle_heading) * dt

        theta = (
            -self.K_x * lane_offset
            - self.K_heading * (ego.heading - self.idle_heading)
            + self.K_xd * ego.x_dot
            - self.K_heading_i * self.I_heading
        )

        e_d = ego.y - nonego.y

        a_speed = self.K_v * (self.idle_speed - ego.speed)

        if e_d < self.spacing and e_d > 0:
            self.I_v_rel += (ego.speed - nonego.speed) * dt

            if (ego.speed - nonego.speed) <= 0:
                self.I_v_rel = (ego.speed - nonego.speed) * dt
                e_d = 0.99 * self.spacing

            a_dist = (
                -self.K_dist * (self.spacing - e_d)
                - self.K_v_rel * (ego.speed - nonego.speed)
                - self.K_dist_d * (self.spacing - e_d) / dt
                - self.K_v_rel_i * self.I_v_rel
            )

        elif e_d < self.spacing and e_d > 0 and (ego.speed - nonego.speed) > 0:
            a_dist = -0.5 * self.K_v_rel * (ego.speed - nonego.speed)
        else:
            self.I_v_rel = 0.0
            a_dist = 1e9

        a = min(a_speed, a_dist)

        theta = np.clip(theta, -10, 10)
        a = np.clip(a, -0.5, 0.5)

        return (theta, a)

    # ------------------------
    # LOAD CONTROLS
    # ------------------------
    def load_controls_from_traj(self, traj):
        self.U_buffer = control.trajectory_to_controls(traj)
        self.t_buffer = traj[:, 0]
        self.U_index = 0

    # ------------------------
    # ASYNC PLANNER (THREAD SAFE)
    # ------------------------
    def prep_path_async(self, screen, center_y=None, ego_speed=None, phase=0):

        if self.planning_in_progress:
            print("Planner already running!")
            return

        surf = pygame.surfarray.array3d(screen)
        surf = np.transpose(surf, (1, 0, 2))
        screen_cv = cv2.cvtColor(surf, cv2.COLOR_RGB2BGR)

        phase_local = int(phase)
        self.planning_in_progress = True

        def worker():
            print(f"Planning started for phase {phase_local}...")
            stime = time.time()

            start = (450.0, 450.0, 0.0)
            if self.planner == "A*":
                center = (325.0, 250.0, 0.0)
            else:
                center = (335.0, 250.0, 0.0)
            goal = (450.0, 25.0, 0.0)
            d2 = np.array(start) + np.array(goal) - np.array(center)

            v0 = float(ego_speed) if ego_speed is not None else float(self.idle_speed)

            if self.planner == "RRT":

                if phase_local == 1:
                    rrt_planner = P_RRTStar(screen_cv, start, center[:2], TURNING_RADIUS//2, RESOLUTION, 1e5)

                elif phase_local == 2:
                    rrt_planner = P_RRTStar(screen_cv, start, d2[:2], TURNING_RADIUS//2, RESOLUTION, 1e5)
                
                else: raise Exception("How could this have even happened")

                baseline, path = rrt_planner.p_rrt_star()
                path = sanitize_rrt_path(path)

            elif self.planner == "A*":
                if phase_local == 1:
                    path = hybrid_a_star_path(start, center, screen_cv)

                elif phase_local == 2:
                    path = hybrid_a_star_path(start, d2, screen_cv)

                else:
                    p1 = hybrid_a_star_path(start, center, screen_cv)
                    p2 = hybrid_a_star_path(p1[-1], goal, screen_cv)
                    path = p1 + p2
            
            else:
                raise ValueError("MotionPlanner.planner must be 'A*' or 'RRT'")

            resampled = smooth_and_resample(path, spacing_m=0.1)
            
            traj, times = parameterize_path_trapezoid(
                resampled,
                v0=v0,
                vf=self.left_lane_speed,
                v_max=6.0,
                a_max=0.5,
                dt=0.02
            )

            if phase_local == 1:
                np.save("data/pathR1.npy", path)
                np.save("data/resampleR1.npy", resampled)
                np.save("data/trajR1.npy", traj)
            elif phase_local == 2:
                np.save("data/pathR2.npy", path)
                np.save("data/resampleR2.npy", resampled)
                np.save("data/trajR2.npy", traj)

            # Doesn't work on worker thread
            # quick_visual_check(path, resampled, traj)

            self.pending_traj = traj
            self.pending_phase = phase_local
            self.planning_in_progress = False

            print("Planning finished in:", time.time() - stime)

        self.planning_thread = threading.Thread(target=worker, daemon=True)
        self.planning_thread.start()

    # ------------------------
    # OVERTAKE EXECUTION
    # ------------------------
    def overtake_step(self, dt_sim=0.02):

        if self.U_buffer is None or self.t_buffer is None:
            return (0, 0)

        t_sim = self.U_index * dt_sim

        if t_sim >= self.t_buffer[-1]:
            return (-1, -1)

        idx_next = np.searchsorted(self.t_buffer, t_sim, side='right')

        if idx_next >= len(self.t_buffer):
            return (-1, -1)

        idx_prev = max(0, idx_next - 1)

        t0, t1 = self.t_buffer[idx_prev], self.t_buffer[idx_next]
        U0, U1 = self.U_buffer[idx_prev], self.U_buffer[idx_next]

        if t1 - t0 < 1e-6:
            U_interp = U0
        else:
            alpha = (t_sim - t0) / (t1 - t0)
            U_interp = (1 - alpha) * U0 + alpha * U1

        self.U_index += 1
        return U_interp.tolist()
