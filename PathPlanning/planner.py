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
from PathPlanning.trajectory import smooth_and_resample, parameterize_path_trapezoid

class CarState(Enum):
    IDLE = 'idle'
    EXECUTING = 'ex'

class MotionPlanner():
    def __init__(self, idle_speed, idle_heading=0.0, spacing=250.0):
        self.state = CarState.IDLE
        self.idle_speed = idle_speed
        self.idle_heading = idle_heading
        self.spacing = spacing
        
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
        # Control buffers (when following a precomputed traj)
        self.U_buffer = None
        self.t_buffer = None
        self.U_index = 0

        self.planning_thread = None
        self.planning_in_progress = False
        self.pending_traj = None
        self.pending_phase = 0
           

    def maintain(self, ego, nonego, lane_offset, dt):
        # LANE CONTROL (steering)
        if abs(ego.heading-self.idle_heading) <= 1e-2:
            self.I_heading = ego.heading-self.idle_heading
        else:
            self.I_heading+=(ego.heading-self.idle_heading)*dt
        theta = -self.K_x*lane_offset - self.K_heading*(ego.heading-self.idle_heading) + self.K_xd*ego.x_dot - self.K_heading_i*self.I_heading
        # DISTANCE
        e_d = ego.y-nonego.y

        # LONGITUDINAL CONTROL
        a_speed = self.K_v * (self.idle_speed - ego.speed)
        if e_d < self.spacing and e_d > 0:
            self.I_v_rel += (ego.speed - nonego.speed)*dt
            if (ego.speed - nonego.speed) <= 0:
                self.I_v_rel = (ego.speed - nonego.speed)*dt
                e_d = 0.99*self.spacing
            a_dist = -self.K_dist * (self.spacing-e_d) + -self.K_v_rel * (ego.speed - nonego.speed) - self.K_dist_d * (self.spacing-e_d)/dt - self.K_v_rel_i* self.I_v_rel
        elif e_d < self.spacing and e_d > 0 and (ego.speed - nonego.speed) > 0:
            a_dist = -0.5*self.K_v_rel * (ego.speed - nonego.speed)
            # else:
            #     self.I_v_rel = (ego.speed - nonego.speed)*dt
            #     a_dist = -5*self.K_v_rel * (ego.speed - nonego.speed) - self.K_v_rel_i * self.I_v_rel
        else:
            self.I_v_rel = 0.0
            a_dist = 1e9

        # SAFETY-CRITICAL CONTROL SELECTION
        a = min(a_speed, a_dist)

        # Clamp outputs
        theta = np.clip(theta, -10, 10)
        a = np.clip(a, -0.5, 0.5)
        return (theta, a)
    
    def load_controls_from_csv(self, csv_file='traj.csv'):
        """Load trajectory CSV and compute U(t) using control.py into this planner instance."""
        traj = control.load_traj_from_csv(csv_file)
        self.U_buffer = control.trajectory_to_controls(traj)  # (N,2)
        self.t_buffer = traj[:, 0]  # timestamps
        self.U_index = 0
        print(f"Loaded {len(self.U_buffer)} control steps from {csv_file}")

    def load_controls_from_traj(self, traj):
        """Load precomputed U(t) into this planner instance from a traj ndarray."""
        self.U_buffer = control.trajectory_to_controls(traj)  # (N,2)

        # simulate forward
        # U = self.U_buffer
        # states = control.simulate_trajectory(traj, U, dt=np.mean(np.gradient(traj[:, 0])))
        # xs = [s.x for s in states]
        # ys = [s.y for s in states]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(traj[:, 1], traj[:, 2], 'r--', label='Reference Path')
        # plt.plot(xs, ys, 'b-', label='Simulated Path')
        # plt.axis('equal')
        # plt.xlabel('x (m)')
        # plt.ylabel('y (m)')
        # plt.title('Trajectory Simulation with Feedforward Controls')
        # plt.legend()
        # plt.show()


        self.t_buffer = traj[:, 0]  # timestamps
        self.U_index = 0

    def prep_path_async(self, screen, center_y: float = None, ego_speed: float = None, phase: int = 0):
        """Launch path planning in a background thread.

        Parameters
        ----------
        screen : surface
            Pygame screen surface (will be copied for CV processing).
        center_y : float, optional
            If provided, the intermediate waypoint's y coordinate (pixels) to use
            when planning the two-phase hybrid A* path. If None, a default
            value is used.
        ego_speed : float, optional
            If provided, the starting speed (v0, in m/s) to use when
            parameterizing the trajectory. If None, falls back to
            `self.idle_speed`.
        """

        if self.planning_in_progress:
            print("Planner already running!")
            return

        # --- Copy screen safely in main thread ---
        surf = pygame.surfarray.array3d(screen)
        surf = np.transpose(surf, (1, 0, 2))
        screen_cv = cv2.cvtColor(surf, cv2.COLOR_RGB2BGR)

        # remember which phase is being planned (0 = both/full, 1 = first half, 2 = second half)
        self.pending_phase = int(phase)
        self.planning_in_progress = True

        def worker():
            print("Planning started in background thread...")
            stime = time.time()

            # determine waypoints; keep x fixed, allow center.y to be set
            start = (450.0, 450.0, 0.0)
            center = (325.0, 250, 0.0)
            goal = (475.0, 25.0, 0.0)

            # Plan depending on requested phase
            if self.pending_phase == 1:
                # only plan first half: start -> center
                phase1 = hybrid_a_star_path(start, center, screen_cv)
                path = phase1
                # parameterize: start speed -> target vf = 6
                v0 = float(ego_speed) if ego_speed is not None else float(self.idle_speed)
                traj, times = parameterize_path_trapezoid(
                    smooth_and_resample(path, spacing_m=0.1),
                    v0=v0,
                    vf=5.0,
                    v_max=6.0,
                    a_max=0.5,
                    dt=0.02
                )
            elif self.pending_phase == 2:
                # only plan second half: center -> goal
                # Use the center point as the nominal start for phase2
                phase1_tmp = hybrid_a_star_path(start, center, screen_cv)
                phase2 = hybrid_a_star_path(phase1_tmp[-1], goal, screen_cv)
                path = phase2
                v0 = float(ego_speed) if ego_speed is not None else float(self.idle_speed)
                traj, times = parameterize_path_trapezoid(
                    smooth_and_resample(path, spacing_m=0.1),
                    v0=v0,
                    vf=5.0,
                    v_max=6.0,
                    a_max=0.5,
                    dt=0.02
                )
            else:
                # default: plan full two-phase path
                phase1 = hybrid_a_star_path(start, center, screen_cv)
                phase2 = hybrid_a_star_path(phase1[-1], goal, screen_cv)
                path = phase1 + phase2
                resampled = smooth_and_resample(path, spacing_m=0.1)
                v0 = float(ego_speed) if ego_speed is not None else float(self.idle_speed)
                traj, times = parameterize_path_trapezoid(
                    resampled,
                    v0=v0,
                    vf=4,
                    v_max=6.0,
                    a_max=0.5,
                    dt=0.02
                )

            self.pending_traj = traj
            # planning finished
            self.planning_in_progress = False
            print("Planning finished in:", time.time() - stime)

        self.planning_thread = threading.Thread(target=worker, daemon=True)
        self.planning_thread.start()


    def overtake_step(self, dt_sim=0.02):
        """
        Return [theta, a] for the current simulation step by interpolating the precomputed control buffer.
        This replaces the previous module-level get_motion_step function.
        """
        if self.U_buffer is None or self.t_buffer is None:
            print("No controls loaded yet, returning U = (0, 0)")
            return (0, 0)

        # current simulation time
        t_sim = self.U_index * dt_sim

        # if we exceeded trajectory time, hold last control
        if t_sim >= self.t_buffer[-1]:
            print("done")
            return (-1, -1)

        # find surrounding indices for interpolation
        idx_next = np.searchsorted(self.t_buffer, t_sim, side='right')
        idx_prev = max(0, idx_next - 1)

        t0, t1 = self.t_buffer[idx_prev], self.t_buffer[idx_next]
        U0, U1 = self.U_buffer[idx_prev], self.U_buffer[idx_next]

        # linear interpolation
        if t1 - t0 < 1e-6:
            U_interp = U0
        else:
            alpha = (t_sim - t0) / (t1 - t0)
            U_interp = (1 - alpha) * U0 + alpha * U1

        self.U_index += 1
        return U_interp.tolist()


if __name__ == '__main__':
    map = cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE)
    center = (350,300)
    goal = (450,77)
    dx,dy, _, _, path = dubins(float(center[0]), float(center[1]), np.deg2rad(-90.0), float(goal[0]), float(goal[1]), np.deg2rad(-90.0), 1/TURNING_RADIUS)
    print(sum(path))
    twodastar = Unconstrained((goal[1],goal[0]),map)
    upath,cost = twodastar.get_unconstrained_path((center[1],center[0]),step_size=5)
    color_map = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
    for i in range(len(dx)):
        cv2.circle(color_map,(int(dx[i]),int(dy[i])),3,(0,255,0),-1)
    # for i in range(len(upath)):
    #     cv2.circle(color_map,(upath[i][1],upath[i][0]),3,(0,0,255),-1)
    cv2.imshow('map+path',color_map)
    print(cost)
    # cv2.imwrite('heuristic.jpg',color_map)
    while True:
        cv2.waitKey(1)