import cv2
import numpy as np
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameters import *
from PathPlanning.hybrida_star import hybrid_a_star_path, dubins, Unconstrained
from PathPlanning.p_rrt_star import P_RRTStar

#TODO: Test for circular import
import PathPlanning.control as control # make sure control.py is in path

# ----------------------
# Global trajectory control buffer
# ----------------------
U_buffer = None
t_buffer = None
U_index = 0

def load_controls_from_csv(csv_file='traj.csv'):
    """
    Load trajectory CSV and compute U(t) using control.py
    """
    global U_buffer, t_buffer, U_index

    traj = control.load_traj_from_csv(csv_file)
    U_buffer = control.trajectory_to_controls(traj)  # (N,2)
    t_buffer = traj[:, 0]  # timestamps
    U_index = 0
    print(f"Loaded {len(U_buffer)} control steps from {csv_file}")

def load_controls_from_traj(traj):
    """
    Load precomputer U(t) into buffer
    """
    global U_buffer, t_buffer, U_index

    U_buffer = control.trajectory_to_controls(traj)  # (N,2)
    t_buffer = traj[:, 0]  # timestamps
    U_index = 0


def get_motion_step(dt_sim=0.02):
    """
    Return [theta, a] for the current simulation step.

    Uses interpolation if dt_sim != dt_traj.
    """
    global U_buffer, t_buffer, U_index

    if U_buffer is None or t_buffer is None:
        print("No controls loaded yet, returning U = (0, 0)")
        return (0,0)

    # current simulation time
    t_sim = U_index * dt_sim

    # if we exceeded trajectory time, hold last control
    if t_sim >= t_buffer[-1]:
        print("done")
        return [0, 0]

    # find surrounding indices for interpolation
    idx_next = np.searchsorted(t_buffer, t_sim, side='right')
    idx_prev = max(0, idx_next - 1)

    t0, t1 = t_buffer[idx_prev], t_buffer[idx_next]
    U0, U1 = U_buffer[idx_prev], U_buffer[idx_next]

    # linear interpolation
    if t1 - t0 < 1e-6:
        U_interp = U0
    else:
        alpha = (t_sim - t0) / (t1 - t0)
        U_interp = (1 - alpha) * U0 + alpha * U1

    U_index += 1
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