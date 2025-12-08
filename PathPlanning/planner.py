import cv2
import numpy as np
import sys
import os
from enum import Enum

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameters import *
from PathPlanning.hybrida_star import hybrid_a_star_path
from PathPlanning.p_rrt_star import P_RRTStar

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

def get_motion_step():
    # Placeholder function to simulate getting motion step from planner
    # In a real scenario, this would interface with the planner module
    theta = 0.0  # No steering angle change
    a = 0.0      # No acceleration change
    return (theta, a)


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