try:
    from PathPlanning.unconstrained import Unconstrained
    from PathPlanning.dubins import plan_dubins_path as dubins
except ModuleNotFoundError:
    from unconstrained import Unconstrained
    from dubins import plan_dubins_path as dubins
# from Kinematics.states import State
import cv2
import numpy as np
# from sim import TURNING_RADIUS
TURNING_RADIUS = 100

def get_motion_step():
    # Placeholder function to simulate getting motion step from planner
    # In a real scenario, this would interface with the planner module
    theta = 0.0  # No steering angle change
    a = 0.0      # No acceleration change
    return (theta, a)

def get_heuristic(curr_state, goal, two_d_astar: Unconstrained):
    path = dubins(curr_state.x, curr_state.y, curr_state.theta, goal.x, goal.y, goal.theta, 1/TURNING_RADIUS)[4]
    h1 = sum(path)
    _, h2 = len(two_d_astar.get_unconstrained_path((curr_state.x, curr_state.y)))
    return max(h1,h2)

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
    for i in range(len(upath)):
        cv2.circle(color_map,(upath[i][1],upath[i][0]),3,(0,0,255),-1)
    cv2.imshow('map+path',color_map)
    print(cost)
    # cv2.imwrite('heuristic.jpg',color_map)
    while True:
        cv2.waitKey(1)