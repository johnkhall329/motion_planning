from unconstrained import Unconstrained
from dubins import plan_dubins_path as dubins
from Kinematics.states import State
from sim import TURNING_RADIUS

def get_motion_step():
    # Placeholder function to simulate getting motion step from planner
    # In a real scenario, this would interface with the planner module
    theta = 0.0  # No steering angle change
    a = 0.0      # No acceleration change
    return (theta, a)

def get_heuristic(curr_state:State, goal:State, two_d_astar: Unconstrained):
    _,_,_,_,h1 = len(dubins(curr_state.x, curr_state.y, curr_state.theta, goal.x, goal.y, goal.theta, TURNING_RADIUS))
    h2 = abs(curr_state.v - goal.v)
    h3 = len(two_d_astar.get_unconstrained_path((curr_state.x, curr_state.y)))
    return max(h1,h2,h3)