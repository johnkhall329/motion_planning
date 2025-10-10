import numpy as np
from queue import PriorityQueue
import math 
import cv2
try:
    from PathPlanning.unconstrained import Unconstrained
    from PathPlanning.dubins import plan_dubins_path as dubins
except ModuleNotFoundError:
    from unconstrained import Unconstrained
    from dubins import plan_dubins_path as dubins

D_HEADING = np.pi/12
RESOLUTION = 10
TURNING_RADIUS = RESOLUTION/D_HEADING
TURN_COST = 5

# a node is a continous (x,y,phi) tuple
# phi is the heading angle in radians 
# x and y are in pixels, with (0,0) as top left corner

def discritize(node, resolution=10, turning_a=D_HEADING):
    """
    Sorts a node into a grid based on resolution and turning angle
    resolution: size of each grid square
    turning_a: angle increment for heading angle

    returns as index of cell location and turning increment
    e.g. (12,9,105 degrees) with resolution 10 and turning_a 15 degrees
    returns (1,0,7)
    cell is 1 in x direction (right), 0 in y direction (down),
    and 7 increments of 15 degrees

    """
    x = node[0]//resolution
    y = node[1]//resolution
    phi = node[2]
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    phi = int(phi/turning_a)
    return (x,y,phi)

def find_neighbors(node, distance=RESOLUTION, turning_a=D_HEADING):
    x,y,phi = node

    # straight
    dx = distance * -math.sin(phi)
    dy = distance * -math.cos(phi)
    straight = (x+dx,y+dy,phi)

    # consts for turning
    r = distance/turning_a
    d = 2 * r * math.sin(turning_a/2)

    # left
    dx = d * -math.sin(phi + turning_a/2)
    dy = d * -math.cos(phi + turning_a/2)
    left = (x+dx,y+dy,phi+turning_a)

    # right
    dx = d * -math.sin(phi - turning_a/2)
    dy = d * -math.cos(phi - turning_a/2)
    right = (x+dx,y+dy,phi-turning_a)


    return (left, straight, right)

def get_heuristic(curr_state, curr_discritized, goal, two_d_astar: Unconstrained, step_size=RESOLUTION):
    path = dubins(curr_state[0], curr_state[1], curr_state[2], goal[0], goal[1], goal[2], 1/TURNING_RADIUS,step_size)[4]
    h1 = sum(path)
    u_path,h2 = two_d_astar.get_unconstrained_path((curr_discritized[1], curr_discritized[0]),step_size)
    return max(h1,h2)

def hybrid_a_star_path(start_loc, goal_loc, map):
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    came_from[discritize(start_loc)] = None
    cost_so_far[discritize(start_loc)] = 0
    frontier.put((0,start_loc))
    goal_discretized = discritize(goal_loc)
    twodastar = Unconstrained((goal_discretized[1],goal_discretized[0]),map)
    while not frontier.empty():
        item = frontier.get()
        curr_node = item[1]
        curr_discritized = discritize(curr_node)

        if curr_discritized == goal_discretized: #will need to do correct goal checking
            return format_path(came_from,curr_node)
        
        for i,next_node in enumerate(find_neighbors(curr_node)):
            new_cost = cost_so_far[curr_discritized] + TURN_COST + RESOLUTION if i%2 == 0 else cost_so_far[curr_discritized] + RESOLUTION # additional costs of moving + turning n shi
            next_discritized = discritize(next_node)
            prev_cost = cost_so_far.get(next_discritized)       
            
            if prev_cost is None or new_cost < prev_cost:    
                cost_so_far[next_discritized] = new_cost
                heuristic = get_heuristic(curr_node,curr_discritized, goal_loc,twodastar)  
                priority = new_cost + heuristic
                frontier.put((priority,next_node))
                came_from[next_discritized] = curr_node
            
    raise ValueError("Unable to find path") 
        
def format_path(came_from, node):
    path = []
    while came_from[node] is not None: # appends nodes in path with goal as beginning
        path.append(node)
        node = came_from[node]
    return path

if __name__ == '__main__':
    map = cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE)
    center = (350,300,0)
    goal = (450,77,0)
    
    path = hybrid_a_star_path(center,goal,map)