import numpy as np
from queue import PriorityQueue
import math
try:
    from PathPlanning.unconstrained import Unconstrained
    from PathPlanning.dubins import plan_dubins_path as dubins
except ModuleNotFoundError:
    from unconstrained import Unconstrained
    from dubins import plan_dubins_path as dubins

D_HEADING = np.pi/12
RESOLUTION = 10
TURNING_RADIUS = RESOLUTION/D_HEADING

# a node is a continous (x,y,phi) tuple
# phi is the heading angle in radians 
# x and y are in pixels, with (0,0) as top left corner

def discritize(node, resolution=10, turning_a=np.pi/12):
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
    phi = (phi//turning_a)
    return (x,y,phi)

def find_neighbors(node):
    # x,y,phi = node
    # return [left, straight, right]
    pass

def get_heuristic(curr_state, goal, two_d_astar: Unconstrained):
    path = dubins(curr_state[0], curr_state[1], curr_state[2], goal[0], goal[1], goal[2], 1/TURNING_RADIUS)[4]
    h1 = sum(path)
    try:
        _, h2 = len(two_d_astar.get_unconstrained_path((curr_state.x, curr_state.y)))
    except ValueError:
        h2 = 1e9
    return max(h1,h2)

def hybrid_a_star_path(start_loc, goal_loc, map):
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    came_from[start_loc] = None
    cost_so_far[start_loc] = 0
    twodastar = Unconstrained((goal_loc[1],goal_loc[0]),map)
    while not frontier.empty():
        item = frontier.get()
        curr_node = item[1]

        if curr_node == goal_loc: #will need to do correct goal checking
            return format_path(came_from,curr_node)
        
        for next_node in find_neighbors(curr_node):
            new_cost = cost_so_far[curr_node] # additional costs of moving + turning n shi
            prev_cost = cost_so_far.get(next_node)       
            
            if prev_cost is None or new_cost < prev_cost:    
                cost_so_far[next_node] = new_cost
                heuristic = get_heuristic(curr_node,goal_loc,twodastar)  
                priority = new_cost + heuristic
                frontier.put((priority,next_node))
                came_from[next_node] = curr_node
            
    raise ValueError("Unable to find path") 
        

def format_path(came_from, node):
    path = []
    while came_from[node] is not None: # appends nodes in path with goal as beginning
        path.append(node)
        node = came_from[node]
    return path

print(discritize((12,9,math.radians(105)))) # (1,0,7)