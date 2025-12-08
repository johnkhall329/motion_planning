import numpy as np
from queue import PriorityQueue
import math
import cv2
import time
import sys
import os
from scipy.spatial import KDTree

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameters import *
from PathPlanning.unconstrained import Unconstrained
from PathPlanning.dubins import plan_dubins_path as dubins

TURN_COST = 10
SHOW_ARROWS = True

def get_bin_road(road_img):
    """
    Take image of road and generate car obstacles and diluted+blurred car obstacles for unconstrained
    
    :param road_img: Image of current road to convert into obstacles
    """
    # road_img = cv2.cvtColor(cv2.transpose(pygame.surfarray.array3d(screen)), cv2.COLOR_RGB2BGR)
    cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
    lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))
    
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    cars_dilute = cv2.dilate(cars,dilute_kernel,iterations=5)
    lines_dilute = cv2.dilate(lines ,dilute_kernel,iterations=2)
    combined = cv2.bitwise_or(cars_dilute, lines_dilute)

    blurred = cv2.GaussianBlur(combined, (45,45),0)
    return cars, blurred

# a node is a continous (x,y,phi) tuple
# phi is the heading angle in radians 
# x and y are in pixels, with (0,0) as top left corner

def round_node(node):
    """
    Take node with floating point error and round to nearest pixel, heading to 2 decimals.
    Used to store nodes in dictionary.
    
    :param node: Node to round
    """
    return (round(node[0]), round(node[1]), round(node[2], 2))

def discretize(node, resolution=RESOLUTION, turning_a=D_HEADING):
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
    x = round(node[0]/resolution)
    y = round(node[1]/resolution)
    phi = node[2]
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    phi = round(phi/turning_a)
    return (x,y,phi)

def find_neighbors(node, distance=RESOLUTION, turning_a=D_HEADING):
    """
    For a node, generate motion primitives to find three other nodes: left, straight, right. There are additional costs for turning.
    
    :param node: Node to find neighbors from
    :param distance: Arc distance away from current node
    :param turning_a: Change in heading during turn
    """
    x,y,phi = node

    # straight
    dx = distance * -math.sin(phi)
    dy = distance * -math.cos(phi)
    straight = (x+dx,y+dy,phi)

    # consts for turning
    r = distance/turning_a
    d = 2 * r * math.sin(turning_a/2)
    # d = RESOLUTION # for debugging

    # left
    dx = d * -math.sin(phi + turning_a/2)
    dy = d * -math.cos(phi + turning_a/2)
    left = (x+dx,y+dy,phi+turning_a)

    # right
    dx = d * -math.sin(phi - turning_a/2)
    dy = d * -math.cos(phi - turning_a/2)
    right = (x+dx,y+dy,phi-turning_a)

    return ((left, TURN_COST), (straight, 0), (right, TURN_COST))

def get_heuristic(curr_state, curr_discritized, goal, two_d_astar: Unconstrained, step_size=RESOLUTION):
    """
    Return max heuristic between path distance of constrained without obstacles (dubins) and unconstrained with obstacles (2D A*).
    
    :param curr_state: Current node to generate path from
    :param curr_discritized: Discretized node location, used for speeding up 2D A*
    :param goal: Goal location to arrive at
    :param two_d_astar: 2D A* search object used for unconstrained search. Goal location is stored as start, use current location as goal and stores history for speed.
    :param step_size: Distance between discretized nodes
    """
    path = dubins(curr_state[0], curr_state[1], curr_state[2]-np.pi/2, goal[0], goal[1], goal[2]-np.pi/2, 1/TURNING_RADIUS)[4]
    h1 = sum(path)
    h2 = two_d_astar.get_unconstrained_path((curr_discritized[1], curr_discritized[0]),step_size)
    return max(h1,h2)

def build_obst_tree(img):
    """
    Generate KD tree containing car obstacle pixels for collision checking
    
    :param img: Binary image of car obstacles
    """
    obstacles_pxs = np.where(img>1)
    obst_tree = KDTree(np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T)
    return obst_tree

def kd_collision_check(object_tree, node):
    """
    Check for collisions between a current node position and any obstacles.
    Returns True if there is a collision.
    
    :param object_tree: KD tree of obstacles
    :param node: Pose to check if in collision
    """
    collision_r = math.ceil(np.hypot(CAR_LENGTH/2, CAR_WIDTH/2))
    car_loc = (node[0]+(math.cos(-np.pi/2-node[2])*CAR_WHEELBASE/2), node[1]+(math.sin(-np.pi/2-node[2]))*CAR_WHEELBASE/2)
    collided_idxs = object_tree.query_ball_point(car_loc, collision_r)
    if len(collided_idxs) > 0:
        car_rect = cv2.RotatedRect(car_loc,(CAR_LENGTH,CAR_WIDTH),np.rad2deg(np.pi/2-node[2]))
        for obst in object_tree.data[collided_idxs]:
            if cv2.pointPolygonTest(car_rect.points(), (obst[0], obst[1]), False) >= 0: 
                return True
    return False

def check_path_collision(objects, node, came_from, cost_so_far):
    """
    Check for collisions between current node and parent. If there is a collision, it will increase the cost to reach the desired node
    
    :param objects: Car obstacle kd tree
    :param node: Current position
    :param came_from: Came from dictionary to get parent node
    :param cost_so_far: Cost map incase of collision
    """
    parent_node  = came_from[round_node(node)]
    collided = False
    dx, dy, dheading, _, _ = dubins(parent_node[0], parent_node[1], -parent_node[2]-np.pi/2, node[0], node[1], -node[2]-np.pi/2, 1/TURNING_RADIUS+1e-3) # Get points along dubins path, slightly increase curvature to prevent full circle
    for path_node in zip(dx, dy, dheading+np.pi/2):
        if kd_collision_check(objects, path_node): 
            collided = True
            break
    if collided: cost_so_far[discretize(node)] = 1e9
    
def format_path(node, came_from):
    """
    Return path from current node to start. Final path starts at starting location and ends at current node
    
    :param node: Current node, most likely the goal
    :param came_from: Dictionary of parent nodes
    """
    path = []
    while came_from[round_node(node)] is not None:
        path.insert(0,node)
        node = came_from[round_node(node)]
    return path

def hybrid_a_star_path(start_loc, goal_loc, screen):
    """
    Generate path from starting location to goal given the current image of the road.
    
    :param start_loc: Starting location of ego car
    :param goal_loc: Goal location for ego car
    :param screen: Image of road containing non-ego cars
    """
    car_img, diluted_img = get_bin_road(screen)
    obst_tree = build_obst_tree(car_img)
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    came_from[round_node(start_loc)] = None
    cost_so_far[discretize(start_loc)] = 0
    frontier.put((0,start_loc))
    goal_discretized = discretize(goal_loc)
    twodastar = Unconstrained((goal_discretized[1],goal_discretized[0]),diluted_img)
    # color_map = cv2.cvtColor(diluted_img,cv2.COLOR_GRAY2BGR)
    # cv2.circle(color_map,(int(goal_loc[0]),int(goal_loc[1])),3, (0,255,0),-1)
    # cv2.circle(color_map,(int(start_loc[0]),int(start_loc[1])),3, (255,0,0),-1)
    itr = 0
    while not frontier.empty():
        item = frontier.get()
        h = item[0]
        curr_node = item[1]
        curr_discritized = discretize(curr_node)
        # cm = color_map.copy()

        if h != 0 and check_path_collision(obst_tree, curr_node, came_from, cost_so_far): # check for collisions between parent node and current node
            continue

        if curr_discritized == goal_discretized: # Goal checking
            path = format_path(curr_node, came_from)      
            return path
        
        # cv2.circle(color_map,(int(curr_node[0]),int(curr_node[1])),3, (0,0,255),-1)    
        for next_neighbor in find_neighbors(curr_node): 
            next_node, turn_cost = next_neighbor
            new_cost = cost_so_far[curr_discritized] + turn_cost + RESOLUTION # Include additional cost of turning
            next_discritized = discretize(next_node)
            prev_cost = cost_so_far.get(next_discritized)       
            
            if prev_cost is None or new_cost < prev_cost:
                cost_so_far[next_discritized] = new_cost
                heuristic = get_heuristic(curr_node,curr_discritized, goal_loc,twodastar)
                priority = new_cost + heuristic
                frontier.put((priority,next_node))
                came_from[round_node(next_node)] = curr_node
            #     for i in range(len(dubinsx)):
            #         cv2.circle(cm,(int(dubinsx[i]),int(dubinsy[i])),3, (0,0,255),-1)
            #     cv2.circle(color_map,(int(next_node[0]),int(next_node[1])),3, (0,0,255),-1)
            # cv2.imshow('Progress', color_map)
            # cv2.waitKey(1)
        itr+=1
    raise ValueError("Unable to find path")
    

if __name__ == '__main__':
    screen = cv2.imread('screen2.jpg', cv2.IMREAD_COLOR)
    start = (450.0,450.0,0.0)
    center = (350.0,240.0,0.0)
    goal = (450.0,25.0,0.0)
    t = time.time()

    cars = cv2.inRange(screen, np.array([0,0,200]), np.array([50,50,255]))
    
    TWO_PHASES = True

    # In 2 phases:
    if TWO_PHASES:
        phase1 = hybrid_a_star_path(start,center,screen)
        phase2 = hybrid_a_star_path(phase1[-1],goal,screen)
        path = phase2 + phase1

    # In 1 phase:
    else:
        path = hybrid_a_star_path(start,goal,screen)

    print("Time taken:", time.time() - t)
    
    # Length of arrow (pixels)
    ARROW_LENGTH = 5

    filepath = 'hybrid_astar_path.npy'
    np.save(filepath, np.array(path))
    print(f"Saved path to {filepath}")

    _, dil = get_bin_road(screen)
    # color_map = cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)
    color_map = screen.copy()
    for loc in path:
        x, y, phi = loc
        center = (int(x), int(y))

        # Draw the center point
        cv2.circle(color_map, center, 3, (255, 0, 0), -1)

        if SHOW_ARROWS:
            # Compute arrow direction (positive heading = CCW)
            dx = -ARROW_LENGTH * math.sin(phi)
            dy = -ARROW_LENGTH * math.cos(phi)
            tip = (int(x + dx), int(y + dy))

            # Draw heading arrow
            cv2.arrowedLine(color_map, center, tip, (0, 0, 255), 2, tipLength=0.4)
        
    cv2.imshow('Progress', color_map)
    while True:
        cv2.waitKey(1)