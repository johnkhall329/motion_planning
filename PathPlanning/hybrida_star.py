import numpy as np
from queue import PriorityQueue
import math 
import cv2
import time
import pygame
from scipy.spatial import KDTree
try:
    from Kinematics.parameters import *
    from PathPlanning.unconstrained import Unconstrained
    from PathPlanning.dubins import plan_dubins_path as dubins
    
except ModuleNotFoundError:
    from unconstrained import Unconstrained
    from dubins import plan_dubins_path as dubins
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    CAR_WIDTH = 40
    CAR_LENGTH = 60
    CAR_WHEELBASE = 40

D_HEADING = np.pi/12
RESOLUTION = 20
TURNING_RADIUS = RESOLUTION/D_HEADING
TURN_COST = 10
SHOW_ARROWS = True

def get_bin_road(road_img):
    # road_img = cv2.cvtColor(cv2.transpose(pygame.surfarray.array3d(screen)), cv2.COLOR_RGB2BGR)
    cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
    lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))
    
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cars_dilute = cv2.dilate(cars,dilute_kernel,iterations=5)
    lines_dilute = cv2.dilate(lines ,dilute_kernel,iterations=2)
    combined = cv2.bitwise_or(cars_dilute, lines_dilute)
    
    # eroded = cv2.erode(combined,erode_kernel)
    # dilated = cv2.dilate(eroded,dilute_kernel,iterations=3)
    blurred = cv2.GaussianBlur(combined, (45,45),0)
    # diluted_road = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return cars, blurred

# a node is a continous (x,y,phi) tuple
# phi is the heading angle in radians 
# x and y are in pixels, with (0,0) as top left corner

def round_node(node):
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
    path = dubins(curr_state[0], curr_state[1], curr_state[2]-np.pi/2, goal[0], goal[1], goal[2]-np.pi/2, 1/TURNING_RADIUS)[4]
    h1 = sum(path)
    h2 = two_d_astar.get_unconstrained_path((curr_discritized[1], curr_discritized[0]),step_size)
    return max(h1,h2)

def build_obst_tree(img):
    obstacles_pxs = np.where(img>1)
    obst_tree = KDTree(np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T)
    return obst_tree

def kd_collision_check(object_tree, node):
    collision_r = math.ceil(np.hypot(CAR_LENGTH/2, CAR_WIDTH/2))
    car_loc = (node[0]+(math.cos(-np.pi/2-node[2])*CAR_WHEELBASE/2), node[1]+(math.sin(-np.pi/2-node[2]))*CAR_WHEELBASE/2)
    collided_idxs = object_tree.query_ball_point(car_loc, collision_r)
    if len(collided_idxs) > 0:
        car_rect = cv2.RotatedRect(car_loc,(CAR_LENGTH,CAR_WIDTH),np.rad2deg(np.pi/2-node[2]))
        for obst in object_tree.data[collided_idxs]:
            if cv2.pointPolygonTest(car_rect.points(), (obst[0], obst[1]), False) >= 0: 
                print("found kd collision")
                return True
    return False

def check_collision(objects, came_from, cost_so_far, node):
    path_img = np.zeros_like(objects)
    path = []
    collided=False
    while came_from[round_node(node)] is not None:
        path.append(node)
        car_loc = cv2.RotatedRect((node[0]+(math.cos(-np.pi/2-node[2])*CAR_WHEELBASE/2), node[1]+(math.sin(-np.pi/2-node[2]))*CAR_WHEELBASE/2),(CAR_LENGTH,CAR_WIDTH),np.rad2deg(np.pi/2-node[2]))
        pts = car_loc.points().astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(path_img,pts,(255,255,255))
        mask = cv2.bitwise_and(objects,path_img)
        if np.any(mask): 
            collided=True
            for collided_node in path:   
                cost_so_far[discretize(collided_node)] = 1e9
            break
        node = came_from[round_node(node)]
    return collided, path

def hybrid_a_star_path(start_loc, goal_loc, screen):
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
    collision_prob_const = 10
    while not frontier.empty():
        item = frontier.get()
        h = item[0]
        curr_node = item[1]
        curr_discritized = discretize(curr_node)
        # cm = color_map.copy()

        if curr_discritized == goal_discretized: #will need to do correct goal checking
            collided, path = check_collision(car_img, came_from, cost_so_far, curr_node)      
            if not collided: 
                return path
            else:
                print('collision')
                continue
        
        if kd_collision_check(obst_tree, curr_node):
            continue
        
        if h!=0 and (collision_prob_const/h > np.random.random()):
            kd_collision_check(obst_tree, curr_node)
            collided,_ = check_collision(car_img, came_from, cost_so_far, curr_node)
            if collided: continue
        
        # cv2.circle(color_map,(int(curr_node[0]),int(curr_node[1])),3, (0,0,255),-1)    
        for next_neighbor in find_neighbors(curr_node):
            next_node, turn_cost = next_neighbor
            new_cost = cost_so_far[curr_discritized] + turn_cost + RESOLUTION
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
        
def format_path(came_from, node):
    path = []
    while came_from[round_node(node)] is not None: # appends nodes in path with goal as beginning
        path.append(node)
        node = came_from[round_node(node)]
    return path

if __name__ == '__main__':
    map = cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE)
    screen = cv2.imread('screen2.jpg', cv2.IMREAD_COLOR)
    start = (450.0,450.0,0.0)
    center = (350.0,240.0,0.0)
    goal = (450.0,25.0,0.0)
    t = time.time()

    cars = cv2.inRange(screen, np.array([0,0,200]), np.array([50,50,255]))
    
    TWO_PHASES = False

    # In 2 phases:
    if TWO_PHASES:
        phase1 = hybrid_a_star_path(start,center,screen)
        phase2 = hybrid_a_star_path(phase1[0],goal,screen)
        path = phase2 + phase1

    # In 1 phase:
    else:
        path = hybrid_a_star_path(start,goal,screen)

    # cv2.imshow('masks', cv2.bitwise_or(cv2.bitwise_or(img1,img2),dil))
    # cv2.imshow('diluted', dil)

    print("Time taken:", time.time() - t)

    # for i, node in enumerate(path):
    #     if i == 0:
    #         continue
    #     # print distance between nodes
    #     prev = path[i-1]
    #     dist = math.sqrt((node[0]-prev[0])**2 + (node[1]-prev[1])**2)
    #     print(f"discretized: {discretize(node)}, actual: {node}")
    #     # print(f"Distance from: {dist:.2f}")
    #     if dist > 10.01:
    #         print("prev node:", prev)
    #         print("curr node:", node)
    
    # Length of arrow (pixels)
    ARROW_LENGTH = 5

    filepath = 'hybrid_astar_path.npy'
    np.save(filepath, np.array(path))
    print(f"Saved path to {filepath}")

    _, dil = get_bin_road(screen)
    color_map = cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)
    # color_map = screen.copy()
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