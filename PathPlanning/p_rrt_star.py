import cv2
import numpy as np
import math
from scipy.spatial import KDTree
import time
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameters import *
from PathPlanning.p_dubins import p_dubins_connect
from PathPlanning.dubins import plan_dubins_path

def get_bin_road(road_img):
    """
    Take image of road and build KD obstacle tree (lines and diluted cars) and shape of road to sample within
    
    :param road_img: Image of current road to convert into obstacles and road sampling
    """
    cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    cars_dilute = cv2.dilate(cars,dilute_kernel,iterations=2) # Dilute size of cars to improve path planning
    lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))
    outside = cv2.inRange(road_img, np.array([25,25,25]), np.array([75,75,75]))
    
    obstacles_pxs = np.where(cv2.bitwise_or(lines, cars_dilute)>1)
    obst_tree = KDTree(np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T)
    
    inv_lines = cv2.bitwise_not(cv2.bitwise_or(lines, outside), 255*np.ones_like(lines)) # Find countour of road
    cnt, _ = cv2.findContours(inv_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    road_i = np.argmax([cv2.contourArea(cont) for cont in cnt])
    return obst_tree, cnt[road_i]

def kd_collision_check(object_tree, collision_r, node):
    """
    Check for collisions between a current node position and any obstacles.
    Returns True if there is a collision.
    
    :param object_tree: KD tree of obstacles: lines/cars
    :param collision_r: Radius containing all points of ego car
    :param node: Pose to check if in collision
    """
    car_loc = (node[0]+(math.cos(-np.pi/2-node[2])*CAR_WHEELBASE/2), node[1]+(math.sin(-np.pi/2-node[2]))*CAR_WHEELBASE/2)
    collided_idxs = object_tree.query_ball_point(car_loc, collision_r)
    if len(collided_idxs) > 0:
        car_rect = cv2.RotatedRect(car_loc,(CAR_LENGTH,CAR_WIDTH),np.rad2deg(np.pi/2-node[2]))
        for obst in object_tree.data[collided_idxs]:
            if cv2.pointPolygonTest(car_rect.points(), (obst[0], obst[1]), False) >= 0: # Use point polygon test to determine if any obstacle points are within car boundaries
                return True
    return False

class P_RRTStar():
    def __init__(self, road_img, start, goal, r, step_size, max_samples, v_update = 10, p_goal=0.1):
        """
        Position based dubins RRT* class. Note points are always only x,y coordinates. Nodes/vertexes are SE2 with x,y,heading.
        
        :param road_img: Image of current road to convertin into obstacles and road sampling
        :param start: Starting position for the algorithm. Must include x,y, and heading
        :param goal: Goal position for the algorithm. Only requires x,y
        :param r: Turning radius of car
        :param step_size: Desired step size between nodes (IS NOT GUARANTEED)
        :param max_samples: Max number of random samples for the algorithm to test
        :param v_update: Number of new nodes added before KD tree is rebuilt
        :param p_goal: Percentage of the random samples to be the goal
        """
        self.img = road_img
        self.start = start
        self.goal = [goal[0], goal[1]]
        self.r = r
        self.step_size = step_size
        self.max_samples = max_samples
        self.v_update = v_update
        self.p_goal = p_goal
        
        self.collision_r = math.ceil(np.hypot(CAR_LENGTH/2, CAR_WIDTH/2))
        self.obst_tree, self.road_cnt = get_bin_road(road_img)
        sample_box = cv2.boxPoints(cv2.minAreaRect(self.road_cnt)) # Shape of road to sample within
        self.min_x, self.min_y = np.min(sample_box, 0) # only sample ahead of current position and behind goal
        self.max_x, self.max_y = np.max(sample_box, 0)
        self.min_y = max(self.min_y, self.goal[1])
        self.max_y = min(self.max_y, self.start[1])
        
        self.V = np.array([start])
        self.unadded_Vs = []
        self.kd_tree = KDTree(self.V[:,:2])
        self.came_from = {0: (None, None)}
        self.cost_so_far = {0: 0}
        self.V_count = 1
        self.sample_count = 0
        
    def add_vertex(self, v):
        """
        Add vertex to list of nodes. Only adds nodes to KD tree every self.v_update to reduce redundant KD tree rebuilding
        
        :param v: New node/vertex to add
        """
        self.V = np.vstack([self.V, v])
        self.unadded_Vs.append(self.V_count)
        self.V_count +=1
        if self.V_count % self.v_update == 0:
            self.kd_tree = KDTree(self.V[:,:2])
            self.unadded_Vs = []
        
    def sample_in_road(self):
        """
        Generate random x,y position within road, and start/goal bounds. Will sample until position with heading 0 is not in collision.
        """
        self.sample_count += 1
        if np.random.random() < self.p_goal:
            return self.goal
        placed = False
        while placed is False:
            sample_x = (self.max_x-self.min_x)*np.random.random() + self.min_x
            sample_y = (self.max_y-self.min_y)*np.random.random() + self.min_y
            placed = cv2.pointPolygonTest(self.road_cnt, (sample_x, sample_y), False) >= 0 and not kd_collision_check(self.obst_tree, self.collision_r, (sample_x, sample_y, 0.0))
        return sample_x, sample_y
    
    def nearest(self, p):
        """
        Query to vertex kd tree to find closest vertex. Will also search unadded vertexes to see if any are closer.
        NOTE: KD Tree will node find closest vertex, not one that is behind the point and has a shorter p_dubins_connect path
              Searching the unadded nodes will automatically filter for nodes positioned behind the point.
        
        :param p: Point to find closest vertex to.
        """
        dist, idx = self.kd_tree.query(p)
        nearest_v = self.V[idx]
        nearest_idx = idx
        for v_idx in self.unadded_Vs:
            v = self.V[v_idx]
            D = np.hypot(p[0]-v[0],p[1]-v[1])
            if D<dist and v[1]>p[1]:
                dist = D
                nearest_v = v
                nearest_idx = v_idx
        return nearest_v, nearest_idx
    
    def near(self, p, scale=3):
        """
        Find nodes nearby to a point. Valid nodes are those behind the point and all nodes contains all nodes within the radius.
        Uses a scale * step_size to generate radius to select within.

        :param p: Point to find nearby nodes to.
        :param scale: Integer scale to multiply step size by.
        """
        near_r = scale*self.step_size
        idxs = self.kd_tree.query_ball_point(p,near_r)
        valid_nodes = []
        all_nodes = []
        for idx in idxs:
            if idx >= self.V.shape[0]: continue
            v = self.V[idx]
            all_nodes.append(idx)
            if v[1]>p[1]: valid_nodes.append(idx)
        for idx in self.unadded_Vs:
            v = self.V[idx]
            all_nodes.append(idx)
            dist = np.hypot(p[0]-v[0],p[1]-v[1])
            if dist <= near_r and v[1]>p[1]: valid_nodes.append(idx)
        return valid_nodes, all_nodes
    
    def steer(self, near, new):
        """
        Given a random point, find new random point within step size in same direction.
        Will return the same random point if within the step size
        
        :param near: Node closest to randomly sampled point
        :param new: Randomly sampled point
        """
        dx = new[0]-near[0]
        dy = new[1]-near[1]
        cost = np.hypot(dx, dy)
        if cost < self.step_size:
            return new
        else:
            x_new = near[0] + self.step_size*(dx)/cost
            y_new = near[1] + self.step_size*(dy)/cost
            return (x_new, y_new)
        
    def rewire(self, v_new, v_new_idx, near_idxs):
        """
        Determine if new vertex allows for cheaper paths further ahead in tree. 
        Will generate paths to each of the nearby nodes, if the path is cheaper and collision free replace
        NOTE: Paths are generated from node to node, so heading is taken into account at goal.
        
        :param v_new: New node to generate paths from.
        :param v_new_idx: Index of new node
        :param near_idxs: Indicies of nearby nodes
        """
        for v_near_idx in near_idxs:
            v_near = self.V[v_near_idx]
            dx, dy, dheading, _,dubins_path = plan_dubins_path(v_new[0], v_new[1], v_new[2]-np.pi/2, v_near[0], v_near[1], v_near[2]-np.pi/2, 1/self.r, step_size=1) # use dubins library instead of p_dubins_connect
            cost = sum(dubins_path)
            if cost+self.cost_so_far[v_new_idx] < self.cost_so_far[v_near_idx]:
                parent_collided = False
                d_path = []
                for node in zip(dx, dy, dheading+np.pi/2):
                    if kd_collision_check(self.obst_tree, self.collision_r, node): 
                        parent_collided = True
                        break
                    d_path.append(np.array(node))
                if not parent_collided:
                    self.cost_so_far[v_near_idx] = cost+self.cost_so_far[v_new_idx]
                    self.came_from[v_near_idx] = (v_new_idx, d_path)
                
    def reconstruct_path(self, goal_idx):
        """
        With the goal found, rebuild the path recursively. Returns the path in terms of just nodes along with interpolated path.
        
        :param goal_idx: Node index of goal
        """
        path = [self.V[goal_idx]]
        total_path = []
        node_idx, d_path = self.came_from[goal_idx]
        while node_idx is not None:
            path.insert(0, self.V[node_idx])
            total_path = d_path + total_path
            node_idx, d_path = self.came_from[node_idx]
        return path, total_path
    
    def p_rrt_star(self, visualize = False):
        """
        Execute P-RRT* algorithm based on current start and goal locations.    
        
        :param visualize: Boolean to determine if sampling is displayed.     
        """
        while self.sample_count < self.max_samples:
            p_rand = self.sample_in_road()
            v_nearest, v_nearest_idx = self.nearest(p_rand)
            p_new = self.steer(v_nearest, p_rand)
            path, cost = p_dubins_connect(v_nearest, p_new, self.r, 1)
            if path is None: continue
            collided = False
            for node in path:
                if kd_collision_check(self.obst_tree, self.collision_r, node): 
                    collided = True
                    break
            total_cost = self.cost_so_far[v_nearest_idx] + cost
            v_parent_idx = v_nearest_idx if not collided else None
            best_path = path if not collided else None
            valid_nearby, all_nearby = self.near(p_new)
            for v_near_idx in valid_nearby:
                if v_near_idx == v_nearest_idx: continue
                v_near = self.V[v_near_idx]
                path, cost = p_dubins_connect(v_near, p_new, self.r, 1)
                if path is None: continue
                if self.cost_so_far[v_near_idx] + cost < total_cost:
                    parent_collided = False
                    for node in path:
                        if kd_collision_check(self.obst_tree, self.collision_r, node): 
                            parent_collided = True
                            break
                    if not parent_collided:
                        total_cost = self.cost_so_far[v_near_idx] + cost
                        v_parent_idx = v_near_idx
                        best_path = path
            if v_parent_idx is None: continue
            v_new = best_path[-1]
            v_new_idx = self.V_count
            self.add_vertex(v_new)
            self.cost_so_far[v_new_idx] = total_cost
            self.came_from[v_new_idx] = (v_parent_idx, best_path)
            self.rewire(v_new, v_new_idx, all_nearby)
            
            if round(v_new[0]) == round(self.goal[0]) and round(v_new[1]) == round(self.goal[1]):
                return self.reconstruct_path(v_new_idx)
            
            if visualize:
                cv2.circle(self.img, (round(v_new[0]), round(v_new[1])), 2, (0,0,255), -1)
                cv2.imshow('progress', self.img)
                cv2.waitKey(1)    
        raise TimeoutError("Unable to find path")

if __name__ == '__main__':
    img = cv2.imread('screen2.jpg', cv2.IMREAD_COLOR)
    start = np.array([450.,450.,0.])
    center = np.array([350., 200.])
    goal = np.array([450., 15.])
    tr = RESOLUTION/(D_HEADING*2)
    
    show_path = False
    
    start_t = time.time()
    rrt = P_RRTStar(img.copy(), start, center, tr, RESOLUTION, 1e5)
    path1, d_path1 = rrt.p_rrt_star(show_path)
    print(f'Path 1: {time.time()-start_t}')
    start_t2 = time.time()
    rrt = P_RRTStar(img.copy(), [path1[-1][0], path1[-1][1], 0], goal, tr, RESOLUTION, 1e5)
    path2, d_path2 = rrt.p_rrt_star(show_path)
    print(f'Path 2: {time.time()-start_t2}')
    print(f'Total Time: {time.time() - start_t}')
    
    d_path = d_path1+d_path2
    for p in d_path:
        cv2.circle(img, p[:2].astype(np.int32), 1, (0,0,255), -1)
    
    path = path1+path2
    for i in range(1,len(path)):
        prev_p = path[i-1]
        p = path[i]
        test_path, _ = p_dubins_connect(prev_p, p[:2], tr, 1)
        cv2.circle(img, p[:2].astype(np.int32), 2, (255,255,255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(1)
                
    # np.save('rrt_path', path)
    cv2.circle(img, start[:2].astype(np.int32), 3, (0,0,0), -1)
    cv2.circle(img, goal[:2].astype(np.int32), 3, (0,255,0), 1)
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    while True:
        cv2.waitKey(1)
    
    