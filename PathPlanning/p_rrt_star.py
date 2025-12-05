import cv2
import numpy as np
import math
from scipy.spatial import KDTree
import time

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 40
CAR_LENGTH = 60
CAR_WHEELBASE = 40

D_HEADING = np.pi/4
RESOLUTION = 20
TURNING_RADIUS = RESOLUTION/D_HEADING
TURN_COST = 10
SHOW_ARROWS = True

from p_dubins import p_dubins_connect
from dubins import plan_dubins_path

def get_bin_road(road_img):
    cars = cv2.inRange(road_img, np.array([0,0,200]), np.array([50,50,255]))
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    cars_dilute = cv2.dilate(cars,dilute_kernel,iterations=2)
    lines = cv2.inRange(road_img, np.array([200,200,200]), np.array([255,255,255]))
    outside = cv2.inRange(road_img, np.array([25,25,25]), np.array([75,75,75]))
    
    obstacles_pxs = np.where(cv2.bitwise_or(lines, cars_dilute)>1)
    obst_tree = KDTree(np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T)
    
    inv_lines = cv2.bitwise_not(cv2.bitwise_or(lines, outside), 255*np.ones_like(lines))
    cnt, _ = cv2.findContours(inv_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    road_i = np.argmax([cv2.contourArea(cont) for cont in cnt])
    return obst_tree, cnt[road_i]

def kd_collision_check(object_tree, collision_r, node):
    car_loc = (node[0]+(math.cos(-np.pi/2-node[2])*CAR_WHEELBASE/2), node[1]+(math.sin(-np.pi/2-node[2]))*CAR_WHEELBASE/2)
    collided_idxs = object_tree.query_ball_point(car_loc, collision_r)
    if len(collided_idxs) > 0:
        car_rect = cv2.RotatedRect(car_loc,(CAR_LENGTH,CAR_WIDTH),np.rad2deg(np.pi/2-node[2]))
        for obst in object_tree.data[collided_idxs]:
            if cv2.pointPolygonTest(car_rect.points(), (obst[0], obst[1]), False) >= 0: 
                return True
    return False

class P_RRTStar():
    def __init__(self, road_img, start, goal, r, step_size, max_samples, v_update = 10, p_goal=0.1):
        self.img = road_img
        self.start = start
        self.goal = goal
        self.r = r
        self.step_size = step_size
        self.max_samples = max_samples
        self.v_update = v_update
        self.p_goal = p_goal
        
        self.collision_r = math.ceil(np.hypot(CAR_LENGTH/2, CAR_WIDTH/2))
        self.obst_tree, self.road_cnt = get_bin_road(road_img)
        sample_box = cv2.boxPoints(cv2.minAreaRect(self.road_cnt))
        self.min_x, self.min_y = np.min(sample_box, 0)
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
        self.V = np.vstack([self.V, v])
        self.unadded_Vs.append(self.V_count)
        self.V_count +=1
        if self.V_count % self.v_update == 0:
            self.kd_tree = KDTree(self.V[:,:2])
            self.unadded_Vs = []
        
    def sample_in_road(self):
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
    
    def near(self, p):
        # new_r = np.hypot(self.goal[0]-self.start[0], self.goal[1]-self.start[1])/(self.r*self.V_count)
        # near_r = max(new_r, 2*self.r)
        near_r = 3*self.step_size
        idxs = self.kd_tree.query_ball_point(p,near_r)
        dists = {}
        for idx in idxs:
            if idx >= self.V.shape[0]: continue
            v = self.V[idx]
            dist = np.hypot(p[0]-v[0],p[1]-v[1])
            if v[1]>p[1]: dists[idx] = dist
        for idx in self.unadded_Vs:
            v = self.V[idx]
            dist = np.hypot(p[0]-v[0],p[1]-v[1])
            if dist <= near_r and v[1]>p[1]: dists[idx] = dist
        dist = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
        return list(dist.keys())
    
    def steer(self, near, new):
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
        for v_near_idx in near_idxs:
            v_near = self.V[v_near_idx]
            dx, dy, dheading, _,dubins_path = plan_dubins_path(v_new[0], v_new[1], v_new[2]-np.pi/2, v_near[0], v_near[1], v_near[2]-np.pi/2, 1/self.r, step_size=1)
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
                    # print('rewired')
                
    def reconstruct_path(self, goal_idx):
        path = [self.V[goal_idx]]
        total_path = []
        node_idx, d_path = self.came_from[goal_idx]
        # total_path += d_path
        while node_idx is not None:
            path.insert(0, self.V[node_idx])
            total_path = d_path + total_path
            node_idx, d_path = self.came_from[node_idx]
        return path, total_path
    
    def p_rrt_star(self):
        while self.sample_count < self.max_samples:
            p_rand = self.sample_in_road()
            v_nearest, v_nearest_idx = self.nearest(p_rand)
            # if round_node(p_rand) == round_node(self.goal):
            #     path, _ = p_dubins_connect(self.start, self.goal, self.r, 1)
            #     for node in path:
            #         if kd_collision_check(self.obst_tree, self.collision_r, node): break
            p_new = self.steer(v_nearest, p_rand)
            path, cost = p_dubins_connect(v_nearest, p_new, self.r, 1)
            if path is None: continue
            collided = False
            for node in path:
                if kd_collision_check(self.obst_tree, self.collision_r, node): 
                    collided = True
                    break
            # if not collided:
            # if all([not kd_collision_check(self.obst_tree, self.collision_r, node) for node in path]):
            total_cost = self.cost_so_far[v_nearest_idx] + cost
            v_parent_idx = v_nearest_idx if not collided else None
            best_path = path if not collided else None
            nearby = self.near(p_new)
            for v_near_idx in nearby:
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
                    # if all([not kd_collision_check(self.obst_tree, self.collision_r, node) for node in path]):
                        total_cost = self.cost_so_far[v_near_idx] + cost
                        v_parent_idx = v_near_idx
                        best_path = path
            if v_parent_idx is None: continue
            v_new = best_path[-1]
            parent_v = self.V[v_parent_idx]
            if np.allclose(parent_v, v_new, atol=1e-2):
                print('huh')
            v_new_idx = self.V_count
            self.add_vertex(v_new)
            self.cost_so_far[v_new_idx] = total_cost
            self.came_from[v_new_idx] = (v_parent_idx, best_path)
            self.rewire(v_new, v_new_idx, nearby)
            # cv2.circle(self.img, (round(v_new[0]), round(v_new[1])), 2, (0,0,255), -1)
            if round(v_new[0]) == round(self.goal[0]) and round(v_new[1]) == round(self.goal[1]):
                return self.reconstruct_path(v_new_idx)
            # cv2.imshow('progress', self.img)
            # cv2.waitKey(1)
            # print(self.sample_count)    
        raise TimeoutError("Unable to find path")

if __name__ == '__main__':
    img = cv2.imread('screen2.jpg', cv2.IMREAD_COLOR)
    start = np.array([450.,450.,0.])
    center = np.array([350., 200.])
    goal = np.array([450., 15.])
    start_t = time.time()
    rrt = P_RRTStar(img.copy(), start, center, TURNING_RADIUS, RESOLUTION, 1e5)
    path1, d_path1 = rrt.p_rrt_star()
    print(f'Path 1: {time.time()-start_t}')
    start_t2 = time.time()
    rrt = P_RRTStar(img.copy(), [path1[-1][0], path1[-1][1], 0], goal, TURNING_RADIUS, RESOLUTION, 1e5)
    path2, d_path2 = rrt.p_rrt_star()
    print(f'Path 2: {time.time()-start_t2}')
    print(f'Total Time: {time.time() - start_t}')
    
    d_path = d_path1+d_path2
    for p in d_path:
        cv2.circle(img, p[:2].astype(np.int32), 1, (0,0,255), -1)
    
    path = path1+path2
    for i in range(1,len(path)):
        prev_p = path[i-1]
        p = path[i]
        # print(p[2])
        test_path, _ = p_dubins_connect(prev_p, p[:2], TURNING_RADIUS, 1)
        cv2.circle(img, p[:2].astype(np.int32), 2, (255,255,255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(1)
                
    cv2.circle(img, start[:2].astype(np.int32), 3, (0,0,0), -1)
    cv2.circle(img, goal[:2].astype(np.int32), 3, (0,255,0), 1)
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    while True:
        cv2.waitKey(1)
    
    