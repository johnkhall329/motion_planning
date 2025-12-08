from queue import PriorityQueue
import math
import numpy as np
import cv2

class Unconstrained():
    '''
    Gets a 2-D path using A* and Euclidean distance heuristic accounting for obstacles. Uses goal state as starting location and stores data as lookup table.
    '''
    def __init__(self, goal_state, map):
        self.goal = goal_state
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[self.goal] = None
        self.cost_so_far[self.goal] = 0
        self.map = map.astype(np.float32)
        self.color_map = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
        self.frontier.put((0,self.goal))
        self.obstacle_scale = 7.5
    
    def get_unconstrained_path(self, start_state, step_size=10):
        """
        Generate cost to arrive at goal from a given starting location.

        :param start_state: Starting location of ego car
        :param step_size: Distance between discretized nodes
        """
        # cv2.circle(self.color_map,(self.goal[1]*step_size,self.goal[0]*step_size),3, (0,255,0),-1)
        # cv2.circle(self.color_map,(start_state[1]*step_size,start_state[0]*step_size),3, (255,0,0),-1)
        if self.cost_so_far.get(start_state) is not None:
            return self.cost_so_far[start_state]
        self.replan_frontier(start_state,step_size)
        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = item[1]
            if self.map[(curr_node[0]*step_size, curr_node[1]*step_size)] > 250: return 1e9
            # cv2.circle(self.color_map,(curr_node[1]*step_size,curr_node[0]*step_size),3, (0,0,255))
         
            if curr_node == start_state: # if it finds the goal, return a formatted path
                # return self.cost_so_far[curr_node]
                return self.cost_so_far[curr_node]
            
            for next_node in [(curr_node[0]-1, curr_node[1]),
                              (curr_node[0], curr_node[1]+1),
                              (curr_node[0]+1, curr_node[1]),
                              (curr_node[0], curr_node[1]-1),
                              (curr_node[0]+1, curr_node[1]+1),
                              (curr_node[0]+1, curr_node[1]-1),
                              (curr_node[0]-1, curr_node[1]+1),
                              (curr_node[0]-1, curr_node[1]-1)]:
                if 0<=next_node[0]*step_size<self.map.shape[0] and 0<=next_node[1]*step_size<self.map.shape[1] and self.map[next_node[0]*step_size, next_node[1]*step_size] < 250:
                    new_cost = self.cost_so_far[curr_node] + self.obstacle_scale*self.map[next_node[0]*step_size, next_node[1]*step_size] + step_size*math.sqrt((curr_node[0]-next_node[0])**2 + (curr_node[1]-next_node[1])**2)
                    prev_cost = self.cost_so_far.get(next_node)
                    # if self.map[next_node] > 150:
                    #     print(new_cost)
                    if prev_cost is None or new_cost < prev_cost: # only adds to queue if unvisited or cheaper to get to
                        self.cost_so_far[next_node] = new_cost
                        heuristic = step_size*math.sqrt((start_state[0]-next_node[0])**2 + (start_state[1]-next_node[1])**2) # Euclidean distance to goal
                        priority = new_cost + heuristic
                        self.frontier.put((priority,next_node))
                        self.came_from[next_node] = curr_node
                        # cv2.circle(self.color_map,(next_node[1]*step_size,next_node[0]*step_size),3, (0,255,255))
            # cv2.imshow('upath', self.color_map)
            # cv2.waitKey(1)
        # print("Unable to find path")
        return 1e9

    def replan_frontier(self,start_state,step_size):
        """
        Given a new starting state, update heuristic values of frontier for priority.
        
        :param start_state: New starting state of ego car (the desired goal of the 2d A*)
        :param step_size: Distance between each discretized point
        """
        new_frontier = PriorityQueue()
        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = item[1]
            if curr_node == self.goal: 
                new_frontier.put((0,curr_node))
                continue

            cost = self.cost_so_far[curr_node]
            new_h = step_size*math.sqrt((start_state[0]-curr_node[0])**2 + (start_state[1]-curr_node[1])**2)
            new_frontier.put((int(cost+new_h),curr_node))
        self.frontier = new_frontier