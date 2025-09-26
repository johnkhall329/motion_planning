from queue import PriorityQueue
import math

class Unconstrained():
    '''
    Gets a 2-D path using A* and Euclidean distance heuristic. Uses goal state as starting location and stores data as lookup table.
    '''
    def __init__(self, goal_state, map):
        self.goal = goal_state
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[self.goal] = None
        self.cost_so_far[self.goal] = 0
        self.map = map
    
    def get_unconstrained_path(self, start_state):
        self.replan_frontier(start_state)
        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = item[1]
            
            if curr_node == self.goal: # if it finds the goal, return a formatted path
                return self.format_path(curr_node)
            
            for next_node in [(curr_node[0]-1, curr_node[1]),
                              (curr_node[0], curr_node[1]+1),
                              (curr_node[0]+1, curr_node[1]),
                              (curr_node[0], curr_node[1]-1),
                              (curr_node[0]+1, curr_node[1]+1),
                              (curr_node[0]+1, curr_node[1]-1),
                              (curr_node[0]-1, curr_node[1]+1),
                              (curr_node[0]-1, curr_node[1]-1)]:
                if 0<=next_node[0]<self.map.shape[0] and 0<=next_node[1]<self.map.shape[1] and self.map[next_node] != 255:
                    new_cost = self.cost_so_far[curr_node] + self.map[next_node] + 1
                    prev_cost = self.cost_so_far.get(next_node)
                    if prev_cost is None or new_cost < prev_cost: # only adds to queue if unvisited or cheaper to get to
                        self.cost_so_far[next_node] = new_cost
                        heuristic = math.sqrt((start_state[0]-next_node[0])**2 + (start_state[1]-next_node[1])**2) # Euclidean distance to goal
                        priority = int(new_cost + heuristic)
                        self.frontier.put((priority,next_node))
                        self.came_from[next_node] = curr_node

    def replan_frontier(self,start_state):
        new_frontier = PriorityQueue()
        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = item[1]

            cost = self.cost_so_far[curr_node]
            new_h = math.sqrt((start_state[0]-curr_node[0])**2 + (start_state[1]-curr_node[1])**2)
            new_frontier.put((int(cost+new_h),curr_node))
        self.frontier = new_frontier

    
    def format_path(self, node):
        path = []
        while self.came_from[node] is not None: # appends nodes in path with goal as beginning
            path.append(node)
            node = self.came_from[node]
        return path