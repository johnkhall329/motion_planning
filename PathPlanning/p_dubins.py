import math
import numpy as np
import cv2

def find_best_tang(tangents, c):
    """
    Given two tangent points and a circle, find the closest kinematically feasible tangent point.
    Returns the index of the closest tangent.
    
    :param tangents: Two points on circle c
    :param c: Coordinates of the circle
    """
    new_thetas = np.zeros(len(tangents))
    for i,t in enumerate(tangents):
        new_theta = -np.pi/2-math.atan2(t[1],t[0])
        if new_theta > np.pi: new_theta -= 2*np.pi
        if new_theta < -np.pi: new_theta += 2*np.pi
        new_thetas[i] = new_theta
    best = 0
    if c[0] > 0: #RIGHT
        best = np.argmax(new_thetas)
    else:
        best = np.argmin(new_thetas)
    return best

def interpolate_path_straight(c, r, x_rand, theta, transform, step_size):
    """
    Return points along a turn-straight path at spacing step_size. Will transform points from local coordinate space into world coordinates.
    Returns list of nodes including heading and total distance of path.
    
    :param c: Coordinates of the circle
    :param r: Turning radius of the car
    :param x_rand: Desired point in local coordinate system
    :param theta: Final arc position on circle
    :param transform: 3x3 transformation matrix from local coordinates to world.
    :param step_size: Distance between nodes in path
    """
    path = []
    c_theta = -np.pi/2 - theta # translate arc distance to y-heading
    if c_theta > np.pi: c_theta -= 2*np.pi
    if c_theta < -np.pi: c_theta += 2*np.pi
    if c[0] > 0: # RIGHT
        start_ang = np.pi # right arc starts at pi
        d_theta = c_theta - start_ang
        if d_theta<0: d_theta += 2*np.pi # total change in arc
    else:
        start_ang = 0 # left arc starts at 0
        d_theta = c_theta-2*np.pi if c_theta > 0 else c_theta
        
    arc_dist = r*abs(d_theta) # distance traveled along arc
    n_points = math.ceil(arc_dist/step_size)+1 # discretize arc into thetas
    thetas = np.linspace(start_ang, start_ang+d_theta, n_points)
    for t in thetas:
        c_point = c+np.array([r*np.cos(t), r*np.sin(t)]) # find point on circle in local coordinate system
        c_point = transform@np.hstack([c_point,1]) # transform to world
        v = transform@np.cross([0,0,np.sign(c[0])],[r*np.cos(t),r*np.sin(t),0]) # find heading using cross product. right turn cw, left ccw
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0]) # convert into y-oriented heading
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)
        
    final_point = c+np.array([r*np.cos(thetas[-1]), r*np.sin(thetas[-1])]) # last point on circle to transition into straight line
    D = np.hypot(x_rand[0]-final_point[0], x_rand[1]-final_point[1]) 
    v = transform@[x_rand[0]-final_point[0], x_rand[1]-final_point[1], 0]
    line_ang = np.atan2(x_rand[1]-final_point[1], x_rand[0]-final_point[0])
    corrected_heading = -np.pi/2 - math.atan2(v[1], v[0])
    if corrected_heading > np.pi: corrected_heading -= 2*np.pi
    if corrected_heading < -np.pi: corrected_heading += 2*np.pi
    n_points = math.ceil(D/step_size) + 1
    rs = np.linspace(0, D, n_points) # discretize radius from final point on circle to goal
    for new_r in rs:
        c_point = final_point+np.array([new_r*np.cos(line_ang), new_r*np.sin(line_ang)])
        c_point = transform@np.hstack([c_point,1])
        c_point[2] = corrected_heading
        path.append(c_point)
    return path, arc_dist+D

def interpolate_path_circle(c, c2, r, x_rand, theta1, theta2, transform, step_size):
    """
    Return points along a turn-turn path at spacing step_size. Will transform points from local coordinate space into world coordinates.
    Returns list of nodes including heading and total distance of path.
    
    :param c: Coordinates of the first turn circle
    :param c2: Coordinates of the second turn circle
    :param r: Turning radius of the car
    :param x_rand: Desired point in local coordinate system
    :param theta1: Final arc position on first circle
    :param theta2: Final arc position on second circle
    :param transform: 3x3 transformation matrix from local coordinates to world.
    :param step_size: Distance between nodes in path
    """
    path = []
    if c[0] > 0: # RIGHT
        start_ang = np.pi # right starts at pi
        d_theta = theta1 + start_ang
    else:
        start_ang = 0
        d_theta = theta1 if theta1<0 else theta1-2*np.pi
        
    arc_dist = r*abs(d_theta)
    n_points = math.ceil(arc_dist/step_size)+1
    thetas = np.linspace(start_ang, start_ang+d_theta, n_points) # discretize arc
    for t in thetas:
        c_point = c+np.array([r*np.cos(t), r*np.sin(t)]) # find point on circle in local coordinate system
        c_point = transform@np.hstack([c_point,1]) # transform to world
        v = transform@np.cross([0,0,np.sign(c[0])],[r*np.cos(t),r*np.sin(t),0]) # find heading using cross product. right turn cw, left ccw
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0]) # convert into y-oriented heading
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)
    
    # Find points along second circle. Only differnce is starting angle
    start_ang = math.atan2(c[1]-c2[1],c[0]-c2[0]) # starting angle of second circle is angle between circle centers
    if c[0] > 0:
        if start_ang < 0:
            d_theta = -theta2
        else: d_theta = -2*np.pi + theta2
    else:
        if start_ang < 0:
            d_theta = theta2
        else: d_theta = 2*np.pi - theta2
    arc_dist2 = r*abs(d_theta)
    n_points = math.ceil(arc_dist2/step_size)+1
    thetas = np.linspace(start_ang, start_ang+d_theta, n_points)
    for t in thetas:
        c_point = c2+np.array([r*np.cos(t), r*np.sin(t)])
        c_point = transform@np.hstack([c_point,1])
        v = transform@np.cross([0,0,np.sign(c2[0])],[r*np.cos(t),r*np.sin(t),0])
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0])
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)    
    return path, arc_dist+arc_dist2

def p_dubins_connect(start, goal, r, step_size, img=None):
    """
    Given start node (with heading) find shortest path to goal position using relaxed dubins paths. Options are turn-straight or turn-turn. 
    This function converts the points into a local coordinate system centered around the start location.
    Returns path and cost to travel.
    Will return None if the start location is ahead of the goal.
    
    :param start: Starting location for path. Must include heading
    :param goal: Goal position to reach
    :param r: Turning radius of car
    :param step_size: Desired step size for path
    :param img: Image to display paths to (OPTIONAL)
    """
    transform = np.array([[np.cos(-start[2]),-np.sin(-start[2]),start[0]], # transformation from local coordinate frame to world
                          [np.sin(-start[2]),np.cos(-start[2]),start[1]],
                          [0,0,1]])
    
    n_x_rand = np.linalg.inv(transform)@np.hstack([goal,1]) # convert random point into local frame

    if (n_x_rand[1]**2 + (abs(n_x_rand[0]) - r)**2) >= r**2 and n_x_rand[1] < 0: # If random point is outside radius of either left/right turns choose turn-straight path
        if n_x_rand[0] >= 0: #RS
            c = np.array([r,0])
        else: #LS
            c = np.array([-r,0])
            
        D = math.sqrt((n_x_rand[0]-c[0])**2 + (n_x_rand[1]-c[1])**2)
        phi = math.atan2((c[0]-n_x_rand[0]), (c[1]-n_x_rand[1]))
        alpha = math.acos(r/D)
        theta = [phi+alpha, phi-alpha]
        t1 = c+[r*math.cos(-np.pi/2-theta[0]), r*math.sin(-np.pi/2-theta[0])] # determine tangent points from goal to circle
        t2 = c+[r*math.cos(-np.pi/2-theta[1]), r*math.sin(-np.pi/2-theta[1])]
        i = find_best_tang([t1,t2], c) # find closest kinematically feasible tangent
        best_theta = theta[i]
        path, length = interpolate_path_straight(c,r,n_x_rand, best_theta, transform, step_size) # generate points along path
        
    elif n_x_rand[1] < 0: # If inside turning radius of either left/right turns, must execute turn-turn to reach.
        if n_x_rand[0] >= 0: #LR
            c = np.array([-r,0])
        else: #RL
            c = np.array([r,0])
        D = math.sqrt((n_x_rand[0]-c[0])**2 + (n_x_rand[1]-c[1])**2)
        psi = math.atan2(c[1]-n_x_rand[1], c[0]-n_x_rand[0]) # find position and arcs to reach point.
        alpha = math.acos((4*r**2 - r**2 - D**2)/(-2*r*D))
        theta2 = math.acos((D**2 - r**2 - 4*r**2)/(-4*r**2)) 
        alpha *= -np.sign(c[0]) # account for cw-ccw turns
        c2 = n_x_rand[:2] + np.array([r*math.cos(psi+alpha),r*math.sin(psi+alpha)])
        theta1 = math.atan2(c2[1]-c[1],c2[0]-c[0]) # find arc of first circle 
        path, length = interpolate_path_circle(c,c2,r,n_x_rand, theta1, theta2, transform, step_size) # generate points along path
    else:
        path, length = None, None
    
    if img is not None and path is not None:
        for p in path:
            cv2.circle(img, p[:2].astype(np.int32), 1, (0,0,255))
                
        cv2.circle(img, (round(start[0]), round(start[1])), 1, (0,255,0), -1)
        cv2.circle(img, (round(goal[0]), round(goal[1])), 1, (255,255,255), 1)
        
        cv2.imshow('paths', img)
        cv2.waitKey(10)
    return path, length

if __name__ == '__main__':
    img = np.zeros((100,100,3),dtype=np.uint8)
    x_init = np.array([15,25,-3*np.pi/4])
    r = 5
    x_rand = np.array([30,23])
    path, length = p_dubins_connect(x_init, x_rand, r, 1, img)
    print(length)