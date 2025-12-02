import math
import numpy as np
import cv2

def find_best_tang(tangents, c):
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
    return tangents[best], best

def interpolate_path_straight(c, r, x_rand, theta, transform, step_size):
    path = []
    c_theta = -np.pi/2 - theta
    if c_theta > np.pi: c_theta -= 2*np.pi
    if c_theta < -np.pi: c_theta += 2*np.pi
    if c[0] > 0:
        start_ang = np.pi
        d_theta = c_theta - start_ang
        if d_theta<0: d_theta += 2*np.pi
    else:
        start_ang = 0
        d_theta = c_theta-2*np.pi if c_theta > 0 else c_theta
        
    arc_dist = r*abs(d_theta)
    n_points = math.ceil(arc_dist/step_size)+1
    thetas = np.linspace(start_ang, start_ang+d_theta, n_points)
    for t in thetas:
        c_point = c+np.array([r*np.cos(t), r*np.sin(t)])
        c_point = transform@np.hstack([c_point,1])
        v = np.cross([0,0,np.sign(c[0])],[r*np.cos(t),r*np.sin(t),0])
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0])
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)
    final_point = c+np.array([r*np.cos(thetas[-1]), r*np.sin(thetas[-1])])
    D = np.hypot(x_rand[0]-final_point[0], x_rand[1]-final_point[1])
    line_ang = np.atan2(x_rand[1]-final_point[1], x_rand[0]-final_point[0])
    corrected_heading = -np.pi/2 - line_ang
    if corrected_heading > np.pi: corrected_heading -= 2*np.pi
    if corrected_heading < -np.pi: corrected_heading += 2*np.pi
    n_points = math.ceil(D/step_size) + 1
    rs = np.linspace(0, D, n_points)
    for new_r in rs:
        c_point = final_point+np.array([new_r*np.cos(line_ang), new_r*np.sin(line_ang)])
        c_point = transform@np.hstack([c_point,1])
        c_point[2] = corrected_heading
        path.append(c_point)
    return path, arc_dist+D

def interpolate_path_circle(c, c2, r, x_rand, theta1, theta2, transform, step_size):
    path = []
    if c[0] > 0:
        start_ang = np.pi
        d_theta = theta1 + start_ang
    else:
        start_ang = 0
        d_theta = theta1 if theta1<0 else theta1-2*np.pi
        
    arc_dist = r*abs(d_theta)
    n_points = math.ceil(arc_dist/step_size)+1
    thetas = np.linspace(start_ang, start_ang+d_theta, n_points)
    for t in thetas:
        c_point = c+np.array([r*np.cos(t), r*np.sin(t)])
        c_point = transform@np.hstack([c_point,1])
        v = np.cross([0,0,np.sign(c[0])],[r*np.cos(t),r*np.sin(t),0])
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0])
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)
    
    start_ang = math.atan2(c[1]-c2[1],c[0]-c2[0])
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
        v = np.cross([0,0,np.sign(c2[0])],[r*np.cos(t),r*np.sin(t),0])
        corrected_heading = -np.pi/2 - math.atan2(v[1], v[0])
        if corrected_heading > np.pi: corrected_heading -= 2*np.pi
        if corrected_heading < -np.pi: corrected_heading += 2*np.pi
        c_point[2] = corrected_heading
        path.append(c_point)    
    return path, arc_dist+arc_dist2

def p_dubins_connect(start, goal, r, step_size, img=None):
    transform = np.array([[np.cos(start[2]),np.sin(start[2]),start[0]],
                          [-np.sin(start[2]),np.cos(start[2]),start[1]],
                          [0,0,1]])
    
    n_x_rand = np.linalg.inv(transform)@np.hstack([goal,1])

    if (n_x_rand[1]**2 + (abs(n_x_rand[0]) - r)**2) >= r**2:
        if n_x_rand[0] >= 0: #RS
            # print("RS")
            c = np.array([r,0])
        else: #LS
            # print("LS")
            c = np.array([-r,0])
        D = math.sqrt((n_x_rand[0]-c[0])**2 + (n_x_rand[1]-c[1])**2)
        phi = math.atan2((c[0]-n_x_rand[0]), (c[1]-n_x_rand[1]))
        alpha = math.acos(r/D)
        theta = [phi+alpha, phi-alpha]
        t1 = c+[r*math.cos(-np.pi/2-theta[0]), r*math.sin(-np.pi/2-theta[0])]
        t2 = c+[r*math.cos(-np.pi/2-theta[1]), r*math.sin(-np.pi/2-theta[1])]
        t,i = find_best_tang([t1,t2], c)
        best_theta = theta[i]
        path, length = interpolate_path_straight(c,r,n_x_rand, best_theta, transform, step_size)
        
    else:
        if n_x_rand[0] >= 0: #LR
            # print("LR")
            c = np.array([-r,0])
        else: #RL
            # print("RL")
            c = np.array([r,0])
        D = math.sqrt((n_x_rand[0]-c[0])**2 + (n_x_rand[1]-c[1])**2)
        psi = math.atan2(c[1]-n_x_rand[1], c[0]-n_x_rand[0])
        alpha = math.acos((4*r**2 - r**2 - D**2)/(-2*r*D))
        theta2 = math.acos((D**2 - r**2 - 4*r**2)/(-4*r**2))
        alpha *= -np.sign(c[0])
        c2 = n_x_rand[:2] + np.array([r*math.cos(psi+alpha),r*math.sin(psi+alpha)])
        theta1 = math.atan2(c2[1]-c[1],c2[0]-c[0])
        path, length = interpolate_path_circle(c,c2,r,n_x_rand, theta1, theta2, transform, step_size)
    
    if img is not None:
        for p in path:
            cv2.circle(img, p[:2].astype(np.int32), 1, (0,0,255))
                
        cv2.circle(img, (round(start[0]), round(start[1])), 1, (0,255,0), -1)
        cv2.circle(img, (round(goal[0]), round(goal[1])), 1, (255,255,255), 1)
        
        cv2.imshow('progress', img)
        cv2.waitKey(1)
    return path, length

if __name__ == '__main__':
    img = np.zeros((100,100,3),dtype=np.uint8)
    x_init = np.array([15,25,0])
    r = 5
    x_rand = np.array([30,23,1])
    path, length = p_dubins_connect(x_init, x_rand, r, img)
    print(length)