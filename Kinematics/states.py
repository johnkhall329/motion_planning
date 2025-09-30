import math
# from sim import CAR_LENGTH, TURNING_RADIUS
CAR_LENGTH = 20

class State():
    def __init__(self,x,y,v,theta):
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta

    def round_state(self):
        return (self.x%30, self.y%30) # WIlL NEED A SCALE

def control_input(state:State,U,dt=0.1):
    theta, a = U
    v_dot = a  # acceleration
    v = state.v + v_dot * dt
    phi_dot = state.v * math.tan(theta) / CAR_LENGTH  # simple bicycle mode l
    theta = state.theta + phi_dot * dt
    x = v * -math.sin(theta)
    y = v * math.cos(theta)
    return State(x,y,v,theta)
    