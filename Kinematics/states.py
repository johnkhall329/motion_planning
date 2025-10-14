import math
# from sim import CAR_LENGTH, TURNING_RADIUS
CAR_LENGTH = 20

class State():
    def __init__(self,x,y,v,phi):
        self.x = x
        self.y = y
        self.v = v
        self.phi = phi

    def round_state(self):
        return (self.x%30, self.y%30) # WILL NEED A SCALE
    
    def __getitem__(self, key):
        return (self.x, self.y, self.v, self.phi)[key]

def control_input(state:State,U,dt=0.1):
    theta, a = U
    v_dot = a  # acceleration
    v = state.v + v_dot * dt
    phi_dot = state.v * math.tan(theta) / CAR_LENGTH  # simple bicycle mode l
    phi = state.phi + phi_dot * dt
    x = v * -math.sin(phi)
    y = v * math.cos(phi)
    return State(x,y,v,phi)
    