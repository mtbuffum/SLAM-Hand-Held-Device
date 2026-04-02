import numpy as np

#forces any angle into (-pi, pi] 
# which is important beause when we have angles like -179 and 179 they are very similar angles but numerically far apart
def wrap_pi(a: float) -> float:
    return (a+np.pi) % (2*np.pi)-np.pi

#The next three functions use the standard rotation matrices about their given axis
#important for the euler_zyx method because we set R = Rz(yaw), Ry(pitch), Rx(roll)
def rotz(yaw: float) -> np.ndarray:
    c, s= np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)

def roty(pitch: float) -> np.ndarray:
    c, s= np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def rotx(roll: float) -> np.ndarray:
    c, s= np.cos(roll), np.sin(roll)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)

# Most important function saying given R extract yaw, pitch and roll such that R = Rz(yaw) * Ry(pitch) * Rx(roll)
def euler_zyx_from_R(R: np.ndarray):
    """
    Returns (yaw,pitch,roll) for zyx convention
    R is body --> world
    """

    # pitch = asin(-R[2,0]) where -R[2,0] is 3rd row (World Z component) and 1st column (body x axis)
    # np.clip juts clips any outliers of the specified range np.clip(a, min, max) if any value in a falls out of that range it gets clipped
    pitch = np.arcsin(np.clip(-R[2,0], -1.0, 1.0)) 
    #handle near gimbal lock 
    # when pitch is around + - 90 degrees cos(pitch) ~0 meaning yaw and roll become etangled and unrecoverable so this is our fall back 
    # the approach is pick roll = 0 and solve yaw from remaining matrix entities
    if abs(np.cos(pitch)) < 1e-6:
        yaw = np.arctan2(-R[0,1], R[1,1])
        roll = 0.0
    # This is the normal case for yaw and roll
    else:
        yaw = np.arctan2(R[1,0], R[0,0]) # np.arctan2 is better then np.arctan because it handles quadrants correctly and avoids division by zero
        roll = np.arctan2(R[2,1], R[2,2])

    return wrap_pi(yaw), wrap_pi(pitch), wrap_pi(roll)

#This function actually constructs the values as R= Rz(yaw)*Ry(pitch)*Rx(roll)
# that form is critical because we want yaw= fused yaw, and pitch/roll = IMU pitch/roll
def R_from_euler_zyx(yaw, pitch, roll):
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

# a is yaw from IMU and b is yaw from Vision, we gently pull IMU yaw to vision yaw each frame
def blend_angles(a, b, beta):
    """
    Blend angles with wrap-around safety
    Returns angle close to a, nudged toward b
    """
    #essentially to blend we find the shortest angular difference
    d= wrap_pi(b-a)
    # and then we step toward b 
    return wrap_pi(a + beta * d)