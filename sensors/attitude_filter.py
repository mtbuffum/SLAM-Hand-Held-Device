import numpy as np

# note same function in integrator
def skew(w):
    """Converts a vector w into a matrix w x Cross product matrix
    It essentially lets us write small-angle rotation updates compactly
    If w is our rotation axis * rate then w x Cross product matrix is the operator that encodes the rotate around w
    """
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0,  wx],
        [-wy, wx, 0],
    ], dtype=np.float64)

def normalize(v, eps= 1e-9):
    n = np.linalg.norm(v)
    return v/ (n + eps)

def rot_from_axis_angle(axis, angle):
    # Rodrigues algorithm
    K = skew(axis)
    I = np.eye(3)
    return I + np.sin(angle)*K + (1 - np.cos(angle)) * (K @ K)


class ComplimentaryAttitude:
    """
    Maintains R_WB (world rotation axis body --> world)
    Uses Gyro integration + accelerometer gravity correction
    """
    def __init__(self, alpha = 0.98):
        self.R_wb = np.eye(3, dtype=np.float64)
        self.last_t_us = None
        self.alpha = alpha

    def update(self, sample):

        t_us = sample["t_us"]
        w = np.array((sample["gx"], sample["gy"], sample["gz"]), dtype=np.float64)
        a = np.array((sample["ax"], sample["ay"], sample["az"]), dtype=np.float64)

        if self.last_t_us is None:
            self.last_t_us = t_us
            return self.R_wb, None
        
        dt = (t_us - self.last_t_us) * 1e-6
        self.last_t_us = t_us

        # Gyro Propogate R < -- R * exp([w] dt)
        dR = np.eye(3) + skew(w) * dt
        self.R_wb = self.R_wb @ dR

        # Re Othronormalize (See notes for info; Pages: 32 - 40)
        U, _, Vt = np.linalg.svd(self.R_wb)
        self.R_wb = U @ Vt

        # Accelerometer correction toward gravity
        g_meas_b = normalize(a)
        g_world = np.array([0, 0, 1.0])

        #predicted gracity direction in body given current R_wb
        # v_b = R_bw v_w, and R_bw = R_wb^T
        g_pred_b = normalize(self.R_wb.T @ g_world)

        #rotation that aligns g_pred_b -> g_meas_b in body frame
        axis = np.cross(g_pred_b, g_meas_b)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))

            #apply small correction
            corr = rot_from_axis_angle(axis, (1 - self.alpha)* angle)
            #correction is in body frame -> right multiply R_wb
            self.R_wb = self.R_wb @ corr

        return self.R_wb, dt


        
