# -------NO LONGER USED UPGRADED WITH ATTITUDE FILTER KEPT FOR LEARNING PURPOSES --------

import numpy as np

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

class GyroIntegrator:
    def __init__(self):
        # R is our current orientation matrix
        self.R = np.eye(3, dtype=np.float64)

        #Stores previous timestamp so we can compute dt
        self.last_t_us = None

    def update(self, sample):
        """
        sample: dict with gx, gy, gz in rad/s and t_us timestamp in microseconds
        Returns updated R and dt or R, None if no dt yet """

        # Pull timestamp & Gyro vector
        t_us = sample["t_us"]
        w = np.array([sample["gx"], sample["gy"], sample["gz"]], dtype=np.float64)


        if self.last_t_us is None:
            self.last_t_us = t_us
            return self.R, None
        
        # Convert microseconds to seconds
        dt = (t_us - self.last_t_us) * 1e-6
        self.last_t_us = t_us

        # this is the small angle approximation essentially in a tiny time dt I rotated by w dt
        self.R = (np.eye(3) + skew(w)*dt) @ self.R
        return self.R, dt