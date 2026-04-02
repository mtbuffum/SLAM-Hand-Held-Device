import numpy as np

class Map3D:
    def __init__(self):
        self.points = []

    def add_points(self, pts_3d):
        for p in pts_3d:
            self.points.append(p.reshape(3,1))

    def get_points_array(self):
        if len(self.points) == 0:
            return None
        return np.hstack(self.points)