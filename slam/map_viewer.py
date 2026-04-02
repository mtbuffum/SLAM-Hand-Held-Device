import numpy as np 
import cv2

class MapViewer:
    def __init__(self, size= 800, scale = 50):
        self.size = size
        self.scale = scale
        self.center= size //2

    def draw(self, trajectory, map_points):
        canvas = np.zeros((self.size, self.size, 3), dtype= np.uint8)

        #Draw map points
        if map_points is not None:
            for p in map_points.T:
                x = int(self.center + p[0] * self.scale)
                z = int (self.center + p[2] * self.scale)

                if 0 <= x < self.size and 0 <= z < self.size:
                    canvas[z, x] = (255, 255, 255)

        # Draw Trajectory
        if len(trajectory) > 1: 
            pts = np.hstack(trajectory)
            xs = pts[0]
            zs = pts[2]

            for i in range(len(xs)-1):
                x1 = int(self.center + xs[i] * self.scale)
                z1 = int(self.center + zs[i] * self.scale)
                x2 = int(self.center + xs[i+1] * self.scale)
                z2 = int(self.center + zs[i+1] * self.scale)
                cv2.line(canvas, (x1, z1), (x2, z2), (0,255,0), 2)
        return canvas