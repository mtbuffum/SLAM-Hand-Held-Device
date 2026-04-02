import matplotlib.pyplot as plt 
import numpy as np

class TrajectoryPlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Camera Trajectory (Top-Down)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.axis("equal")

    def update(self, trajectory):
        if len(trajectory) < 2:
            return
        
        pts = np.hstack(trajectory)
        x = pts[0, :]
        z = pts[2, :]

        self.ax.clear()
        self.ax.plot(x, z, "-b")
        self.ax.scatter(x[-1], z[-1], c="r", label="Current")
        self.ax.legend()
        self.ax.set_title("Camera Trajectory (Top-Down)")
        self.ax.axis("equal")
        plt.pause(0.001)
        
