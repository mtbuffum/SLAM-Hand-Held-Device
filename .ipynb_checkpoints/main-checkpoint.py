import cv2
import numpy as np
from sensors.camera import Camera
from odometry.feature_tracker import FeatureTracker
from odometry.visual_odometry import VisualOdometry
from visualization.plotter import TrajectoryPlotter


def main():
    # Intrinsnic array see page 10 & 11 in PDF notes to understand more
    K = np.array([
        [700,   0, 320],
        [  0, 700, 240],
        [  0,   0,   1]
    ], dtype=np.float64)

    # Global pose
    #Global Rotation
    R_wc = np.eye(3)
    # Global Position
    p_wc = np.zeros((3,1))
    # List of postions for plotting
    trajectory = []

    cam = Camera()
    tracker = FeatureTracker()
    vo = VisualOdometry(K)
    plotter = TrajectoryPlotter()

    print("Press q to quit")
    while True:
        frame = cam.read()
        if frame is None:
            print("failed to read frame.")
            

        vis, n_tracks, prev_pts, curr_pts = tracker.process(frame)
        
        if prev_pts is not None and curr_pts is not None:
            R, t, inliers = vo.estimate_motion(prev_pts, curr_pts)
            if inliers is not None:
                inlier_count = int(inliers.sum())
                total = len(prev_pts)
                ratio = inlier_count / max(total, 1)
                cv2.putText(vis, f"inliers: {inlier_count}/{total} ({ratio:.2f})", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if R is not None:
                # Arbituary scale 
                scale = 0.05

                #update position and rotation
                p_wc = p_wc + scale * (R_wc @ t)
                R_wc = R @ R_wc

                trajectory.append(p_wc.copy())

                plotter.update(trajectory)
                
                cv2.putText(
                    vis, "VO: motion Estimated", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
        
        cv2.imshow("Feature Tracking", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()