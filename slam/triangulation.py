import cv2
import numpy as np

def triangulate_points(K, R, t, pts1, pts2):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R,t))

    pts1 = pts1.T
    pts2 = pts2.T

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d[:3] / pts4d[3]

    return pts3d.T

def reprojection_filter(K, R, t, pts3d, pts1, pts2, max_err_px=2.5):
    # P1 and P2 for reprojection
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # Homogeneous
    X = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T  # (4,N)

    x1 = (P1 @ X); x1 = (x1[:2] / x1[2]).T
    x2 = (P2 @ X); x2 = (x2[:2] / x2[2]).T

    e1 = np.linalg.norm(x1 - pts1.reshape(-1,2), axis=1)
    e2 = np.linalg.norm(x2 - pts2.reshape(-1,2), axis=1)

    mask = (e1 < max_err_px) & (e2 < max_err_px)
    return mask