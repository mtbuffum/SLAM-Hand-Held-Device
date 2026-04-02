import cv2, numpy as np

CHECKER = (9, 6)         # inner corners
SQUARE = 0.024           # meters (measure yours!)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

objp = np.zeros((CHECKER[0]*CHECKER[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKER[0], 0:CHECKER[1]].T.reshape(-1,2)
objp *= SQUARE

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(1)  # adjust index if needed 
print("Press SPACE to capture when corners are detected. Press ESC to finish.")

while True:
    ret, frame = cap.read()
    if not ret: continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ok, corners = cv2.findChessboardCorners(gray, CHECKER)
    vis = frame.copy()

    if ok:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis, CHECKER, corners2, ok)

    cv2.imshow("calib", vis)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # ESC
        break
    if k == 32 and ok:  # SPACE
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        print("Captured", len(objpoints))

cap.release()
cv2.destroyAllWindows()

assert len(objpoints) >= 10, "Need at least ~10 good captures"

h, w = gray.shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print("RMS reprojection error:", ret)
print("K:\n", K)
print("dist:\n", dist.ravel())

np.savez("camera_calib.npz", K=K, dist=dist, wh=np.array([w,h]))
print("Saved camera_calib.npz")