import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, max_corners=800, quality_level=0.01, min_distance=8):
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize= 7,
        )

        self.lk_params = dict(
            winSize=(21,21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 30, 0.01),
        )

        self.prev_gray = None
        self.prev_pts = None
    
    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def _detect(self,gray):
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        return pts

    def process(self, frame_bgr):
        """Returns:
                vis_frame: frame with tracks drawn
                n_tracks: number of successfully tracked points
                prev_pts_good, curr_pts_good: (N,2) float arrays (or None)
                """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        #detect features
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return frame_bgr, 0, None, None
        
        #if we lost features re detect
        if self.prev_pts is None or len(self.prev_pts) < 100:
            self.prev_pts = self._detect(self.prev_gray)

        if self.prev_pts is None:
            #no features found at all
            self.prev_gray = gray
            return frame_bgr, 0, None, None
        
        #track features
        curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
        
        status = status.reshape(-1)
        prev_good = self.prev_pts[status ==1].reshape(-1,2)
        curr_good = curr_pts[status == 1].reshape(-1,2)

        # back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        #     gray, self.prev_gray, curr_pts, None, **self.lk_params
        # )

        # back_status = back_status.reshape(-1)
        # fb_err = np.linalg.norm(self.prev_pts.reshape(-1,2) - back_pts.reshape(-1,2), axis = 1)

        # good = (status == 1) & (back_status) & (fb_err < 1.0)
        # prev_good = self.prev_pts.reshape(-1,2)[good]
        # curr_good = curr_pts.reshape(-1,2)[good]

        vis = frame_bgr.copy()

        #draw tracks
        for (x0, y0), (x1,y1) in zip(prev_good, curr_good):
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 1)
            cv2.circle(vis, (int(x1), int(y1)), 2, (0,255,0), -1)
        
        n_tracks = len(curr_good)

        #update state
        self.prev_gray = gray
        self.prev_pts = curr_good.reshape(-1,1,2).astype(np.float32)

        return vis, n_tracks, prev_good, curr_good