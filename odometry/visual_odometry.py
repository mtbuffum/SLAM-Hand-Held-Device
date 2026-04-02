import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, K):
        self.K = K

    def estimate_motion(self, pts_prev, pts_curr, parallax_thresh_px=2.0):
        """
        pts_prev, pts_curr: (N,2) arrays of matched points in PIXELS
        Returns:
          R (3,3), t (3,1), mask_pose (N,1 uint8), stats dict
        """

        if pts_prev is None or pts_curr is None:
            return None, None, None, {"reason": "no_points"}

        pts_prev = np.asarray(pts_prev, dtype=np.float32).reshape(-1, 2)
        pts_curr = np.asarray(pts_curr, dtype=np.float32).reshape(-1, 2)

        N = len(pts_prev)
        if N < 8:
            return None, None, None, {"reason": "too_few_points", "N": N}

        # --- Parallax metric (median flow magnitude) ---
        flow = pts_curr - pts_prev
        parallax = float(np.median(np.linalg.norm(flow, axis=1)))

        if parallax < 1.5:
            return None, None, None, {"reason": "low_parallax", "parallax_px": parallax, "N": N}
        
        # --- Estimate Essential Matrix (RANSAC) ---
        E, mask_E = cv2.findEssentialMat(
            pts_prev, pts_curr,
            self.K,
            method=cv2.USAC_MAGSAC,
            prob=0.999,
            threshold=3.0,   # 1–2 px is the right regime for 640x480-ish
        )

        if E is None or mask_E is None:
            return None, None, None, {"reason": "E_failed", "parallax": parallax}

        mask_E = mask_E.reshape(-1, 1).astype(np.uint8)
        mask_E = (mask_E > 0).astype(np.uint8)

        # Handle multiple E solutions: E can be 3x3 or 3x(3k)
        E_candidates = []
        if E.shape == (3, 3):
            E_candidates = [E]
        else:
            # e.g. (3, 9), (3, 12), ...
            for i in range(E.shape[1] // 3):
                E_candidates.append(E[:, 3*i:3*(i+1)])

        best = None
        best_inliers = -1

        for Ei in E_candidates:
            # IMPORTANT: pass the mask into recoverPose
            _, R, t, mask_pose = cv2.recoverPose(
                Ei, pts_prev, pts_curr, self.K, mask=mask_E
            )
            if mask_pose is None:
                continue

            inl = int(mask_pose.sum())  # mask_pose is 0/1
            if inl > best_inliers:
                best_inliers = inl
                best = (R, t, mask_pose)

        if best is None:
            return None, None, None, {"reason": "recoverPose_failed", "parallax": parallax}

        R, t, mask_pose = best
        mask_pose = mask_pose.reshape(-1, 1).astype(np.uint8)

        # --- Stats (this will immediately tell us if things are real) ---
        n_inliers_E = int(mask_E.sum() // 255) if mask_E.max() == 255 else int(mask_E.sum())
        n_inliers_pose = int(mask_pose.sum())  # 0/1
        
        # if n_inliers_E < 30:
        #     return None, None, None, {"reason" : "low_inliers"}
        stats = {
            "N": N,
            "parallax_px": parallax,
            "inliers_E": n_inliers_E,
            "inliers_pose": n_inliers_pose,
            "ratio_E": n_inliers_E / max(1, N),
            "ratio_pose": n_inliers_pose / max(1, N),
            "detR": float(np.linalg.det(R)),
            "translation_norm": float(np.linalg.norm(t)),
            "parallax_gate_pass": bool(parallax >= parallax_thresh_px),
        }

        # --- Optional: if parallax too small, kill translation (rotation-only update) ---
        if parallax < parallax_thresh_px:
            t = np.zeros((3, 1), dtype=np.float64)

        return R, t, mask_pose, stats