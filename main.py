import cv2
import numpy as np
from sensors.camera import Camera
from odometry.feature_tracker import FeatureTracker
from odometry.visual_odometry import VisualOdometry
from sensors.imu import IMUReader
from sensors.attitude_filter import ComplimentaryAttitude
from sensors.yaw_fusion import euler_zyx_from_R, R_from_euler_zyx, blend_angles
from slam.triangulation import triangulate_points, reprojection_filter
from slam.map import Map3D
from slam.map_viewer import MapViewer


def main():
    # Camera calibration
    cal = np.load("camera_calib.npz")
    K = cal["K"]
    dist = cal["dist"]

    # Global pose
    R_wc = np.eye(3, dtype=np.float64)
    p_wc = np.zeros((3, 1), dtype=np.float64)
    trajectory = []

    # Vision-only orientation accumulator
    R_vo_wb = np.eye(3, dtype=np.float64)

    # Fusion / update params
    beta_yaw = 0.05
    frame_idx = 0

    # Demo-scale translation from relative motion
    k_parallax = 0.002
    scale_ema = 0.0
    scale_alpha = 0.9

    cam = Camera()
    tracker = FeatureTracker()
    vo = VisualOdometry(K)
    imu = IMUReader(port="COM3", baud=115200)
    att = ComplimentaryAttitude(alpha=0.98)
    world_map = Map3D()
    viewer = MapViewer(size=800, scale=30)

    print("Press q to quit")

    while True:
        frame = cam.read()
        if frame is None:
            print("failed to read frame.")
            continue

        frame = cv2.undistort(frame, K, dist)
        frame_idx += 1

        vis, n_tracks, prev_pts, curr_pts = tracker.process(frame)

        sample = imu.read_latest()
        if sample is not None:
            R_imu, dt = att.update(sample)
        else:
            R_imu = att.R_wb

        # Default IMU-only orientation values so overlays always work
        yaw_imu, pitch_imu, roll_imu = euler_zyx_from_R(R_imu)
        yaw_vo = 0.0
        yaw_fused = yaw_imu
        R_fused = R_from_euler_zyx(yaw_fused, pitch_imu, roll_imu)

        # Default VO/debug state
        R, t, inliers, stats = None, None, None, {"reason": "no_tracks"}
        good_vo = False

        if prev_pts is not None and curr_pts is not None:
            R, t, inliers, stats = vo.estimate_motion(prev_pts, curr_pts)

            # Decide if this frame is trustworthy enough to use
            good_vo = (
                R is not None
                and t is not None
                and inliers is not None
                and stats is not None
                and stats.get("parallax_px", 0.0) >= 2.0
                and stats.get("inliers_pose", 0) >= 50
                and stats.get("ratio_pose", 0.0) >= 0.20
            )

            if R is not None:
                # Accumulate vision orientation
                R_vo_wb = R_vo_wb @ R
                yaw_vo, _, _ = euler_zyx_from_R(R_vo_wb)

            # Blend yaw only on trustworthy VO frames
            if good_vo:
                yaw_fused = blend_angles(yaw_imu, yaw_vo, beta_yaw)
            else:
                yaw_fused = yaw_imu

            # Rebuild fused orientation: yaw from fused, pitch/roll from IMU
            R_fused = R_from_euler_zyx(yaw_fused, pitch_imu, roll_imu)

            # Save previous pose for mapping world transform
            R_wc_prev = R_wc.copy()
            p_wc_prev = p_wc.copy()

            # Always update orientation from fused estimate
            R_wc = R_fused

            # Only update translation on good VO frames
            if good_vo:
                raw_scale = k_parallax * stats["parallax_px"]
                raw_scale = float(np.clip(raw_scale, 0.0, 0.20))
                scale_ema = scale_alpha * scale_ema + (1.0 - scale_alpha) * raw_scale
                p_wc = p_wc + scale_ema * (R_wc @ t)

            trajectory.append(p_wc.copy())

            # Sparse mapping only on good keyframes
            if good_vo and frame_idx % 10 == 0:
                pts1_in = prev_pts[inliers.ravel() == 1]
                pts2_in = curr_pts[inliers.ravel() == 1]

                if len(pts1_in) >= 8:
                    pts3d = triangulate_points(K, R, t, pts1_in, pts2_in)
                    n_raw = len(pts3d)

                    mask = np.isfinite(pts3d).all(axis=1)
                    mask &= (pts3d[:, 2] > 0) & (pts3d[:, 2] < 8.0)
                    mask &= (np.linalg.norm(pts3d, axis=1) < 15.0)

                    pts3d = pts3d[mask]
                    pts1_m = pts1_in[mask].reshape(-1, 2)
                    pts2_m = pts2_in[mask].reshape(-1, 2)
                    n_keep = len(pts3d)

                    if len(pts3d) > 0:
                        mask2 = reprojection_filter(
                            K, R, t, pts3d, pts1_m, pts2_m, max_err_px=2.5
                        )
                        pts3d = pts3d[mask2]
                        n_reproj = len(pts3d)

                        if len(pts3d) > 0:
                            # Transform points using previous camera pose (camera 1 frame)
                            pts3d_w = (R_wc_prev @ pts3d.T) + p_wc_prev
                            world_map.add_points(pts3d_w.T)
                    else:
                        n_reproj = 0
                else:
                    n_raw, n_keep, n_reproj = 0, 0, 0

                cv2.putText(
                    vis,
                    f"triangulated: {n_raw}  kept: {n_keep}  reproj: {n_reproj}",
                    (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        # Draw map every frame
        map_points = world_map.get_points_array()
        map_img = viewer.draw(trajectory, map_points)
        cv2.imshow("SLAM Map", map_img)

        # Overlays
        cv2.putText(
            vis,
            f"VO reason: {stats.get('reason', 'ok')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"tracks: {n_tracks}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"E_inliers: {stats.get('inliers_E', 0)}  pose_inliers: {stats.get('inliers_pose', 0)}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"parallax: {stats.get('parallax_px', 0.0):.2f}px  pose_ratio: {stats.get('ratio_pose', 0.0):.2f}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"yaw_imu: {np.degrees(yaw_imu):6.1f}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"yaw_vo : {np.degrees(yaw_vo):6.1f}",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"yaw_fus: {np.degrees(yaw_fused):6.1f}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            vis,
            f"scale_ema: {scale_ema:.4f}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"good_vo: {'YES' if good_vo else 'NO'}",
            (10, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if good_vo else (0, 0, 255),
            2,
        )

        if good_vo:
            cv2.putText(
                vis,
                "VO: motion trusted",
                (10, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                vis,
                "VO: frame rejected",
                (10, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if sample is not None:
            cv2.putText(
                vis,
                "IMU: OK",
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Feature Tracking", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()