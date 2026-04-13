"""LiDAR projection and distance-estimation helper functions."""

import open3d as o3d
import numpy as np


def load_pcd_points(pcd_path):
    """Load points from a PCD file.

    Args:
        pcd_path: Path to the PCD file.

    Returns:
        (N, 3) numpy array of point coordinates as floats.
    """
    pc = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pc.points, dtype=float)


def project_lidar_points(points_lidar, K, T_cam_lidar, img_shape):
    """Project 3-D LiDAR points into the camera image plane.

    Applies the extrinsic transform T_cam_lidar (4x4) to convert points from
    the LiDAR frame to the camera frame, then applies the intrinsic matrix K
    to obtain pixel coordinates (u, v).  Points behind the camera (z <= 0) and
    points that fall outside the image boundary are flagged as invalid.

    Args:
        points_lidar:  (N, 3) array of LiDAR points in the sensor frame.
        K:             (3, 3) camera intrinsic matrix.
        T_cam_lidar:   (4, 4) extrinsic transform from LiDAR to camera frame.
        img_shape:     (H, W[, C]) shape of the target image.

    Returns:
        Tuple containing:
            pixels: (N, 2) array of (u, v) pixel coordinates (NaN for invalid).
            depths: (N,) 2-D Euclidean distance from the LiDAR origin (x-y plane).
            in_image: (N,) boolean mask; True where a point projects into the image.
    """
    n = points_lidar.shape[0]
    # Homogenise points and transform to camera frame
    pts_h = np.hstack([points_lidar, np.ones((n, 1))])
    pts_cam = (T_cam_lidar @ pts_h.T).T
    pts_cam = pts_cam[:, :3]

    h, w = img_shape[:2]
    z = pts_cam[:, 2]
    # Only project points in front of the camera (positive depth)
    valid_z = z > 1e-6

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    u[valid_z] = fx * pts_cam[valid_z, 0] / z[valid_z] + cx
    v[valid_z] = fy * pts_cam[valid_z, 1] / z[valid_z] + cy

    in_image = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )

    # 2-D ground-plane distance (ignores height) as a proxy for range
    depths = np.sqrt(points_lidar[:, 0] ** 2 + points_lidar[:, 1] ** 2)
    return np.column_stack([u, v]), depths, in_image


def estimate_distance(points_lidar, K, T_cam_lidar, cone_mask):
    """Estimate the distance to a cone using LiDAR points projected onto its mask.

    Projects the LiDAR scan into the image plane, finds which points land inside
    the cone silhouette mask, and returns the median 2-D range of those hits.
    The median is used instead of the mean to be robust against a small number of
    spurious close or far returns (e.g. reflections or ground points).

    Args:
        points_lidar:  (N, 3) LiDAR points in the sensor frame.
        K:             (3, 3) camera intrinsic matrix.
        T_cam_lidar:   (4, 4) LiDAR-to-camera extrinsic transform.
        cone_mask:     (H, W) binary mask where non-zero pixels belong to the cone.

    Returns:
        Median distance in metres, or None if no LiDAR points hit the mask.
    """
    if points_lidar.shape[0] == 0:
        return None

    pixels, depths, valid = project_lidar_points(points_lidar, K, T_cam_lidar, cone_mask.shape)
    if not np.any(valid):
        return None

    h, w = cone_mask.shape[:2]
    ui = np.round(pixels[valid, 0]).astype(int)
    vi = np.round(pixels[valid, 1]).astype(int)
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    # Check which projected points fall inside the cone region mask
    on_mask = np.zeros(len(ui), dtype=bool)
    on_mask[in_bounds] = cone_mask[vi[in_bounds], ui[in_bounds]] > 0

    hit_depths = depths[valid][on_mask]
    if len(hit_depths) == 0:
        return None

    return float(np.median(hit_depths))