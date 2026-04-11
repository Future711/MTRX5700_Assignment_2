import open3d as o3d
import numpy as np


def load_pcd_points(pcd_path):
    pc = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pc.points, dtype=float)


def project_lidar_points(points_lidar, K, T_cam_lidar, img_shape):
    n = points_lidar.shape[0]
    pts_h = np.hstack([points_lidar, np.ones((n, 1))])
    pts_cam = (T_cam_lidar @ pts_h.T).T
    pts_cam = pts_cam[:, :3]

    h, w = img_shape[:2]
    z = pts_cam[:, 2]
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

    depths = np.sqrt(points_lidar[:, 0] ** 2 + points_lidar[:, 1] ** 2)
    return np.column_stack([u, v]), depths, in_image


def estimate_distance(points_lidar, K, T_cam_lidar, cone_mask):
    if points_lidar.shape[0] == 0:
        return None

    pixels, depths, valid = project_lidar_points(points_lidar, K, T_cam_lidar, cone_mask.shape)
    if not np.any(valid):
        return None

    h, w = cone_mask.shape[:2]
    ui = np.round(pixels[valid, 0]).astype(int)
    vi = np.round(pixels[valid, 1]).astype(int)
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    on_mask = np.zeros(len(ui), dtype=bool)
    on_mask[in_bounds] = cone_mask[vi[in_bounds], ui[in_bounds]] > 0

    hit_depths = depths[valid][on_mask]
    if len(hit_depths) == 0:
        return None

    return float(np.median(hit_depths))