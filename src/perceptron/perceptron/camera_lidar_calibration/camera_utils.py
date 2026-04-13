"""Camera calibration utility functions.

This module loads camera intrinsics and camera-to-LiDAR extrinsics from a
JSON calibration file used by the perception pipeline.
"""

import json
import numpy as np


def load_calibration(calib_file):
    """Load camera calibration parameters from a JSON file.

    The JSON must contain:
        "K"           -- 3x3 camera intrinsic matrix (focal lengths and principal point).
        "T_cam_lidar" -- 4x4 extrinsic transform from the LiDAR frame to the camera frame.

    Args:
        calib_file: Path to the JSON calibration file.

    Returns:
        Tuple containing:
            K: (3, 3) numpy float array.
            T_cam_lidar: (4, 4) numpy float array.
    """
    with open(calib_file, "r") as f:
        calib = json.load(f)
    K = np.array(calib["K"], dtype=float)
    T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=float)

    return K, T_cam_lidar