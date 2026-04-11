import json
import numpy as np

def load_calibration(calib_file):
    with open(calib_file, "r") as f:
        calib = json.load(f)
    K = np.array(calib["K"], dtype=float)
    T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=float)
    
    return K, T_cam_lidar