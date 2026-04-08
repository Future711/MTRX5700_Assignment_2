"""
Visualise LiDAR point projections on the image to diagnose T_cam_lidar.
Run: python3 debug_projection.py [index]
"""
import os, sys, json
import cv2
import numpy as np
import open3d as o3d

IMAGE_DIR  = "lidar_matched_images/"
PCD_DIR    = "pointcloud_left_1/"
CALIB_FILE = "calibration.json"

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

images = sorted(os.listdir(IMAGE_DIR))
pcds   = sorted(os.listdir(PCD_DIR))

frame  = cv2.imread(os.path.join(IMAGE_DIR, images[idx]))
pc     = o3d.io.read_point_cloud(os.path.join(PCD_DIR, pcds[idx]))
pts    = np.asarray(pc.points, dtype=float)

with open(CALIB_FILE) as f:
    calib = json.load(f)
K           = np.array(calib["K"],          dtype=float)
T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=float)

print(f"Image : {images[idx]}  shape={frame.shape}")
print(f"PCD   : {pcds[idx]}    points={pts.shape[0]}")
print(f"LiDAR x range: {pts[:,0].min():.2f} to {pts[:,0].max():.2f}")
print(f"LiDAR y range: {pts[:,1].min():.2f} to {pts[:,1].max():.2f}")
print(f"LiDAR z range: {pts[:,2].min():.2f} to {pts[:,2].max():.2f}")

n    = pts.shape[0]
pts_h    = np.hstack([pts, np.ones((n, 1))])
pts_cam  = (T_cam_lidar @ pts_h.T).T[:, :3]

print(f"\nCamera-frame X range: {pts_cam[:,0].min():.2f} to {pts_cam[:,0].max():.2f}")
print(f"Camera-frame Y range: {pts_cam[:,1].min():.2f} to {pts_cam[:,1].max():.2f}")
print(f"Camera-frame Z range: {pts_cam[:,2].min():.2f} to {pts_cam[:,2].max():.2f}")

H, W = frame.shape[:2]
Z = pts_cam[:, 2]
valid = Z > 1e-6
print(f"\nPoints with Z>0 (project forward): {valid.sum()} / {n}")

vis = frame.copy()
projected = 0
for i in np.where(valid)[0]:
    u = int(K[0,0] * pts_cam[i,0] / pts_cam[i,2] + K[0,2])
    v = int(K[1,1] * pts_cam[i,1] / pts_cam[i,2] + K[1,2])
    if 0 <= u < W and 0 <= v < H:
        cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
        projected += 1

print(f"Points projected inside image: {projected}")
cv2.imwrite("debug_projection.png", vis)
print("Saved debug_projection.png")
