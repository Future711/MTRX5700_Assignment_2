import os
import sys
import cv2
import csv
import json
import numpy as np
import open3d as o3d

from traffic_sign_classification.inference import inference, CLASS_NAMES

# directory of lidar filtered images (dinhs one with every third image)
IMAGE_DIR  = "lidar_matched_images_2/"

# Directory to the point cloud files
PCD_DIR    = "pointcloud_left_2/"

CALIB_FILE = "calibration.json"
OUTPUT_DIR   = "output_distances/"
OUTPUT_SIGNS = "output_signs_distances/"
OUTPUT_CSV   = "cylinder_distances.csv"

#Need to get this from the drive as well
MODEL_PATH   = "ckpt_rmsprop_lr_0.005_bs_64_ep_100_True.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_SIGNS, exist_ok=True)

# ---------------------------------------------------------------------------
# Load calibration
# ---------------------------------------------------------------------------
with open(CALIB_FILE) as f:
    calib = json.load(f)

K           = np.array(calib["K"],          dtype=float)
T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=float)

# ---------------------------------------------------------------------------
# Cone detection (same parameters as detect_simple.py)
# ---------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

MIN_ORANGE_PX = 10
POLY_DEG = 2


def threshold_orange(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    mask1 = cv2.inRange(hsv, np.array([0,  100, 50]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


def fit_cone_bounds(orange_mask: np.ndarray):
    """Fit polynomials to left/right cone edges. Returns (left_poly, right_poly) or (None, None)."""
    H, W = orange_mask.shape
    rows, left_x, right_x = [], [], []
    for r in range(H):
        cols = np.where(orange_mask[r] > 0)[0]
        if len(cols) >= MIN_ORANGE_PX:
            rows.append(r)
            left_x.append(cols[0])
            right_x.append(cols[-1])
    if len(rows) < POLY_DEG + 1:
        return None, None
    rows = np.array(rows)
    left_poly  = np.poly1d(np.polyfit(rows, left_x,  POLY_DEG))
    right_poly = np.poly1d(np.polyfit(rows, right_x, POLY_DEG))
    return left_poly, right_poly


def apply_silhouette(orange_mask: np.ndarray, left_poly, right_poly) -> np.ndarray:
    """White out everything outside the fitted cone silhouette in the orange mask."""
    H, W = orange_mask.shape
    result = orange_mask.copy()
    all_rows = np.arange(H)
    left_bounds  = np.clip(left_poly(all_rows).astype(int),  0, W - 1)
    right_bounds = np.clip(right_poly(all_rows).astype(int), 0, W - 1)
    for r in range(H):
        if left_bounds[r] > 0:
            result[r, :left_bounds[r]] = 255
        if right_bounds[r] < W - 1:
            result[r, right_bounds[r] + 1:] = 255
    return result


def detect_sign(cone_crop):
    """Find the sign on a cone crop using v2 polynomial silhouette method.
    Finds the largest non-orange region inside the cone outline. Returns the sign crop or None."""
    orange_mask = threshold_orange(cone_crop)
    left_poly, right_poly = fit_cone_bounds(orange_mask)
    if left_poly is None:
        return None

    silhouette = apply_silhouette(orange_mask, left_poly, right_poly)
    # Invert: non-orange regions inside the cone become white
    inverted = cv2.bitwise_not(silhouette)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Pad to square with white
    side = max(w, h)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    x_off = (side - w) // 2
    y_off = (side - h) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cone_crop[y:y+h, x:x+w]
    return square


def build_silhouette_mask(shape, left_poly, right_poly) -> np.ndarray:
    """Filled mask between the two polynomial cone edges."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    all_rows = np.arange(H)
    left_bounds  = np.clip(left_poly(all_rows).astype(int),  0, W - 1)
    right_bounds = np.clip(right_poly(all_rows).astype(int), 0, W - 1)
    for r in range(H):
        l, rb = left_bounds[r], right_bounds[r]
        if l <= rb:
            mask[r, l:rb + 1] = 255
    return mask

def detect_cones(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    mask1 = cv2.inRange(hsv, np.array([0,   105, 40]), np.array([8,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 105, 40]), np.array([180, 255, 255]))
    colour_mask = cv2.bitwise_or(mask1, mask2)
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN,  np.ones((3, 3)))

    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 0.5 * w:
            continue
        if area / (w * h) < 0.18:
            continue
        boxes.append((x, y, w, h))

    # Merge boxes that overlap horizontally (stacked fragments = same cylinder)
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)
        for i, (x1, y1, w1, h1) in enumerate(boxes):
            if used[i]:
                continue
            mx, my, mx2, my2 = x1, y1, x1 + w1, y1 + h1
            for j in range(len(boxes)):
                if i == j or used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                jx2, jy2 = x2 + w2, y2 + h2
                if x2 < mx2 + 20 and jx2 > mx - 20:
                    mx  = min(mx,  x2)
                    my  = min(my,  y2)
                    mx2 = max(mx2, jx2)
                    my2 = max(my2, jy2)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append((mx, my, mx2 - mx, my2 - my))
        boxes = merged

    return boxes, colour_mask


# ---------------------------------------------------------------------------
# LiDAR projection and distance estimation
# ---------------------------------------------------------------------------
def load_pcd_points(pcd_path):
    """Load XYZ points from a PCD file. Returns (N,3) array."""
    pc = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pc.points, dtype=float)


def project_lidar_points(points_lidar, K, T_cam_lidar, img_shape):
    """
    Project LiDAR points into image coordinates.
    Returns:
        pixels      (N, 2) float  [u, v]
        depths      (N,)   float  distance in LiDAR frame
        valid       (N,)   bool
    """
    n = points_lidar.shape[0]
    pts_h = np.hstack([points_lidar, np.ones((n, 1))])   # (N,4)
    pts_cam = (T_cam_lidar @ pts_h.T).T                  # (N,4)
    pts_cam = pts_cam[:, :3]                             # (N,3)

    H, W = img_shape[:2]
    Z = pts_cam[:, 2]
    valid_z = Z > 1e-6

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    u[valid_z] = fx * pts_cam[valid_z, 0] / Z[valid_z] + cx
    v[valid_z] = fy * pts_cam[valid_z, 1] / Z[valid_z] + cy

    in_image = (
        valid_z &
        np.isfinite(u) & np.isfinite(v) &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    depths = np.sqrt(points_lidar[:, 0]**2 + points_lidar[:, 1]**2)
    return np.column_stack([u, v]), depths, in_image


def estimate_distance(points_lidar, K, T_cam_lidar, cone_mask):
    """Return median distance (m) of LiDAR points that project onto cone_mask, or None."""
    if points_lidar.shape[0] == 0:
        return None

    pixels, depths, valid = project_lidar_points(points_lidar, K, T_cam_lidar, cone_mask.shape)
    if not np.any(valid):
        return None

    H, W = cone_mask.shape[:2]
    ui = np.round(pixels[valid, 0]).astype(int)
    vi = np.round(pixels[valid, 1]).astype(int)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    on_mask = np.zeros(len(ui), dtype=bool)
    on_mask[in_bounds] = cone_mask[vi[in_bounds], ui[in_bounds]] > 0

    hit_depths = depths[valid][on_mask]
    if len(hit_depths) == 0:
        return None

    return float(np.median(hit_depths))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
images = sorted(os.listdir(IMAGE_DIR))
pcds   = sorted(os.listdir(PCD_DIR))

# Pair by index (first image ↔ first pcd)
pairs = list(zip(images, pcds))

csv_rows = []

for idx, (img_name, pcd_name) in enumerate(pairs):
    frame = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    if frame is None:
        continue

    points_lidar = load_pcd_points(os.path.join(PCD_DIR, pcd_name))
    boxes, colour_mask = detect_cones(frame)

    annotated = frame.copy()
    for cone_idx, (x, y, w, h) in enumerate(boxes):
        # Try to fit a tight cone silhouette using polynomial edge fitting.
        # Falls back to the full bounding box if there aren't enough orange pixels.
        cone_crop = frame[y:y+h, x:x+w]
        orange_mask = threshold_orange(cone_crop)
        left_poly, right_poly = fit_cone_bounds(orange_mask)

        cone_region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if left_poly is not None:
            silhouette = build_silhouette_mask((h, w), left_poly, right_poly)
            cone_region_mask[y:y+h, x:x+w] = silhouette
        else:
            cone_region_mask[y:y+h, x:x+w] = 255

        dist = estimate_distance(points_lidar, K, T_cam_lidar, cone_region_mask)

        # Draw box and distance label
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{dist:.2f}m" if dist is not None else "no dist"
        cv2.putText(annotated, label, (x + 4, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        sign = detect_sign(cone_crop)
        sign_label = None
        if sign is not None:
            sign_name = f"{os.path.splitext(img_name)[0]}_cone{cone_idx}_sign.png"
            sign_path = os.path.join(OUTPUT_SIGNS, sign_name)
            cv2.imwrite(sign_path, sign)
            predicted_idx = inference(MODEL_PATH, sign_path)
            sign_label = CLASS_NAMES[predicted_idx]
            cv2.putText(annotated, sign_label, (x + 4, y + h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        csv_rows.append({
            "frame":    idx,
            "image":    img_name,
            "pcd":      pcd_name,
            "cone_idx": cone_idx,
            "x": x, "y": y, "w": w, "h": h,
            "distance_m": f"{dist:.4f}" if dist is not None else "",
            "sign":     sign_label or "",
            "status":   "ok" if dist is not None else "no_lidar_hit",
        })

    base = os.path.splitext(img_name)[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_dist.png"), annotated)
    print(f"[{idx:04d}] {img_name}  cones={len(boxes)}  "
          + "  ".join(
              f"cone{i}={r['distance_m']}m" if r['distance_m'] else f"cone{i}=no_hit"
              for i, r in enumerate(csv_rows[-len(boxes):])
          ))

# Write CSV
fieldnames = ["frame", "image", "pcd", "cone_idx", "x", "y", "w", "h", "distance_m", "sign", "status"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nDone. Annotated images → {OUTPUT_DIR}")
print(f"CSV → {OUTPUT_CSV}")
