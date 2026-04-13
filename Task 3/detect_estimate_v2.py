import os
import sys
import cv2
import csv
import json
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Vision Task"))
from inference import inference, CLASS_NAMES

# directory of lidar filtered images (dinhs one with every third image)
IMAGE_DIR  = "lidar_matched_multiple_3/"

# Directory to the point cloud files
PCD_DIR    = "pointcloud_multi_3/"

CALIB_FILE = "calibration.json"
OUTPUT_DIR   = "output_distances/"
OUTPUT_SIGNS = "output_signs_distances/"
OUTPUT_CSV   = "cylinder_distances.csv"

VIEW_RESULTS = True
START_INDEX = 0

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
    # CLAHE normalises local brightness so orange cones are detected consistently
    # regardless of shadows or changing lighting conditions
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    # Orange wraps around 0/180 in OpenCV's 0-180 hue scale, so two ranges are
    # needed: low-hue (0-20) and high-hue (160-180) to capture both sides
    mask1 = cv2.inRange(hsv, np.array([0,  100, 50]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


def fit_cone_bounds(orange_mask: np.ndarray):
    """Fit polynomials to left/right cone edges. Returns (left_poly, right_poly) or (None, None)."""
    H, W = orange_mask.shape
    rows, left_x, right_x = [], [], []
    for r in range(H):
        cols = np.where(orange_mask[r] > 0)[0]
        # Skip rows with too few orange pixels — likely noise rather than cone edge
        if len(cols) >= MIN_ORANGE_PX:
            rows.append(r)
            left_x.append(cols[0])   # leftmost orange pixel on this row
            right_x.append(cols[-1]) # rightmost orange pixel on this row
    # Need at least POLY_DEG+1 points to uniquely determine a degree-2 polynomial
    if len(rows) < POLY_DEG + 1:
        return None, None
    rows = np.array(rows)
    # Fit a quadratic (degree 2) to capture the slight curve of the cone silhouette
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
        # Set pixels outside the cone outline to white (255) so that when this
        # mask is inverted, only the interior non-orange region (the sign) remains
        if left_bounds[r] > 0:
            result[r, :left_bounds[r]] = 255
        if right_bounds[r] < W - 1:
            result[r, right_bounds[r] + 1:] = 255
    return result


def detect_sign(cone_crop):
    """Find the sign on a cone crop using v2 polynomial silhouette method.
    Returns (sign_crop, cone_mask_debug, sign_mask_debug).
    cone_mask_debug: orange binary mask with fitted polynomials.
    sign_mask_debug: inverted silhouette (non-orange inside cone) with bounding rect of detected sign.
    sign_crop is None if no sign found."""
    H, W = cone_crop.shape[:2]
    orange_mask = threshold_orange(cone_crop)
    left_poly, right_poly = fit_cone_bounds(orange_mask)

    # Cone mask debug: orange binary mask + polynomials
    cone_mask_debug = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
    if left_poly is not None:
        rows = np.arange(H)
        left_pts  = np.column_stack([np.clip(left_poly(rows).astype(int),  0, W-1), rows])
        right_pts = np.column_stack([np.clip(right_poly(rows).astype(int), 0, W-1), rows])
        cv2.polylines(cone_mask_debug, [left_pts],  False, (0, 255, 0), 1)
        cv2.polylines(cone_mask_debug, [right_pts], False, (0, 255, 0), 1)

    if left_poly is None:
        return None, cone_mask_debug, np.zeros((H, W, 3), dtype=np.uint8)

    silhouette = apply_silhouette(orange_mask, left_poly, right_poly)
    # Invert: orange pixels and outside-cone pixels are now black, leaving only
    # the non-orange interior of the cone (where the sign would be) as white
    inverted = cv2.bitwise_not(silhouette)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sign mask debug: inverted silhouette with bounding rect drawn
    sign_mask_debug = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    if not contours:
        return None, cone_mask_debug, sign_mask_debug

    # The sign is assumed to be the largest non-orange blob inside the cone
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(sign_mask_debug, (x, y), (x + w, y + h), (0, 255, 255), 1)

    # Pad the crop to a square so the classifier always receives a consistent
    # aspect ratio regardless of the sign's natural shape
    side = max(w, h)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    x_off = (side - w) // 2
    y_off = (side - h) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cone_crop[y:y+h, x:x+w]
    return square, cone_mask_debug, sign_mask_debug


def build_side_panel(crops, thumb_w=140):
    """Build a vertical strip: cone | cone mask+poly | sign mask+rect | sign crop."""
    GAP = 4
    NUM_COLS = 4
    panel_w = thumb_w * NUM_COLS + GAP * (NUM_COLS - 1)
    items = []

    def scale_to_w(img, w):
        h, iw = img.shape[:2]
        return cv2.resize(img, (w, max(1, int(h * w / iw))))

    def pad_h(img, h):
        diff = h - img.shape[0]
        return np.vstack([img, np.zeros((diff, img.shape[1], 3), dtype=np.uint8)]) if diff > 0 else img

    for i, (cone_crop, sign, cone_mask_debug, sign_mask_debug) in enumerate(crops):
        cone_thumb      = scale_to_w(cone_crop,       thumb_w)
        cone_mask_thumb = scale_to_w(cone_mask_debug, thumb_w)
        sign_mask_thumb = scale_to_w(sign_mask_debug, thumb_w)

        if sign is not None:
            sign_thumb = scale_to_w(sign, thumb_w)
        else:
            sign_thumb = np.zeros((cone_thumb.shape[0], thumb_w, 3), dtype=np.uint8)
            cv2.putText(sign_thumb, "no sign", (4, max(12, cone_thumb.shape[0] // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        row_h = max(cone_thumb.shape[0], cone_mask_thumb.shape[0], sign_mask_thumb.shape[0], sign_thumb.shape[0])
        cone_thumb      = pad_h(cone_thumb,      row_h)
        cone_mask_thumb = pad_h(cone_mask_thumb, row_h)
        sign_mask_thumb = pad_h(sign_mask_thumb, row_h)
        sign_thumb      = pad_h(sign_thumb,      row_h)

        label_bar = np.zeros((20, panel_w, 3), dtype=np.uint8)
        for col_i, text in enumerate([f"Cone {i}", "Cone mask", "Sign mask", "Sign"]):
            cv2.putText(label_bar, text, (col_i * (thumb_w + GAP) + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        g = np.full((row_h, GAP, 3), 40, dtype=np.uint8)
        row = np.hstack([cone_thumb, g, cone_mask_thumb, g, sign_mask_thumb, g, sign_thumb])

        items.append(label_bar)
        items.append(row)
        items.append(np.full((6, panel_w, 3), 40, dtype=np.uint8))

    if not items:
        return np.zeros((1, panel_w, 3), dtype=np.uint8)

    return np.vstack(items)


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

    # Slightly tighter hue/saturation than threshold_orange to reduce false
    # positives at the full-frame level (signs can break this up at crop level)
    mask1 = cv2.inRange(hsv, np.array([0,   105, 40]), np.array([8,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 105, 40]), np.array([180, 255, 255]))
    colour_mask = cv2.bitwise_or(mask1, mask2)
    # CLOSE fills small gaps between orange fragments (e.g. the sign breaking the cone)
    # OPEN then removes small isolated noise blobs
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN,  np.ones((3, 3)))

    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Discard tiny blobs that are clearly not a cone
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Cones are taller than they are wide — reject flat detections
        if h < 0.5 * w:
            continue
        # Low fill ratio means the contour is a thin ring/arc, not a solid cone body
        if area / (w * h) < 0.18:
            continue
        boxes.append((x, y, w, h))

    # Merge boxes that overlap horizontally (stacked fragments = same cylinder).
    # A sign on the cone can split it into multiple contours; iterating until
    # stable handles chains of three or more fragments.
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
                # 20-pixel tolerance accounts for small horizontal gaps between fragments
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
    # Convert to homogeneous coordinates so the 4x4 extrinsic matrix T_cam_lidar
    # can be applied in a single matrix multiply (handles both rotation and translation)
    pts_h = np.hstack([points_lidar, np.ones((n, 1))])   # (N,4)
    pts_cam = (T_cam_lidar @ pts_h.T).T                  # (N,4)
    pts_cam = pts_cam[:, :3]                             # (N,3)

    H, W = img_shape[:2]
    Z = pts_cam[:, 2]
    # Only project points in front of the camera; Z <= 0 would cause divide-by-zero
    # or project behind the image plane
    valid_z = Z > 1e-6

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    # Standard pinhole projection: u = fx * X/Z + cx
    u[valid_z] = fx * pts_cam[valid_z, 0] / Z[valid_z] + cx
    v[valid_z] = fy * pts_cam[valid_z, 1] / Z[valid_z] + cy

    # Final validity mask: must be in front of camera AND within image bounds
    in_image = (
        valid_z &
        np.isfinite(u) & np.isfinite(v) &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    # Use 2D radial distance (x-y plane only) as the range metric — this is the
    # ground-plane distance to the cone, ignoring vertical offset
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
    # in_bounds guards against any rounding that pushed pixels just outside the image
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    on_mask = np.zeros(len(ui), dtype=bool)
    # Look up each projected pixel in the cone silhouette mask — only keep points
    # that land on the cone's filled outline region
    on_mask[in_bounds] = cone_mask[vi[in_bounds], ui[in_bounds]] > 0

    hit_depths = depths[valid][on_mask]
    if len(hit_depths) == 0:
        return None

    # Median is more robust than mean — a few spurious points (e.g. ground returns
    # near the cone base) won't skew the reported distance
    return float(np.median(hit_depths))


# ---------------------------------------------------------------------------
# Main loop with interactive viewer
# ---------------------------------------------------------------------------
images = sorted(os.listdir(IMAGE_DIR))
pcds   = sorted(os.listdir(PCD_DIR))

# Pair by index (first image ↔ first pcd)
pairs = list(zip(images, pcds))

csv_rows = []
saved_rows = set()   # avoid duplicate CSV rows if you step backward/forward
idx = START_INDEX

while 0 <= idx < len(pairs):
    img_name, pcd_name = pairs[idx]

    frame = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    if frame is None:
        print(f"Skipping unreadable image: {img_name}")
        idx += 1
        continue

    points_lidar = load_pcd_points(os.path.join(PCD_DIR, pcd_name))
    boxes, colour_mask = detect_cones(frame)

    annotated = frame.copy()
    frame_rows = []
    cone_crops_display = []   # (cone_crop, sign_img_or_None)

    for cone_idx, (x, y, w, h) in enumerate(boxes):
        cone_crop = frame[y:y+h, x:x+w]
        orange_mask = threshold_orange(cone_crop)
        left_poly, right_poly = fit_cone_bounds(orange_mask)

        # Build a full-frame mask for LiDAR projection: use the tight polynomial
        # silhouette if available, otherwise fall back to the whole bounding box
        cone_region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if left_poly is not None:
            silhouette = build_silhouette_mask((h, w), left_poly, right_poly)
            cone_region_mask[y:y+h, x:x+w] = silhouette
        else:
            cone_region_mask[y:y+h, x:x+w] = 255

        dist = estimate_distance(points_lidar, K, T_cam_lidar, cone_region_mask)

        # Draw bbox and distance
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{dist:.2f}m" if dist is not None else "no dist"
        cv2.putText(
            annotated, label, (x + 4, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2
        )

        sign, cone_mask_debug, sign_mask_debug = detect_sign(cone_crop)
        cone_crops_display.append((cone_crop.copy(), sign.copy() if sign is not None else None, cone_mask_debug, sign_mask_debug))
        sign_label = None
        if sign is not None:
            sign_name = f"{os.path.splitext(img_name)[0]}_cone{cone_idx}_sign.png"
            sign_path = os.path.join(OUTPUT_SIGNS, sign_name)
            cv2.imwrite(sign_path, sign)
            predicted_idx = inference(MODEL_PATH, sign_path)
            sign_label = CLASS_NAMES[predicted_idx]
            cv2.putText(
                annotated, sign_label, (x + 4, min(frame.shape[0] - 10, y + h + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
            )

        row = {
            "frame": idx,
            "image": img_name,
            "pcd": pcd_name,
            "cone_idx": cone_idx,
            "x": x, "y": y, "w": w, "h": h,
            "distance_m": f"{dist:.4f}" if dist is not None else "",
            "sign": sign_label or "",
            "status": "ok" if dist is not None else "no_lidar_hit",
        }
        frame_rows.append(row)

    # Save annotated image once per frame
    base = os.path.splitext(img_name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_dist.png")
    cv2.imwrite(out_path, annotated)

    # Only append unique rows once, so stepping backward doesn't duplicate CSV entries
    for row in frame_rows:
        key = (row["image"], row["pcd"], row["cone_idx"])
        if key not in saved_rows:
            csv_rows.append(row)
            saved_rows.add(key)

    # Console summary
    print(
        f"[{idx:04d}] {img_name}  cones={len(boxes)}  " +
        "  ".join(
            f"cone{i}={r['distance_m']}m" if r['distance_m'] else f"cone{i}=no_hit"
            for i, r in enumerate(frame_rows)
        )
    )

    if not VIEW_RESULTS:
        idx += 1
        continue

    # Viewer overlay text
    info1 = f"Frame {idx+1}/{len(pairs)} | image={img_name} | pcd={pcd_name}"
    info2 = "n/right: next   b/left: back   q/esc: quit"
    cv2.putText(annotated, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(annotated, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    side_panel = build_side_panel(cone_crops_display)
    h_main, h_side = annotated.shape[0], side_panel.shape[0]
    max_h = max(h_main, h_side)
    # Both images must be the same height before hstack — pad the shorter one with black
    if h_main < max_h:
        annotated = np.vstack([annotated, np.zeros((max_h - h_main, annotated.shape[1], 3), dtype=np.uint8)])
    if h_side < max_h:
        side_panel = np.vstack([side_panel, np.zeros((max_h - h_side, side_panel.shape[1], 3), dtype=np.uint8)])
    display = np.hstack([annotated, side_panel])
    cv2.imshow("Task 3 Viewer", display)
    key = cv2.waitKey(0) & 0xFF

    # q or Esc -> quit
    if key in [ord('q'), 27]:
        break
    # b -> previous
    elif key == ord('b'):
        idx = max(0, idx - 1)
    # n, Enter, Space -> next
    else:
        idx += 1

cv2.destroyAllWindows()

# Write CSV once at the end
fieldnames = ["frame", "image", "pcd", "cone_idx", "x", "y", "w", "h", "distance_m", "sign", "status"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

from collections import Counter

# Count cones per frame (include frames with 0 detections)
cones_per_frame = Counter(r["image"] for r in csv_rows)
all_images = [name for name, _ in pairs]
cone_count_dist = Counter(cones_per_frame.get(img, 0) for img in all_images)

sign_counts = Counter(r["sign"] for r in csv_rows if r["sign"])

print(f"\nDone. Annotated images → {OUTPUT_DIR}")
print(f"CSV → {OUTPUT_CSV}")
print("\nCones detected per frame:")
for n in sorted(cone_count_dist):
    print(f"  {n} cone{'s' if n != 1 else ''}: {cone_count_dist[n]} frame{'s' if cone_count_dist[n] != 1 else ''}")
print("\nSign detections:")
for sign_name, count in sorted(sign_counts.items()):
    print(f"  {sign_name}: {count}")
if not sign_counts:
    print("  (none)")