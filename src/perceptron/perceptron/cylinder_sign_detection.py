"""cylinder and traffic-sign extraction from camera images.

This module detects orange cylinders in a BGR frame and extracts the mounted
traffic sign crop for downstream classification.
"""

import cv2
import numpy as np
from PIL import Image


# CLAHE (Contrast Limited Adaptive Histogram Equalisation) applied to the V
# channel to normalise brightness variations before colour thresholding.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Minimum number of orange pixels required in a row to count that row when
# fitting the cylinder boundary polynomials.
MIN_ORANGE_PX = 10
# Degree of the polynomial used to model the left/right cylinder edges.
POLY_DEG = 2


def threshold_orange(img: np.ndarray) -> np.ndarray:
    """Return a binary mask of orange pixels in *img*.

    Converts the image to HSV, equalises the V channel with CLAHE to handle
    changing lighting, then combines two hue ranges that together cover the
    full orange/red wrap-around in HSV (0-20 and 160-180 degrees).

    Args:
        img: Input BGR image.

    Returns:
        Binary mask where orange pixels are non-zero.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    mask1 = cv2.inRange(hsv, np.array([0,  100, 50]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)

def fit_cylinder_bounds(orange_mask: np.ndarray):
    """Fit polynomial curves to left and right cylinder boundaries.

    Args:
        orange_mask: Binary mask containing cylinder-colored pixels.

    Returns:
        Tuple containing:
            left_poly: numpy.poly1d for the left boundary, or None.
            right_poly: numpy.poly1d for the right boundary, or None.
    """
    H, _ = orange_mask.shape
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
    """Mask pixels outside the fitted cylinder silhouette.

    Args:
        orange_mask: Binary cylinder color mask.
        left_poly: Polynomial describing the left cylinder edge.
        right_poly: Polynomial describing the right cylinder edge.

    Returns:
        Mask where pixels outside the silhouette are set to white.
    """
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

def build_silhouette_mask(shape, left_poly, right_poly) -> np.ndarray:
    """Create a filled binary mask between two cylinder boundary polynomials.

    Args:
        shape: Output mask shape as (height, width).
        left_poly: Polynomial describing the left cylinder edge.
        right_poly: Polynomial describing the right cylinder edge.

    Returns:
        Binary mask filled between left and right boundaries.
    """
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

def detect_cylinders(frame):
    """Detect orange traffic cylinders in a BGR camera frame.

    Pipeline:
      1. Convert to HSV and equalise brightness with CLAHE.
      2. Build a colour mask covering the orange/red hue wrap-around.
      3. Apply morphological close (fills gaps) then open (removes noise).
      4. Extract contours and filter by minimum area, aspect ratio (must be
         taller than wide), and fill ratio (rejects thin stray edges).
      5. Iteratively merge boxes that overlap horizontally so that cylinder
         fragments caused by a sign occluding part of the cylinder are
         reunited into a single bounding box.

    Returns:
        Tuple containing:
            boxes: List of (x, y, w, h) cylinder bounding boxes.
            colour_mask: Binary orange mask used for contour extraction.
    """
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
        # Reject blobs that are wider than tall (not cylinder-like)
        if h < 0.5 * w:
            continue
        # Reject very sparse blobs (e.g. thin lines or road markings)
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
                # Allow a 20-pixel horizontal slack to catch near-adjacent fragments
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

def detect_sign(cylinder_crop, orange_mask=None, left_poly=None, right_poly=None):
    """Isolate and return the traffic sign mounted on a cylinder crop as a PIL Image.

    The sign sits *inside* the cylinder silhouette but is not orange itself.
    Strategy:
      1. Build (or reuse) the orange mask and cylinder boundary polynomials.
      2. Apply the silhouette mask to the orange mask, setting pixels outside
         the cylinder boundaries to 255 (white/orange).
      3. Invert so that non-orange regions inside the cylinder become white blobs.
      4. Find the largest such blob - that is the sign face.
      5. Crop the sign, centre it in a square black canvas and return a PIL
         Image ready for the classifier (which expects square inputs).

    Args:
        cylinder_crop: BGR image crop containing one cylinder.
        orange_mask: Optional precomputed cylinder-color mask.
        left_poly: Optional precomputed left boundary polynomial.
        right_poly: Optional precomputed right boundary polynomial.

    Returns:
        PIL image containing the extracted square sign crop, or None if sign
        extraction fails.
    """
    if orange_mask is None:
        orange_mask = threshold_orange(cylinder_crop)

    if left_poly is None or right_poly is None:
        left_poly, right_poly = fit_cylinder_bounds(orange_mask)

    if left_poly is None:
        return None

    # Mask out everything outside the cylinder shape, then invert to find non-orange regions
    silhouette = apply_silhouette(orange_mask, left_poly, right_poly)
    inverted = cv2.bitwise_not(silhouette)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # The largest non-orange blob inside the cylinder is assumed to be the sign
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Centre the sign crop in a square canvas so the classifier receives a
    # consistent aspect ratio regardless of the sign's own proportions.
    side = max(w, h)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    x_off = (side - w) // 2
    y_off = (side - h) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cylinder_crop[y:y + h, x:x + w]
    square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square_pil = Image.fromarray(square_rgb)
    return square_pil