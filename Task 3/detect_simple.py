import os
import cv2
import numpy as np

# IMAGE_DIR = "images_right_2/"
# CONE_DIR = "output_cones_4/"
# SIGN_DIR = "output_signs_4/"

IMAGE_DIR = "multi_cone_3_images/"
CONE_DIR = "multi_output_cones_3/"
SIGN_DIR = "multioutput_signs_1/"


os.makedirs(CONE_DIR, exist_ok=True)
os.makedirs(SIGN_DIR, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def detect_cones(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Normalise brightness with CLAHE so colour thresholds work under poor lighting
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    # Red/orange hue ranges
    mask1 = cv2.inRange(hsv, np.array([0,   105, 40]), np.array([8,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 105, 40]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # Merge boxes that are vertically close and horizontally overlapping
    # Merge boxes that overlap horizontally (same column = same cylinder)
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
                if x2 < mx2 + 20 and jx2 > mx - 20:  # horizontal ranges overlap or within 20px
                    mx  = min(mx,  x2)
                    my  = min(my,  y2)
                    mx2 = max(mx2, jx2)
                    my2 = max(my2, jy2)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append((mx, my, mx2 - mx, my2 - my))
        boxes = merged
    return merged


def detect_sign(cone_crop):
    """Find the blue sign with white border on a cone crop. Returns the sign crop or None."""
    hsv = cv2.cvtColor(cone_crop, cv2.COLOR_BGR2HSV)
    H, W = cone_crop.shape[:2]

    # Find the blue face of the sign
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 30]), np.array([135, 255, 255]))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  np.ones((3, 3)))

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0  # higher = more square and more solid
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < 0.7 or aspect > 1.4:
            continue

        solidity = area / (w * h)
        if solidity < 0.55:
            continue

        # Score: prefer compact square blobs over large elongated ones
        squareness = 1.0 - abs(aspect - 1.0)
        score = squareness * solidity
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    # Force a square crop around the centre of the detected region
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half = side // 2
    pad = max(4, int(side * 0.15))
    x1 = max(0, cx - half - pad)
    y1 = max(0, cy - half - pad)
    x2 = min(W, cx + half + pad)
    y2 = min(H, cy + half + pad)
    return cone_crop[y1:y2, x1:x2]


total = 0
correct = 0

for img_name in sorted(os.listdir(IMAGE_DIR)):
    frame = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    if frame is None:
        continue

    boxes = detect_cones(frame)

    base = os.path.splitext(img_name)[0]
    n_signs = 0
    annotated = frame.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw bright green bounding box on the original image
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Still crop for sign detection
        side = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        half = side // 2
        fH, fW = frame.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(fW, cx + half)
        y2 = min(fH, cy + half)
        cone_crop = frame[y1:y2, x1:x2]

        sign = detect_sign(cone_crop)
        if sign is not None:
            n_signs += 1

    cv2.imwrite(os.path.join(CONE_DIR, f"{base}_annotated.png"), annotated)

    total += 1
    if len(boxes) == 1:
        correct += 1
    print(f"{img_name}: {len(boxes)} cone(s), {n_signs} sign(s) {'✓' if len(boxes) == 1 else '✗'}")

accuracy = correct / total * 100 if total > 0 else 0
print(f"\nDone. Cones → {CONE_DIR}  Signs → {SIGN_DIR}")
print(f"Accuracy: {correct}/{total} images with exactly 1 cone detected ({accuracy:.1f}%)")
