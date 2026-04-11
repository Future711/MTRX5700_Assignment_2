import cv2
import numpy as np
import os

INPUT_DIR = "Task 3/ConeImages/CroppedImages/images_traffic_sign_multiple1"
OUTPUT_DIR = "Task 3/ConeImages/ThresholdAgain/images_traffic_sign_multiple1"

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def threshold_orange(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    # Orange/red hue wraps around 0 in HSV
    mask1 = cv2.inRange(hsv, np.array([0,  100, 50]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


MIN_ORANGE_PX = 10   # minimum orange pixels in a row to be considered reliable
POLY_DEG = 2         # polynomial degree for fitting cone edges


def fit_cone_bounds(orange_mask: np.ndarray):
    """Fit polynomials to left/right cone edges using only reliable orange rows.
    Returns (left_poly, right_poly) as np.poly1d objects, plus reliable row mask."""
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


def apply_silhouette(img: np.ndarray, left_poly, right_poly) -> np.ndarray:
    """White out everything outside the fitted cone silhouette."""
    H, W = img.shape[:2]
    result = img.copy()
    all_rows = np.arange(H)
    left_bounds  = np.clip(left_poly(all_rows).astype(int),  0, W - 1)
    right_bounds = np.clip(right_poly(all_rows).astype(int), 0, W - 1)

    for r in range(H):
        if left_bounds[r] > 0:
            result[r, :left_bounds[r]] = 255
        if right_bounds[r] < W - 1:
            result[r, right_bounds[r] + 1:] = 255
    return result


def draw_poly_lines(img: np.ndarray, left_poly, right_poly) -> np.ndarray:
    """Draw the fitted left/right polynomial lines on the image."""
    H = img.shape[0]
    debug = img.copy()
    for r in range(H):
        lx = int(np.clip(left_poly(r),  0, img.shape[1] - 1))
        rx = int(np.clip(right_poly(r), 0, img.shape[1] - 1))
        cv2.circle(debug, (lx, r), 1, (0, 255, 0), -1)
        cv2.circle(debug, (rx, r), 1, (0, 255, 0), -1)
    return debug


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    for fname in sorted(os.listdir(INPUT_DIR)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        in_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Skipping {fname}: could not read")
            continue
        orange_mask = threshold_orange(img)
        left_poly, right_poly = fit_cone_bounds(orange_mask)
        if left_poly is None:
            print(f"Skipping {fname}: not enough orange pixels to fit")
            continue
        silhouette = apply_silhouette(orange_mask, left_poly, right_poly)

        # Invert so black areas become white for contour finding
        inverted = cv2.bitwise_not(silhouette)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found: {fname}")
            continue
        largest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest)
        crop = img[y:y+h, x:x+w]

        # Pad to square with white
        side = max(w, h)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        x_off = (side - w) // 2
        y_off = (side - h) // 2
        square[y_off:y_off+h, x_off:x_off+w] = crop

        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, square)
        print(f"Saved {out_path}")