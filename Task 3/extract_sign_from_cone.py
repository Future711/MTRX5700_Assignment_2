import cv2
import numpy as np

# FILE_PATH = "TeamBagsandData2/cropped_cones/Pasted image.png"
FILE_PATH = "TeamBagsandData2/cropped_cones/Pasted image (2).png"


def binarise(img: np.ndarray) -> np.ndarray:
    """Detect edges using Canny after Gaussian blur. Returns the edge map."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 10, 50)
    return edges


def find_circle(edges: np.ndarray):
    """Fit a circle to edge map using Hough transform. Returns (x, y, r) or None."""
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=0,
    )
    if circles is None:
        return None
    # Take the circle with the most votes (best fit)
    circles = np.round(circles[0]).astype(int)
    return circles[0]


def crop(img: np.ndarray, circle) -> np.ndarray:
    """Crop the image to the bounding square of the detected circle."""
    x, y, r = circle
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(img.shape[1], x + r), min(img.shape[0], y + r)
    return img[y1:y2, x1:x2]


def extract_sign(img: np.ndarray) -> np.ndarray:
    """Full pipeline: binarise -> find circle -> crop."""
    edges = binarise(img)
    circle = find_circle(edges)
    if circle is None:
        raise ValueError("No circle found in image")
    return crop(img, circle)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    for fname in os.listdir(input_dir):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Skipping {fname}: could not read")
            continue
        edges = binarise(img)
        circle = find_circle(edges)
        debug = img.copy()
        if circle is not None:
            x, y, r = circle
            cv2.circle(debug, (x, y), r, (0, 255, 0), 2)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, debug)
        print(f"Saved {out_path}")
