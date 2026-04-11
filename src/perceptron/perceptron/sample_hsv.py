import cv2
import numpy as np
import sys
import os

IMAGE_DIR = "images_left_1/"
images = sorted(os.listdir(IMAGE_DIR))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

if len(sys.argv) > 1 and not sys.argv[1].isdigit():
    img_path = sys.argv[1]
else:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    img_path = os.path.join(IMAGE_DIR, images[idx])
frame = cv2.imread(img_path)
print(f"Loaded: {img_path}  (pass a file path or image index as arg)")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = clahe.apply(v)
hsv_clahe = cv2.merge([h, s, v])

display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        H, S, V = hsv_clahe[y, x]
        B, G, R = frame[y, x]
        print(f"  pixel ({x:4d}, {y:4d})  HSV=({H:3d}, {S:3d}, {V:3d})  BGR=({B:3d},{G:3d},{R:3d})")

cv2.namedWindow("sample_hsv - click to sample, q to quit")
cv2.setMouseCallback("sample_hsv - click to sample, q to quit", on_click)

while True:
    cv2.imshow("sample_hsv - click to sample, q to quit", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
