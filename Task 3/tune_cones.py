import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

IMAGE_DIR = "images_left_1/"
images = sorted(os.listdir(IMAGE_DIR))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def build_mask(frame, hue_max, sat_min, val_min):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    m1 = cv2.inRange(hsv, np.array([0,        sat_min, val_min]),
                          np.array([hue_max,   255,     255]))
    m2 = cv2.inRange(hsv, np.array([180 - 15, sat_min, val_min]),
                          np.array([180,       255,     255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3)))
    return mask

def get_boxes(mask, min_area, aspect, solidity):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < aspect * w:
            continue
        sol = area / (w * h)
        if sol < solidity:
            continue
        boxes.append((x, y, w, h, sol))
    return boxes

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plt.subplots_adjust(left=0.05, bottom=0.38, right=0.98, top=0.95)
ax_img, ax_mask = axes

# Initial values
init = dict(img_idx=0, hue_max=25, sat_min=60, val_min=40,
            min_area=800, aspect=1.5, solidity=0.35)

frame = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mask  = build_mask(frame, init['hue_max'], init['sat_min'], init['val_min'])

im_img  = ax_img.imshow(rgb)
im_mask = ax_mask.imshow(mask, cmap='gray')
ax_img.set_title(images[0])
ax_mask.set_title("Mask")
ax_img.axis('off'); ax_mask.axis('off')

# Sliders  (left, bottom, width, height)
sliders = {}
slider_defs = [
    ("img_idx",  "Image",      0,   len(images) - 1, init['img_idx'],  True),
    ("hue_max",  "Hue max",    1,   50,               init['hue_max'],  False),
    ("sat_min",  "Sat min",    0,   255,              init['sat_min'],  False),
    ("val_min",  "Val min",    0,   255,              init['val_min'],  False),
    ("min_area", "Min area",   100, 5000,             init['min_area'], False),
    ("aspect",   "Aspect×10",  5,   50,               int(init['aspect'] * 10), False),
    ("solidity", "Solidity%",  5,   100,              int(init['solidity'] * 100), False),
]
for i, (key, label, vmin, vmax, vinit, is_int) in enumerate(slider_defs):
    ax_sl = plt.axes([0.1, 0.29 - i * 0.038, 0.8, 0.025])
    sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=1 if is_int else None)
    sliders[key] = sl

patch_handles = []

def redraw(_=None):
    global patch_handles
    idx      = int(sliders['img_idx'].val)
    hue_max  = int(sliders['hue_max'].val)
    sat_min  = int(sliders['sat_min'].val)
    val_min  = int(sliders['val_min'].val)
    min_area = int(sliders['min_area'].val)
    aspect   = sliders['aspect'].val / 10.0
    solidity = sliders['solidity'].val / 100.0

    f = cv2.imread(os.path.join(IMAGE_DIR, images[idx]))
    if f is None:
        return
    m = build_mask(f, hue_max, sat_min, val_min)
    boxes = get_boxes(m, min_area, aspect, solidity)

    im_img.set_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    im_mask.set_data(m)
    ax_img.set_title(f"{images[idx]}  —  {len(boxes)} cone(s)")

    for p in patch_handles:
        p.remove()
    patch_handles = []
    for (x, y, w, h, sol) in boxes:
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='orange', facecolor='none')
        ax_img.add_patch(rect)
        patch_handles.append(rect)
        t = ax_img.text(x, y - 4, f"sol={sol:.2f}", color='orange', fontsize=7)
        patch_handles.append(t)

    fig.canvas.draw_idle()

for sl in sliders.values():
    sl.on_changed(redraw)

ax_print = plt.axes([0.4, 0.01, 0.2, 0.04])
btn = Button(ax_print, 'Print settings')

def print_settings(_):
    print(f"\n--- Settings for image {images[int(sliders['img_idx'].val)]} ---")
    print(f"hue_max  = {int(sliders['hue_max'].val)}")
    print(f"sat_min  = {int(sliders['sat_min'].val)}")
    print(f"val_min  = {int(sliders['val_min'].val)}")
    print(f"min_area = {int(sliders['min_area'].val)}")
    print(f"aspect   = {sliders['aspect'].val / 10.0}")
    print(f"solidity = {sliders['solidity'].val / 100.0}")

btn.on_clicked(print_settings)

redraw()
plt.show()
