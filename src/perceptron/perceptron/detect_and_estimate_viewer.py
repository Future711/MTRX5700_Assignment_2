#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np
import torch

from camera_lidar_calibration.camera_utils import load_calibration
from camera_lidar_calibration.lidar_utils import load_pcd_points, estimate_distance
from traffic_sign_classification.inference import load_model, inference
from cylinder_sign_detection import threshold_orange, fit_cone_bounds, build_silhouette_mask, detect_cones, detect_sign


# Default paths kept aligned with detect_and_estimate.py
IMAGE_DIR = "/home/hdqquang/Projects/MTRX5700_Assignment_2/traffic_sign_right_1/images"
PCD_DIR = "/home/hdqquang/Projects/MTRX5700_Assignment_2/traffic_sign_right_1/pointclouds"
CALIB_FILE = "calibration.json"
MODEL_PATH = "traffic_sign_classification/checkpoint/ckpt_rmsprop_lr_0.005_bs_64_ep_100_True_v1.pth"


def annotate_pair(image_path, pcd_path, model, device, K, T_cam_lidar):
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    points_lidar = load_pcd_points(pcd_path)
    boxes_result = detect_cones(frame)
    boxes = boxes_result[0] if isinstance(boxes_result, tuple) else boxes_result

    annotated = frame.copy()
    cone_summaries = []

    for cone_idx, (x, y, w, h) in enumerate(boxes):
        cone_crop = frame[y:y + h, x:x + w]
        orange_mask = threshold_orange(cone_crop)
        left_poly, right_poly = fit_cone_bounds(orange_mask)

        cone_region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if left_poly is not None:
            silhouette = build_silhouette_mask((h, w), left_poly, right_poly)
            cone_region_mask[y:y + h, x:x + w] = silhouette
        else:
            cone_region_mask[y:y + h, x:x + w] = 255

        dist = estimate_distance(points_lidar, K, T_cam_lidar, cone_region_mask)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dist_label = f"{dist:.2f}m" if dist is not None else "no dist"
        cv2.putText(
            annotated,
            dist_label,
            (x + 4, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        sign_label = None
        sign = detect_sign(cone_crop, orange_mask=orange_mask, left_poly=left_poly, right_poly=right_poly)
        if sign is not None:
            sign_label = inference(model, device, sign)
            cv2.putText(
                annotated,
                sign_label,
                (x + 4, y + h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        cone_summaries.append(
            f"cone{cone_idx}: {dist_label}, {sign_label or 'no sign'}"
        )

    return annotated, cone_summaries, len(boxes)


def main():
    parser = argparse.ArgumentParser(
        description="Display annotated cone/sign/distance results without saving images."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--pcd-dir", default=PCD_DIR)
    parser.add_argument("--calib", default=CALIB_FILE)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    K, T_cam_lidar = load_calibration(args.calib)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    images = sorted(os.listdir(args.image_dir))
    pcds = sorted(os.listdir(args.pcd_dir))
    pairs = list(zip(images, pcds))

    if not pairs:
        print("No image/pcd pairs found.")
        return

    idx = 0
    cache = {}

    cv2.namedWindow("detect_and_estimate_viewer", cv2.WINDOW_NORMAL)

    while True:
        if idx not in cache:
            img_name, pcd_name = pairs[idx]
            image_path = os.path.join(args.image_dir, img_name)
            pcd_path = os.path.join(args.pcd_dir, pcd_name)
            annotated, summaries, n_cones = annotate_pair(image_path, pcd_path, model, device, K, T_cam_lidar)
            cache[idx] = (annotated, summaries, n_cones, img_name, pcd_name)

        annotated, summaries, n_cones, img_name, pcd_name = cache[idx]
        frame_to_show = annotated.copy()

        header = f"[{idx + 1}/{len(pairs)}] {img_name} | {pcd_name} | cones={n_cones}"
        cv2.putText(frame_to_show, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(
            frame_to_show,
            "Controls: q=quit, w=prev, e=next",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("detect_and_estimate_viewer", frame_to_show)

        if summaries:
            print(f"\n{header}")
            for s in summaries:
                print(f"  - {s}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("e"):
            idx = min(idx + 1, len(pairs) - 1)
            continue
        if key == ord("w"):
            idx = max(idx - 1, 0)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
