import json
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class ScanData:
    scan_id: int
    timestamp: float
    angle_min: float
    angle_increment: float
    ranges: np.ndarray
    range_min: float
    range_max: float


@dataclass
class CylinderDetection:
    cylinder_id: str
    mask_pixels: np.ndarray  # shape (N, 2), columns [x, y]


@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    cylinders: List[CylinderDetection]


@dataclass
class ClusterResult:
    beam_indices: np.ndarray
    ranges: np.ndarray
    pixels: np.ndarray
    points_lidar: np.ndarray


def load_calibration(calibration_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(calibration_path, "r") as f:
        data = json.load(f)

    K = np.asarray(data["K"], dtype=float)
    T_cam_lidar = np.asarray(data["T_cam_lidar"], dtype=float)

    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")
    if T_cam_lidar.shape != (4, 4):
        raise ValueError("T_cam_lidar must be 4x4")

    return K, T_cam_lidar


def load_frames(frames_path: str) -> List[FrameData]:
    with open(frames_path, "r") as f:
        raw = json.load(f)

    frames: List[FrameData] = []
    for item in raw:
        cylinders = []
        for cyl in item["cylinders"]:
            mask_pixels = np.asarray(cyl["mask_pixels"], dtype=int)
            if mask_pixels.ndim != 2 or mask_pixels.shape[1] != 2:
                raise ValueError(
                    f"mask_pixels for cylinder {cyl['cylinder_id']} must be Nx2 [[x,y], ...]"
                )
            cylinders.append(
                CylinderDetection(
                    cylinder_id=str(cyl["cylinder_id"]),
                    mask_pixels=mask_pixels
                )
            )

        frames.append(
            FrameData(
                frame_id=int(item["frame_id"]),
                timestamp=float(item["timestamp"]),
                image_width=int(item["image_width"]),
                image_height=int(item["image_height"]),
                cylinders=cylinders
            )
        )

    return frames


def load_scans(scans_path: str) -> List[ScanData]:
    with open(scans_path, "r") as f:
        raw = json.load(f)

    scans: List[ScanData] = []
    for i, item in enumerate(raw):
        scans.append(
            ScanData(
                scan_id=int(item.get("scan_id", i)),
                timestamp=float(item["timestamp"]),
                angle_min=float(item["angle_min"]),
                angle_increment=float(item["angle_increment"]),
                ranges=np.asarray(item["ranges"], dtype=float),
                range_min=float(item.get("range_min", 0.05)),
                range_max=float(item.get("range_max", 100.0)),
            )
        )
    return scans


def build_mask(mask_pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Build a boolean mask from a list of absolute image pixels [x, y].
    """
    mask = np.zeros((height, width), dtype=bool)

    x = mask_pixels[:, 0]
    y = mask_pixels[:, 1]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]

    mask[y, x] = True
    return mask


def associate_nearest_scan(
    frame_timestamp: float,
    scans: List[ScanData],
    max_time_diff: float = 0.25
) -> Optional[ScanData]:
    """
    Return nearest scan in time, or None if the nearest scan is too far away.
    """
    if not scans:
        return None

    best_scan = min(scans, key=lambda s: abs(s.timestamp - frame_timestamp))
    dt = abs(best_scan.timestamp - frame_timestamp)

    if dt > max_time_diff:
        return None
    return best_scan


def laser_scan_to_points(scan: ScanData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert LaserScan to LiDAR-frame points.

    Returns:
        points_lidar: (N, 3)
        valid_ranges: (N,)
        valid_beam_indices: (N,)
    """
    n = len(scan.ranges)
    beam_indices = np.arange(n, dtype=int)
    angles = scan.angle_min + beam_indices * scan.angle_increment

    valid = (
        np.isfinite(scan.ranges) &
        (scan.ranges >= scan.range_min) &
        (scan.ranges <= scan.range_max)
    )

    r = scan.ranges[valid]
    a = angles[valid]
    idx = beam_indices[valid]

    x = r * np.cos(a)
    y = r * np.sin(a)
    z = np.zeros_like(x)

    points_lidar = np.column_stack([x, y, z])
    return points_lidar, r, idx


def transform_points(T_cam_lidar: np.ndarray, points_lidar: np.ndarray) -> np.ndarray:
    """
    p_cam = T_cam_lidar @ p_lidar_h
    """
    n = points_lidar.shape[0]
    pts_h = np.hstack([points_lidar, np.ones((n, 1), dtype=float)])
    pts_cam_h = (T_cam_lidar @ pts_h.T).T
    return pts_cam_h[:, :3]


def project_points(K: np.ndarray, points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame points into image pixels.

    Returns:
        pixels: (N, 2) with columns [u, v]
        valid_z: (N,) boolean mask
    """
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    valid_z = Z > 1e-6

    u = np.full_like(Z, np.nan, dtype=float)
    v = np.full_like(Z, np.nan, dtype=float)

    u[valid_z] = fx * X[valid_z] / Z[valid_z] + cx
    v[valid_z] = fy * Y[valid_z] / Z[valid_z] + cy

    pixels = np.column_stack([u, v])
    return pixels, valid_z


def get_candidate_points_for_mask(
    scan: ScanData,
    K: np.ndarray,
    T_cam_lidar: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep LiDAR points whose projected pixels fall on the orange mask.

    Returns:
        candidate_beam_indices
        candidate_ranges
        candidate_pixels
        candidate_points_lidar
    """
    height, width = mask.shape

    points_lidar, valid_ranges, valid_beam_indices = laser_scan_to_points(scan)
    points_cam = transform_points(T_cam_lidar, points_lidar)
    pixels, valid_z = project_points(K, points_cam)

    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)

    in_image = (
        valid_z &
        np.isfinite(pixels[:, 0]) &
        np.isfinite(pixels[:, 1]) &
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    on_mask = np.zeros(len(points_lidar), dtype=bool)
    on_mask[in_image] = mask[v[in_image], u[in_image]]

    candidate_beam_indices = valid_beam_indices[on_mask]
    candidate_ranges = valid_ranges[on_mask]
    candidate_pixels = pixels[on_mask]
    candidate_points_lidar = points_lidar[on_mask]

    return (
        candidate_beam_indices,
        candidate_ranges,
        candidate_pixels,
        candidate_points_lidar
    )


def split_into_contiguous_clusters(
    beam_indices: np.ndarray,
    ranges: np.ndarray,
    pixels: np.ndarray,
    points_lidar: np.ndarray,
    max_beam_gap: int = 2,
    max_range_jump: float = 0.12
) -> List[ClusterResult]:
    """
    Split candidate points into LiDAR clusters using:
    - beam adjacency
    - range continuity

    max_beam_gap=2 handles occasional missing beams.
    """
    if len(beam_indices) == 0:
        return []

    order = np.argsort(beam_indices)
    beam_indices = beam_indices[order]
    ranges = ranges[order]
    pixels = pixels[order]
    points_lidar = points_lidar[order]

    clusters = []
    start = 0

    for i in range(len(beam_indices) - 1):
        beam_gap = beam_indices[i + 1] - beam_indices[i]
        range_jump = abs(ranges[i + 1] - ranges[i])

        split_here = (beam_gap > max_beam_gap) or (range_jump > max_range_jump)

        if split_here:
            clusters.append(
                ClusterResult(
                    beam_indices=beam_indices[start:i + 1],
                    ranges=ranges[start:i + 1],
                    pixels=pixels[start:i + 1],
                    points_lidar=points_lidar[start:i + 1]
                )
            )
            start = i + 1

    clusters.append(
        ClusterResult(
            beam_indices=beam_indices[start:],
            ranges=ranges[start:],
            pixels=pixels[start:],
            points_lidar=points_lidar[start:]
        )
    )

    return clusters


def choose_best_cluster(
    clusters: List[ClusterResult],
    min_points: int = 2
) -> Optional[ClusterResult]:
    """
    Choose the nearest valid cluster.

    Preference:
    1. nearest mean range
    2. larger cluster if similar
    """
    valid_clusters = [c for c in clusters if len(c.ranges) >= min_points]
    if not valid_clusters:
        return None

    def score(c: ClusterResult):
        return (float(np.mean(c.ranges)), -len(c.ranges))

    return min(valid_clusters, key=score)


def compute_cluster_stats(cluster: ClusterResult) -> Dict[str, float]:
    ranges = cluster.ranges
    return {
        "num_points": int(len(ranges)),
        "mean_distance_m": float(np.mean(ranges)),
        "std_distance_m": float(np.std(ranges)),
        "min_distance_m": float(np.min(ranges)),
        "max_distance_m": float(np.max(ranges)),
        "median_distance_m": float(np.median(ranges)),
    }


def process_all_frames(
    frames: List[FrameData],
    scans: List[ScanData],
    K: np.ndarray,
    T_cam_lidar: np.ndarray,
    max_time_diff: float = 0.25,
    max_beam_gap: int = 2,
    max_range_jump: float = 0.12,
    min_cluster_points: int = 2
) -> List[Dict]:
    rows = []

    for frame in frames:
        matched_scan = associate_nearest_scan(
            frame_timestamp=frame.timestamp,
            scans=scans,
            max_time_diff=max_time_diff
        )

        if matched_scan is None:
            for cyl in frame.cylinders:
                rows.append({
                    "frame_id": frame.frame_id,
                    "frame_timestamp": frame.timestamp,
                    "scan_id": "",
                    "scan_timestamp": "",
                    "time_diff_s": "",
                    "cylinder_id": cyl.cylinder_id,
                    "num_candidate_points": 0,
                    "num_cluster_points": 0,
                    "mean_distance_m": "",
                    "std_distance_m": "",
                    "min_distance_m": "",
                    "max_distance_m": "",
                    "median_distance_m": "",
                    "status": "no_nearby_scan"
                })
            continue

        time_diff = abs(frame.timestamp - matched_scan.timestamp)

        for cyl in frame.cylinders:
            mask = build_mask(
                mask_pixels=cyl.mask_pixels,
                width=frame.image_width,
                height=frame.image_height
            )

            beam_idx, cand_ranges, cand_pixels, cand_points_lidar = get_candidate_points_for_mask(
                scan=matched_scan,
                K=K,
                T_cam_lidar=T_cam_lidar,
                mask=mask
            )

            if len(cand_ranges) == 0:
                rows.append({
                    "frame_id": frame.frame_id,
                    "frame_timestamp": frame.timestamp,
                    "scan_id": matched_scan.scan_id,
                    "scan_timestamp": matched_scan.timestamp,
                    "time_diff_s": time_diff,
                    "cylinder_id": cyl.cylinder_id,
                    "num_candidate_points": 0,
                    "num_cluster_points": 0,
                    "mean_distance_m": "",
                    "std_distance_m": "",
                    "min_distance_m": "",
                    "max_distance_m": "",
                    "median_distance_m": "",
                    "status": "no_points_on_mask"
                })
                continue

            clusters = split_into_contiguous_clusters(
                beam_indices=beam_idx,
                ranges=cand_ranges,
                pixels=cand_pixels,
                points_lidar=cand_points_lidar,
                max_beam_gap=max_beam_gap,
                max_range_jump=max_range_jump
            )

            best_cluster = choose_best_cluster(
                clusters=clusters,
                min_points=min_cluster_points
            )

            if best_cluster is None:
                rows.append({
                    "frame_id": frame.frame_id,
                    "frame_timestamp": frame.timestamp,
                    "scan_id": matched_scan.scan_id,
                    "scan_timestamp": matched_scan.timestamp,
                    "time_diff_s": time_diff,
                    "cylinder_id": cyl.cylinder_id,
                    "num_candidate_points": int(len(cand_ranges)),
                    "num_cluster_points": 0,
                    "mean_distance_m": "",
                    "std_distance_m": "",
                    "min_distance_m": "",
                    "max_distance_m": "",
                    "median_distance_m": "",
                    "status": "no_valid_cluster"
                })
                continue

            stats = compute_cluster_stats(best_cluster)

            rows.append({
                "frame_id": frame.frame_id,
                "frame_timestamp": frame.timestamp,
                "scan_id": matched_scan.scan_id,
                "scan_timestamp": matched_scan.timestamp,
                "time_diff_s": time_diff,
                "cylinder_id": cyl.cylinder_id,
                "num_candidate_points": int(len(cand_ranges)),
                "num_cluster_points": stats["num_points"],
                "mean_distance_m": stats["mean_distance_m"],
                "std_distance_m": stats["std_distance_m"],
                "min_distance_m": stats["min_distance_m"],
                "max_distance_m": stats["max_distance_m"],
                "median_distance_m": stats["median_distance_m"],
                "status": "ok"
            })

    return rows


def save_csv(rows: List[Dict], output_csv: str) -> None:
    if not rows:
        return

    fieldnames = [
        "frame_id",
        "frame_timestamp",
        "scan_id",
        "scan_timestamp",
        "time_diff_s",
        "cylinder_id",
        "num_candidate_points",
        "num_cluster_points",
        "mean_distance_m",
        "std_distance_m",
        "min_distance_m",
        "max_distance_m",
        "median_distance_m",
        "status",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, help="Path to calibration.json")
    parser.add_argument("--frames", required=True, help="Path to frames.json")
    parser.add_argument("--scans", required=True, help="Path to scans.json")
    parser.add_argument("--output", default="cylinder_distances.csv", help="Output CSV path")

    parser.add_argument("--max-time-diff", type=float, default=0.25,
                        help="Maximum allowed time difference between frame and scan in seconds")
    parser.add_argument("--max-beam-gap", type=int, default=2,
                        help="Maximum allowed beam index gap within a LiDAR cluster")
    parser.add_argument("--max-range-jump", type=float, default=0.12,
                        help="Maximum allowed range jump (m) within a LiDAR cluster")
    parser.add_argument("--min-cluster-points", type=int, default=2,
                        help="Minimum number of points required for a valid cluster")

    args = parser.parse_args()

    K, T_cam_lidar = load_calibration(args.calibration)
    frames = load_frames(args.frames)
    scans = load_scans(args.scans)

    rows = process_all_frames(
        frames=frames,
        scans=scans,
        K=K,
        T_cam_lidar=T_cam_lidar,
        max_time_diff=args.max_time_diff,
        max_beam_gap=args.max_beam_gap,
        max_range_jump=args.max_range_jump,
        min_cluster_points=args.min_cluster_points
    )

    save_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()