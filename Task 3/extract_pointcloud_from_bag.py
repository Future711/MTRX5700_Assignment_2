#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract PointCloud messages from a rosbag2 and write PCD files."""

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud


def _iter_pointcloud_messages(
    bag_uri: str,
    topic: str,
    storage_id: str,
) -> Iterable[Tuple[int, PointCloud]]:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    while reader.has_next():
        topic_name, serialized_data, timestamp_ns = reader.read_next()
        if topic_name != topic:
            continue
        yield timestamp_ns, deserialize_message(serialized_data, PointCloud)


def _write_pcd_with_open3d(path: str, cloud: PointCloud) -> None:
    pts = cloud.points
    n = len(pts)

    xyz = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(pts):
        xyz[i, 0] = p.x
        xyz[i, 1] = p.y
        xyz[i, 2] = p.z

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Map a common "intensity" channel to grayscale colors if present.
    for ch in cloud.channels:
        if ch.name == "intensity" and len(ch.values) == n:
            intens = np.asarray(ch.values, dtype=np.float32)
            max_val = float(np.max(intens)) if intens.size > 0 else 0.0
            if max_val > 0.0:
                intens = intens / max_val
            colors = np.stack([intens, intens, intens], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            break

    o3d.io.write_point_cloud(path, pcd, write_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sensor_msgs/msg/PointCloud messages from a rosbag2 into .pcd files."
    )
    parser.add_argument("bag_uri", help="Input rosbag2 directory (uri).")
    parser.add_argument("output_dir", help="Output directory to write PCD files into.")

    parser.add_argument(
        "--topic",
        default="/pointcloud2d",
        help="PointCloud topic to extract (default: /pointcloud2d).",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help="rosbag2 storage id (e.g. mcap, sqlite3). Default: mcap.",
    )
    parser.add_argument(
        "--prefix",
        default="cloud",
        help="Output filename prefix (default: cloud).",
    )
    parser.add_argument(
        "--use-timestamp",
        action="store_true",
        help="Name files as <prefix>_<timestamp_ns>.pcd instead of a counter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after writing N clouds (0 = no limit).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    for timestamp_ns, cloud in _iter_pointcloud_messages(
        bag_uri=args.bag_uri,
        topic=args.topic,
        storage_id=args.storage_id,
    ):
        if args.use_timestamp:
            filename = f"{args.prefix}_{timestamp_ns}.pcd"
        else:
            filename = f"{args.prefix}_{count:06d}.pcd"

        out_path = os.path.join(args.output_dir, filename)
        _write_pcd_with_open3d(out_path, cloud)

        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Wrote {count} PCD file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
