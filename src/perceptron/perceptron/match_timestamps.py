#!/usr/bin/env python3
"""Match image and pointcloud timestamps from a rosbag using nearest-neighbour.

Outputs:
  timestamps_img.txt  - all image timestamps found in the bag
  timestamps.txt      - image timestamps matched to each pointcloud scan
  timestamps_matched.txt - tab-separated pairs: pc_ts  img_ts  delta_ms
"""

import argparse
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud


def read_timestamps_from_bag(bag_uri, img_topic, pc_topic, storage_id="mcap"):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    img_ts = []
    pc_ts = []

    while reader.has_next():
        topic, _, timestamp_ns = reader.read_next()
        if topic == img_topic:
            img_ts.append(timestamp_ns)
        elif topic == pc_topic:
            pc_ts.append(timestamp_ns)

    return sorted(img_ts), sorted(pc_ts)


def nearest_match(pc_timestamps, img_timestamps, max_delta_ms=100):
    """For each PC timestamp find the nearest image timestamp."""
    pairs = []
    img_arr = img_timestamps
    j = 0
    for pc_t in pc_timestamps:
        # Advance j while the next image timestamp is closer
        while j + 1 < len(img_arr) and abs(img_arr[j + 1] - pc_t) < abs(img_arr[j] - pc_t):
            j += 1
        delta_ns = abs(img_arr[j] - pc_t)
        delta_ms = delta_ns / 1e6
        if delta_ms <= max_delta_ms:
            pairs.append((pc_t, img_arr[j], delta_ms))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Match image and pointcloud timestamps.")
    parser.add_argument("bag_uri", help="Path to rosbag2 directory.")
    parser.add_argument("--img-topic", default="/camera/image_raw")
    parser.add_argument("--pc-topic", default="/pointcloud2d")
    parser.add_argument("--max-delta-ms", type=float, default=100,
                        help="Maximum allowed time difference in ms (default: 100).")
    args = parser.parse_args()

    print("Reading timestamps from bag...")
    img_ts, pc_ts = read_timestamps_from_bag(args.bag_uri, args.img_topic, args.pc_topic)
    print(f"  Found {len(img_ts)} image messages, {len(pc_ts)} pointcloud messages.")

    # Write all image timestamps
    # with open("timestamps_img.txt", "w") as f:
    #     for t in img_ts:
    #         f.write(f"{t}\n")
    # print(f"Wrote timestamps_img.txt ({len(img_ts)} entries)")

    # Match
    pairs = nearest_match(pc_ts, img_ts, max_delta_ms=args.max_delta_ms)
    # print(f"Matched {len(pairs)} pairs (max delta: {args.max_delta_ms} ms).")

    # Keep one image timestamp per matched pointcloud message (no deduplication).
    matched_img_ts = [img_t for _, img_t, _ in pairs]

    # Write matched image timestamps (used by extract_images_from_bag.py)
    with open("timestamps.txt", "w") as f:
        for t in matched_img_ts:
            f.write(f"{t}\n")
    print(f"Wrote timestamps.txt ({len(matched_img_ts)} image timestamps)")

    # Write full pair table
    # with open("timestamps_matched.txt", "w") as f:
    #     f.write("pc_timestamp\timg_timestamp\tdelta_ms\n")
    #     for pc_t, img_t, delta in sorted(pairs):
    #         f.write(f"{pc_t}\t{img_t}\t{delta:.3f}\n")
    # print("Wrote timestamps_matched.txt (pc_ts <-> img_ts pairs with delta)")


if __name__ == "__main__":
    main()
