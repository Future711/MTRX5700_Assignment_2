#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import numpy as np

import cv2

import rclpy
import rosbag2_py

import cv_bridge
from sensor_msgs.msg import Image

from rclpy.serialization import deserialize_message


def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")

    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    # print("Extract images from " + args.bag_file + " on topic " + args.image_topic + " into " + args.output_dir)

    bag_file = args.bag_file
    output_dir = args.output_dir
    image_topic = args.image_topic

    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
        print("Created directory"+output_dir)
    else: 
        print(output_dir+" already exists.")


    output_dir_image = output_dir+"/images/"
    if os.path.isdir(output_dir_image) == False:
        os.makedirs(output_dir_image)
        print("Created directory"+output_dir_image)
    else: 
        print(output_dir_image+" already exists.")

    # bag = rosbag.Bag(args.bag_file, "r")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_file,
        storage_id='mcap')

    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    cvbridge = cv_bridge.CvBridge()

    count = 0
    while reader.has_next():
        msg = reader.read_next()
        if msg[0] == image_topic:
            decoded_data = deserialize_message(msg[1], Image) # get serialized version of message and decode it
            # Convert to an 8-bit, viewable format before writing.
            # 'passthrough' can yield 16-bit/Bayer/YUV which looks "blank" in normal viewers.
            enc = (decoded_data.encoding or "").lower()
            if enc in {"16uc1", "mono16"}:
                cv_image_16 = cvbridge.imgmsg_to_cv2(decoded_data, desired_encoding="passthrough")
                cv_image = cv2.convertScaleAbs(cv_image_16, alpha=(255.0 / 65535.0))
            elif enc == "rgb8":
                cv_rgb = cvbridge.imgmsg_to_cv2(decoded_data, desired_encoding="rgb8")
                cv_image = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
            else:
                cv_image = cvbridge.imgmsg_to_cv2(decoded_data, desired_encoding="bgr8")
            t = msg[2]

            # Do this to selectively save images from the cv.imshow window
            # cv2.imshow("Frame", cv_image)
            # cv2.waitKey()

            # Do this to just save everything to a directory
            ok = cv2.imwrite(os.path.join(output_dir_image, f"{t}.png"), cv_image)
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
            # cv2.imwrite(os.path.join(output_dir_image, "%i.png" % t), cv_image)
            # print("Wrote image" + str(count) + ": " + str(t) + ".png")

            count += 1


if __name__ == '__main__':
    main()
