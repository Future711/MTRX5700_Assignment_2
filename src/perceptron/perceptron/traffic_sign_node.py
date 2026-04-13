#!/usr/bin/env python3
"""ROS 2 node for cone detection, sign classification, and distance estimation."""

import json
import os

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, PointCloud
from std_msgs.msg import Float32MultiArray, String


from .cylinder_sign_detection import (
    build_silhouette_mask,
    detect_cones,
    detect_sign,
    fit_cone_bounds,
    threshold_orange,
)
from .camera_lidar_calibration.camera_utils import load_calibration
from .camera_lidar_calibration.lidar_utils import estimate_distance
from .traffic_sign_classification.inference import inference, load_model


class TrafficSignNode(Node):
    """ROS 2 node that detects orange traffic cylinders, reads their signs, and
    estimates distances using a fused camera-LiDAR pipeline.

    Subscriptions:
        /camera/image_raw/compressed  -- live compressed camera frames.
        /pointcloud2d                 -- 2-D LiDAR point cloud (x, y, z).

    Publications:
        /perceptron/viewer       -- annotated camera image with bounding boxes.
        /perceptron/sign_labels  -- JSON list of classified sign names per frame.
        /perceptron/distances_m  -- Float32MultiArray of per-cone distances (m).
        /perceptron/detections   -- JSON array of full detection records.
    """

    def __init__(self):
        """Initialise subscribers, publishers, model, and calibration resources."""
        super().__init__("traffic_sign_node")

        this_dir = os.path.dirname(os.path.abspath(__file__))
        default_calib = os.path.join(this_dir, "calibration.json")
        default_model = os.path.join(
            this_dir,
            "traffic_sign_classification",
            "checkpoint",
            "ckpt_rmsprop_lr_0.005_bs_64_ep_100_True_v1.pth",
        )

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("compressed_image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("pointcloud_topic", "/pointcloud2d")
        self.declare_parameter("viewer_topic", "/perceptron/viewer")
        self.declare_parameter("labels_topic", "/perceptron/sign_labels")
        self.declare_parameter("distances_topic", "/perceptron/distances_m")
        self.declare_parameter("detections_topic", "/perceptron/detections")
        self.declare_parameter("calibration_file", default_calib)
        self.declare_parameter("model_path", default_model)

        image_topic = self.get_parameter("image_topic").value
        compressed_image_topic = self.get_parameter("compressed_image_topic").value
        pointcloud_topic = self.get_parameter("pointcloud_topic").value
        viewer_topic = self.get_parameter("viewer_topic").value
        labels_topic = self.get_parameter("labels_topic").value
        distances_topic = self.get_parameter("distances_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        calibration_file = self.get_parameter("calibration_file").value
        model_path = self.get_parameter("model_path").value

        self.k, self.t_cam_lidar = load_calibration(calibration_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, self.device)

        self.bridge = CvBridge()
        self.latest_points = np.empty((0, 3), dtype=float)
        self.latest_cloud_stamp_ns = None

        # Raw image only for recorded bags. For live camera feed, use compressed image topic to save bandwidth.
        # self.image_sub = self.create_subscription(
        #     Image,
        #     image_topic,
        #     self.image_callback,
        #     qos_profile_sensor_data,
        # )
        self.compressed_image_sub = self.create_subscription(
            CompressedImage,
            compressed_image_topic,
            self.compressed_image_callback,
            qos_profile_sensor_data,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud,
            pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        # Viewer should only be used to demonstrate the functionality of the node at the current stage, and is not required for the core detection pipeline. It can be turned off to save bandwidth if needed.
        self.viewer_pub = self.create_publisher(Image, viewer_topic, qos_profile_sensor_data)
        self.labels_pub = self.create_publisher(String, labels_topic, 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, distances_topic, 10)
        self.detections_pub = self.create_publisher(String, detections_topic, 10)

        self.get_logger().info(
            f"Subscribed: {image_topic}, {compressed_image_topic}, {pointcloud_topic} | Publishing: {viewer_topic}, {labels_topic}, {distances_topic}, {detections_topic}"
        )

    def _stamp_to_ns(self, stamp) -> int:
        """Convert a ROS header stamp to nanoseconds.

        Args:
            stamp: ROS time stamp object with sec and nanosec fields.

        Returns:
            Integer timestamp in nanoseconds.
        """
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def pointcloud_callback(self, msg: PointCloud):
        """Cache the most recent LiDAR point cloud as an (N, 3) numpy array.

        The point cloud is stored so that the next image callback can use the
        latest available scan without needing to synchronise the two topics.
        The cloud timestamp is saved separately so detection records can report
        how fresh the LiDAR data was relative to the image.

        Args:
            msg: Incoming PointCloud message.

        Returns:
            None.
        """
        if not msg.points:
            self.latest_points = np.empty((0, 3), dtype=float)
        else:
            pts = np.zeros((len(msg.points), 3), dtype=float)
            for i, p in enumerate(msg.points):
                pts[i, 0] = p.x
                pts[i, 1] = p.y
                pts[i, 2] = p.z
            self.latest_points = pts

        if msg.header.stamp.sec or msg.header.stamp.nanosec:
            self.latest_cloud_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        else:
            self.latest_cloud_stamp_ns = None

    def image_callback(self, msg: Image):
        """Decode a raw ROS Image message and process it.

        Args:
            msg: Incoming raw image message.

        Returns:
            None.
        """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._process_frame(msg.header, frame)

    def compressed_image_callback(self, msg: CompressedImage):
        """Decode a CompressedImage message and process it.

        Args:
            msg: Incoming compressed image message.

        Returns:
            None.
        """
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("Failed to decode CompressedImage frame")
            return
        self._process_frame(msg.header, frame)

    def _process_frame(self, header, frame):
        """Run the full detection pipeline on one camera frame.

        For each detected cone the method:
          1. Crops the cone region from the frame.
          2. Fits polynomial bounds to the orange body to define the silhouette.
          3. Projects the cached LiDAR scan into the image plane and reads off
             the median depth of points landing inside the cone silhouette mask.
          4. Extracts the sign crop (non-orange area inside the silhouette) and
             runs the ResNet-18 classifier to obtain a class label.
          5. Annotates the frame with bounding boxes, distances, and labels.

        Publishes the annotated image, sign labels, distances, and full
        detection records (including timestamps for both sensors).

        Args:
            header: ROS message header for the image frame.
            frame: BGR image array.

        Returns:
            None.
        """
        points_lidar = self.latest_points.copy()

        boxes_result = detect_cones(frame)
        boxes = boxes_result[0] if isinstance(boxes_result, tuple) else boxes_result

        annotated = frame.copy()
        labels = []
        distances = []
        detections = []

        image_stamp_ns = None
        if header.stamp.sec or header.stamp.nanosec:
            image_stamp_ns = self._stamp_to_ns(header.stamp)

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

            dist = estimate_distance(points_lidar, self.k, self.t_cam_lidar, cone_region_mask)
            distances.append(float("nan") if dist is None else float(dist))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dist_label = f"{dist:.2f}m" if dist is not None else "no dist"
            cv2.putText(annotated, dist_label, (x + 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            sign_label = ""
            sign_crop = detect_sign(cone_crop, orange_mask=orange_mask, left_poly=left_poly, right_poly=right_poly)
            if sign_crop is not None:
                sign_label = inference(self.model, self.device, sign_crop)
                cv2.putText(
                    annotated,
                    sign_label,
                    (x + 4, y + h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            labels.append(sign_label)
            detections.append(
                {
                    "cone_idx": cone_idx,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "distance_m": None if dist is None else float(dist),
                    "sign_label": sign_label,
                }
            )
        # cv2.imshow("Annotated", annotated)
        # cv2.waitKey(1)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = header
        self.viewer_pub.publish(annotated_msg)

        labels_payload = {
            "stamp_ns": image_stamp_ns,
            "labels": labels,
        }
        self.labels_pub.publish(String(data=json.dumps(labels_payload)))

        self.distances_pub.publish(Float32MultiArray(data=distances))

        detections_payload = {
            "stamp_ns": image_stamp_ns,
            "cloud_stamp_ns": self.latest_cloud_stamp_ns,
            "detections": detections,
        }
        self.detections_pub.publish(String(data=json.dumps(detections_payload)))


def main(args=None):
    """Run the traffic sign node process.

    Args:
        args: Optional ROS argument list.

    Returns:
        None.
    """
    rclpy.init(args=args)
    node = TrafficSignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
