# MTRX5700 Assignment 2 — Perceptron: Camera-LiDAR Perception Pipeline

A robotics perception system that detects orange traffic cones, classifies the signs mounted on them, and estimates their distance using fused camera and LiDAR data.

---

## Overview

The project is split into three components that build on each other:

| Component | Location | Purpose |
|-----------|----------|---------|
| Traffic Sign Classifier | `Vision Task/` | Train a ResNet18 CNN to classify traffic signs |
| Standalone Detector | `Task 3/detect_estimate_v2.py` | Offline cone detection + sign classification + distance estimation |
| ROS 2 Node | `src/perceptron/` | Real-time deployment of the full pipeline as a ROS 2 node |

### Full Pipeline

```
Camera Intrinsic Calibration  ─┐
                                ├─► calibration.json ─► LiDAR Projection ─► Distance Estimate
Camera-LiDAR Extrinsic Cal.   ─┘

Camera Frame ─► Cone Detection ─► Sign Crop ─► ResNet18 Classifier ─► Sign Label
```

---

## Repository Structure

```
Assignment 2/
├── Vision Task/                    # Task 1 — Traffic sign classifier (student exercises)
│   ├── README.md                   # Detailed instructions and TODOs
│   ├── train_final.py              # Training entry point
│   ├── network.py                  # ResNet18 architecture
│   ├── dataset.py                  # Data loading & preprocessing
│   ├── inference.py                # Model inference helper
│   └── vis_utils.py                # Visualisation utilities
│
├── Task 3/                         # Task 3 — Offline detector/estimator
│   ├── detect_estimate_v2.py       # Main script (cone detect + sign classify + LiDAR distance)
│   ├── calibration.json            # Camera intrinsics (K) and extrinsics (T_cam_lidar)
│   ├── ckpt_*.pth                  # Trained ResNet18 checkpoint
│   ├── lidar_matched_images*/      # Matched camera frames
│   ├── pointcloud_*/               # Corresponding LiDAR point clouds (.pcd)
│   ├── output_distances/           # Annotated output images
│   ├── output_signs_distances/     # Cropped sign images
│   └── cylinder_distances.csv      # Per-cone distance log
│
└── src/perceptron/                 # Task 2/4 — ROS 2 package
    └── perceptron/
        ├── traffic_sign_node.py            # ROS 2 node (main entry point)
        ├── cylinder_sign_detection.py      # Cone & sign detection module
        ├── calibration.json                # Calibration parameters
        ├── camera_lidar_calibration/       # Calibration tools
        │   ├── cam_intrinsic.py            # Camera intrinsic calibration (checkerboard)
        │   ├── cam_lidar_2d_icp.py         # Camera-LiDAR extrinsic calibration (ICP)
        │   ├── icp_2d.py                   # 2D ICP algorithm
        │   ├── gui.py                      # GUI for manual point selection
        │   ├── camera_utils.py             # Calibration file I/O
        │   └── lidar_utils.py              # LiDAR projection utilities
        └── traffic_sign_classification/    # Classifier (mirrors Vision Task/)
            ├── network.py
            ├── dataset.py
            ├── inference.py
            ├── train_final.py
            └── vis_utils.py
```

---

## Dependencies

### Python packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy scipy matplotlib pandas seaborn scikit-learn tqdm open3d
```

### ROS 2 (for the `src/perceptron` node only)

- ROS 2 Humble or later
- Additional ROS packages: `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`

---

## Component Details

### 1. Traffic Sign Classifier (`Vision Task/`)

Trains a **ResNet18** CNN on a filtered subset of the [GTSRB dataset](https://benchmark.ini.rub.de/) to classify 5 traffic sign types:

| Label | Class |
|-------|-------|
| 0 | Stop |
| 1 | Turn right |
| 2 | Turn left |
| 3 | Ahead only |
| 4 | Roundabout mandatory |

**Train the model:**

```bash
cd "Vision Task"
python train_final.py --optimizer rmsprop --lr 0.005 --batch_size 64 --epochs 100
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.01 | Learning rate |
| `--batch_size` | 64 | Batch size |
| `--epochs` | 100 | Training epochs |
| `--optimizer` | rmsprop | `sgd` / `adam` / `rmsprop` |
| `--scheduler` | cosine | `cosine` / `step` / `none` |
| `--resume` / `-r` | — | Resume from checkpoint |

Outputs are saved to `results/` (plots) and `checkpoint/` (model weights).

**Best result achieved:** 97.22% test accuracy (100 epochs, RMSprop, lr=0.005, with augmentation).

---

### 2. Calibration Tools (`src/perceptron/perceptron/camera_lidar_calibration/`)

Produces the `calibration.json` file used by both the standalone detector and the ROS 2 node.

#### Camera intrinsic calibration

Uses checkerboard images to estimate the camera matrix **K** and distortion coefficients.

```bash
python cam_intrinsic.py <image_dir> --output calibration.json
```

#### Camera-LiDAR extrinsic calibration

Aligns camera and LiDAR coordinate frames using manually selected checkerboard correspondences and 2D ICP.

```bash
python cam_lidar_2d_icp.py
```

The output `calibration.json` contains:
- `K` — 3×3 camera intrinsic matrix
- `T_cam_lidar` — 4×4 rigid-body transform from LiDAR frame to camera frame

---

### 3. Standalone Detector (`Task 3/detect_estimate_v2.py`)

Processes pre-recorded matched camera frames and LiDAR point clouds to detect cones, classify their signs, and estimate distances. Results are written to `output_distances/` and `cylinder_distances.csv`.

**Configuration** (edit the constants at the top of the file):

| Variable | Description |
|----------|-------------|
| `IMAGE_DIR` | Directory of matched camera frames |
| `PCD_DIR` | Directory of corresponding `.pcd` files |
| `CALIB_FILE` | Path to `calibration.json` |
| `MODEL_PATH` | Path to trained `.pth` checkpoint |
| `VIEW_RESULTS` | `True` to open the interactive viewer |
| `START_INDEX` | Frame index to start from |

**Run:**

```bash
cd "Task 3"
python detect_estimate_v2.py
```

**Interactive viewer controls:**

| Key | Action |
|-----|--------|
| `n` / `Space` / `Enter` | Next frame |
| `b` | Previous frame |
| `q` / `Esc` | Quit |

**Detection pipeline:**

1. **Cone detection** — HSV colour thresholding (with CLAHE) → morphological cleanup → bounding box extraction → iterative merge of fragmented blobs
2. **Cone tracking** — Greedy nearest-centroid tracker assigns stable IDs across frames
3. **Sign detection** — Polynomial silhouette fit to the orange cone edges; the largest non-orange blob inside the silhouette is the sign crop
4. **Sign classification** — ResNet18 inference on the extracted sign crop
5. **Distance estimation** — LiDAR points projected into the image via pinhole model; median of points landing on the cone silhouette mask gives the ground-plane range

**CSV output columns:** `frame`, `image`, `pcd`, `cone_idx`, `track_id`, `x`, `y`, `w`, `h`, `distance_m`, `sign`, `status`

---

### 4. ROS 2 Node (`src/perceptron/`)

Deploys the full pipeline for real-time operation.

**Build and run:**

```bash
cd <ros2_ws>
colcon build --packages-select perceptron
source install/setup.bash
ros2 run perceptron traffic_sign_node
```

**Subscriptions:**

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw/compressed` | `CompressedImage` | Live camera frames |
| `/pointcloud2d` | `PointCloud` | 2-D LiDAR scans |

**Publications:**

| Topic | Type | Description |
|-------|------|-------------|
| `/perceptron/viewer` | `Image` | Annotated camera frame |
| `/perceptron/sign_labels` | `String` | JSON list of sign names |
| `/perceptron/distances_m` | `Float32MultiArray` | Per-cone distances (m) |
| `/perceptron/detections` | `String` | JSON array of full detection records |

**Node parameters** (set via `ros2 run ... --ros-args -p <param>:=<value>`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image_raw` | Raw image topic |
| `compressed_image_topic` | `/camera/image_raw/compressed` | Compressed image topic |
| `pointcloud_topic` | `/pointcloud2d` | LiDAR topic |

---

## Quick-Start Checklist

1. Install Python dependencies (see above).
2. Download the GTSRB dataset pickles (`train.p`, `valid.p`, `test.p`) from Google Drive and place them in `Vision Task/`.
3. Train the classifier: `python "Vision Task/train_final.py"` — or use the pre-trained checkpoint in `Task 3/`.
4. Copy the trained checkpoint to `Task 3/` and update `MODEL_PATH` in `detect_estimate_v2.py`.
5. Run the standalone detector: `python "Task 3/detect_estimate_v2.py"`.
6. (Optional) Build and run the ROS 2 node for live operation.
