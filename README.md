# 🏆 NXP AIM India 2025 — Team LITEHeads

> **Regional Finalist · 2nd Place** — NXP Autonomous Intelligent Machines (AIM) India 2025

An autonomous warehouse robot solution built on the **NXP B3RB** rover platform, capable of navigating a simulated warehouse, detecting shelves via SLAM maps, decoding QR codes, and identifying shelf objects using a quantized YOLO model — all running in real time on ROS 2.

---

## 📽️ Challenge Overview

The **NXP AIM India Warehouse Challenge** tasks a B3RB rover with a sequential "treasure hunt" across a simulated warehouse:

1. Detect the first shelf from a SLAM-generated occupancy map
2. Navigate to it autonomously using the Nav2 stack
3. Decode a QR code that encodes the next shelf's direction angle (heuristic)
4. Identify objects on the shelf using YOLO-based detection
5. Publish QR + object data to reveal the next curtained shelf
6. Repeat for all N shelves in the minimum time with zero collisions

Scoring rewards object identification accuracy, QR decoding, time efficiency, and penalizes collisions and false detections.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Robot OS | ROS 2 Humble (Python) |
| Autonomous Navigation | Nav2 stack (NavigateToPose action client) |
| Mapping | SLAM Toolbox (occupancy grid) |
| Object Detection | YOLO11n float32 TFLite + custom NMS |
| QR Decoding | OpenCV + pyzbar |
| Shelf Detection | SciPy connected-component labeling + scikit-learn PCA |
| Simulation | Gazebo (CogniPilot AIRY / B3RB SIL) |
| Visualization | Foxglove Studio, matplotlib (draw map node) |
| GUI | Tkinter progress table |

---

## 📂 Repository Structure

```
.
├── NXP_AIM_INDIA_2025/                  # ROS 2 Python package
│   ├── b3rb_ros_aim_india/
│   │   ├── b3rb_ros_warehouse.py        # Main navigation & exploration node
│   │   ├── b3rb_ros_object_recog.py     # YOLO11n TFLite inference node
│   │   ├── b3rb_ros_draw_map.py         # SLAM map visualizer (debug)
│   │   └── b3rb_ros_model_remove.py     # Curtain removal (evaluation)
│   ├── resource/
│   │   ├── yolo11n_float32.tflite       # YOLO11n float32 model
│   │   ├── yolov5n-int8.tflite          # YOLOv5n INT8 quantized model
│   │   └── coco.yaml                    # COCO class labels
│   ├── setup.py
│   └── package.xml
└── config/                              # Tuned Nav2 & SLAM configurations
    ├── nav2.yaml                        # Nav2 stack parameters
    ├── slam.yaml                        # SLAM Toolbox parameters
    ├── nav_to_pose_bt.xml               # Behavior tree — single goal
    └── nav_through_poses_bt.xml         # Behavior tree — waypoints
```

---

## 🧠 Key Algorithms & Implementation

### Shelf Detection (SLAM-based)
- Subscribes to `/map` (SLAM occupancy grid) and `/global_costmap/costmap`
- Identifies connected high-occupancy clusters (SciPy `label` + `center_of_mass`) that match shelf dimensions
- Uses **PCA** (scikit-learn) to estimate each shelf's orientation from its point cloud
- Deduplicates detections across map updates with `add_shelf_if_unique`

### Heuristic-Guided Sequential Navigation
- The first shelf's direction is given as an `initial_angle` parameter
- Each QR code encodes the **angle** (0–360°) from the current position to the next shelf
- `find_shelves_in_direction(point, angle, tolerance=25°)` narrows candidates along the heuristic ray
- Falls back to **frontier-based exploration** (boundary between explored/unknown space) when no shelf is found, with a directional dot-product filter to stay aligned with the heuristic

### QR Code Decoding
- Subscribes to `/camera/image_raw/compressed`
- Preprocesses frames with OpenCV (grayscale, threshold) before passing to **pyzbar**
- Parses the decoded string to extract shelf ID and next-shelf angle (`qr_angle_decoded`)
- Publishes annotated debug images on `/debug_images/qr_code`

### Object Recognition (YOLO11n TFLite)
- Loads `yolo11n_float32.tflite` via `tflite_runtime`
- Custom `non_max_suppression` + `xywh2xyxy` post-processing pipeline
- Publishes detected objects on `/shelf_objects` (`synapse_msgs/WarehouseShelf`)
- Debug images published on `/debug_images/object_recog`

### Collision Recovery
- Tracks consecutive navigation failures against `recovery_threshold`
- On trigger: cancels current Nav2 goal → engages manual rover mode (`/cerebri/in/joy`) to back away from the obstacle → resumes autonomous navigation

---

## ⚙️ Setup & Running

> **Prerequisites:** Ubuntu 22.04, CogniPilot AIRY release installed. See the [official install guide](https://airy.cognipilot.org/getting_started/install/).

### 1. Clone into Cranium
```bash
git clone https://github.com/<your-username>/LITEHeads_NXP_AIM_INDIA_2025.git \
    ~/cognipilot/cranium/src/NXP_AIM_INDIA_2025
```

### 2. Install Python dependencies
```bash
pip install \
    torch==2.3.0 torchvision==0.18.0 numpy==1.26.4 \
    opencv-python==4.11.0.86 scipy==1.15.1 scikit-learn==1.5.2 \
    tk==0.1.0 pyzbar==0.1.9 matplotlib==3.5.1 \
    pyyaml==6.0.2 tflite-runtime==2.14.0
```

### 3. Build and launch simulation
```bash
cd ~/cognipilot/cranium/
colcon build
source install/setup.bash

# Example — Warehouse 2 (4 shelves)
ros2 launch b3rb_gz_bringup sil.launch.py \
    world:=nxp_aim_india_2025/warehouse_2 \
    warehouse_id:=2 shelf_count:=4 \
    initial_angle:=040.6 x:=0.0 y:=-7.0 yaw:=1.57
```

Available worlds:

| World | Shelves | Initial Angle | Start Pose |
|---|---|---|---|
| `warehouse_1` | 2 | 135.0° | (0, 0, 0°) |
| `warehouse_2` | 4 | 40.6° | (0, −7, 90°) |
| `warehouse_3` | 3 | 45.0° | (5, −2, 180°) |
| `warehouse_4` | 5 | 45.0° | (5, −2, 180°) |

### 4. (Optional) Visualize SLAM map
```bash
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_aim_india visualize
```

---

## 🏅 Competition Result

**NXP AIM India 2025 — Regional Finalist, 2nd Place**  
Team **LITEHeads** · BITS Pilani

---

## 📄 License

Licensed under the [Apache 2.0 License](NXP_AIM_INDIA_2025/LICENSE).  
Original framework by [NXP HoverGames](https://github.com/NXPHoverGames/NXP_AIM_INDIA_2025); competition implementation by Team LITEHeads.
