# SPILL

**S**ize, **P**ose, and **I**nternal **L**iquid **L**evel estimation of transparent glassware from a single view.

[Paper](https://arxiv.org/abs/...) · [HuggingFace Demo](https://huggingface.co/spaces/Louadria/SPILL) · [Dataset](https://doi.org/10.5281/zenodo.17288314)

[![Watch the demo](https://img.youtube.com/vi/gcHi0ebrDps/0.jpg)](https://www.youtube.com/watch?v=gcHi0ebrDps)

## Installation

```bash
git clone https://github.com/Louadria/SPILL.git
cd SPILL
pip install -e .
```

## Quick Start

### 2D Keypoint Detection (RGB only)

```bash
python examples/detect_image.py --image examples/data/glass.jpg
```

```python
from spill import GlassDetector
import cv2

detector = GlassDetector(
    keypoint_checkpoint="checkpoints/wild_glasses.ckpt",
    yolo_model_path="checkpoints/yolov8m.pt",
    device="cuda",
)

image = cv2.imread("path/to/image.jpg")
for kp in detector.detect(image):
    print(f"Glass at bbox {kp.bounding_box}")
    print(f"  bottom_front:  {kp.bottom_front}")
    print(f"  top_front:     {kp.top_front}")
    print(f"  top_left:      {kp.top_left}")
    print(f"  top_right:     {kp.top_right}")
    print(f"  fluid_level:   {kp.fluid_level}")
```

### Full 3D Pipeline

```bash
python examples/demo_3d.py --image examples/data/glass.jpg --depth examples/data/glass_depth.png
```

```python
from spill import GlassDetector, reconstruct_glass_3d, detect_table_height
import numpy as np

detector = GlassDetector(...)  # as above
keypoints_list = detector.detect(image)

# Camera parameters (from calibration)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
X_World_Camera = ...  # 4x4 camera-to-world transform
table_height = detect_table_height(depth_map, K, X_World_Camera)

# Reconstruct all glasses at once
glasses = reconstruct_glass_3d(keypoints_list, K, table_height, X_World_Camera)
for g in glasses:
    if g is None:
        continue
    print(f"radius={g.radius*100:.1f}cm height={g.height*100:.1f}cm fluid={g.fluid_percentage*100:.0f}%")
```

### Live Camera Demo

Real-time glass localization with a ZED or RealSense camera:

```bash
# ZED camera
python examples/live_demo.py --camera zed --serial 943222073454

# RealSense camera
python examples/live_demo.py --camera realsense --serial 943222073454

# USB webcam (2D detection only — no depth)
python examples/live_demo.py --camera usb --device 0
```

## API Reference

### GlassDetector

```python
GlassDetector(
    keypoint_checkpoint: str,   # Path to .ckpt file
    yolo_model_path: str,       # Path to YOLO weights (.pt)
    device: str = "cuda",       # "cuda" or "cpu"
    glass_classes: list[str] = None,  # ["cup", "vase", "wine glass"]
)
```

**Methods:**
- `detect(image) -> List[GlassKeypoints]` — detect glasses in a BGR image

### reconstruct_glass_3d(keypoints, camera_intrinsics, table_height, X_World_Camera) -> GlassDetection | List[GlassDetection]

Reconstruct 3D glass properties from 2D keypoints. Accepts a single `GlassKeypoints` or a list — returns a `GlassDetection` or list accordingly (with `None` for any that fail). Each result has `center_3d`, `radius`, `height`, `glass_angle`, `fluid_percentage`, and `reprojection_error`.

### detect_table_height(depth_map, camera_intrinsics, X_World_Camera) -> float

Estimate table height (meters) from a depth map using histogram mode estimation.

## Architecture

1. **Object Detection** — YOLOv8m detects glass bounding boxes (cup, vase, wine glass)
2. **Keypoint Detection** — MaxViT-Unet predicts semantic keypoints per glass: bottom front, top front, top left, top right, fluid level
3. **3D Reconstruction** — Geometric optimization back-projects 2D keypoints to 3D using camera intrinsics and the table plane, then iteratively refines radius, height, and tilt angle

## Citation

```bibtex
@article{adriaens2025spill,
  title={SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending},
  author={Adriaens, Louis et al.},
  year={2025}
}
```

## License

MIT — see [LICENSE](LICENSE)
