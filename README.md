# SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending

Robotic perception of transparent objects presents unique challenges due to their refractive properties, lack of texture, and limitations of conventional RGB-D sensors in capturing reliable depth information. These challenges significantly hinder robotic manipulation capabilities in real-world settings such as household assistance, hospitality, and healthcare.

To address these issues, we propose **SPILL**: A lightweight perception pipeline for **S**ize, **P**ose, and **I**nternal **L**iquid **L**evel estimation of unknown transparent glassware using a single view. SPILL combines object detection with semantic keypoint detection and operates without requiring object-specific 3D models or depth completion. We demonstrate its effectiveness in autonomous robotic pouring tasks.

Additionally, to enhance the robustness and generalization of keypoint detection to diverse real-world scenarios, we introduce *Glasses-in-the-Wild*, a new dataset that captures a wide variety of glass types in realistic environments. Evaluated on a robot manipulator, SPILL achieves a **93.6% success rate** across 500 autonomous pours with 20 unseen glasses in three diverse real-world scenes.

We further demonstrate robustness through multiple live public events in real-world, human-centered environments. In one recorded session, the robot autonomously served 62 drinks with a **98.3% success rate**.
These results demonstrate that task-relevant keypoint detection enables scalable, real-world transparent object interaction, paving the way for practical applications in service and assistive robotics — without spilling a drop.

[![Watch the demo](https://img.youtube.com/vi/gcHi0ebrDps/0.jpg)](https://www.youtube.com/watch?v=gcHi0ebrDps)

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Louadria/SPILL)

## Installation

```bash
# Clone the repository
git clone https://github.com/Louadria/SPILL.git
cd SPILL

# Install as a Python package (editable mode for development)
pip install -e .

# Optional: visualization tools (Open3D, Matplotlib)
pip install -e ".[visualize]"
```

## Quick Start

### 2D Keypoint Detection (RGB only)

```python
from spill import GlassDetector
import cv2

# Load the detector
detector = GlassDetector(
    keypoint_checkpoint="checkpoints/wild_glasses.ckpt",
    yolo_model_path="checkpoints/yolov8m.pt",
    device="cuda",  # or "cpu"
)

# Detect glasses in an image
image = cv2.imread("path/to/image.jpg")
keypoints_list = detector.detect(image)

for kp in keypoints_list:
    print(f"Glass at bbox {kp.bounding_box}")
    print(f"  Bottom front: {kp.bottom_front}")
    print(f"  Top front: {kp.top_front}")
    print(f"  Top left: {kp.top_left}")
    print(f"  Top right: {kp.top_right}")
    if kp.fluid_level is not None:
        print(f"  Fluid level: {kp.fluid_level}")
```

### Full 3D Reconstruction (requires depth + camera calibration)

```python
from spill import GlassDetector, reconstruct_glass_3d, detect_table_height
import numpy as np

# ... load detector and image as above ...

# Camera parameters (from your RGB-D camera calibration)
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1],
])
X_World_Camera = np.eye(4)  # camera-to-world transform

# Estimate table height from depth map
depth_map = load_depth(...)  # HxW array in meters
table_height = detect_table_height(depth_map, K, X_World_Camera)

# Reconstruct 3D glass properties
for kp in keypoints_list:
    glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
    if glass:
        print(f"Glass: radius={glass.radius*100:.1f}cm, "
              f"height={glass.height*100:.1f}cm, "
              f"fluid={glass.fluid_percentage*100:.0f}%")
```

See `examples/` for complete working scripts:
- `examples/detect_image.py` — detect keypoints on a single image
- `examples/demo_3d.py` — full 3D pipeline with depth map
- `examples/test_mobi.py` — integration with CriticBarista/MOBI robot

## API Reference

### `GlassDetector`

Main entry point for glass detection.

```python
GlassDetector(
    keypoint_checkpoint: str,   # Path to .ckpt file
    yolo_model_path: str,       # Path to YOLO weights (.pt)
    device: str = "cuda",       # "cuda" or "cpu"
    glass_classes: list[str] = None,  # ["cup", "vase", "wine glass"]
)
```

**Methods:**
- `detect(image) -> List[GlassKeypoints]` — Detect glasses and return 2D keypoints

### `reconstruct_glass_3d(keypoints, camera_intrinsics, table_height, X_World_Camera) -> GlassDetection`

Reconstruct 3D glass properties from 2D keypoints, camera intrinsics, and table plane.

### `detect_table_height(depth_map, camera_intrinsics, X_World_Camera) -> float`

Estimate table height from a depth map using histogram mode estimation.

## Glasses-in-the-Wild Dataset

A crowdsourced dataset of transparent glassware in diverse domestic and real-world environments, annotated with bounding box and keypoints.

Available at: [10.5281/zenodo.17288314](https://doi.org/10.5281/zenodo.17288314)

## Architecture

The SPILL pipeline has two stages:

1. **Object Detection** — YOLOv8m detects glass bounding boxes (COCO classes: cup, vase, wine glass)
2. **Keypoint Detection** — A MaxViT-Unet model predicts 5 semantic keypoints per glass:
   - Bottom front, top front, top left, top right (structural)
   - Fluid level (liquid surface)
3. **3D Reconstruction** — Geometric optimization back-projects 2D keypoints to 3D using camera intrinsics and the table plane, then iteratively refines radius, height, and tilt angle.

## Integration with CriticBarista / MOBI

SPILL was developed as part of the CriticBarista project for the MOBI robotic bartender. The library is designed as a drop-in replacement for the `GlassLocalizer` in the CriticBarista pipeline.

See `examples/test_mobi.py` for integration instructions.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{adriaens2025spill,
  title={SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending},
  author={Adriaens, Louis et al.},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
