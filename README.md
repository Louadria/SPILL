# SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending

[![HuggingFace Demo](https://img.shields.io/badge/%F0%9F%A4%9D-HuggingFace%20Demo-blue)](https://huggingface.co/spaces/Louadria/SPILL)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)](https://doi.org/10.5281/zenodo.17288314)

Robotic perception of transparent objects presents unique challenges due to their refractive properties, lack of texture, and limitations of conventional RGB-D sensors in capturing reliable depth information. These challenges significantly hinder robotic manipulation capabilities in real-world settings such as household assistance, hospitality, and healthcare.

To address these issues, we propose **SPILL**: A lightweight perception pipeline for **S**ize, **P**ose, and **I**nternal **L**iquid **L**evel estimation of unknown transparent glassware using a single view. SPILL combines object detection with semantic keypoint detection and operates without requiring object-specific 3D models or depth completion. We demonstrate its effectiveness in autonomous robotic pouring tasks.

Additionally, to enhance the robustness and generalization of keypoint detection to diverse real-world scenarios, we introduce *Glasses-in-the-Wild*, a new dataset that captures a wide variety of glass types in realistic environments. Evaluated on a robot manipulator, SPILL achieves a **93.6% success rate** across 500 autonomous pours with 20 unseen glasses in three diverse real-world scenes.

We further demonstrate robustness through multiple live public events in real-world, human-centered environments. In one recorded session, the robot autonomously served 62 drinks with a **98.3% success rate**.
These results demonstrate that task-relevant keypoint detection enables scalable, real-world transparent object interaction, paving the way for practical applications in service and assistive robotics — without spilling a drop.

[![Watch the demo](https://img.youtube.com/vi/gcHi0ebrDps/0.jpg)](https://www.youtube.com/watch?v=gcHi0ebrDps)

---

## Try it online

[**HuggingFace Space**](https://huggingface.co/spaces/Louadria/SPILL) — upload any image and see glass keypoints detected in real time (YOLOv8 + semantic keypoint detection, runs on CPU).

---

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `torch`, `torchvision`, `ultralytics`, `opencv-python`, `scipy`, `timm`.

The pre-trained checkpoints (`wild_glasses.ckpt` + `yolov8m.pt`) are included in this repo (Git LFS).

## Quick start

```python
from spill import GlassDetector

detector = GlassDetector()  # loads checkpoints automatically

# 2D keypoint detection from any RGB image (OpenCV/BGR format)
keypoints_list = detector.detect(image)  # returns list of GlassKeypoints

for kp in keypoints_list:
    print(f"bottom_front: {kp.bottom_front}")
    print(f"top_front:    {kp.top_front}")
    print(f"top_left:     {kp.top_left}")
    print(f"top_right:    {kp.top_right}")
    print(f"fluid_level:  {kp.fluid_level}")
```

## 3D reconstruction

Once you have 2D keypoints, reconstruct full 3D glass properties (radius, height, tilt angle, fluid percentage):

```python
from spill import reconstruct_glass_3d, detect_table_height

# Option 1 — if you know camera pose (X_World_Camera):
#   Use detect_table_height() with a depth map to find the table plane.
table_height = detect_table_height(
    depth_map, camera_intrinsics, X_World_Camera
)
glasses = reconstruct_glass_3d(
    keypoints_list, camera_intrinsics, table_height, X_World_Camera
)

# Option 2 — no camera pose needed:
#   Fit the table plane directly from a point cloud using RANSAC
#   (you only need camera intrinsics to project depth -> 3D).
#   The plane normal and distance become the reference frame, so you
#   never need to know X_World_Camera explicitly.
#   See examples/demo_3d.py for a full working example.

for glass in glasses:
    print(f"radius:       {glass.radius:.3f} m")
    print(f"height:       {glass.height:.3f} m")
    print(f"fluid_level:  {glass.fluid_percentage:.1%}")
    print(f"center_3d:    {glass.center_3d}")
```

**Key insight:** You do not need a hand-measured `X_World_Camera` transform. The table plane can be recovered from a depth map or point cloud alone (RANSAC plane fit), and `reconstruct_glass_3d()` works relative to that plane. This makes the pipeline practical for single-camera setups where the camera pose is unknown or drifting.

## Glasses-in-the-Wild Dataset

A crowdsourced dataset of transparent glassware in diverse domestic and real-world environments, annotated with bounding boxes and keypoints.

Available at: [10.5281/zenodo.17288314](https://doi.org/10.5281/zenodo.17288314)

## Citation

```bibtex
@article{adriaens2025spill,
  title={SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending},
  author={Adriaens, Louis et al.},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
