# SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending

Robotic perception of transparent objects presents unique challenges due to their refractive properties, lack of texture, and limitations of conventional RGB-D sensors in capturing reliable depth information. These challenges significantly hinder robotic manipulation capabilities in real-world settings such as household assistance, hospitality, and healthcare. 

To address these issues, we propose **SPILL**: A lightweight perception pipeline for **S**ize, **P**ose, and **I**nternal **L**iquid **L**evel estimation of unknown transparent glassware using a single view. SPILL combines object detection with semantic keypoint detection and operates without requiring object-specific 3D models or depth completion. We demonstrate its effectiveness in autonomous robotic pouring tasks. 

Additionally, to enhance the robustness and generalization of keypoint detection to diverse real-world scenarios, we introduce *Glasses-in-the-Wild*, a new dataset that captures a wide variety of glass types in realistic environments. Evaluated on a robot manipulator, SPILL achieves a **93.6% success rate** across 500 autonomous pours with 20 unseen glasses in three diverse real-world scenes. 

We further demonstrate robustness through multiple live public events in real-world, human-centered environments. In one recorded session, the robot autonomously served 62 drinks with a **98.3% success rate**.  
These results demonstrate that task-relevant keypoint detection enables scalable, real-world transparent object interaction, paving the way for practical applications in service and assistive robotics — without spilling a drop.


[![Watch the demo](https://img.youtube.com/vi/gcHi0ebrDps/0.jpg)](https://www.youtube.com/watch?v=gcHi0ebrDps)

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Louadria/SPILL)

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


# Usage

The core perception functions are provided in `glassloc.py`. Two primary functions are:

## `localize_table(point_cloud, X_Platform_Camera, platform_height)`

Detects the table plane in a point cloud and returns its height in the platform frame.

## `localize_glass(image, table_height, X_Platform_Camera, platform_height)`

Detects all glasses in an RGB image and returns their 3D positions in the platform frame.

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
