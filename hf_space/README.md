---
title: "SPILL - Glass Detection & 3D Reconstruction"
emoji: 🥤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
short_description: Detect glassware keypoints and reconstruct 3D cylinders
---

# SPILL: Glass Detection & 3D Reconstruction

Interactive demo for **SPILL** (Size, Pose, and Internal Liquid Level Estimation of transparent glassware).

Upload an image to detect glasses and their semantic keypoints:
- Bottom front, top front, top left, top right corners
- Fluid level (liquid surface)

## Features

**2D Reconstruction tab** — Detect glass keypoints from any image. Shows annotated keypoints with pixel coordinates.

**3D Reconstruction tab** — Full 3D cylinder reconstruction from a single RGB image. Uses DepthAnythingV2 for monocular depth estimation, RANSAC for table plane detection, and the SPILL cylinder solver for radius, height, tilt angle, and fluid level. No depth camera required!

## How to use

1. Upload or paste an image containing glasses
2. **2D tab:** Click "Detect Glasses" to see keypoints overlaid
3. **3D tab:** Click "Reconstruct 3D" to get full 3D reconstruction with interactive 3D visualization

## Paper & Code

- **Paper:** SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending
- **GitHub:** https://github.com/Louadria/SPILL
- **Dataset:** [Glasses-in-the-Wild](https://doi.org/10.5281/zenodo.17288314)

## How 3D Reconstruction Works

1. **Keypoint detection** — YOLOv8 + custom keypoint model finds glass structure in 2D
2. **Monocular depth** — DepthAnythingV2-Large estimates depth from a single RGB image
3. **Table plane** — Open3D RANSAC finds the dominant plane (table surface) in the depth map
4. **Cylinder estimation** — SPILL solver reconstructs 3D glass parameters (radius, height, tilt, fluid level)

See the GitHub repository for the complete pipeline integration with real RGB-D cameras.
