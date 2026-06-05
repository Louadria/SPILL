---
title: "SPILL - Glass Keypoint Detection"
emoji: 🥤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
short_description: Detect transparent glassware keypoints
---

# SPILL: Glass Keypoint Detection

Interactive demo for **SPILL** (Size, Pose, and Internal Liquid Level Level Estimation of transparent glassware).

Upload an image to detect glasses and their semantic keypoints:
- Bottom front, top front, top left, top right corners
- Fluid level (liquid surface)

## How to use

1. Upload or paste an image containing glasses
2. Click "Detect Glasses"
3. See annotated keypoints overlaid on the image

## Paper & Code

- **Paper:** SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending
- **GitHub:** https://github.com/Louadria/SPILL
- **Dataset:** [Glasses-in-the-Wild](https://doi.org/10.5281/zenodo.17288314)

## Notes

This demo shows the **2D keypoint detection** component. The full 3D reconstruction pipeline requires:
- An RGB-D camera (depth sensor)
- Calibrated camera intrinsics
- Camera-to-world transformation matrix

See the GitHub repository for the complete pipeline integration.
