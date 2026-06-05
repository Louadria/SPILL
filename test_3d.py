#!/usr/bin/env python3
"""
Test the 3D reconstruction pipeline from the HF Space.
Skips 2D detection (Blackwell GPU + PyTorch mismatch on localhost).
Tests: monocular depth, RANSAC plane, SPILL cylinder, Plotly figure.
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent / "hf_space"
sys.path.insert(0, str(BASE))

from spill import Monocular3DReconstructor
from spill import depth_overlay, create_3d_figure, build_3d_info
from spill.monocular_3d import (
    estimate_intrinsics,
    ransac_table_plane,
    build_world_transform,
    depth_to_colormap,
)
from spill.types import GlassKeypoints

# Force CPU to avoid Blackwell GPU + PyTorch mismatch (local only)
os.environ["SPILL_FORCE_CUDA"] = ""

print("=== Step 1: Create synthetic test image with table + glass ===")
h, w = 480, 640
img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
# Background: gradient (wall above, table below)
for y in range(h):
    t = y / h
    img_bgr[y, :, 0] = int(100 * (1 - t))
    img_bgr[y, :, 1] = int(140 * (1 - t))
    img_bgr[y, :, 2] = int(200 * (1 - t))
# Table: gray surface
table_y = int(h * 0.55)
img_bgr[table_y:, :, :] = [128, 128, 128]
# Glass: transparent cylinder shape
cx, cy = w // 2, table_y - 50
glass_w, glass_h = 60, 150
glass_top = cy - glass_h // 2
glass_bot = cy + glass_h // 2
cv2.rectangle(img_bgr, (cx - glass_w // 2, glass_top),
              (cx + glass_w // 2, glass_bot), (200, 200, 200), 2)
cv2.ellipse(img_bgr, (cx, glass_top), (glass_w // 2, 10), 0, 0, 360, (200, 200, 200), 2)
fluid_y = glass_top + int(glass_h * 0.6)
cv2.line(img_bgr, (cx - glass_w // 2, fluid_y),
         (cx + glass_w // 2, fluid_y), (150, 200, 255), 1)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"Test image shape: {img_bgr.shape}")

print("\n=== Step 2: Load 3D reconstructor ===")
reconstructor = Monocular3DReconstructor(depth_model_size="small", device="cpu")
print("3D reconstructor loaded OK")

print("\n=== Step 3: Create synthetic keypoints ===")
keypoints = [GlassKeypoints(
    bottom_front=np.array([float(cx), float(glass_bot - 5)]),
    top_front=np.array([float(cx), float(glass_top + 5)]),
    top_left=np.array([float(cx - glass_w // 2 + 3), float(glass_top + 5)]),
    top_right=np.array([float(cx + glass_w // 2 - 3), float(glass_top + 5)]),
    fluid_level=np.array([float(cx), float(fluid_y)]),
    fluid_level_2=None,
    bounding_box=(cx - glass_w // 2, glass_top, cx + glass_w // 2, glass_bot),
    yolo_bounding_box=(cx - glass_w // 2, glass_top, cx + glass_w // 2, glass_bot),
)]
print(f"Keypoints: bf={keypoints[0].bottom_front}, tf={keypoints[0].top_front}")

print("\n=== Step 4: Monocular 3D reconstruction ===")
glasses, depth_map, info = reconstructor.reconstruct(img_bgr, keypoints)
print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
print(f"Plane normal: {info['plane_normal']}")
print(f"Plane d: {info['plane_d']:.4f}")
print(f"Table height: {info['table_height']:.4f}")
print(f"Got {len(glasses)} glass reconstruction result(s)")

for i, g in enumerate(glasses):
    if g is not None:
        print(f"Glass #{i+1}:")
        print(f"  Radius:   {g.radius * 1000:.1f} mm")
        print(f"  Height:   {g.height * 1000:.1f} mm")
        print(f"  Tilt:     {np.rad2deg(g.glass_angle):.1f} deg")
        print(f"  Fluid:    {g.fluid_percentage * 100:.0f}%")
        print(f"  Center:   ({g.center_3d[0]:.3f}, {g.center_3d[1]:.3f}, {g.center_3d[2]:.3f}) m")
        print(f"  Repr err: {g.reprojection_error:.1f} px")
    else:
        print(f"Glass #{i+1}: reconstruction failed")

print("\n=== Step 5: Depth colormap & overlay ===")
colormap = depth_to_colormap(depth_map)
print(f"Colormap shape: {colormap.shape}, dtype: {colormap.dtype}")
overlay = depth_overlay(img_rgb, depth_map)
print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")

print("\n=== Step 6: 3D Plotly figure ===")
fig = create_3d_figure(glasses, info, img_bgr.shape)
print(f"Figure has {len(fig.data)} traces:")
for trace in fig.data:
    print(f"  - {trace.name}: {type(trace).__name__}")

print("\n=== Step 7: Info text ===")
info_text = build_3d_info(glasses, keypoints, info)
print(info_text)

print("\n=== Step 8: Utility functions ===")
K = estimate_intrinsics(h, w)
print(f"Intrinsics: K =\n{K}")

plane_n, plane_d = ransac_table_plane(depth_map, K, keypoints)
print(f"RANSAC: n={plane_n}, d={plane_d:.4f}")

X_wc, table_h = build_world_transform(plane_n, plane_d)
print(f"World transform: table_h={table_h:.4f}, X_wc=\n{X_wc}")

print("\n=== ALL TESTS PASSED ===")
