#!/usr/bin/env python3
"""
Test / integration script for SPILL.

Quickly verify that models load and inference runs correctly.
Also shows how to integrate SPILL into CriticBarista's pour pipeline.

Usage:
    # Self-contained test (synthetic image, placeholder camera params)
    python examples/test_mobi.py

    # Test with a real image
    python examples/test_mobi.py --image path/to/glass.jpg

    # Test with image + depth (full 3D pipeline)
    python examples/test_mobi.py --image path/to/glass.jpg --depth path/to/depth.png
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from spill import GlassDetector, reconstruct_glass_3d, detect_table_height


def test_with_image(args):
    """Test detection (and optional 3D reconstruction) on a saved image."""
    # Load detector
    print("Loading models...")
    t0 = time.monotonic()
    detector = GlassDetector(
        keypoint_checkpoint=args.checkpoint,
        yolo_model_path=args.yolo,
        device=args.device,
    )
    print(f"Loaded in {(time.monotonic() - t0)*1000:.0f} ms\n")

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not read {args.image}")
        sys.exit(1)
    print(f"Image: {image.shape[1]}x{image.shape[0]}")

    # Run detection
    print("Detecting...")
    t0 = time.monotonic()
    keypoints_list = detector.detect(image)
    print(f"Detection: {(time.monotonic() - t0)*1000:.0f} ms")
    print(f"Found {len(keypoints_list)} glass(es)\n")

    for i, kp in enumerate(keypoints_list):
        print(f"--- Glass #{i+1} ---")
        print(f"  bottom_front:  ({kp.bottom_front[0]:.0f}, {kp.bottom_front[1]:.0f})")
        print(f"  top_front:     ({kp.top_front[0]:.0f}, {kp.top_front[1]:.0f})")
        print(f"  top_left:      ({kp.top_left[0]:.0f}, {kp.top_left[1]:.0f})")
        print(f"  top_right:     ({kp.top_right[0]:.0f}, {kp.top_right[1]:.0f})")
        if kp.fluid_level is not None:
            print(f"  fluid_level:   ({kp.fluid_level[0]:.0f}, {kp.fluid_level[1]:.0f})")
        if kp.fluid_level_2 is not None:
            print(f"  fluid_level_2: ({kp.fluid_level_2[0]:.0f}, {kp.fluid_level_2[1]:.0f})")

    # 3D reconstruction if depth provided
    if args.depth:
        depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"ERROR: Could not read {args.depth}")
            sys.exit(1)
        depth_map = depth_raw.astype(np.float32) / 1000.0

        K = np.array([
            [631.0, 0.0, 320.0],
            [0.0, 631.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        X_World_Camera = np.eye(4)

        if args.table_height is None:
            table_height = detect_table_height(depth_map, K, X_World_Camera)
            print(f"\nEstimated table height: {table_height*100:.1f} cm")
        else:
            table_height = args.table_height
            print(f"\nUsing provided table height: {table_height*100:.1f} cm")

        print("Reconstructing 3D...")
        glasses = reconstruct_glass_3d(keypoints_list, K, table_height, X_World_Camera)
        for i, glass in enumerate(glasses):
            if glass is None:
                print(f"\nGlass #{i+1}: 3D reconstruction failed")
                continue
            print(f"\nGlass #{i+1}:")
            print(f"  radius={glass.radius*100:.1f}cm  height={glass.height*100:.1f}cm")
            print(f"  tilt={np.rad2deg(glass.glass_angle):.1f}deg  fluid={glass.fluid_percentage*100:.0f}%")
            print(f"  reprojection_error={glass.reprojection_error:.1f}px")


def show_criticbarista_integration():
    """Print integration instructions for CriticBarista."""
    print("""
=== CriticBarista Integration Guide ===

Replace GlassLocalizer calls in pour_drinkz.py with:

    from spill import GlassDetector, reconstruct_glass_3d, detect_table_height

    # In __init__:
    self._spill = GlassDetector(
        keypoint_checkpoint="/path/to/wild_glasses.ckpt",
        yolo_model_path="/path/to/yolov8m.pt",
        device="cuda",
    )

    # In the detection loop (replaces self.glass_localizer.localize_glass):
    K = self.wrist_camera.intrinsics_matrix()
    depth_map = self.wrist_camera._retrieve_depth_map()
    image = self.wrist_camera._retrieve_rgb_image_as_int()

    table_height = detect_table_height(depth_map, K, X_World_Camera)
    keypoints_list = self._spill.detect(image)

    glasses = []
    for kp in keypoints_list:
        glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
        if glass:
            # Build GlassInfo for downstream code:
            X_W_glass = np.eye(4)
            X_W_glass[:3, 3] = (X_World_Camera @ np.append(glass.center_3d, 1))[:3]
            from airo_barista.perception.glassloc.GlassInfo import GlassInfo
            glasses.append(GlassInfo(
                X_W_glass, glass.radius, glass.height,
                glass.fluid_percentage, f"glass_{len(glasses)}"
            ))
""")


def main():
    parser = argparse.ArgumentParser(description="SPILL test / integration script")
    parser.add_argument("--image", type=str, default=None, help="Test image path")
    parser.add_argument("--depth", type=str, default=None, help="Depth map path (16-bit PNG, mm)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Keypoint checkpoint")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--table-height", type=float, default=None, help="Override table height (m)")
    parser.add_argument("--show-integration", action="store_true",
                        help="Show CriticBarista integration instructions")
    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(base_dir / "checkpoints" / "wild_glasses.ckpt")
    if args.yolo is None:
        args.yolo = str(base_dir / "checkpoints" / "yolov8m.pt")

    if args.show_integration:
        show_criticbarista_integration()
        return

    # If no image provided, use a synthetic one
    if args.image is None:
        print("No image provided. Creating synthetic test image...")
        image = np.zeros((480, 640, 3), dtype=np.uint8) + 128
        cv2.rectangle(image, (250, 100), (390, 400), (200, 200, 200), -1)
        tmp_path = "/tmp/spill_test_image.jpg"
        cv2.imwrite(tmp_path, image)
        args.image = tmp_path
        print(f"Saved synthetic image to {tmp_path}\n")

    test_with_image(args)


if __name__ == "__main__":
    main()
