#!/usr/bin/env python3
"""
Example: Full 3D glass reconstruction from RGB + depth.

This demonstrates the complete SPILL pipeline:
1. YOLO detects glass bounding boxes
2. Keypoint detector predicts 2D semantic keypoints
3. Table height is estimated from the depth map
4. 3D properties (radius, height, tilt, fluid level) are reconstructed

You need:
  - An RGB image
  - A matching depth map (16-bit PNG, values in mm)
  - Camera intrinsics (K matrix) from your calibration

Usage:
    python examples/demo_3d.py --image examples/data/glass.jpg --depth examples/data/glass_depth.png

For placeholder camera params, adjust --fx, --fy, --cx, --cy and --table-height.
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from spill import GlassDetector, reconstruct_glass_3d, detect_table_height


def main():
    parser = argparse.ArgumentParser(description="Full 3D glass reconstruction demo")
    parser.add_argument("--image", required=True, help="RGB image path")
    parser.add_argument("--depth", required=True, help="Depth map path (16-bit PNG, mm)")
    parser.add_argument("--checkpoint", default=None, help="Path to keypoint checkpoint")
    parser.add_argument("--yolo", default=None, help="Path to YOLO model")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    # Camera intrinsics (override with your calibration)
    parser.add_argument("--fx", type=float, default=631.0, help="Focal length x")
    parser.add_argument("--fy", type=float, default=631.0, help="Focal length y")
    parser.add_argument("--cx", type=float, default=320.0, help="Principal point x")
    parser.add_argument("--cy", type=float, default=240.0, help="Principal point y")
    # Camera pose (camera IS the world frame for standalone demo)
    parser.add_argument("--table-height", type=float, default=None,
                        help="Override auto-detected table height (meters)")
    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(base_dir / "checkpoints" / "wild_glasses.ckpt")
    if args.yolo is None:
        args.yolo = str(base_dir / "checkpoints" / "yolov8m.pt")

    # Camera intrinsics
    K = np.array([
        [args.fx, 0.0, args.cx],
        [0.0, args.fy, args.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Camera-to-world transform (identity = camera is world frame)
    X_World_Camera = np.eye(4)

    # Load models
    print(f"Loading models on {args.device}...")
    detector = GlassDetector(
        keypoint_checkpoint=args.checkpoint,
        yolo_model_path=args.yolo,
        device=args.device,
    )
    print("Models loaded!\n")

    # Load image + depth
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not read {args.image}")
        return
    print(f"Image: {image.shape[1]}x{image.shape[0]}")

    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        print(f"ERROR: Could not read {args.depth}")
        return
    depth_map = depth_raw.astype(np.float32) / 1000.0  # mm -> meters
    print(f"Depth: {depth_map.shape[1]}x{depth_map.shape[0]}, range: {depth_map.min():.3f}-{depth_map.max():.3f}m\n")

    # Estimate table height
    if args.table_height is not None:
        table_height = args.table_height
        print(f"Using provided table height: {table_height*100:.1f} cm")
    else:
        print("Estimating table height from depth map...")
        table_height = detect_table_height(depth_map, K, X_World_Camera)
        print(f"Estimated table height: {table_height*100:.1f} cm")

    # Detect glasses
    print("\nDetecting glasses...")
    keypoints_list = detector.detect(image)
    print(f"Found {len(keypoints_list)} glass(es)\n")

    # 3D reconstruction
    print("\nReconstructing 3D...")
    glasses = reconstruct_glass_3d(keypoints_list, K, table_height, X_World_Camera)
    for i, glass in enumerate(glasses):
        if glass is None:
            print(f"\nGlass #{i+1}: 3D reconstruction failed")
            continue
        print(f"\n--- Glass #{i+1} ---")
        print(f"  3D Reconstruction:")
        print(f"    Center (cam frame): {glass.center_3d}")
        print(f"    Radius: {glass.radius*100:.1f} cm")
        print(f"    Height: {glass.height*100:.1f} cm")
        print(f"    Tilt:   {np.rad2deg(glass.glass_angle):.1f} deg")
        print(f"    Fluid:  {glass.fluid_percentage*100:.0f}%")
        print(f"    Reprojection error: {glass.reprojection_error:.1f} px")


if __name__ == "__main__":
    main()
