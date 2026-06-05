#!/usr/bin/env python3
"""
Example: Full 3D pipeline (requires depth camera / pre-recorded data).

This shows how to integrate SPILL into a robot pipeline like CriticBarista/MOBI.
You need:
  - RGB image from your camera
  - Depth map (or point cloud) for table detection
  - Camera intrinsics (K matrix)
  - Camera-to-world transform (X_World_Camera)

For this example, we use placeholder values. Replace with your actual camera data.

Usage:
    python examples/demo_3d.py --image path/to/image.jpg --depth path/to/depth.png
"""
import argparse
import numpy as np
from pathlib import Path

from spill import GlassDetector, reconstruct_glass_3d, detect_table_height


def load_camera_pose():
    """
    Load camera intrinsics and extrinsics from your robot setup.

    In CriticBarista, this comes from:
      - wrist_camera.intrinsics_matrix()  -> 3x3 K matrix
      - robot.get_tcp_pose() @ X_Tcp_WristCamera  -> X_World_Camera

    Replace these with your actual values!
    """
    # Example: Intel RealSense D435 (adjust to your calibration)
    camera_intrinsics = np.array([
        [631.0, 0.0, 320.0],
        [0.0, 631.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    # Example: camera looking down at table at ~45 degrees, 0.5m away
    X_World_Camera = np.eye(4)
    X_World_Camera[0, 3] = 0.0      # X offset
    X_World_Camera[1, 3] = 0.0      # Y offset
    X_World_Camera[2, 3] = 0.5      # Z offset (depth)
    # Add rotation as needed

    return camera_intrinsics, X_World_Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="RGB image")
    parser.add_argument("--depth", required=True, help="Depth map (16-bit PNG, meters)")
    parser.add_argument("--checkpoint", default="checkpoints/wild_glasses.ckpt")
    parser.add_argument("--yolo", default="checkpoints/yolov8m.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 1. Load models
    print("Loading models...")
    detector = GlassDetector(
        keypoint_checkpoint=args.checkpoint,
        yolo_model_path=args.yolo,
        device=args.device,
    )

    # 2. Load camera parameters
    K, X_World_Camera = load_camera_pose()
    print(f"Camera intrinsics:\n{K}")
    print(f"X_World_Camera:\n{X_World_Camera}")

    # 3. Load image and depth
    import cv2
    image = cv2.imread(args.image)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    # Convert to meters (adjust scale factor for your depth sensor)
    depth_map = depth.astype(np.float32) / 1000.0  # mm -> meters

    # 4. Detect table height from depth map
    print("Detecting table height...")
    table_height = detect_table_height(depth_map, K, X_World_Camera)
    print(f"Table height: {table_height:.3f}m")

    # 5. Detect glasses (2D keypoints)
    print("Detecting glasses...")
    keypoints_list = detector.detect(image)
    print(f"Found {len(keypoints_list)} glass(es)")

    # 6. Reconstruct 3D properties
    for i, kp in enumerate(keypoints_list):
        glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
        if glass:
            print(f"\nGlass {i+1}:")
            print(f"  Center (camera frame): {glass.center_3d}")
            print(f"  Radius: {glass.radius*100:.1f} cm")
            print(f"  Height: {glass.height*100:.1f} cm")
            print(f"  Tilt: {np.rad2deg(glass.glass_angle):.1f} deg")
            print(f"  Fluid level: {glass.fluid_percentage*100:.0f}%")
            print(f"  Reprojection error: {glass.reprojection_error:.1f} px")

            # Transform to world frame if needed
            center_world = (X_World_Camera @ np.append(glass.center_3d, 1.0))[:3]
            print(f"  Center (world frame): {center_world}")


if __name__ == "__main__":
    main()
