#!/usr/bin/env python3
"""
MOBI Integration Example - CriticBarista Pipeline

This shows how to use SPILL as a drop-in replacement for the GlassLocalizer
in the CriticBarista/MOBI robot pipeline.

In your airo_barista code, instead of:
    from airo_barista.perception.glassloc.glassloc import GlassLocalizer

You can use:
    from spill import GlassDetector, reconstruct_glass_3d, detect_table_height

And replace the pipeline calls:

OLD (CriticBarista):
    table_height = mobi.glass_localizer.estimate_table_height_from_depth_map(
        depth_map, X_World_Camera)
    glasses = mobi.glass_localizer.localize_glass(image, table_height, X_World_Camera)

NEW (SPILL library):
    table_height = detect_table_height(depth_map, K, X_World_Camera)
    keypoints_list = detector.detect(image)
    glasses = [reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
               for kp in keypoints_list]

---

This script demonstrates the full pipeline assuming you're running on the MOBI robot
with the CriticBarista environment. Run from the airo-barista directory with:

    cd /home/louadria/demos/CriticBarista/airo-barista
    python /home/louadria/demos/SPILL/examples/test_mobi.py

Requires:
    - critic-barista conda environment
    - SPILL installed (pip install -e /home/louadria/demos/SPILL)
"""
import sys
import os
import numpy as np
import cv2
import time

# Add SPILL to path if not installed
SPILL_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
if not os.path.exists(os.path.join(SPILL_PATH, "spill")):
    SPILL_PATH = None

if SPILL_PATH and SPILL_PATH not in sys.path:
    sys.path.insert(0, SPILL_PATH)

from spill import GlassDetector, reconstruct_glass_3d, detect_table_height


def test_with_saved_data():
    """
    Test SPILL with a saved image (no robot needed).

    This is useful for quick verification that the library loads and runs.
    """
    print("=" * 60)
    print("SPILL Library Test - Static Image")
    print("=" * 60)

    # Paths - adjust to your checkpoint locations
    checkpoint = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "wild_glasses.ckpt")
    yolo_model = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "yolov8m.pt")

    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        print("Run from the SPILL repo root or adjust paths.")
        return

    # Create detector
    print(f"\nLoading detector from {checkpoint}...")
    t0 = time.time()
    detector = GlassDetector(
        keypoint_checkpoint=checkpoint,
        yolo_model_path=yolo_model,
        device="cuda",
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Create a dummy test image (or use a real one)
    # For a real test, replace with: image = cv2.imread("path/to/glass/image.jpg")
    test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        print(f"Loaded test image: {image.shape}")
    else:
        # Create a synthetic test image
        image = np.zeros((480, 640, 3), dtype=np.uint8) + 128
        # Draw a glass-like shape
        cv2.rectangle(image, (250, 100), (390, 400), (200, 200, 200), -1)
        cv2.putText(image, "TEST IMAGE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (0, 0, 255), 2)
        print("Using synthetic test image")

    # Run detection
    print("\nRunning detection...")
    t0 = time.time()
    keypoints_list = detector.detect(image)
    elapsed = time.time() - t0
    print(f"Detection took {elapsed*1000:.0f}ms")
    print(f"Found {len(keypoints_list)} glass(es)")

    # Try 3D reconstruction with placeholder parameters
    K = np.array([
        [631.0, 0.0, 320.0],
        [0.0, 631.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    X_World_Camera = np.eye(4)
    X_World_Camera[2, 3] = 0.5
    table_height = 0.75  # typical table height

    for i, kp in enumerate(keypoints_list):
        glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
        if glass:
            print(f"\nGlass {i+1}:")
            print(f"  Radius: {glass.radius*100:.1f} cm")
            print(f"  Height: {glass.height*100:.1f} cm")
            print(f"  Fluid: {glass.fluid_percentage*100:.0f}%")
            print(f"  Error: {glass.reprojection_error:.1f} px")

    print("\n" + "=" * 60)
    print("Test complete!")


def integrate_into_criticbarista():
    """
    Shows how to integrate SPILL into CriticBarista's pour_drinkz.py flow.

    Replace this section in pour_drinkz.py:

        table_height = self.glass_localizer.estimate_table_height_from_depth_map(
            depth_map, X_World_Camera)
        glasses = self.glass_localizer.localize_glass(image, table_height, X_World_Camera)

    With:

        from spill import GlassDetector, reconstruct_glass_3d, detect_table_height

        # Initialize once (e.g., in __init__)
        self._spill_detector = GlassDetector(
            keypoint_checkpoint="/path/to/wild_glasses.ckpt",
            yolo_model_path="/path/to/yolov8m.pt",
            device="cuda",
        )

        # In the detection loop:
        K = self.wrist_camera.intrinsics_matrix()
        table_height = detect_table_height(depth_map, K, X_World_Camera)
        keypoints_list = self._spill_detector.detect(image)

        glasses = []
        for kp in keypoints_list:
            glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
            if glass:
                # Convert to the format expected by downstream code
                glasses.append([
                    glass.center_3d,        # top_middle_3d
                    glass.radius,
                    glass.height,
                    glass.fluid_percentage,
                ])

    See demo_3d.py for a complete working example.
    """
    print("See function docstring for integration instructions.")


if __name__ == "__main__":
    test_with_saved_data()
