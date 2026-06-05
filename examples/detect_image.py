#!/usr/bin/env python3
"""
Example: Detect glasses and visualize keypoints on a single image.

Usage:
    python examples/detect_image.py --image path/to/image.jpg [--output result.jpg]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from spill import GlassDetector


def draw_keypoints(image, keypoints_list):
    """Draw detected keypoints on the image."""
    kp_colors = {
        "bottom_front": (0, 0, 255),
        "top_front": (0, 255, 0),
        "top_left": (255, 0, 0),
        "top_right": (255, 0, 255),
        "fluid_level": (255, 255, 0),
    }

    for idx, kp in enumerate(keypoints_list):
        # Draw bounding box
        x1, y1, x2, y2 = kp.bounding_box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Glass #{idx+1}", (int(x1), int(y1) - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw keypoints
        all_kps = [
            ("bottom_front", kp.bottom_front),
            ("top_front", kp.top_front),
            ("top_left", kp.top_left),
            ("top_right", kp.top_right),
            ("fluid_level", kp.fluid_level),
            ("fluid_level_2", kp.fluid_level_2),
        ]

        for name, pt in all_kps:
            if pt is not None:
                color = kp_colors.get(name, (255, 255, 255))
                cv2.circle(image, (int(pt[0]), int(pt[1])), 5, color, -1)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=None, help="Path to keypoint checkpoint")
    parser.add_argument("--yolo", default=None, help="Path to YOLO model")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None, help="Save output image")
    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(base_dir / "checkpoints" / "wild_glasses.ckpt")
    if args.yolo is None:
        args.yolo = str(base_dir / "checkpoints" / "yolov8m.pt")

    # Load detector
    print(f"Loading models on {args.device}...")
    detector = GlassDetector(
        keypoint_checkpoint=args.checkpoint,
        yolo_model_path=args.yolo,
        device=args.device,
    )

    # Detect
    print(f"Detecting glasses in {args.image}...")
    image = cv2.imread(args.image)
    keypoints_list = detector.detect(image)

    # Print results
    if keypoints_list:
        print(f"\nFound {len(keypoints_list)} glass(es):")
        for i, kp in enumerate(keypoints_list):
            print(f"\n--- Glass #{i+1} ---")
            print(f"  bottom_front:  ({kp.bottom_front[0]:.0f}, {kp.bottom_front[1]:.0f})")
            print(f"  top_front:     ({kp.top_front[0]:.0f}, {kp.top_front[1]:.0f})")
            print(f"  top_left:      ({kp.top_left[0]:.0f}, {kp.top_left[1]:.0f})")
            print(f"  top_right:     ({kp.top_right[0]:.0f}, {kp.top_right[1]:.0f})")
            if kp.fluid_level is not None:
                print(f"  fluid_level:   ({kp.fluid_level[0]:.0f}, {kp.fluid_level[1]:.0f})")
            if kp.fluid_level_2 is not None:
                print(f"  fluid_level_2: ({kp.fluid_level_2[0]:.0f}, {kp.fluid_level_2[1]:.0f})")

        # Visualize
        result = draw_keypoints(image.copy(), keypoints_list)
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"\nSaved to {args.output}")
        else:
            cv2.imshow("SPILL Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No glasses detected.")


if __name__ == "__main__":
    main()
