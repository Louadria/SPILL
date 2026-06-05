#!/usr/bin/env python3
"""
Example: Detect glasses and visualize keypoints.

Usage:
    python examples/detect_image.py --image path/to/image.jpg [--checkpoint checkpoints/wild_glasses.ckpt]

Requires: pip install spill-glassloc
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from spill import GlassDetector


def draw_keypoints(image, keypoints_list):
    """Draw detected keypoints on the image."""
    kp_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    kp_names = ["bot_front", "top_front", "top_left", "top_right", "fluid", "fluid2"]

    for idx, kp in enumerate(keypoints_list):
        # Draw bounding box
        x1, y1, x2, y2 = kp.bounding_box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Glass {idx+1}", (int(x1), int(y1) - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw keypoints
        all_kps = [kp.bottom_front, kp.top_front, kp.top_left, kp.top_right,
                    kp.fluid_level, kp.fluid_level_2]

        for j, pt in enumerate(all_kps):
            if pt is not None:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 5, kp_colors[j % len(kp_colors)], -1)
                cv2.putText(image, kp_names[j], (int(pt[0]) + 8, int(pt[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, kp_colors[j % len(kp_colors)], 1)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="checkpoints/wild_glasses.ckpt",
                        help="Path to keypoint checkpoint")
    parser.add_argument("--yolo", default="checkpoints/yolov8m.pt",
                        help="Path to YOLO model")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None, help="Save output image")
    args = parser.parse_args()

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

    # Visualize
    if keypoints_list:
        result = draw_keypoints(image.copy(), keypoints_list)
        print(f"Found {len(keypoints_list)} glass(es)!")

        if args.output:
            cv2.imwrite(args.output, result)
            print(f"Saved to {args.output}")
        else:
            cv2.imshow("SPILL Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No glasses detected.")


if __name__ == "__main__":
    main()
