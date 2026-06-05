"""Glass detection: YOLO bounding boxes + keypoint detection.

Usage:
    detector = GlassDetector(checkpoint_path, yolo_path)
    keypoints_list = detector.detect(image)
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from .model.load_checkpoint import load_keypoint_detector
from .model.yolo import GlassDetector as YOLOGlassDetector
from .model.heatmap import get_keypoints_from_heatmap_batch_maxpool
from .types import GlassKeypoints


KEYPOINT_NAMES = ["bottom_front", "top_front", "top_left", "top_right", "fluid_level"]


class GlassDetector:
    """Detect glasses and their keypoints in RGB images.

    Args:
        keypoint_checkpoint: Path to the keypoint detector checkpoint (.ckpt)
        yolo_model_path: Path to YOLO model weights (e.g., yolov8m.pt)
        device: Torch device string ("cuda" or "cpu")
        glass_classes: COCO class names to detect as glasses
    """

    def __init__(
        self,
        keypoint_checkpoint: str,
        yolo_model_path: str,
        device: str = "cuda",
        glass_classes: Optional[List[str]] = None,
    ):
        if glass_classes is None:
            glass_classes = ["cup", "vase", "wine glass"]

        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._classes = glass_classes

        # Load models
        self._yolo = YOLOGlassDetector(yolo_model_path, device=str(self._device))
        self._kp_model = load_keypoint_detector(keypoint_checkpoint, device=str(self._device))

    def detect(self, image: np.ndarray) -> List[GlassKeypoints]:
        """Detect glasses and their keypoints in an RGB image.

        Args:
            image: BGR image (H, W, 3) as numpy array

        Returns:
            List of GlassKeypoints, one per detected glass
        """
        # Step 1: Detect bounding boxes
        boxes = self._yolo.detect(image, classes=self._classes)
        if boxes is None:
            return []

        # Step 2: Crop, resize, and run keypoint detection
        results = []
        crops = []
        crop_coords = []
        yolo_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            padding_x = int(64 * (x2 - x1) / 256)
            padding_y = int(64 * (y2 - y1) / 256)
            padded_x1, padded_y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
            padded_x2, padded_y2 = min(image.shape[1], x2 + padding_x), min(image.shape[0], y2 + padding_y)
            crop = image[int(padded_y1):int(padded_y2), int(padded_x1):int(padded_x2)]
            crops.append(cv2.resize(crop, (256, 256)))
            crop_coords.append((padded_x1, padded_y1, padded_x2, padded_y2))
            yolo_boxes.append((x1, y1, x2, y2))

        # Batch inference
        crops_tensor = torch.as_tensor(
            np.array(crops), device=self._device, dtype=torch.float32
        ).permute(0, 3, 1, 2) / 255.0

        with torch.no_grad():
            heatmaps = self._kp_model(crops_tensor)

        # Extract keypoints from heatmaps
        kp_results = get_keypoints_from_heatmap_batch_maxpool(
            heatmaps, max_keypoints=20, min_keypoint_pixel_distance=5
        )

        for i, coords in enumerate(crop_coords):
            kp_data = kp_results[i]  # all channels for glass i: [bottom_front, top_front, top_left, top_right, fluid_level]
            x1, y1, x2, y2 = coords

            def map_kp(kp_256):
                """Map keypoint from 256x256 crop to original image."""
                x = kp_256[0] / 256 * (x2 - x1) + x1
                y = kp_256[1] / 256 * (y2 - y1) + y1
                return np.array([x, y])

            # Extract 4 structural keypoints
            struct_kps = []
            for j in range(4):
                if kp_data[j]:
                    struct_kps.append(map_kp(kp_data[j][0]))
                else:
                    break

            if len(struct_kps) < 4:
                continue  # Skip incomplete detections

            # Extract optional fluid levels
            fluid = None
            fluid_2 = None
            if len(kp_data) > 4 and kp_data[4]:
                fluid = map_kp(kp_data[4][0])
                if len(kp_data[4]) > 1:
                    fluid_2 = map_kp(kp_data[4][1])

            yolo_box = yolo_boxes[i]
            glass_kp = GlassKeypoints(
                bottom_front=struct_kps[0],
                top_front=struct_kps[1],
                top_left=struct_kps[2],
                top_right=struct_kps[3],
                fluid_level=fluid,
                fluid_level_2=fluid_2,
                bounding_box=yolo_box,
                yolo_bounding_box=yolo_box,
            )
            results.append(glass_kp)

        return results
