"""YOLO object detection wrapper."""
from pathlib import Path
from typing import List, Optional
import numpy as np
from ultralytics import YOLO


class GlassDetector:
    """Detects glass objects using YOLOv8."""

    GLASS_CLASSES = ["cup", "vase", "wine glass"]

    def __init__(self, model_path: str, device: str = "cuda"):
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device

    def detect(self, image: np.ndarray, classes: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Detect glasses in an image.

        Args:
            image: BGR image (H, W, 3)
            classes: Glass classes to detect (defaults to all)

        Returns:
            Array of shape (N, 4) with xyxy coordinates, or None if no detections
        """
        if classes is None:
            classes = self.GLASS_CLASSES

        results = self._model(image, verbose=False)
        result = results[0]

        bounding_boxes = []
        for box in result.boxes:
            if self._model.names[int(box.cls)] in classes:
                box_coords = box.xyxyn.cpu().numpy()[0]
                box_coords[0] *= image.shape[1]
                box_coords[1] *= image.shape[0]
                box_coords[2] *= image.shape[1]
                box_coords[3] *= image.shape[0]
                bounding_boxes.append(box_coords)

        return np.stack(bounding_boxes) if bounding_boxes else None
