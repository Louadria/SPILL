import numpy as np
from airo_camera_toolkit.utils.image_converter import ImageConverter
from ultralytics import YOLO
from typing import List
from airo_typing import OpenCVIntImageType
import torch
from torchvision.transforms.functional import to_tensor

from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint, load_from_checkpoint

class GlassDetector:
    def __init__(self, classes: List[str], keypoint_detector: str) -> None:
        self._classes = classes
        self._detector = YOLO("checkpoints/yolov8m.pt")
        self._detector.to("cuda")
        self._keypoint_detector = load_from_checkpoint(keypoint_detector)
        self._keypoint_detector.eval()
        self._keypoint_detector.cuda()

    def get_glass_bounding_boxes(self, image: OpenCVIntImageType) -> np.ndarray | None:
        """Run object detection, detecting glasses ("cup" and "wine glass") and return the bounding boxes.

        Args:
            image: The image to detect on.

        Returns:
            An array of shape (N, 4) containing the xyxy coordinates in the original image shape of the bounding box(es), or None if there are no detections."""
        results = self._detector(image)
        result = results[0]
        bounding_boxes = []
        for i, box in enumerate(result.boxes):
            if self._detector.names[int(box.cls)] in self._classes:
                box = box.xyxyn.cpu().numpy()
                box[0, 0] *= image.shape[1]
                box[0, 1] *= image.shape[0]
                box[0, 2] *= image.shape[1]
                box[0, 3] *= image.shape[0]
                bounding_boxes.append(box)
        return np.stack(bounding_boxes) if len(bounding_boxes) > 0 else None

    def keypoint_detector_local_inference(self, image: np.ndarray, device="cuda"):
        """inference on a single image as if you would load the image from disk or get it from a camera.
        Returns a list of the extracted keypoints for each channel.


        """
        # assert model is in eval mode! (important for batch norm layers)
        assert self._keypoint_detector.training == False, "model should be in eval mode for inference"

        # convert image to tensor with correct shape (channels, height, width) and convert to floats in range [0,1]
        # add batch dimension
        # and move to device
        image = to_tensor(image).unsqueeze(0).to(device)

        # pass through model
        with torch.no_grad():
            heatmaps = self._keypoint_detector(image).squeeze(0)

        # extract keypoints from heatmaps
        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps.unsqueeze(0))[0]

        return predicted_keypoints