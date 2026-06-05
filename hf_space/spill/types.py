"""Data types for SPILL."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GlassKeypoints:
    """2D keypoints detected on a cropped glass image.

    Attributes:
        bottom_front: (x, y) - Bottom front of the glass
        top_front: (x, y) - Top front rim of the glass
        top_left: (x, y) - Top left rim of the glass
        top_right: (x, y) - Top right rim of the glass
        fluid_level: Optional[(x, y)] - Fluid level if detected
        fluid_level_2: Optional[(x, y)] - Second fluid level if detected
        bounding_box: (x1, y1, x2, y2) - Bounding box in original image
    """
    bottom_front: np.ndarray
    top_front: np.ndarray
    top_left: np.ndarray
    top_right: np.ndarray
    fluid_level: Optional[np.ndarray] = None
    fluid_level_2: Optional[np.ndarray] = None
    bounding_box: tuple = (0, 0, 0, 0)
    yolo_bounding_box: tuple = (0, 0, 0, 0)


@dataclass
class GlassDetection:
    """3D glass detection result.

    Attributes:
        center_3d: (x, y, z) - Center of the glass top in camera coordinates (meters)
        radius: float - Glass radius in meters
        height: float - Glass height in meters
        glass_angle: float - Tilt angle in radians (0 = perfectly upright)
        fluid_percentage: float - Fill level from 0.0 (empty) to 1.0 (full)
        keypoints: GlassKeypoints - The 2D keypoints used for reconstruction
        reprojection_error: float - Pixel error of the 3D->2D reprojection
    """
    center_3d: np.ndarray
    radius: float
    height: float
    glass_angle: float = 0.0
    fluid_percentage: float = 1.0
    keypoints: Optional['GlassKeypoints'] = None
    reprojection_error: float = 0.0
