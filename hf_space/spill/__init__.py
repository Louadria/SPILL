"""SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware.

A lightweight perception pipeline for detecting transparent glassware and estimating
their 3D properties from a single RGB image and camera parameters.

Paper: https://github.com/Louadria/SPILL
Dataset: https://doi.org/10.5281/zenodo.17288314
"""

from .types import GlassDetection, GlassKeypoints
from .detect import GlassDetector
from .reconstruct import reconstruct_glass_3d, detect_table_height

__all__ = [
    "GlassDetection",
    "GlassKeypoints",
    "GlassDetector",
    "reconstruct_glass_3d",
    "detect_table_height",
]
