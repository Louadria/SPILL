"""3D reconstruction from 2D keypoints.

Given 2D keypoints, camera intrinsics, and a table plane, reconstruct
the 3D position, size, and pose of transparent glassware.
"""
import numpy as np
from scipy.optimize import fsolve
from typing import List, Optional

from .types import GlassDetection, GlassKeypoints


def reconstruct_glass_3d(
    keypoints,  # GlassKeypoints or List[GlassKeypoints]
    camera_intrinsics: np.ndarray,
    table_height: float,
    X_World_Camera: np.ndarray,
):
    """Reconstruct 3D glass properties from 2D keypoints.

    Accepts a single GlassKeypoints or a list. Returns a GlassDetection or list
    of GlassDetection (with None for any that failed reconstruction).

    Args:
        keypoints: GlassKeypoints or List[GlassKeypoints] from detector.detect()
        camera_intrinsics: 3x3 camera intrinsic matrix K
        table_height: Height of the table in world frame (meters)
        X_World_Camera: 4x4 transformation from camera to world frame

    Returns:
        GlassDetection (single) or List[GlassDetection] (batch), with None for failures
    """
    if isinstance(keypoints, list):
        return [
            _reconstruct_single(kp, camera_intrinsics, table_height, X_World_Camera)
            for kp in keypoints
        ]
    return _reconstruct_single(keypoints, camera_intrinsics, table_height, X_World_Camera)


def _reconstruct_single(
    keypoints: GlassKeypoints,
    camera_intrinsics: np.ndarray,
    table_height: float,
    X_World_Camera: np.ndarray,
) -> Optional[GlassDetection]:
    K = camera_intrinsics
    bottom_front_2d = keypoints.bottom_front
    top_front_2d = keypoints.top_front
    top_left_2d = keypoints.top_left
    top_right_2d = keypoints.top_right

    # Compute table plane in camera frame
    [a, b, c, d] = np.transpose(X_World_Camera) @ np.array([0.0, 0.0, -1.0, table_height])
    n = np.array([a, b, c])

    # --- Back-project bottom front to 3D ---
    bottom_front_ray = np.linalg.inv(K) @ np.array([bottom_front_2d[0], bottom_front_2d[1], 1.0])
    denominator = n @ bottom_front_ray
    if np.isclose(denominator, 0):
        return None
    bottom_front_3d = bottom_front_ray * -d / denominator

    # Compute basis vectors
    backwards_vector_3d = bottom_front_3d - n * np.dot(bottom_front_3d, n)
    backwards_vector_3d /= np.linalg.norm(backwards_vector_3d)
    side_vector_3d = np.cross(n, backwards_vector_3d)
    side_vector_3d /= np.linalg.norm(side_vector_3d)

    # --- Initial height and radius estimates ---
    fx, fy = K[0, 0], K[1, 1]
    glass_depth = np.linalg.norm(bottom_front_3d)
    width_2d = np.linalg.norm(top_left_2d - top_right_2d)
    height_2d = np.linalg.norm(top_front_2d - bottom_front_2d)

    height_3d = height_2d * glass_depth * 2 / (fx + fy) / np.sqrt(
        1 - (np.dot(bottom_front_3d, n) / np.linalg.norm(bottom_front_3d)) ** 2
    )
    radius_3d = width_2d / 2 * glass_depth * 2 / (fx + fy)

    # --- Iterative refinement ---
    old_height_3d = 0.0
    old_radius_3d = 0.0
    old_height_depth = glass_depth
    old_radius_depth = glass_depth

    while np.abs(old_height_3d - height_3d) > 0.0005 or np.abs(old_radius_3d - radius_3d) > 0.0005:
        old_height_3d = height_3d
        old_radius_3d = radius_3d
        height_depth = np.linalg.norm(bottom_front_3d - height_3d * n)
        radius_depth = np.linalg.norm(bottom_front_3d - height_3d * n + radius_3d * backwards_vector_3d)
        height_3d *= height_depth / old_height_depth
        radius_3d *= radius_depth / old_radius_depth
        old_height_depth = height_depth
        old_radius_depth = radius_depth

    # --- Optimize lateral offset ---
    def get_bottom_front_error_side(p):
        bf = bottom_front_3d + p * side_vector_3d
        tl = bf + radius_3d * backwards_vector_3d - height_3d * n - radius_3d * side_vector_3d
        tr = bf + radius_3d * backwards_vector_3d - height_3d * n + radius_3d * side_vector_3d
        tf = bf - height_3d * n
        proj = lambda pt: (K @ pt)[:2] / (K @ pt)[2]
        return (np.linalg.norm(proj(tl) - top_left_2d) +
                np.linalg.norm(proj(tr) - top_right_2d) +
                np.linalg.norm(proj(tf) - top_front_2d))

    p = fsolve(get_bottom_front_error_side, 0.0)
    bottom_front_3d = bottom_front_3d + p * side_vector_3d

    # --- Optimize radius ---
    def get_radius_error(r):
        tl = bottom_front_3d + r * backwards_vector_3d - height_3d * n - r * side_vector_3d
        tr = bottom_front_3d + r * backwards_vector_3d - height_3d * n + r * side_vector_3d
        tm = bottom_front_3d + r * backwards_vector_3d - height_3d * n
        tf = bottom_front_3d - height_3d * n
        proj = lambda pt: (K @ pt)[:2] / (K @ pt)[2]
        top_middle_2d = (top_left_2d + top_right_2d) / 2
        return (np.linalg.norm(proj(tl) - proj(tr)) - width_2d +
                np.linalg.norm(proj(tm) - proj(tf)) - np.linalg.norm(top_middle_2d - top_front_2d))

    new_radius = fsolve(get_radius_error, radius_3d)[0]
    if 0.005 < new_radius < 0.5:
        radius_3d = new_radius

    # --- Optimize shape (tilt angle + height) ---
    def opt_shape(p):
        glass_angle = p[0]
        h = p[1]
        tf = bottom_front_3d - h * n - np.tan(glass_angle) * h * backwards_vector_3d
        tm = tf + radius_3d * backwards_vector_3d
        tl = tm - radius_3d * side_vector_3d
        tr = tm + radius_3d * side_vector_3d
        proj = lambda pt: (K @ pt)[:2] / (K @ pt)[2]
        top_middle_2d = (top_left_2d + top_right_2d) / 2
        return (np.linalg.norm(proj(tl) - top_left_2d) +
                np.linalg.norm(proj(tr) - top_right_2d) +
                np.linalg.norm(proj(tm) - top_middle_2d) +
                np.linalg.norm(proj(tf) - top_front_2d),
                np.deg2rad(3) - p[0])

    p = fsolve(opt_shape, np.array([np.deg2rad(3), height_3d]))
    if 0 < p[0] < np.deg2rad(10) and 0 < p[1] < 0.3:
        glass_angle = p[0]
        height_3d = p[1]
    else:
        glass_angle = 0.0

    # Sanity check
    if not (0.005 < radius_3d < 1.0 and 0.005 < height_3d < 1.0):
        return None

    # --- Compute 3D keypoints ---
    top_middle_3d = (bottom_front_3d + radius_3d * backwards_vector_3d
                     - height_3d * n - np.tan(glass_angle) * height_3d * backwards_vector_3d)

    # --- Fluid level ---
    fluid_percentage = 1.0
    if keypoints.fluid_level is not None:
        fluid_percentage = np.linalg.norm(keypoints.fluid_level - bottom_front_2d) / height_2d
        if keypoints.fluid_level_2 is not None:
            fluid_2_pct = np.linalg.norm(keypoints.fluid_level_2 - bottom_front_2d) / height_2d
            fluid_percentage = max(fluid_percentage, fluid_2_pct)

    # --- Reprojection error ---
    def get_final_error():
        tl = bottom_front_3d + radius_3d * backwards_vector_3d - height_3d * n - np.tan(glass_angle) * height_3d * backwards_vector_3d - radius_3d * side_vector_3d
        tr = bottom_front_3d + radius_3d * backwards_vector_3d - height_3d * n - np.tan(glass_angle) * height_3d * backwards_vector_3d + radius_3d * side_vector_3d
        tf = bottom_front_3d - height_3d * n - np.tan(glass_angle) * height_3d * backwards_vector_3d
        bf = bottom_front_3d
        proj = lambda pt: (K @ pt)[:2] / (K @ pt)[2]
        return (np.linalg.norm(proj(tl) - top_left_2d) +
                np.linalg.norm(proj(tr) - top_right_2d) +
                np.linalg.norm(proj(tf) - top_front_2d) +
                np.linalg.norm(proj(bf) - bottom_front_2d))

    reprojection_error = get_final_error()

    return GlassDetection(
        center_3d=top_middle_3d,
        radius=radius_3d,
        height=height_3d,
        glass_angle=glass_angle,
        fluid_percentage=fluid_percentage,
        keypoints=keypoints,
        reprojection_error=reprojection_error,
    )


def detect_table_height(
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    X_World_Camera: np.ndarray,
    crop_height_min: float = 0.3,
    crop_height_max: float = 1.3,
) -> float:
    """Estimate table height from a depth map.

    Projects depth map pixels to 3D world coordinates and finds the
    dominant Z height (table surface) using histogram mode estimation.

    Args:
        depth_map: Depth image (H, W) in meters
        camera_intrinsics: 3x3 camera intrinsic matrix
        X_World_Camera: 4x4 camera-to-world transformation
        crop_height_min: Minimum height to consider (meters)
        crop_height_max: Maximum height to consider (meters)

    Returns:
        Estimated table height in world frame (meters)
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    h, w = depth_map.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Project to camera coordinates
    z_c = depth_map
    x_c = (uu - cx) * z_c / fx
    y_c = (vv - cy) * z_c / fy

    # Transform to world
    ones = np.ones_like(z_c)
    points_cam = np.stack((x_c, y_c, z_c, ones), axis=-1).reshape(-1, 4)
    points_world = (X_World_Camera @ points_cam.T).T[:, :3]
    heights = points_world[:, 2]

    # Find dominant height
    hist, bin_edges = np.histogram(
        heights, bins=int((crop_height_max - crop_height_min) / 0.001),
        range=(crop_height_min, crop_height_max)
    )
    max_bin = np.argmax(hist)
    table_height = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2 - 0.005

    return float(table_height)
