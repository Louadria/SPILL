import time
from typing import List

import cv2
import numpy as np
import open3d as o3d
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import OpenCVIntImageType, NumpyFloatImageType


class CameraVisualizer:
    def __init__(self, intrinsics: np.ndarray):
        self._intrinsics = intrinsics
        self._image = None
        self._image_original = None
        self._mask = None
        # cv2.namedWindow("RealSense RGB", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("RealSense RGB", 1280, 720)  # Set the desired width and height

    def update(self, image: OpenCVIntImageType) -> None:
        """Update the image to display.

        Args:
            image: The image to display

        Returns:
            None"""
        self._image = image
        self._image_original = image.copy()

    def draw_bounding_boxes(self, bounding_boxes: np.ndarray | None, name="unknown") -> None:
        """Draw bounding boxes on the image.

        Args:
            bounding_boxes: The bounding boxes to draw.
            name: What label to put on the bounding boxes.

        Returns:
            None"""
        if bounding_boxes is None:
            return

        # Draw bounding boxes
        for bounding_box in bounding_boxes:
            print(f"Drawing bounding box: {bounding_box}")
            x1, y1, x2, y2 = bounding_box[0]
            cv2.rectangle(self._image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(self._image, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def draw_3d_bounding_boxes(self, bounding_boxes: List[o3d.geometry.OrientedBoundingBox] | None, name="unknown") -> None:
        """Draw 3D Oriented Bounding Boxes on the image.

        Args:
            bounding_boxes: The bounding boxes to draw.
            name: What label to put on the bounding boxes.

        Returns:
            None"""
        if bounding_boxes is None:
            return

        # Draw 3D bounding boxes
        for bounding_box in bounding_boxes:
            print(f"Drawing 3D bounding box: {bounding_box}")
            vertices_2d = []
            for vertex in np.asarray(bounding_box.get_box_points()):
                x, y, z = vertex
                x, y, z = self._intrinsics @ np.array([x, y, z])
                vertices_2d.append([x / z, y / z])
            vertices_2d = np.array(vertices_2d)

            for i in range(4):
                cv2.line(self._image, tuple(vertices_2d[i].astype(int)), tuple(vertices_2d[(i + 1) % 4].astype(int)), (0, 255, 0), 2)
                cv2.line(self._image, tuple(vertices_2d[i + 4].astype(int)), tuple(vertices_2d[(i + 1) % 4 + 4].astype(int)), (0, 255, 0), 2)
                cv2.line(self._image, tuple(vertices_2d[i].astype(int)), tuple(vertices_2d[i + 4].astype(int)), (0, 255, 0), 2)

            cv2.putText(self._image, name, tuple(vertices_2d[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def draw_glass_sizes(self, pos, radius, height, glass_angle, x, y):
        drawpos_x, drawpos_y = pos
        cv2.putText(self._image, f"D: {radius*2*100:.2f}cm", (int(drawpos_x), int(drawpos_y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(self._image, f"H: {height*100:.2f}cm", (int(drawpos_x), int(drawpos_y - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(self._image, f"A: {np.rad2deg(glass_angle):.2f}", (int(drawpos_x), int(drawpos_y - 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(self._image, f"XY: ({x*100:.2f},{y*100:.2f})cm", (int(drawpos_x), int(drawpos_y - 90)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_segmentation_mask(self, mask: NumpyFloatImageType) -> None:
        """Draw the segmentation mask on the image.

        Args:
            mask: The mask to draw.

        Returns:
            None"""
        if mask is None:
            return

        # Draw the mask
        if self._mask is None:
            self._mask = mask
        else:
            self._mask = np.maximum(self._mask, mask)

    def draw_keypoints(self, keypoints: np.ndarray) -> None:
        """Draw keypoints on the image.

        Args:
            keypoints: The keypoints to draw.

        Returns:
            None"""
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)] * (len(keypoints) + 4 // 5)
        for color_idx, keypoint in enumerate(keypoints):
            x, y = keypoint
            cv2.circle(self._image, (int(x), int(y)), 5, colors[color_idx], -1)

    def add_triangle_meshes(self, meshes: list[o3d.t.geometry.TriangleMesh], X_Platform_Camera = None):
        for mesh, color in zip(meshes, [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * (len(meshes) + 2 // 3)):
            vertices_2d = []
            for vertex in np.asarray(mesh.vertices):
                x, y, z = vertex
                # if X_Platform_Camera is not None:
                #     vertex = X_Platform_Camera @ np.array([x, y, z, 1])
                #     x, y, z = vertex[:3]
                x, y, z = self._intrinsics @ np.array([x, y, z])
                vertices_2d.append([x / z, y / z])
            vertices_2d = np.array(vertices_2d)

            for triangle in np.asarray(mesh.triangles):
                cv2.line(self._image, tuple(vertices_2d[triangle[0]].astype(int)), tuple(vertices_2d[triangle[1]].astype(int)), color, 1)
                cv2.line(self._image, tuple(vertices_2d[triangle[1]].astype(int)), tuple(vertices_2d[triangle[2]].astype(int)), color, 1)
                cv2.line(self._image, tuple(vertices_2d[triangle[2]].astype(int)), tuple(vertices_2d[triangle[0]].astype(int)), color, 1)

    def show_triangle_meshes(self, meshes: list[o3d.t.geometry.TriangleMesh], X_Platform_Camera = None):
        self.add_triangle_meshes(meshes, X_Platform_Camera)
        # first resize to 1280x720
        self._image = cv2.resize(self._image, (1280, 720))
        cv2.imshow("Triangle mesh visualization", self._image)

    def show(self, blocking=False) -> int:
        """Show the image.

        Returns:
            The key pressed by the user"""
        if self._mask is not None:
            self._image = cv2.addWeighted(self._image, 0.8, cv2.cvtColor((self._mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.2, 0)
            self._mask = None

        if self._image is not None:
            cv2.imshow("RealSense RGB", self._image)
        return cv2.waitKey(0 if blocking else 1)

    def save_image(self):
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
        cv2.imwrite(f"detection_screenshots/image_{current_time}.png", self._image)
        cv2.imwrite(f"detection_screenshots/image_{current_time}_original.png", self._image_original)

    def draw_fluid_lines(self, bottom_front_2d, top_front_2d, front_center_2d, width_2d, fluid_level_2d, fluid_2nd_level_2d):
        top_bottom_vector_2d = (top_front_2d - bottom_front_2d) / np.linalg.norm(top_front_2d - bottom_front_2d)
        side_vector_2d = np.array([top_bottom_vector_2d[1], -top_bottom_vector_2d[0]])
        cv2.line(self._image, (
            int(front_center_2d[0] - np.linalg.norm(front_center_2d - top_front_2d) * top_bottom_vector_2d[0]),
            int(front_center_2d[1] - np.linalg.norm(front_center_2d - top_front_2d) * top_bottom_vector_2d[1])),
                 (int(front_center_2d[0] + np.linalg.norm(front_center_2d - bottom_front_2d) *
                      top_bottom_vector_2d[0]), int(
                     front_center_2d[1] + np.linalg.norm(front_center_2d - bottom_front_2d) *
                     top_bottom_vector_2d[1])),
                 (0, 255, 0), 2)
        if fluid_level_2d is not None:
            cv2.line(self._image, (int(fluid_level_2d[0] - width_2d / 2 * side_vector_2d[0]),
                             int(fluid_level_2d[1] - width_2d / 2 * side_vector_2d[1])),
                     (int(fluid_level_2d[0] + width_2d / 2 * side_vector_2d[0]),
                      int(fluid_level_2d[1] + width_2d / 2 * side_vector_2d[1])),
                     (255, 0, 0), 2)
            if fluid_2nd_level_2d is not None:
                cv2.line(self._image, (int(fluid_2nd_level_2d[0] - width_2d / 2 * side_vector_2d[0]),
                                 int(fluid_2nd_level_2d[1] - width_2d / 2 * side_vector_2d[1])),
                         (int(fluid_2nd_level_2d[0] + width_2d / 2 * side_vector_2d[0]),
                          int(fluid_2nd_level_2d[1] + width_2d / 2 * side_vector_2d[1])),
                         (255, 0, 255), 2)
        else:
            cv2.line(self._image, (int(front_center_2d[0] - width_2d / 2 * side_vector_2d[0]),
                             int(front_center_2d[1] - width_2d / 2 * side_vector_2d[1])),
                     (int(front_center_2d[0] + width_2d / 2 * side_vector_2d[0]),
                      int(front_center_2d[1] + width_2d / 2 * side_vector_2d[1])),
                     (255, 0, 0), 2)