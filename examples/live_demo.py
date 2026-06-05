#!/usr/bin/env python3
"""
Live camera demo: real-time glass localization with ZED or RealSense.

Connects to an RGB-D camera, estimates table height from the depth map,
and continuously localizes glasses with full 3D reconstruction.

Usage:
    # ZED camera
    python examples/live_demo.py --camera zed

    # RealSense camera
    python examples/live_demo.py --camera realsense

    # USB webcam (2D detection only -- no depth / no 3D)
    python examples/live_demo.py --camera usb

Requires:
    - SPILL installed (pip install -e .)
    - pyzed or pyrealsense2 for respective modes
    - opencv-python for display
"""
import argparse
import cv2
import numpy as np
import time
from pathlib import Path

from spill import GlassDetector, reconstruct_glass_3d, detect_table_height


# --- Camera backends ---

def create_zed_camera(serial=None):
    """Create a ZED camera and return (grab_fn, depth_fn, K)."""
    import pyzed.sl as sl

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    if serial:
        init_params.input.set_from_serial_number(serial)

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    runtime_params = sl.RuntimeParameters()
    calib = zed.get_camera_information().camera_calibration_parameters
    K = np.array(calib.camera_data, dtype=np.float32).reshape(3, 3)

    def grab():
        if zed.grab(runtime_params, sl.TIMEOUT_WAIT_ONE) != sl.ERROR_CODE.SUCCESS:
            return None, None
        image = sl.Mat()
        depth = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.RGB)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        img_np = image.get_data()
        depth_np = depth.get_data().astype(np.float32) / 1000.0
        return img_np, depth_np

    def close():
        zed.close()

    return grab, K, close


def create_realsense_camera(serial=None):
    """Create a RealSense camera and return (grab_fn, depth_fn, K)."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if serial:
        config.enable_device(serial)

    profile = pipeline.start(config)
    rs_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [rs_intr.fx, 0, rs_intr.ppx],
        [0, rs_intr.fy, rs_intr.ppy],
        [0, 0, 1],
    ], dtype=np.float32)

    def grab():
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None, None
        img_np = np.asanyarray(color.get_data())
        depth_np = np.asanyarray(depth.get_data()).astype(np.float32) / 1000.0
        return img_np, depth_np

    def close():
        pipeline.stop()

    return grab, K, close


def create_usb_camera(device_id=0):
    """Create a USB webcam (RGB only, no depth)."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open USB camera {device_id}")

    def grab():
        ret, img = cap.read()
        if not ret:
            return None, None
        return img, None

    def close():
        cap.release()

    return grab, None, close


# --- Visualization ---

KP_COLORS = {
    "bottom_front": (0, 0, 255),
    "top_front": (0, 255, 0),
    "top_left": (255, 0, 0),
    "top_right": (255, 0, 255),
    "fluid_level": (255, 255, 0),
}


def draw_keypoints(image, keypoints_list):
    """Draw keypoints and bounding boxes on the image."""
    output = image.copy()
    for idx, kp in enumerate(keypoints_list):
        x1, y1, x2, y2 = kp.bounding_box
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(output, f"Glass #{idx+1}", (int(x1), max(int(y1) - 5, 15)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for name, pt in [
            ("bottom_front", kp.bottom_front),
            ("top_front", kp.top_front),
            ("top_left", kp.top_left),
            ("top_right", kp.top_right),
            ("fluid_level", kp.fluid_level),
        ]:
            if pt is not None:
                color = KP_COLORS.get(name, (255, 255, 255))
                cv2.circle(output, (int(pt[0]), int(pt[1])), 5, color, -1)

    return output


def main():
    parser = argparse.ArgumentParser(description="Live glass localization demo")
    parser.add_argument("--camera", choices=["zed", "realsense", "usb"], default="zed",
                        help="Camera type (default: zed)")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial number")
    parser.add_argument("--device", type=int, default=0, help="USB device ID (for --camera usb)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to keypoint checkpoint")
    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO model")
    parser.add_argument("--device-gpu", type=str, default="cuda", help="Device for inference (cuda/cpu)")
    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(base_dir / "checkpoints" / "wild_glasses.ckpt")
    if args.yolo is None:
        args.yolo = str(base_dir / "checkpoints" / "yolov8m.pt")

    # Load detector
    print(f"Loading SPILL models on {args.device_gpu}...")
    detector = GlassDetector(
        keypoint_checkpoint=args.checkpoint,
        yolo_model_path=args.yolo,
        device=args.device_gpu,
    )
    print("Models loaded!")

    # Connect camera
    print(f"Connecting to {args.camera} camera...")
    camera_init = {
        "zed": create_zed_camera,
        "realsense": create_realsense_camera,
        "usb": create_usb_camera,
    }[args.camera]

    if args.camera == "usb":
        grab_fn, K, close_fn = camera_init(args.device)
    else:
        grab_fn, K, close_fn = camera_init(args.serial)

    has_depth = K is not None
    if has_depth:
        print(f"Camera connected with depth. K:\n{K}")
    else:
        print(f"USB webcam connected (2D detection only)")

    # Camera-to-world transform (identity = camera IS the world frame)
    X_World_Camera = np.eye(4)

    print("\nControls: q=quit, r=re-estimate table, +/-=adjust table height")
    print("Table height is auto-estimated from depth on first frame.\n")

    table_height = None

    while True:
        t_start = time.monotonic()

        # Capture
        image, depth_map = grab_fn()
        if image is None:
            continue

        # Auto-estimate table height once
        if has_depth and table_height is None:
            table_height = detect_table_height(depth_map, K, X_World_Camera)
            print(f"Auto-estimated table height: {table_height*100:.1f} cm")

        # Detect
        t_detect = time.monotonic()
        keypoints_list = detector.detect(image)
        detect_ms = (time.monotonic() - t_detect) * 1000

        # 3D reconstruction
        glasses_3d = []
        if has_depth and table_height is not None:
            for kp in keypoints_list:
                glass = reconstruct_glass_3d(kp, K, table_height, X_World_Camera)
                if glass:
                    glasses_3d.append(glass)

        # Draw
        output = draw_keypoints(image, keypoints_list)

        # Overlay info
        fps = 1.0 / max(time.monotonic() - t_start, 0.001)
        info_lines = [f"FPS: {fps:.1f}  |  Detection: {detect_ms:.0f}ms"]
        if has_depth and table_height is not None:
            info_lines.append(f"Table: {table_height*100:.1f}cm")

        for i, glass in enumerate(glasses_3d):
            info_lines.append(
                f"Glass #{i+1}: r={glass.radius*100:.1f}cm h={glass.height*100:.1f}cm "
                f"fluid={glass.fluid_percentage*100:.0f}%"
            )

        y_offset = 25
        for line in info_lines:
            cv2.putText(output, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 20

        cv2.imshow("SPILL Live Demo", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and has_depth and depth_map is not None:
            table_height = detect_table_height(depth_map, K, X_World_Camera)
            print(f"Re-estimated table height: {table_height*100:.1f} cm")
        elif key in (ord("+"), ord("=")) and table_height is not None:
            table_height += 0.005
            print(f"Manual table height: {table_height*100:.1f} cm")
        elif key == ord("-") and table_height is not None:
            table_height -= 0.005
            print(f"Manual table height: {table_height*100:.1f} cm")

    cv2.destroyAllWindows()
    close_fn()
    print("Bye!")


if __name__ == "__main__":
    main()
