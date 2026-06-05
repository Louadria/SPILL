"""
HuggingFace Space app for SPILL: Glass Keypoint Detection + 3D Reconstruction

Gradio demo with two tabs:
1. 2D Reconstruction — detect keypoints on uploaded images
2. 3D Reconstruction — monocular depth estimation + cylinder reconstruction

To deploy:
1. Create a new Space on HuggingFace with `gradio` SDK
2. Upload this directory contents
3. Or run: `huggingface-cli login` then push from terminal
"""
import os
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path

# Import SPILL library (installed via requirements.txt)
from spill import GlassDetector, Monocular3DReconstructor
from spill import depth_overlay, create_3d_figure, build_3d_info

# Model paths - checkpoints are stored in the repo
BASE_DIR = Path(__file__).parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "wild_glasses.ckpt"
YOLO_PATH = BASE_DIR / "checkpoints" / "yolov8m.pt"

# Detect device — default to CPU for safety (Blackwell GPU + old PyTorch issue).
# HF Spaces set SPILL_FORCE_CUDA=1 in the Dockerfile to enable GPU.
if os.environ.get("SPILL_FORCE_CUDA"):
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Loading SPILL models on {DEVICE}...")
detector = GlassDetector(
    keypoint_checkpoint=str(CHECKPOINT_PATH),
    yolo_model_path=str(YOLO_PATH),
    device=DEVICE,
)
print("2D models loaded!")

# Lazy-load 3D reconstructor on first use (saves GPU VRAM at startup)
_reconstructor = None


def get_reconstructor():
    global _reconstructor
    if _reconstructor is None:
        print("[3D] Lazy-loading 3D reconstructor...")
        _reconstructor = Monocular3DReconstructor(
            depth_model_size="large",
            device=DEVICE,
        )
    return _reconstructor


KP_COLORS = {
    "bottom_front": (255, 0, 0),      # Red
    "top_front": (0, 255, 0),          # Green
    "top_left": (0, 0, 255),           # Blue
    "top_right": (255, 0, 255),        # Magenta
    "fluid_level": (0, 255, 255),      # Cyan
    "fluid_level_2": (255, 255, 0),    # Yellow
}


def draw_detections(image, keypoints_list):
    """Draw keypoints and bounding boxes on the image."""
    output = image.copy()

    for idx, kp in enumerate(keypoints_list):
        # Draw original YOLO bounding box (no padding)
        x1, y1, x2, y2 = kp.bounding_box
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(output, f"Glass #{idx+1}", (int(x1), int(y1) - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw keypoints (no text labels on image)
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
                color = KP_COLORS[name]
                cv2.circle(output, (int(pt[0]), int(pt[1])), 6, color, -1)
                cv2.circle(output, (int(pt[0]), int(pt[1])), 8, color, 1)

    return output


def detect_glasses_2d(image):
    """2D tab callback: detect glasses and return annotated image + info."""
    if image is None:
        return None, "Please upload an image."

    # Gradio gives us RGB, convert to BGR for OpenCV
    if isinstance(image, dict):
        image = image["image"]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect
    keypoints_list = detector.detect(image_bgr)

    if not keypoints_list:
        output = image.copy()
        return output, "No glasses detected. Try an image with clear glasses/cups/wine glasses."

    # Draw
    output_rgb = draw_detections(image_bgr, keypoints_list)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)

    # Build info text — per-keypoint pixel coordinates (CriticBarista style)
    info_lines = [f"Found {len(keypoints_list)} glass(es):\n"]
    for i, kp in enumerate(keypoints_list):
        info_lines.append(f"\n--- Glass #{i+1} ---")
        info_lines.append(f"  bottom_front:  ({kp.bottom_front[0]:.0f}, {kp.bottom_front[1]:.0f})")
        info_lines.append(f"  top_front:     ({kp.top_front[0]:.0f}, {kp.top_front[1]:.0f})")
        info_lines.append(f"  top_left:      ({kp.top_left[0]:.0f}, {kp.top_left[1]:.0f})")
        info_lines.append(f"  top_right:     ({kp.top_right[0]:.0f}, {kp.top_right[1]:.0f})")
        if kp.fluid_level is not None:
            info_lines.append(f"  fluid_level:   ({kp.fluid_level[0]:.0f}, {kp.fluid_level[1]:.0f})")
        if kp.fluid_level_2 is not None:
            info_lines.append(f"  fluid_level_2: ({kp.fluid_level_2[0]:.0f}, {kp.fluid_level_2[1]:.0f})")

    info = "\n".join(info_lines)
    return output_rgb, info


def detect_glasses_3d(image):
    """3D tab callback: full monocular 3D reconstruction."""
    if image is None:
        return None, None, None, "Please upload an image."

    if isinstance(image, dict):
        image = image["image"]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Step 1: 2D keypoint detection
    keypoints_list = detector.detect(image_bgr)

    if not keypoints_list:
        output = image.copy()
        return (output, None, None,
                "No glasses detected. Try an image with clear glasses/cups/wine glasses.")

    # Step 2: Load 3D reconstructor (lazy)
    reconstructor = get_reconstructor()

    # Step 3: Full 3D reconstruction
    glasses, depth_map, info = reconstructor.reconstruct(image_bgr, keypoints_list)

    # Step 4: Annotated image (2D keypoints overlaid)
    annotated_rgb = draw_detections(image_bgr, keypoints_list)
    annotated_rgb = cv2.cvtColor(annotated_rgb, cv2.COLOR_BGR2RGB)

    # Step 5: Depth overlay
    depth_vis = depth_overlay(image, depth_map)

    # Step 6: 3D plot
    plot = create_3d_figure(glasses, info, image.shape)

    # Step 7: Info text
    info_text = build_3d_info(glasses, keypoints_list, info)

    return annotated_rgb, depth_vis, plot, info_text


# ── Build Gradio interface ──────────────────────────────────────

DESCRIPTION = """
# SPILL: Glass Detection & 3D Reconstruction

Detect **transparent glassware** in images using semantic keypoint detection — and reconstruct full 3D cylinders from a single RGB image.

**How it works:**
1. YOLOv8 detects glass bounding boxes (cups, vases, wine glasses)
2. A keypoint detector predicts structural points + fluid level on each glass
3. (3D tab) DepthAnythingV2 estimates monocular depth → RANSAC finds the table plane → cylinder estimation gives radius, height, tilt, and fluid level

**Key points:**
- 🔴 Bottom Front — base of the glass facing the camera
- 🟢 Top Front — rim edge facing the camera
- 🔵 Top Left — left edge of the rim
- 🟣 Top Right — right edge of the rim
- 🟡 Fluid Level — liquid surface detected by the model
- 🟠 Fluid Level (alt) — secondary fluid level candidate (shown when the model detects multiple peaks on the fluid level heatmap; useful when the first peak is uncertain, so downstream use cases can conservatively pick the highest or lowest value)

**3D Reconstruction** uses DepthAnythingV2-Large for monocular depth estimation, Open3D RANSAC for plane detection, and the SPILL cylinder solver for full 3D parameters.
"""

with gr.Blocks(title="SPILL Glass Detection & 3D Reconstruction") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ── 2D Detection Tab ──
        with gr.Tab("2D Reconstruction"):
            gr.Markdown(
                "### 2D Keypoint Detection\n\n"
                "Upload an image and see detected glass keypoints overlaid. "
                "No depth camera needed."
            )
            with gr.Row():
                input_2d = gr.Image(type="numpy", label="Upload Image", sources=["upload", "clipboard"])
                output_2d = gr.Image(type="numpy", label="Detection Result")
            detect_2d_btn = gr.Button("Detect Glasses", variant="primary")
            info_2d = gr.Textbox(label="Detection Info")
            detect_2d_btn.click(
                fn=detect_glasses_2d,
                inputs=input_2d,
                outputs=[output_2d, info_2d],
            )

        # ── 3D Reconstruction Tab ──
        with gr.Tab("3D Reconstruction"):
            gr.Markdown(
                "### Monocular 3D Reconstruction\n\n"
                "From a single RGB image: estimate depth (DepthAnythingV2), find the table plane (RANSAC), "
                "and reconstruct 3D glass cylinders — radius, height, tilt angle, and fluid level. "
                "No depth camera required!\n\n"
                "**Best results:** place the glass on a flat surface (table), keep the camera roughly level."
            )
            with gr.Row():
                input_3d = gr.Image(type="numpy", label="Upload Image", sources=["upload", "clipboard"])
                output_3d_annotated = gr.Image(type="numpy", label="Keypoints Overlay")
            with gr.Row():
                output_3d_depth = gr.Image(type="numpy", label="Depth Estimate")
                output_3d_plot = gr.Plot(label="3D Reconstruction")
            detect_3d_btn = gr.Button("Reconstruct 3D", variant="primary")
            info_3d = gr.Textbox(label="3D Reconstruction Info")
            detect_3d_btn.click(
                fn=detect_glasses_3d,
                inputs=input_3d,
                outputs=[output_3d_annotated, output_3d_depth, output_3d_plot, info_3d],
            )

    gr.Markdown("""
    ---
    **Paper:** [SPILL: Size, Pose, and Internal Liquid Level Estimation](https://github.com/Louadria/SPILL)
    | **Dataset:** [Glasses-in-the-Wild](https://doi.org/10.5281/zenodo.17288314)
    | **Code:** [Louadria/SPILL](https://github.com/Louadria/SPILL)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
