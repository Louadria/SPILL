"""
HuggingFace Space app for SPILL: Glass Keypoint Detection

Gradio demo that lets users upload images and see detected glass keypoints.
Uses CPU inference since HF Spaces (free tier) don't always have GPU.

To deploy:
1. Create a new Space on HuggingFace with `gradio` SDK
2. Upload this directory contents
3. Or run: `huggingface-cli login` then push from terminal
"""
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path

# Import SPILL library (installed via requirements.txt)
from spill import GlassDetector

# Model paths - checkpoints are stored in the repo
BASE_DIR = Path(__file__).parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "wild_glasses.ckpt"
YOLO_PATH = BASE_DIR / "checkpoints" / "yolov8m.pt"

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SPILL models on {DEVICE}...")
detector = GlassDetector(
    keypoint_checkpoint=str(CHECKPOINT_PATH),
    yolo_model_path=str(YOLO_PATH),
    device=DEVICE,
)
print("Models loaded!")


KP_COLORS = {
    "bottom_front": (255, 0, 0),      # Red
    "top_front": (0, 255, 0),          # Green
    "top_left": (0, 0, 255),           # Blue
    "top_right": (255, 0, 255),        # Magenta
    "fluid_level": (0, 255, 255),      # Cyan
    "fluid_level_2": (255, 255, 0),    # Yellow
}

KP_LABELS = {
    "bottom_front": "Bottom Front",
    "top_front": "Top Front",
    "top_left": "Top Left",
    "top_right": "Top Right",
    "fluid_level": "Fluid Level",
    "fluid_level_2": "Fluid Level (alt)",
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


def detect_glasses(image):
    """Gradio callback: detect glasses and return annotated image + info."""
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


# Build Gradio interface
DESCRIPTION = """
# SPILL: Glass Keypoint Detection

Detect **transparent glassware** in images using semantic keypoint detection.

**How it works:**
1. YOLOv8 detects glass bounding boxes (cups, vases, wine glasses)
2. A keypoint detector predicts structural points + fluid level on each glass

**Key points:**
- 🔴 Bottom Front — base of the glass facing the camera
- 🟢 Top Front — rim edge facing the camera
- 🔵 Top Left — left edge of the rim
- 🟣 Top Right — right edge of the rim
- 🟡 Fluid Level — liquid surface detected by the model
- 🟠 Fluid Level (alt) — secondary fluid level candidate (shown when the model detects multiple peaks on the fluid level heatmap; useful when the first peak is uncertain, so downstream use cases can conservatively pick the highest or lowest value)

Full 3D reconstruction requires a depth camera and calibrated transforms — see the [GitHub repo](https://github.com/Louadria/SPILL) for the complete pipeline.
"""

with gr.Blocks(title="SPILL Glass Detection") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        input_img = gr.Image(type="numpy", label="Upload Image", sources=["upload", "clipboard"])
        output_img = gr.Image(type="numpy", label="Detection Result")

    detect_btn = gr.Button("Detect Glasses", variant="primary")
    info_output = gr.Textbox(label="Detection Info")

    detect_btn.click(
        fn=detect_glasses,
        inputs=input_img,
        outputs=[output_img, info_output],
    )

    gr.Markdown("""
    ---
    **Paper:** [SPILL: Size, Pose, and Internal Liquid Level Estimation](https://github.com/Louadria/SPILL)
    | **Dataset:** [Glasses-in-the-Wild](https://doi.org/10.5281/zenodo.17288314)
    | **Code:** [Louadria/SPILL](https://github.com/Louadria/SPILL)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
