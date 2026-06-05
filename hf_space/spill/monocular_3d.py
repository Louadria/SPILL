"""
Monocular 3D reconstruction from single RGB images.

Pipeline:
1. DepthAnythingV2 estimates relative depth from a single RGB image
2. Open3D RANSAC finds the table plane from the depth point cloud
3. SPILL cylinder estimation reconstructs 3D glass parameters
4. Plotly creates an interactive 3D visualization

No depth camera required — works from any photo.
"""

import os
import numpy as np
import torch
import cv2
import open3d as o3d
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict

from .types import GlassKeypoints, GlassDetection


# ── Depth estimation ──────────────────────────────────────────────

class MonocularDepthEstimator:
    """Estimate monocular depth using DepthAnythingV2."""

    CONFIGS = {
        'small': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
            'repo': 'depth-anything/Depth-Anything-V2-Small',
            'file': 'depth_anything_v2_vits.pth',
        },
        'large': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024],
            'repo': 'depth-anything/Depth-Anything-V2-Large',
            'file': 'depth_anything_v2_vitl.pth',
        },
    }

    @staticmethod
    def _pick_device(device: Optional[str] = None) -> torch.device:
        """Pick a working device, falling back from cuda -> cpu.

        Avoids touching CUDA if the GPU is known-incompatible (e.g. Blackwell
        with old PyTorch) to prevent contaminating the CUDA runtime context.
        """
        if device is not None:
            return torch.device(device)
        # Never auto-pick CUDA here — let the caller (HF Space app) pass
        # an explicit 'cuda' string.  Auto-detect defaults to CPU so that
        # local dev machines with mismatched CUDA drivers still work.
        if os.environ.get('SPILL_FORCE_CUDA'):
            return torch.device('cuda')
        return torch.device('cpu')

    def __init__(self, size: str = 'small', device: Optional[str] = None):
        self._device = self._pick_device(device)
        self._size = size

        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        cfg = self.CONFIGS[size]
        print(f"[3D] Loading DepthAnythingV2-{size.upper()} on {device} ...")

        self._model = DepthAnythingV2(
            encoder=cfg['encoder'],
            features=cfg['features'],
            out_channels=cfg['out_channels'],
        )
        self._model.to(self._device)

        ckpt = hf_hub_download(repo_id=cfg['repo'], filename=cfg['file'])
        self._model.load_state_dict(
            torch.load(ckpt, map_location=self._device, weights_only=True)
        )
        self._model.eval()

        # Monkey-patch image2tensor so it uses our device instead of hardcoded CUDA
        # DepthAnythingV2.image2tensor hardcodes CUDA detection which fails on broken GPUs
        _orig_image2tensor = self._model.image2tensor
        _device = self._device

        def _patched_image2tensor(raw_image, input_size=518):
            # Import transform components
            import cv2 as _cv2
            from depth_anything_v2.util.transform import (
                NormalizeImage, PrepareForNet, Resize,
            )
            class Compose:
                def __init__(self, transforms): self.transforms = transforms
                def __call__(self, d):
                    for t in self.transforms: d = t(d)
                    return d
            transform = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=_cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            h, w = raw_image.shape[:2]
            image = _cv2.cvtColor(raw_image, _cv2.COLOR_BGR2RGB) / 255.0
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(_device)
            return image, (h, w)

        self._model.image2tensor = _patched_image2tensor
        print(f"[3D] Depth model ready.")

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return relative depth map (H,W) float32, higher = farther."""
        with torch.no_grad():
            return self._model.infer_image(image_bgr)


# ── Camera intrinsics heuristic ───────────────────────────────────

def estimate_intrinsics(h: int, w: int) -> np.ndarray:
    """Pinhole intrinsics from image size (50 mm equiv focal length)."""
    f = 1.2 * max(w, h)
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0, 1.0]], dtype=np.float64)


# ── RANSAC table plane ────────────────────────────────────────────

def ransac_table_plane(
    depth: np.ndarray,
    K: np.ndarray,
    keypoints: Optional[List[GlassKeypoints]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Find the dominant table plane in the depth map via RANSAC.

    Returns (normal, d) where  normal·x + d = 0  (normal unit, points toward camera).
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
    Z = depth.astype(np.float64)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    flat_z = Z.ravel()

    # Keep valid, non-outlier depths
    valid = flat_z > 0
    if valid.sum() > 0:
        lo, hi = np.percentile(flat_z[valid], [5, 95])
        valid &= (flat_z > lo) & (flat_z < hi)

    # Focus on bottom 40 % of the image (table region)
    bottom = (v.ravel() > h * 0.55)
    valid &= bottom

    # Also keep a ROI around each glass bottom-front keypoint
    if keypoints:
        roi = np.zeros(h * w, dtype=bool)
        for kp in keypoints:
            bx, by = kp.bottom_front
            r = max(40, int(0.08 * h))
            for row in range(max(0, int(by) - r), min(h, int(by) + r)):
                for col in range(max(0, int(bx) - r), min(w, int(bx) + r)):
                    roi[row * w + col] = True
        valid |= (roi & (flat_z > 0))

    pts = pts[valid]
    if len(pts) < 50:
        # Fallback: all valid bottom-half points
        pts = pts[v.ravel() > h * 0.5]
    if len(pts) < 10:
        raise ValueError(f"Only {len(pts)} valid points for RANSAC — cannot find table plane.")

    # Open3D plane segmentation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    scale = np.percentile(np.linalg.norm(pts, axis=1), 75)
    thr = max(scale * 0.08, 0.01)

    model, inliers = pcd.segment_plane(
        distance_threshold=thr, ransac_n=3, num_iterations=3000
    )
    a, b, c, d = model
    n = np.array([a, b, c])

    # Normal should point toward camera (negative Z for forward-facing)
    if n[2] > 0:
        n = -n
        d = -d

    mag = np.linalg.norm(n)
    return n / mag, d / mag


# ── World-frame construction ──────────────────────────────────────

def build_world_transform(
    plane_normal: np.ndarray, plane_d: float
) -> Tuple[np.ndarray, float]:
    """
    Build X_World_Camera and table_height so the RANSAC plane matches
    the world-frame plane z = table_height.

    World Z (up) is aligned with -plane_normal in camera frame.
    """
    up_cam = -plane_normal  # world Z in camera coords

    # Right vector (arbitrary, perpendicular to up)
    if abs(up_cam[2]) < 0.9:
        right = np.cross(up_cam, [0, 0, 1])
    else:
        right = np.cross(up_cam, [1, 0, 0])
    right /= np.linalg.norm(right)

    forward = np.cross(right, up_cam)
    forward /= np.linalg.norm(forward)

    R = np.column_stack([forward, right, up_cam])  # 3x3
    X = np.eye(4)
    X[:3, :3] = R
    # Translation = 0 (world origin at camera)

    return X, float(plane_d)


# ── Full monocular 3D reconstruction ──────────────────────────────

class Monocular3DReconstructor:
    """
    End-to-end monocular 3D glass reconstruction.

    Parameters
    ----------
    depth_model_size : str
        'small' (default, fast) or 'large' (slower, more accurate depth).
    device : str or None
        'cuda', 'cpu', or None for auto-detect.
    """

    def __init__(self, depth_model_size: str = 'small', device: Optional[str] = None):
        self._depth = MonocularDepthEstimator(size=depth_model_size, device=device)

    def reconstruct(
        self,
        image_bgr: np.ndarray,
        keypoints: List[GlassKeypoints],
        K: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[GlassDetection]], np.ndarray, Dict]:
        """
        Reconstruct 3D glass from a single RGB image.

        Returns
        -------
        glasses : list[GlassDetection | None]
        depth   : (H,W) depth map (scaled to real-world meters)
        info    : dict with plane_normal, plane_d, K, table_height, X_World_Camera
        """
        # 1. Monocular depth (relative, arbitrary scale)
        depth_rel = self._depth.estimate(image_bgr)

        # 2. Camera intrinsics
        if K is None:
            K = estimate_intrinsics(*image_bgr.shape[:2])

        # 3. Table plane via RANSAC (in relative depth units)
        plane_n, plane_d_rel = ransac_table_plane(depth_rel, K, keypoints)

        # 4. Scale relative depth to real-world meters.
        # DepthAnythingV2 only gives depth up to an unknown scale factor.
        # Assume a typical table height of ~0.75 m and scale accordingly.
        ASSUMED_TABLE_HEIGHT = 0.75  # meters
        scale = ASSUMED_TABLE_HEIGHT / max(abs(plane_d_rel), 1e-6)
        depth = depth_rel * scale
        plane_d = plane_d_rel * scale

        # 5. World transform (now in real-world meters)
        X_wc, table_h = build_world_transform(plane_n, plane_d)

        # 6. SPILL cylinder estimation
        from .reconstruct import reconstruct_glass_3d
        raw = reconstruct_glass_3d(keypoints, K, table_h, X_wc)
        glasses = raw if isinstance(raw, list) else [raw]

        info = dict(
            plane_normal=plane_n,
            plane_d=plane_d,
            K=K,
            table_height=table_h,
            X_World_Camera=X_wc,
        )
        return glasses, depth, info


# ── Depth colormap ────────────────────────────────────────────────

def depth_to_colormap(depth: np.ndarray, cmap: str = 'viridis') -> np.ndarray:
    """Return (H,W,3) uint8 colored depth map."""
    d = depth.ravel()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    normed = (depth - d_min) / (d_max - d_min)
    cm = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cm


def depth_overlay(image_rgb: np.ndarray, depth: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend depth colormap over the original RGB image."""
    colored = depth_to_colormap(depth)
    img = image_rgb.astype(np.float32)
    dep = colored.astype(np.float32)
    blended = img * (1 - alpha) + dep * alpha
    return blended.clip(0, 255).astype(np.uint8)


# ── 3D Plotly visualization ───────────────────────────────────────

_KP_COLORS = {
    'bottom_front': '#FF4444',
    'top_front': '#44CC44',
    'top_left': '#4488FF',
    'top_right': '#CC44CC',
    'fluid_level': '#FFCC44',
}

_Glass_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE']


def cylinder_mesh(
    bottom_center: np.ndarray,
    top_center: np.ndarray,
    radius: float,
    up: np.ndarray,          # world up = away from table, in camera frame
    n_segments: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Build triangle mesh for a cylinder (wireframe surface)."""
    side = np.cross(up, [0, 0, 1]) if abs(up[2]) < 0.9 else np.cross(up, [1, 0, 0])
    side /= np.linalg.norm(side)
    fwd = np.cross(side, up)
    fwd /= np.linalg.norm(fwd)

    theta = np.linspace(0, 2 * np.pi, n_segments + 1)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    bx = bottom_center[0] + radius * (cos_t * side[0] + sin_t * fwd[0])
    by = bottom_center[1] + radius * (cos_t * side[1] + sin_t * fwd[1])
    bz = bottom_center[2] + radius * (cos_t * side[2] + sin_t * fwd[2])

    tx = top_center[0]     + radius * (cos_t * side[0] + sin_t * fwd[0])
    ty = top_center[1]     + radius * (cos_t * side[1] + sin_t * fwd[1])
    tz = top_center[2]     + radius * (cos_t * side[2] + sin_t * fwd[2])

    # Build triangles (side walls only, no caps)
    vx, vy, vz = [], [], []
    ti, tj, tk = [], [], []
    for i in range(n_segments):
        # Quad → 2 triangles
        idx = len(vx)
        vx.extend([bx[i], bx[i + 1], tx[i], tx[i + 1]])
        vy.extend([by[i], by[i + 1], ty[i], ty[i + 1]])
        vz.extend([bz[i], bz[i + 1], tz[i], tz[i + 1]])
        ti.extend([idx, idx + 1, idx + 2, idx + 1, idx + 3, idx + 2])
        tj.extend([idx + 1, idx + 3, idx + 2, idx + 3, idx + 2, idx + 2])
        tk.extend([idx + 2, idx + 2, idx + 2, idx + 2, idx + 2, idx + 2])

    return np.array(vx), np.array(vy), np.array(vz), ti, tj, tk


def create_3d_figure(
    glasses: List[Optional[GlassDetection]],
    info: Dict,
    image_shape: tuple,
) -> go.Figure:
    """Interactive 3D Plotly figure with table plane + glass cylinders."""
    fig = go.Figure()
    h, w = image_shape[:2]
    n = info['plane_normal']
    d = info['plane_d']

    # ── Table plane ──
    extent = max(0.25, abs(d) * 1.8)
    grid = np.linspace(-extent, extent, 6)
    gx, gy = np.meshgrid(grid, grid)
    gz = -(n[0] * gx + n[1] * gy + d) / n[2]

    gx_f, gy_f, gz_f = gx.ravel(), gy.ravel(), gz.ravel()
    ng = len(grid)
    pi, pj, pk = [], [], []
    for i in range(ng - 1):
        for j in range(ng - 1):
            a = i * ng + j
            pi.extend([a, a + ng, a + 1, a + ng, a + ng + 1, a + 1])
            pj.extend([a + 1, a + ng, a + 1, a + ng + 1, a + ng + 1, a + 1])
            pk.extend([0, 0, 0, 0, 0, 0])

    fig.add_trace(go.Mesh3d(
        x=gx_f, y=gy_f, z=gz_f,
        i=pi, j=pj, k=pk,
        opacity=0.10, color='#888888', flatshading=True,
        name='Table plane', showlegend=True,
    ))

    # ── Cylinder per glass ──
    up_cam = -n  # "up" away from table
    for idx, g in enumerate(glasses):
        if g is None:
            continue
        color = _Glass_COLORS[idx % len(_Glass_COLORS)]
        r = g.radius
        top_c = g.center_3d
        bot_c = top_c + g.height * n  # bottom = top toward table

        cx, cy, cz, ci, cj, ck = cylinder_mesh(bot_c, top_c, r, up_cam)
        fig.add_trace(go.Mesh3d(
            x=cx, y=cy, z=cz,
            i=ci, j=cj, k=ck,
            opacity=0.35, color=color, flatshading=True,
            name=f'Glass #{idx + 1}',
        ))

        # ── 3D keypoints ──
        side = np.cross(up_cam, [0, 0, 1]) if abs(up_cam[2]) < 0.9 else np.cross(up_cam, [1, 0, 0])
        side /= np.linalg.norm(side)
        fwd = np.cross(side, up_cam)
        fwd /= np.linalg.norm(fwd)

        kps_3d = [
            ('bottom_front', bot_c - r * fwd),
            ('top_front',    top_c - r * fwd),
            ('top_left',     top_c - r * side),
            ('top_right',    top_c + r * side),
        ]
        for name, pt in kps_3d:
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode='markers',
                marker=dict(size=5, color=_KP_COLORS[name]),
                name=name if idx == 0 else None,
            ))

    # ── Camera marker ──
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text=['📷 Camera'],
        textposition='top center',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Camera',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        template='plotly_white',
    )
    return fig


# ── Info text builder ─────────────────────────────────────────────

def build_3d_info(
    glasses: List[Optional[GlassDetection]],
    keypoints: List[GlassKeypoints],
    info: Dict,
) -> str:
    """Human-readable reconstruction summary (CriticBarista style)."""
    lines = [f"Found {len(keypoints)} glass(es) — 3D reconstruction from single RGB image:\n"]

    for i, g in enumerate(glasses):
        lines.append(f"\n--- Glass #{i + 1} ---")
        if g is None:
            lines.append("  3D reconstruction failed (check image / try closer view)")
            continue
        lines.append(f"  Radius:       {g.radius * 1000:.1f} mm")
        lines.append(f"  Height:       {g.height * 1000:.1f} mm")
        lines.append(f"  Tilt angle:   {np.rad2deg(g.glass_angle):.1f} deg")
        lines.append(f"  Fluid level:  {g.fluid_percentage * 100:.0f}%")
        lines.append(f"  Reprojection: {g.reprojection_error:.1f} px")
        lines.append(f"  Center 3D:    ({g.center_3d[0]:.3f}, {g.center_3d[1]:.3f}, {g.center_3d[2]:.3f}) m")

    lines.append(f"\nTable plane:  n=({info['plane_normal'][0]:.3f}, {info['plane_normal'][1]:.3f}, {info['plane_normal'][2]:.3f})")
    lines.append(f"Table dist:   {info['plane_d']:.3f} m (camera frame)")
    lines.append(f"Camera K:     f={info['K'][0, 0]:.0f}, c=({info['K'][0, 2]:.0f}, {info['K'][1, 2]:.0f})")

    return '\n'.join(lines)
