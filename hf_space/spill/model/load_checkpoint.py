"""Load keypoint detector checkpoint.

Adapted from keypoint_detection.utils.load_checkpoints.
"""
import torch
from .backbone import MaxVitUnet
from .detector import KeypointDetector


def load_keypoint_detector(checkpoint_path: str, device: str = "cpu") -> KeypointDetector:
    """Load a KeypointDetector from a PyTorch Lightning checkpoint file.

    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to load model on

    Returns:
        KeypointDetector model ready for inference
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create backbone
    backbone = MaxVitUnet()

    # Create detector
    n_heatmaps = len(checkpoint["hyper_parameters"]["keypoint_channel_configuration"])
    n_channels_out = backbone.get_n_channels_out()
    model = KeypointDetector(
        backbone=backbone,
        n_heatmaps=n_heatmaps,
        n_channels_out=n_channels_out,
    )

    # Load state dict - need to map from LightningModule keys
    # The checkpoint state_dict has 'unnormalized_model.' prefix
    state_dict = {k: v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model
