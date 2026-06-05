"""Keypoint detector model - inference only.

Adapted from the keypoint_detection package (KeypointDetector LightningModule).
"""
import torch
import torch.nn as nn


class KeypointDetector(nn.Module):
    """Keypoint detector using spatial heatmaps.

    Takes RGB images and outputs heatmap predictions for each keypoint channel.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_heatmaps: int,
        n_channels_out: int,
    ):
        """
        Args:
            backbone: Feature extractor backbone
            n_heatmaps: Number of keypoint channels to predict
            n_channels_out: Number of output channels from backbone
        """
        super().__init__()

        head = nn.Conv2d(
            in_channels=n_channels_out,
            out_channels=n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )
        head.bias.data.fill_(-4)

        self.unnormalized_model = nn.Sequential(
            backbone,
            head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, H, W) with values in [0, 1]

        Returns:
            Heatmap tensor of shape (N, n_heatmaps, H, W) with sigmoid activation
        """
        return torch.sigmoid(self.unnormalized_model(x))
