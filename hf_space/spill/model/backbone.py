import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class UpSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_skip_channels_in, n_channels_out, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_skip_channels_in + n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )
        self.norm1 = nn.BatchNorm2d(n_channels_out)
        self.relu1 = nn.ReLU()

    def forward(self, x, x_skip):
        x = nn.functional.interpolate(x, scale_factor=2.0)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        return x

class MaxVitUnet(nn.Module):
    FEATURE_CONFIG = [
        {"down": 2, "channels": 64},
        {"down": 4, "channels": 64},
        {"down": 8, "channels": 128},
        {"down": 16, "channels": 256},
        {"down": 32, "channels": 512},
    ]
    MODEL_NAME = "maxvit_nano_rw_256"
    feature_layers = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = timm.create_model(self.MODEL_NAME, pretrained=True, num_classes=0)
        self.feature_extractor = create_feature_extractor(self.encoder, self.feature_layers)
        self.decoder_blocks = nn.ModuleList()
        for config_skip, config_in in zip(self.FEATURE_CONFIG, self.FEATURE_CONFIG[1:]):
            block = UpSamplingBlock(config_in["channels"], config_skip["channels"], config_skip["channels"], 3)
            self.decoder_blocks.append(block)
        self.final_conv = nn.Conv2d(
            self.FEATURE_CONFIG[0]["channels"], self.FEATURE_CONFIG[0]["channels"], 3, padding="same"
        )
        self.final_upsampling_block = UpSamplingBlock(
            self.FEATURE_CONFIG[0]["channels"], 3, self.FEATURE_CONFIG[0]["channels"], 3
        )

    def forward(self, x):
        orig_x = torch.clone(x)
        features = list(self.feature_extractor(x).values())
        x = features.pop(-1)
        for block in self.decoder_blocks[::-1]:
            x = block(x, features.pop(-1))
        x = self.final_upsampling_block(x, orig_x)
        return x

    def get_n_channels_out(self):
        return self.FEATURE_CONFIG[0]["channels"]
