"""PID-UNet v5: spatial PID allocation on the unchanged v2 backbone.

No new loss, RGB correction head, threshold or test-specific inference.
"""
import torch
from torch import nn
from models import register_model
from models.pid_unet_v2 import PIDUNetV2

class SpatialPIDGating(nn.Module):
    """Global PID prior plus bounded, position-dependent logit corrections."""
    def __init__(self, global_controller, channels):
        super().__init__()
        self.global_controller = global_controller
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1))
        # Preserve the global controller at initialization.
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def forward(self, context):
        prior = torch.cat(self.global_controller(context), dim=1)
        logits = prior.clamp_min(1e-6).log() + self.spatial(context).tanh()
        gains = logits.softmax(dim=1)
        return gains[:, :1], gains[:, 1:2], gains[:, 2:3]

@register_model("pid_unet_v5")
class PIDUNetV5(PIDUNetV2):
    def __init__(self, spatial_pid=True):
        super().__init__()
        if spatial_pid:
            for name in ('bag1', 'bag2', 'bag3', 'bag4'):
                block = getattr(self, name)
                block.pid_gate = SpatialPIDGating(
                    block.pid_gate, block.proj_i[0].in_channels)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000
