"""Feature-context refinement of v2. Experimental; validate before test."""
import torch
from torch import nn
from models import register_model
from models.pid_unet_v2 import PIDUNetV2

class ContextRefinementHead(nn.Module):
    """Signed bounded correction from local and surrounding decoder features."""
    def __init__(self, coarse_head, channels=32):
        super().__init__()
        self.coarse = coarse_head
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.correction = nn.Sequential(
            nn.Conv2d(channels * 3, 24, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.Conv2d(24, 1, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, features):
        coarse = self.coarse(features)
        correction = self.correction(torch.cat(
            (features, self.local(features), self.context(features)), dim=1))
        # Limit logit correction so a texture response cannot dominate coarse semantics.
        return coarse + correction.tanh()

@register_model("pid_unet_v4")
class PIDUNetV4(PIDUNetV2):
    def __init__(self):
        super().__init__()
        self.head1 = ContextRefinementHead(self.head1)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000
