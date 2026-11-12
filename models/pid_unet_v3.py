"""PID-UNet v3: uncertainty-guided residual refinement of v2 logits.

Experimental, single-model inference; no accuracy superiority is assumed.
The correction can add or remove foreground, with a nonzero attention floor.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models import register_model
from models.pid_unet_v2 import PIDUNetV2

@register_model("pid_unet_v3")
class PIDUNetV3(PIDUNetV2):
    def __init__(self, refinement=True):
        super().__init__()
        self.refinement = refinement
        # Local RGB detail plus coarse prediction, confidence, and boundary.
        self.detail = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1))
        # Start with a small learned correction rather than changing v2 abruptly.
        nn.init.normal_(self.detail[-1].weight, std=.001)
        nn.init.zeros_(self.detail[-1].bias)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000

    def forward(self, x, return_aux=False):
        outputs, boundary = super().forward(x, return_aux=True)
        if self.refinement:
            coarse = outputs[-1]
            probability = coarse.sigmoid()
            uncertainty = 4 * probability * (1 - probability)
            edge = F.interpolate(boundary, size=x.shape[-2:],
                                 mode="bilinear", align_corners=False).sigmoid()
            detail = self.detail(torch.cat((x, probability, uncertainty, edge), dim=1))
            attention = .25 + .75 * torch.maximum(uncertainty, edge)
            refined = coarse + attention * detail
            # Keep the tuple contract. All decoder stages remain supervised;
            # the full-resolution head receives gradients via refined logits.
            outputs = (*outputs[:-1], refined)
        return (outputs, boundary) if return_aux else outputs
