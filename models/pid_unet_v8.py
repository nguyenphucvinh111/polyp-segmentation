"""V2 with spatial control of the integral path at the two finest decoder stages.

All weights are initialized from scratch. The new gate is initially identity;
the existing encoder, P/D gates, fusion convolutions, heads and losses are kept.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models import register_model
from models.pid_unet_v2 import BoundedPIDBlock, PIDUNetV2


class IntegralSpatialGate(nn.Module):
    """A bounded spatial multiplier informed by skip, context and boundary features."""

    def __init__(self, channels, hidden=16, strength=.5):
        super().__init__()
        if not 0 < strength < 1:
            raise ValueError('Gate strength must be between 0 and 1')
        self.strength = strength
        # Separate projections avoid concatenating three full-resolution feature maps.
        self.skip = nn.Conv2d(channels, hidden, 1, bias=False)
        self.context = nn.Conv2d(channels, hidden, 1, bias=False)
        self.difference = nn.Conv2d(channels, hidden, 1, bias=False)
        self.boundary = nn.Conv2d(1, hidden, 1, bias=False)
        self.local = nn.Sequential(
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, p, integral, boundary_logits):
        features = self.skip(p) + self.context(integral)
        features = features + self.difference((p - integral).abs())
        features = features + self.boundary(boundary_logits.sigmoid())
        correction = self.local(features)
        with torch.autocast(device_type=p.device.type, enabled=False):
            scale = 1 + self.strength * correction.float().tanh()
        return scale.to(integral.dtype)


class IntegralControlledPIDBlock(BoundedPIDBlock):
    def __init__(self, p_channels, i_channels, out_channels):
        super().__init__(p_channels, i_channels, out_channels)
        self.integral_gate = IntegralSpatialGate(out_channels)

    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode='bilinear', align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode='bilinear', align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, integral = self.proj_p(p_feat), self.proj_i(i_feat)
        floor = .6 * self.alpha.sigmoid()
        gate = floor + (1 - floor) * sal
        derivative = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        kp, ki, kd = self.pid_gate(i_feat)

        controlled = integral * self.integral_gate(p, integral, boundary_map)
        # Apply the same control to BOTH integral routes, including the direct shortcut.
        features = kp * p * gate + ki * controlled + kd * derivative
        return self.fuse(self.se(features)) + controlled


@register_model('pid_unet_v8')
class PIDUNetV8(PIDUNetV2):
    """Independent v2 experiment for integral gating; no v7 decoder is used."""

    strict_checkpoint_loading = True

    def __init__(self, integral_gating=True):
        super().__init__()
        if integral_gating:
            for name in ('bag1', 'bag2'):
                old = getattr(self, name)
                block = IntegralControlledPIDBlock(
                    old.proj_p[0].in_channels, old.proj_i[0].in_channels,
                    old.proj_p[0].out_channels,
                )
                # These are v2's freshly initialized weights, not pretrained weights.
                missing, unexpected = block.load_state_dict(old.state_dict(), strict=False)
                if unexpected or any(not key.startswith('integral_gate.') for key in missing):
                    raise RuntimeError('Unexpected difference from the v2 decoder')
                setattr(self, name, block)
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError('PID-UNet v8 exceeds the 1M parameter budget')
