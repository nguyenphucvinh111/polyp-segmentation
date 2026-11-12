"""Ablatable PID-UNet with bounded gates and global context."""
import math
import torch
from torch import nn
from models import register_model
from models.pid_unet import EnhancedTinyPIDUNet, SoftPIDBAG

class BoundedPIDBlock(SoftPIDBAG):
    def forward(self, p_feat, i_feat, boundary_map):
        import torch.nn.functional as F
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode="bilinear", align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        gate = .6 * self.alpha.sigmoid()
        gate = gate + (1 - gate) * sal
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        kp, ki, kd = self.pid_gate(i_feat)
        return self.fuse(self.se(kp*p*gate + ki*i + kd*d)) + i

class GlobalContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 24, 1), nn.ReLU(), nn.Conv2d(24, channels, 1))
    def forward(self, x):
        return x + self.net(x)

@register_model("pid_unet_v2")
class PIDUNetV2(EnhancedTinyPIDUNet):
    def __init__(self, bounded_gates=True, global_context=True):
        super().__init__()
        if bounded_gates:
            for name in ("bag1", "bag2", "bag3", "bag4"):
                block = getattr(self, name)
                replacement = BoundedPIDBlock(block.proj_p[0].in_channels,
                                               block.proj_i[0].in_channels,
                                               block.proj_p[0].out_channels)
                replacement.load_state_dict(block.state_dict())
                with torch.no_grad():
                    replacement.alpha.fill_(math.log(.35 / .25))
                setattr(self, name, replacement)
        if global_context:
            self.bottleneck = nn.Sequential(self.bottleneck, GlobalContext(192))
        assert sum(p.numel() for p in self.parameters()) < 1_000_000
