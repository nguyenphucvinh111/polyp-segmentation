"""
Ablation Study Variants for PID-UNet (v2 Architecture).

Each variant isolates a specific clinical/architectural hypothesis:
1. pid_unet_v2_nod: w/o Derivative (D) branch (Kd = 0, no boundary modulation)
2. pid_unet_v2_noi: w/o Integral (I) branch (Ki = 0, only P + D)
3. pid_unet_v2_static: Static/Equal PID gains (Kp = Ki = Kd = 1/3) vs Dynamic Adaptive Controller
4. pid_unet_v2_hardgate: Hard Saliency Gate (gate = sal) vs Bounded Soft Gate
5. pid_unet_v2_stddown: Standard Strided Conv Downsampler vs Info-Preserving Downsampler
6. pid_unet_v2_nogc: w/o Bottleneck Global Context module
7. pid_unet_v2_nobound: w/o Bounded Gate (uses original SoftPIDBAG learnable alpha)

All variants strictly maintain < 1.0M parameter budget and conform to the
PID-UNet v2 training recipe (AdamW, lr 3e-4, cosine decay, boundary supervision).
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from models import register_model
from models.pid_unet_v2 import PIDUNetV2, BoundedPIDBlock
from models._hardgate_v2 import PIDUNetV2 as FrozenHardgateV2


# ===========================================================================
# 1. Component Blocks for Ablation
# ===========================================================================

class NoDPIDBlock(BoundedPIDBlock):
    """PID block without Derivative (D) branch (Kd = 0)."""
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        gate = .6 * self.alpha.sigmoid()
        gate = gate + (1 - gate) * sal
        
        # Derivative is eliminated; only P and I contribute
        kp, ki, _ = self.pid_gate(i_feat)
        k_sum = kp + ki + 1e-6
        kp, ki = kp / k_sum, ki / k_sum
        return self.fuse(self.se(kp * p * gate + ki * i)) + i


class NoIPIDBlock(BoundedPIDBlock):
    """PID block without Integral (I) branch in PID attention (Ki = 0)."""
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode="bilinear", align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        gate = .6 * self.alpha.sigmoid()
        gate = gate + (1 - gate) * sal
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        
        # Integral is eliminated in PID combination; only P and D contribute
        kp, _, kd = self.pid_gate(i_feat)
        k_sum = kp + kd + 1e-6
        kp, kd = kp / k_sum, kd / k_sum
        return self.fuse(self.se(kp * p * gate + kd * d)) + i


class StaticPIDBlock(BoundedPIDBlock):
    """PID block with Static/Equal weights (Kp = Ki = Kd = 1/3) instead of Dynamic Gating."""
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode="bilinear", align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        gate = .6 * self.alpha.sigmoid()
        gate = gate + (1 - gate) * sal
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        
        # Static equal weights
        return self.fuse(self.se((1.0 / 3.0) * p * gate + (1.0 / 3.0) * i + (1.0 / 3.0) * d)) + i


class HardGatePIDBlock(BoundedPIDBlock):
    """PID block with Hard Saliency Gating (gate = sal) without learned soft baseline floor."""
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode="bilinear", align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        
        # Hard gate directly from saliency prediction
        gate = sal
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        kp, ki, kd = self.pid_gate(i_feat)
        return self.fuse(self.se(kp * p * gate + ki * i + kd * d)) + i


class StandardDownsample(nn.Module):
    """Standard Strided Convolution Downsampler (for ablation of InfoPreservingDownsample)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ===========================================================================
# 2. Registered Ablation Models
# ===========================================================================

@register_model("pid_unet_v2_nogc")
class PIDUNetV2_NoGC(PIDUNetV2):
    """Ablation: w/o Bottleneck Global Context Module."""
    def __init__(self):
        super().__init__(bounded_gates=True, global_context=False)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_nobound")
class PIDUNetV2_NoBound(PIDUNetV2):
    """Ablation: w/o Bounded Gate (reverts to original SoftPIDBAG learnable alpha)."""
    def __init__(self):
        super().__init__(bounded_gates=False, global_context=True)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_nod")
class PIDUNetV2_NoD(PIDUNetV2):
    """Ablation: w/o Derivative (D) branch (Kd = 0)."""
    def __init__(self):
        super().__init__(bounded_gates=False, global_context=True)
        for name in ("bag1", "bag2", "bag3", "bag4"):
            block = getattr(self, name)
            rep = NoDPIDBlock(
                block.proj_p[0].in_channels,
                block.proj_i[0].in_channels,
                block.proj_p[0].out_channels
            )
            rep.load_state_dict(block.state_dict())
            with torch.no_grad():
                rep.alpha.fill_(math.log(.35 / .25))
            setattr(self, name, rep)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_noi")
class PIDUNetV2_NoI(PIDUNetV2):
    """Ablation: w/o Integral (I) branch in PID attention (Ki = 0)."""
    def __init__(self):
        super().__init__(bounded_gates=False, global_context=True)
        for name in ("bag1", "bag2", "bag3", "bag4"):
            block = getattr(self, name)
            rep = NoIPIDBlock(
                block.proj_p[0].in_channels,
                block.proj_i[0].in_channels,
                block.proj_p[0].out_channels
            )
            rep.load_state_dict(block.state_dict())
            with torch.no_grad():
                rep.alpha.fill_(math.log(.35 / .25))
            setattr(self, name, rep)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_static")
class PIDUNetV2_Static(PIDUNetV2):
    """Ablation: Static/Equal PID gains (Kp = Ki = Kd = 1/3) instead of Adaptive Controller."""
    def __init__(self):
        super().__init__(bounded_gates=False, global_context=True)
        for name in ("bag1", "bag2", "bag3", "bag4"):
            block = getattr(self, name)
            rep = StaticPIDBlock(
                block.proj_p[0].in_channels,
                block.proj_i[0].in_channels,
                block.proj_p[0].out_channels
            )
            rep.load_state_dict(block.state_dict())
            with torch.no_grad():
                rep.alpha.fill_(math.log(.35 / .25))
            setattr(self, name, rep)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_hardgate")
class PIDUNetV2_HardGate(FrozenHardgateV2):
    """Ablation: Hard Saliency Gating (gate = sal) without learned soft baseline floor."""
    def __init__(self):
        super().__init__(bounded_gates=False, global_context=True)
        for name in ("bag1", "bag2", "bag3", "bag4"):
            block = getattr(self, name)
            rep = HardGatePIDBlock(
                block.proj_p[0].in_channels,
                block.proj_i[0].in_channels,
                block.proj_p[0].out_channels
            )
            rep.load_state_dict(block.state_dict())
            setattr(self, name, rep)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000


@register_model("pid_unet_v2_stddown")
class PIDUNetV2_StdDown(PIDUNetV2):
    """Ablation: Standard Strided Conv Downsampler instead of Info-Preserving Downsampler."""
    def __init__(self):
        super().__init__(bounded_gates=True, global_context=True)
        c1, c2, c3, c4, c5 = 32, 64, 96, 128, 192
        self.down1 = StandardDownsample(c1, c2)
        self.down2 = StandardDownsample(c2, c3)
        self.down3 = StandardDownsample(c3, c4)
        self.down4 = StandardDownsample(c4, c5)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000
