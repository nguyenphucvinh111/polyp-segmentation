"""Hardgate implementation frozen from d667f89 for reproducible ablations."""
import torch
from torch import nn
import torch.nn.functional as F
from models import register_model
from models._hardgate_v2 import PIDUNetV2, BoundedPIDBlock

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


class PIDUNetV2_HardGate(PIDUNetV2):
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


