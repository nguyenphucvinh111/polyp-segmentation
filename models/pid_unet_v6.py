"""v2 with information-preserving, learnable PID channel fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from models import register_model
from models.pid_unet_v2 import PIDUNetV2, BoundedPIDBlock

class ChannelFusionPIDBlock(BoundedPIDBlock):
    def __init__(self, p_channels, i_channels, out_channels):
        super().__init__(p_channels, i_channels, out_channels)
        self.mix = nn.Conv2d(3 * out_channels, out_channels, 1, bias=False)
        # Exactly reproduce weighted addition initially; learn cross-channel
        # interactions separately for each of P, I and D during training.
        with torch.no_grad():
            identity = torch.eye(out_channels).view(out_channels, out_channels, 1, 1)
            self.mix.weight.copy_(torch.cat((identity, identity, identity), dim=1))

    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode='bilinear', align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode='bilinear', align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        floor = .6 * self.alpha.sigmoid()
        gate = floor + (1 - floor) * sal
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        kp, ki, kd = self.pid_gate(i_feat)
        fused = self.mix(torch.cat((kp * p * gate, ki * i, kd * d), dim=1))
        return self.fuse(self.se(fused)) + i

@register_model('pid_unet_v6')
class PIDUNetV6(PIDUNetV2):
    def __init__(self):
        super().__init__()
        for name in ('bag1', 'bag2', 'bag3', 'bag4'):
            old = getattr(self, name)
            block = ChannelFusionPIDBlock(old.proj_p[0].in_channels,
                                          old.proj_i[0].in_channels,
                                          old.proj_p[0].out_channels)
            incompatible = block.load_state_dict(old.state_dict(), strict=False)
            assert incompatible.missing_keys == ['mix.weight']
            assert not incompatible.unexpected_keys
            setattr(self, name, block)
        assert sum(p.numel() for p in self.parameters()) < 1_000_000
