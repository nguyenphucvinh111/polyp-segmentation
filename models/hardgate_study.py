"""One-factor ablations of the frozen hybrid-bottleneck hardgate model."""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from models import register_model
from models.pid_unet_z_hardgate import PIDUNetV2_HardGate, HardGatePIDBlock

VARIANTS = ('full', 'soft_floor', 'no_derivative', 'no_integral', 'fixed_gains', 'no_boundary_gate', 'no_global', 'no_boundary_loss')

class StudyBlock(HardGatePIDBlock):
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode='bilinear', align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode='bilinear', align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        floor = .6 * self.alpha.sigmoid() if self.variant == 'soft_floor' else 0
        gate = floor + (1-floor)*sal
        d = (p-self.diff_pool(p)).abs()*gate
        d = d * (1 if self.variant == 'no_boundary_gate' else boundary_map.sigmoid())
        if self.variant == 'no_derivative':
            d = torch.zeros_like(d)
        if self.variant == 'no_integral':
            i = torch.zeros_like(i)
        kp, ki, kd = (1/3, 1/3, 1/3) if self.variant == 'fixed_gains' else self.pid_gate(i_feat)
        return self.fuse(self.se(kp*p*gate + ki*i + kd*d)) + i

class HardgateStudy(PIDUNetV2_HardGate):
    strict_checkpoint_loading = True
    def __init__(self, variant='full'):
        super().__init__()
        self.checkpoint_architecture = f'hardgate_study.v1.{variant}'
        if variant == 'no_global':
            # Only remove the additive v2 context; hybrid RFB pooling stays.
            self.bottleneck[1] = nn.Identity()
        elif variant not in ('full', 'no_boundary_loss'):
            with torch.random.fork_rng(devices=[]):
                for name in ('bag1','bag2','bag3','bag4'):
                    old = getattr(self, name)
                    block = StudyBlock(old.proj_p[0].in_channels, old.proj_i[0].in_channels,
                                       old.proj_p[0].out_channels)
                    block.load_state_dict(old.state_dict())
                    block.variant = variant
                    if variant == 'soft_floor':
                        with torch.no_grad():
                            block.alpha.fill_(torch.log(torch.tensor(.35/.25)))
                    setattr(self, name, block)

for variant in VARIANTS:
    register_model('pid_hardgate_ablation_'+variant)(partial(HardgateStudy, variant=variant))
