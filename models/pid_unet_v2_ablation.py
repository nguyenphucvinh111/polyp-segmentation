"""Controlled functional ablations; original pid_unet_v2 is untouched.

All variants first initialize the complete v2. Surviving weights AND the CPU
RNG state after construction match the control for a given seed. Boundary
supervision and four segmentation heads are retained in every variant.
"""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pid_v2_ablation import MODEL_NAMES, VARIANTS
from models import register_model
from models.pid_unet_v2 import BoundedPIDBlock, PIDUNetV2


class UniformPIDGains(nn.Module):
    def forward(self, context):
        gain = context.new_full((context.shape[0], 1, 1, 1), 1 / 3)
        return gain, gain, gain


class AblationPIDBlock(BoundedPIDBlock):
    def forward(self, p_feat, i_feat, boundary_map):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode='bilinear', align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode='bilinear', align_corners=True)
        sal = self.saliency_gate(i_feat)
        p, i = self.proj_p(p_feat), self.proj_i(i_feat)
        floor = 0.0 if self.variant == 'no_gate_floor' else .6 * self.alpha.sigmoid()
        gate = floor + (1 - floor) * sal
        if self.variant == 'no_derivative':
            d = torch.zeros_like(p)
        else:
            boundary_gate = 1.0 if self.variant == 'no_boundary_gate' else boundary_map.sigmoid()
            d = (p - self.diff_pool(p)).abs() * boundary_gate * gate
        kp, ki, kd = self.pid_gate(i_feat)
        return self.fuse(self.se(kp * p * gate + ki * i + kd * d)) + i


class PIDUNetV2Ablation(PIDUNetV2):
    strict_checkpoint_loading = True

    def __init__(self, variant='full'):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown v2 ablation: {variant}')
        super().__init__()
        self.variant = variant
        self.checkpoint_architecture = f'pid_unet_v2_ablation.v1.{variant}'
        if variant == 'no_global':
            # Keep bottleneck[0] and its state_dict names identical to v2.
            self.bottleneck[1] = nn.Identity()
        elif variant != 'full':
            # Constructor draws must not alter the data/dropout RNG sequence.
            with torch.random.fork_rng(devices=[]):
                for name in ('bag1', 'bag2', 'bag3', 'bag4'):
                    original = getattr(self, name)
                    block = AblationPIDBlock(original.proj_p[0].in_channels,
                                             original.proj_i[0].in_channels,
                                             original.proj_p[0].out_channels)
                    block.load_state_dict(original.state_dict(), strict=True)
                    block.variant = variant
                    if variant == 'fixed_gains':
                        block.pid_gate = UniformPIDGains()
                    if variant == 'no_gate_floor':
                        block.register_parameter('alpha', None)
                    setattr(self, name, block)


for _variant, _model_name in MODEL_NAMES.items():
    register_model(_model_name)(partial(PIDUNetV2Ablation, variant=_variant))
