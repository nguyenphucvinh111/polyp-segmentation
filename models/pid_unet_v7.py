"""Scratch-trained PID-UNet with supervised semantic guidance and multi-scale decoding.

Retains v2's encoder, context bottleneck, boundary branch and four output heads.
Decoder ideas are related to attention-gated U-Net and multi-scale depthwise
decoding; this is not a reproduction or a claim of a novel attention primitive:
https://arxiv.org/abs/1804.03999
https://arxiv.org/abs/2405.06880
"""

import torch
from torch import nn
from torch.nn import functional as F

from models import register_model
from models.pid_unet_v2 import PIDUNetV2, BoundedPIDBlock


class MultiScaleRefinement(nn.Module):
    """Dense 3/5/7 spatial neighborhoods with pointwise channel mixing."""

    def __init__(self, channels):
        super().__init__()
        self.project_in = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel, padding=kernel // 2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ) for kernel in (3, 5, 7)
        ])
        self.project_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features):
        projected = self.project_in(features)
        combined = sum(branch(projected) for branch in self.branches) / len(self.branches)
        return features + self.project_out(combined)


class SemanticPIDBlock(BoundedPIDBlock):
    def __init__(self, p_channels, i_channels, out_channels,
                 multiscale=True, semantic_guidance=True):
        super().__init__(p_channels, i_channels, out_channels)
        self.semantic_guidance = semantic_guidance
        if multiscale:
            self.fuse = MultiScaleRefinement(out_channels)
        width = 16
        self.local_p = nn.Conv2d(out_channels, width, 1, bias=False)
        self.local_i = nn.Conv2d(out_channels, width, 1, bias=False)
        self.local_difference = nn.Conv2d(out_channels, width, 1, bias=False)
        self.local_gate = nn.Sequential(
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

    def semantic_mask(self, p, integral, prior, coarse_logits=None):
        local = self.local_p(p) + self.local_i(integral)
        local = local + self.local_difference((p - integral).abs())
        correction = self.local_gate(local)
        with torch.autocast(device_type=p.device.type, enabled=False):
            probability = prior.float()
            if coarse_logits is not None:
                coarse = F.interpolate(coarse_logits.float(), size=p.shape[2:],
                                       mode='bilinear', align_corners=True).sigmoid()
                probability = .5 * (probability + coarse)
            # Bounded local log-odds adjustment; compute in FP32 even under AMP.
            log_odds = torch.logit(probability.clamp(1e-4, 1 - 1e-4))
            mask = (log_odds + correction.float().tanh()).sigmoid()
        return mask.to(p.dtype)

    def forward(self, p_feat, i_feat, boundary_map, coarse_logits=None):
        size = p_feat.shape[2:]
        i_feat = F.interpolate(i_feat, size=size, mode='bilinear', align_corners=True)
        boundary_map = F.interpolate(boundary_map, size=size, mode='bilinear', align_corners=True)
        p, integral = self.proj_p(p_feat), self.proj_i(i_feat)
        saliency = self.saliency_gate(i_feat)
        if self.semantic_guidance:
            saliency = self.semantic_mask(p, integral, saliency, coarse_logits)
        floor = .6 * self.alpha.sigmoid()
        gate = floor + (1 - floor) * saliency
        derivative = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        kp, ki, kd = self.pid_gate(i_feat)
        features = kp * p * gate + ki * integral + kd * derivative
        return self.fuse(self.se(features)) + integral


@register_model('pid_unet_v7')
class PIDUNetV7(PIDUNetV2):
    """Independent v2-based architecture; all weights are initialized from scratch.

    Disable multiscale and semantic_guidance together for the v2-function ablation.
    Output and auxiliary boundary interfaces match v2 exactly.
    """

    strict_checkpoint_loading = True

    def __init__(self, multiscale=True, semantic_guidance=True):
        super().__init__()
        for name in ('bag1', 'bag2', 'bag3', 'bag4'):
            old = getattr(self, name)
            block = SemanticPIDBlock(old.proj_p[0].in_channels,
                                     old.proj_i[0].in_channels,
                                     old.proj_p[0].out_channels,
                                     multiscale, semantic_guidance)
            # Retain v2's initialization of the shared paths, including alpha.
            shared = {key: value for key, value in old.state_dict().items()
                      if not (multiscale and key.startswith('fuse.'))}
            incompatible = block.load_state_dict(shared, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(f'Unexpected v2 keys: {incompatible.unexpected_keys}')
            setattr(self, name, block)
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError('PID-UNet v7 exceeds the 1M parameter budget')

    def forward(self, x, return_aux=False):
        size = x.shape[2:]
        e1 = self.enc1(self.inc(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        integral = self.bottleneck(self.down4(e4))
        boundary = self.boundary_branch(e1, e2, e3, e4, integral)
        d4 = self.bag4(e4, integral, boundary)
        logit4 = self.head4(d4)
        d3 = self.bag3(e3, d4, boundary, logit4)
        logit3 = self.head3(d3)
        d2 = self.bag2(e2, d3, boundary, logit3)
        logit2 = self.head2(d2)
        d1 = self.bag1(e1, d2, boundary, logit2)
        outputs = tuple(F.interpolate(logit, size=size, mode='bilinear', align_corners=True)
                        for logit in (logit4, logit3, logit2)) + (self.head1(d1),)
        return (outputs, boundary) if return_aux else outputs
