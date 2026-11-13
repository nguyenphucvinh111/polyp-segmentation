"""PID-UNet v2 with progressive foreground/background context refinement.

The v2 encoder, PID blocks, heads and boundary branch are retained. Existing
coarse segmentation heads supply soft region assignments for two context
tokens per image. No ground truth enters forward(), and no extra loss is used.

Region aggregation is inspired by Object-Contextual Representations (OCR):
https://arxiv.org/abs/1909.11065. This is a small binary, cross-scale adaptation,
not an OCR reproduction or a claim of a new attention mechanism.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models import register_model
from models.pid_unet_v2 import PIDUNetV2


class ForegroundBackgroundContext(nn.Module):
    """Refine local features using soft foreground and background prototypes.

Attention has two keys per image, not a quadratic pixel-to-pixel matrix.
The residual path stays open even when coarse predictions miss an object.
"""

    def __init__(self, channels, coarse_channels, width=24, pool_factor=1):
        super().__init__()
        self.pool_factor = pool_factor
        self.query = nn.Sequential(
            nn.Conv2d(channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
        )
        self.key_value = nn.Sequential(
            nn.Conv2d(coarse_channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
        )
        # Local appearance, object context, their difference, coarse probability
        # and foreground affinity are all available to the residual predictor.
        self.refine = nn.Sequential(
            nn.Conv2d(3 * width + 2, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, channels, 1),
        )
        # Start at the v2 function, then learn semantic corrections. Earlier
        # layers receive gradients once this final projection leaves zero.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def region_context(query, source, coarse_logits):
        """Return attended region features and foreground affinity in FP32."""
        # Spatial sums and normalized attention must not overflow in FP16 AMP.
        with torch.autocast(device_type=query.device.type, enabled=False):
            probability = coarse_logits.float().sigmoid()
            weights = torch.cat((probability, 1 - probability), dim=1).flatten(2)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            prototypes = torch.bmm(weights, source.float().flatten(2).transpose(1, 2))
            pixels = query.float().flatten(2).transpose(1, 2)
            similarity = torch.bmm(
                F.normalize(pixels, dim=-1, eps=1e-6),
                F.normalize(prototypes, dim=-1, eps=1e-6).transpose(1, 2),
            )
            attention = (4.0 * similarity).softmax(dim=-1)
            context = torch.bmm(attention, prototypes).transpose(1, 2)
            context = context.reshape_as(query)
            affinity = attention[..., 0].reshape(query.shape[0], 1, *query.shape[2:])
        return context, affinity

    def forward(self, features, coarse_features, coarse_logits):
        work = features
        if self.pool_factor > 1:
            work = F.avg_pool2d(work, self.pool_factor, self.pool_factor)
        query = self.query(work)
        source = self.key_value(coarse_features)
        context, affinity = self.region_context(query, source, coarse_logits)
        probability = F.interpolate(
            coarse_logits.float().sigmoid(), size=query.shape[2:],
            mode="bilinear", align_corners=True,
        )
        context = context.to(query.dtype)
        correction = self.refine(torch.cat(
            (query, context, query - context,
             probability.to(query.dtype), affinity.to(query.dtype)), dim=1,
        ))
        if correction.shape[2:] != features.shape[2:]:
            correction = F.interpolate(
                correction, size=features.shape[2:], mode="bilinear", align_corners=True,
            )
        return features + correction


@register_model("pid_unet_v2_plus")
class PIDUNetV2Plus(PIDUNetV2):
    """v2 plus semantic context at decoder scales 1/4, 1/2 and 1.

    ``context_stages=()`` disables context for an architecture ablation.
    Default output and ``return_aux=True`` have exactly the v2 interfaces.
    """

    strict_checkpoint_loading = True

    def __init__(self, context_stages=(3, 2, 1)):
        super().__init__()
        if not set(context_stages).issubset({1, 2, 3}):
            raise ValueError("context_stages must contain only 1, 2 or 3")
        self.context_stages = tuple(context_stages)
        self.context3 = ForegroundBackgroundContext(96, 128)
        self.context2 = ForegroundBackgroundContext(64, 96)
        # Evaluate the semantic residual at half resolution; keep v2's full
        # resolution detail path intact, limiting extra memory on a T4.
        self.context1 = ForegroundBackgroundContext(32, 64, pool_factor=2)
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError("PID-UNet v2 Plus exceeds the 1M parameter budget")

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
        d3 = self.bag3(e3, d4, boundary)
        if 3 in self.context_stages:
            d3 = self.context3(d3, d4, logit4)
        logit3 = self.head3(d3)
        d2 = self.bag2(e2, d3, boundary)
        if 2 in self.context_stages:
            d2 = self.context2(d2, d3, logit3)
        logit2 = self.head2(d2)
        d1 = self.bag1(e1, d2, boundary)
        if 1 in self.context_stages:
            d1 = self.context1(d1, d2, logit2)

        outputs = tuple(F.interpolate(logit, size=size, mode="bilinear", align_corners=True)
                        for logit in (logit4, logit3, logit2)) + (self.head1(d1),)
        return (outputs, boundary) if return_aux else outputs
