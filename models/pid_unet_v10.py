"""v2 with a spatial cross-scale context bridge, trained entirely from scratch.

The bridge refines encoder features BEFORE boundary prediction and PID decoding.
It keeps spatial context tokens instead of conditioning on a predicted mask.
All v2 modules, output heads and losses are retained. Zero output projections
start the added branch at identity; no checkpoint or pretrained backbone is used.

Pooled attention and spatial feed-forward layers are established ideas:
P2T https://arxiv.org/abs/2106.12011
SegFormer https://arxiv.org/abs/2105.15203
This small CNN bridge is an adaptation, not a reproduction or a new attention
primitive. Accuracy gains must be measured after training.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models import register_model
from models.pid_unet_v2 import PIDUNetV2


class SpatialAttention(nn.Module):
    """SDPA isolated so the profiler also counts both attention matrix products."""

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

    @staticmethod
    def count_macs(module, inputs, output):
        # Same multiply-accumulate convention as THOP's Conv/Linear counters.
        # Softmax/scaling are not included in this estimate.
        query, key, value = inputs
        batch, heads, queries, key_width = query.shape
        keys, value_width = key.shape[-2], value.shape[-1]
        module.total_ops += batch * heads * queries * keys * (key_width + value_width)


class PooledContextMixer(nn.Module):
    """Full spatial queries, at most 85 pooled keys, and a local convolution path.

    LayerNorm acts across channels at each pixel in the NEW branch only. It does
    not replace v2's BatchNorm or normalize away each image's spatial appearance.
    The depthwise 3x3 convolutions provide local positional information. No fixed
    image size, absolute coordinates, foreground prior or test-set rule is used.
    """

    def __init__(self, width=64, heads=4, grids=(8, 4, 2, 1)):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('width must be positive and divisible by heads')
        if not grids or any(grid < 1 for grid in grids):
            raise ValueError('grids must contain positive pooling sizes')
        self.heads = heads
        self.grids = tuple(grids)
        self.position = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.attention = SpatialAttention()
        self.project = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, groups=2 * width),
            nn.GELU(),
            nn.Conv2d(2 * width, width, 1),
        )

    def forward(self, features):
        batch, width, height, columns = features.shape
        features = features + self.position(features)
        pixels = features.flatten(2).transpose(1, 2)
        normalized = self.norm1(pixels)
        spatial = normalized.transpose(1, 2).reshape(batch, width, height, columns)
        memory = torch.cat([
            F.adaptive_avg_pool2d(spatial, (min(grid, height), min(grid, columns))).flatten(2)
            for grid in self.grids
        ], dim=2).transpose(1, 2)

        head_width = width // self.heads
        query = self.query(normalized).reshape(batch, -1, self.heads, head_width).transpose(1, 2)
        key, value = self.key_value(memory).chunk(2, dim=-1)
        key = key.reshape(batch, -1, self.heads, head_width).transpose(1, 2)
        value = value.reshape(batch, -1, self.heads, head_width).transpose(1, 2)
        context = self.attention(query, key, value).transpose(1, 2).reshape(batch, -1, width)
        pixels = pixels + self.project(context)
        spatial = self.norm2(pixels).transpose(1, 2).reshape(batch, width, height, columns)
        local = self.feed_forward(spatial)
        return pixels.transpose(1, 2).reshape(batch, width, height, columns) + local


class CrossScaleContextBridge(nn.Module):
    """Combine 1/4, 1/8 and 1/16 features, then return signed residuals to each.

    Queries stay at 1/4 resolution to preserve local detail. Global attention
    uses spatially pooled tokens, so its cost is O(HW * 85), not O((HW)^2).
    Residuals are features, not a probability multiplier: they can supply new
    evidence to every segmentation head and the existing boundary branch.
    """

    def __init__(self, channels=(96, 128, 192), width=64, depth=2):
        super().__init__()
        self.inputs = nn.ModuleList(nn.Conv2d(channels_in, width, 1) for channels_in in channels)
        self.mixers = nn.Sequential(*(PooledContextMixer(width) for _ in range(depth)))
        self.outputs = nn.ModuleList(nn.Conv2d(width, channels_out, 1) for channels_out in channels)
        for projection in self.outputs:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, e3, e4, integral):
        features = (e3, e4, integral)
        size = e3.shape[2:]
        projected = [F.interpolate(layer(feature), size=size, mode='bilinear', align_corners=True)
                     for layer, feature in zip(self.inputs, features)]
        context = self.mixers(sum(projected) / len(projected))
        # Pool before each 1x1 output projection to avoid large intermediate maps.
        return tuple(feature + layer(F.adaptive_avg_pool2d(context, feature.shape[2:]))
                     for layer, feature in zip(self.outputs, features))


@register_model('pid_unet_v10')
class PIDUNetV10(PIDUNetV2):
    """Original v2 plus one ablatable cross-scale spatial context bridge.

    ``context_bridge=False`` constructs exactly v2's parameter graph. The
    default keeps v2's initial predictions as well as its original modules.
    Train from scratch using the unchanged PID v2 family recipe.
    """

    strict_checkpoint_loading = True

    def __init__(self, context_bridge=True):
        super().__init__()
        self.context_bridge = CrossScaleContextBridge() if context_bridge else None
        if sum(parameter.numel() for parameter in self.parameters()) >= 1_000_000:
            raise ValueError('PID-UNet v10 exceeds the 1M parameter budget')

    @staticmethod
    def profiling_custom_ops():
        return {SpatialAttention: SpatialAttention.count_macs}

    def forward(self, x, return_aux=False):
        size = x.shape[2:]
        e1 = self.enc1(self.inc(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        integral = self.bottleneck(self.down4(e4))
        if self.context_bridge is not None:
            e3, e4, integral = self.context_bridge(e3, e4, integral)
        boundary = self.boundary_branch(e1, e2, e3, e4, integral)

        d4 = self.bag4(e4, integral, boundary)
        d3 = self.bag3(e3, d4, boundary)
        d2 = self.bag2(e2, d3, boundary)
        d1 = self.bag1(e1, d2, boundary)
        outputs = tuple(F.interpolate(head(feature), size=size, mode='bilinear', align_corners=True)
                        for head, feature in ((self.head4, d4), (self.head3, d3), (self.head2, d2)))
        outputs += (self.head1(d1),)
        return (outputs, boundary) if return_aux else outputs
