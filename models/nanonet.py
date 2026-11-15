"""
NanoNet: Real-Time Polyp Segmentation in Video Capsule Endoscopy and Colonoscopy.

Reference:
    Jha, D., Tomar, N. K., Ali, S., Riegler, M. A., Johansen, H. D., Johansen, D., & Halvorsen, P. (2021).
    NanoNet: Real-Time Polyp Segmentation in Video Capsule Endoscopy and Colonoscopy.
    IEEE International Symposium on Computer-Based Medical Systems (CBMS 2021).
    arXiv:2104.11138
    Official repository: https://github.com/DebeshJha/NanoNet

Architectural Foundations:
    1. Efficient MobileNetV2 Backbone:
       - Uses depthwise separable convolutions and linear inverted bottlenecks.
       - Extracts multi-scale representations at strides 2, 4, 8, 16, and 32.
    2. Modified Residual Decoder with Squeeze-and-Excitation (SE):
       - Progressively aggregates high-level semantic context with shallow spatial detail.
       - Squeeze-and-Excitation channel gating dynamically recalibrates feature activations.
    3. Variants:
       - NanoNet-A: decoder channels [128, 64, 32] -> Highest representation capacity
       - NanoNet-B: decoder channels [96, 64, 32]  -> Balanced speed and accuracy
       - NanoNet-C: decoder channels [32, 24, 16]  -> Ultra-compact variant

Registered as: 'nanonet', 'nanonet_a', 'nanonet_b', 'nanonet_c'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# MobileNetV2 Building Blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=False)
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block with Depthwise Separable Convolution."""
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation & Decoder Residual Block
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel attention via Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_c = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_c, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_c, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DecoderResidualBlock(nn.Module):
    """
    Modified Residual Block with SE Channel Attention.
    Faithful to NanoNet's decoder formulation.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.se = SqueezeExcitation(out_c)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.se(self.conv(x))
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# NanoNet Architecture
# ---------------------------------------------------------------------------

class NanoNetBase(nn.Module):
    """Base NanoNet architecture parameterizable by decoder channel variants."""
    def __init__(self, num_classes: int = 1, variant: str = 'a'):
        super().__init__()
        variant = variant.lower()
        if variant == 'c':
            dec_channels = [32, 24, 16]
        elif variant == 'b':
            dec_channels = [96, 64, 32]
        else:
            dec_channels = [128, 64, 32]

        # MobileNetV2 Backbone Stages
        # Stage 1: 1/2 resolution (16 channels)
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            InvertedResidual(32, 16, stride=1, expand_ratio=1)
        )
        # Stage 2: 1/4 resolution (24 channels)
        self.stage2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6)
        )
        # Stage 3: 1/8 resolution (32 channels)
        self.stage3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6)
        )
        # Stage 4: 1/16 resolution (96 channels)
        self.stage4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
        # Stage 5 / Bridge: 1/32 resolution (320 channels)
        self.stage5 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 320, stride=1, expand_ratio=6)
        )

        # Decoder Blocks
        c5, c4, c3, c2, c1 = 320, 96, 32, 24, 16
        dc1, dc2, dc3 = dec_channels[0], dec_channels[1], dec_channels[2]

        # Dec 4: fuses stage5 (1/32) and stage4 (1/16)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DecoderResidualBlock(c5 + c4, dc1)

        # Dec 3: fuses dec4 (1/16) and stage3 (1/8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DecoderResidualBlock(dc1 + c3, dc2)

        # Dec 2: fuses dec3 (1/8) and stage2 (1/4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DecoderResidualBlock(dc2 + c2, dc3)

        # Dec 1: fuses dec2 (1/4) and stage1 (1/2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DecoderResidualBlock(dc3 + c1, dc3)

        # Final prediction head
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(dc3, num_classes, kernel_size=1)

        # Deep supervision heads during training
        self.head4 = nn.Conv2d(dc1, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(dc2, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(dc3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        orig_sz = x.shape[2:]

        # Encoder Path
        s1 = self.stage1(x)       # (B, 16, 176, 176)
        s2 = self.stage2(s1)      # (B, 24, 88, 88)
        s3 = self.stage3(s2)      # (B, 32, 44, 44)
        s4 = self.stage4(s3)      # (B, 96, 22, 22)
        s5 = self.stage5(s4)      # (B, 320, 11, 11)

        # Decoder Path
        d4 = self.dec4(torch.cat([self.up4(s5), s4], dim=1)) # 22x22
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1)) # 44x44
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1)) # 88x88
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1)) # 176x176

        out = self.final_conv(self.final_up(d1))              # 352x352

        if self.training:
            # 4-scale deep supervision (outputs[-1] is out, evaluated by benchmark)
            p4 = F.interpolate(self.head4(d4), size=orig_sz, mode='bilinear', align_corners=True)
            p3 = F.interpolate(self.head3(d3), size=orig_sz, mode='bilinear', align_corners=True)
            p2 = F.interpolate(self.head2(d2), size=orig_sz, mode='bilinear', align_corners=True)
            return p4, p3, p2, out
        return out


@register_model('nanonet')
@register_model('nanonet_a')
class NanoNetA(NanoNetBase):
    def __init__(self, num_classes: int = 1):
        super().__init__(num_classes=num_classes, variant='a')


@register_model('nanonet_b')
class NanoNetB(NanoNetBase):
    def __init__(self, num_classes: int = 1):
        super().__init__(num_classes=num_classes, variant='b')


@register_model('nanonet_c')
class NanoNetC(NanoNetBase):
    def __init__(self, num_classes: int = 1):
        super().__init__(num_classes=num_classes, variant='c')
