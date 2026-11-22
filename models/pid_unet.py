"""PIDUnet: exact frozen hardgate construction, including RNG consumption.
Copy this file into models/ as pid_unet.py. Training-name mapping
in _pid_family.py and train.py must also include pid_unet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) Channel Recalibration Module
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for dynamic channel-wise feature recalibration.
    Suppresses normal mucosal background channels while boosting lesion-sensitive responses.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ---------------------------------------------------------------------------
# Asymmetric Factorized Convolution (3x1 followed by 1x3)
# ---------------------------------------------------------------------------

class AsymmetricConv(nn.Module):
    """
    Factorized 3x1 and 1x3 convolution with dilation.
    Reduces parameter count by 33% compared to standard 3x3 while doubling activation depth.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


# ---------------------------------------------------------------------------
# Information-Preserving Downsampler (Dual-Path Conv + MaxPool Concat)
# ---------------------------------------------------------------------------

class InfoPreservingDownsample(nn.Module):
    """
    Prevents loss of diminutive polyps during resolution reduction.
    Combines learned strided convolution with localized max pooling.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c_conv = out_channels // 2
        c_pool = out_channels - c_conv

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, c_conv, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_conv)
        )
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, c_pool, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_pool)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv_path(x), self.pool_path(x)], dim=1)
        return self.act(out)


# ---------------------------------------------------------------------------
# Multi-Scale Asymmetric Receptive Field Block with Optional Global Context
# ---------------------------------------------------------------------------

class AsymmetricRFB(nn.Module):
    """
    Parallel multi-scale dilated asymmetric convolutions with Squeeze-and-Excitation,
    optional global context pooling, optional spatial dropout, and residual shortcut.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: tuple = (1, 2, 4),
        dropout_p: float = 0.0,
        use_global: bool = False
    ):
        super().__init__()
        self.use_global = use_global
        num_branches = len(dilations) + (1 if use_global else 0)
        branch_c = out_channels // num_branches
        rem_c = out_channels - branch_c * (num_branches - 1)

        self.branches = nn.ModuleList()
        for idx, d in enumerate(dilations):
            bc = rem_c if (idx == num_branches - 1 and not use_global) else branch_c
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, bc, kernel_size=1, bias=False),
                nn.BatchNorm2d(bc),
                nn.ReLU(inplace=True),
                AsymmetricConv(bc, bc, dilation=d)
            ))

        if use_global:
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, rem_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(rem_c),
                nn.ReLU(inplace=True)
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_channels)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else nn.Identity()
        )
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        if self.use_global:
            g = F.interpolate(self.global_branch(x), size=x.shape[2:], mode='bilinear', align_corners=True)
            feats.append(g)
        out = self.fuse(torch.cat(feats, dim=1)) + self.shortcut(x)
        return self.drop(out)


# ---------------------------------------------------------------------------
# Dynamic Self-Tuning PID Controller Module
# ---------------------------------------------------------------------------

class AdaptivePIDGating(nn.Module):
    """
    Dynamic Self-Tuning PID Controller Module:
    Learns dynamic gains (Kp, Ki, Kd) based on the image's global feature context.
    - Diminutive / hard lesions (ETIS): Automatically boosts Kd (boundary) and Kp (fine spatial detail).
    - Large / obvious lesions: Boosts Ki (integral global semantics) for uniform coverage.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3, bias=False),
            nn.Softmax(dim=1)  # Kp + Ki + Kd = 1.0
        )

    def forward(self, context_feat: torch.Tensor):
        b = context_feat.size(0)
        flat = self.pool(context_feat).view(b, -1)
        gains = self.fc(flat)
        kp = gains[:, 0].view(b, 1, 1, 1)
        ki = gains[:, 1].view(b, 1, 1, 1)
        kd = gains[:, 2].view(b, 1, 1, 1)
        return kp, ki, kd


# ---------------------------------------------------------------------------
# Soft Semantic PID Attention Block (SoftPIDBAG v5)
# ---------------------------------------------------------------------------

class SoftPIDBAG(nn.Module):
    """
    Soft Semantic PID Attention Guidance Block (Flagship v5):
    1. Proportional (P): Soft-gated skip features with learnable baseline alpha:
       `p_gated = p_proj * (alpha + (1-alpha)*sal_mask)`
    2. Derivative (D): High-frequency discrete Laplacian (|P - AvgPool(P)|) modulated
       by boundary attention AND soft gate to prevent silencing tiny lesions in ETIS.
    3. Integral (I): Accumulated multi-scale context from the deeper decoder stage.
    4. Adaptive PID Controller: Allocates frame-specific weights (Kp, Ki, Kd).
    5. SE Channel Recalibration & Residual Shortcut (+ i_proj).
    """
    def __init__(self, p_channels: int, i_channels: int, out_channels: int):
        super().__init__()
        self.proj_p = nn.Sequential(
            nn.Conv2d(p_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_i = nn.Sequential(
            nn.Conv2d(i_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Saliency gate from I context: filters out false-alarm mucosal folds
        mid_c = max(16, out_channels // 2)
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(i_channels, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Learnable soft gate floor: init 0.35 (preserves signal in flat/diminutive lesions)
        self.alpha = nn.Parameter(torch.tensor(0.35))
        # High-pass difference operator for Derivative (D)
        self.diff_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pid_gate = AdaptivePIDGating(i_channels)
        self.se = SqueezeExcitation(out_channels)
        self.fuse = AsymmetricConv(out_channels, out_channels, dilation=1)

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        h, w = p_feat.shape[2:]
        if i_feat.shape[2:] != (h, w):
            i_feat = F.interpolate(i_feat, size=(h, w), mode='bilinear', align_corners=True)
        if boundary_map.shape[2:] != (h, w):
            boundary_map = F.interpolate(boundary_map, size=(h, w), mode='bilinear', align_corners=True)

        sal_mask = self.saliency_gate(i_feat)
        p_proj = self.proj_p(p_feat)
        i_proj = self.proj_i(i_feat)

        # 1. P Component: Soft gate — retains min `alpha` baseline even in ambiguous background
        alpha = torch.clamp(self.alpha, 0.0, 0.6)
        soft_gate = alpha + (1.0 - alpha) * sal_mask
        p_gated = p_proj * soft_gate

        # 2. D Component: Discrete Laplacian difference modulated by boundary attention & soft gate
        p_diff = torch.abs(p_proj - self.diff_pool(p_proj))
        edge_att = torch.sigmoid(boundary_map)
        d_comp = p_diff * edge_att * soft_gate

        # 3. Dynamic PID Controller Weighting
        kp, ki, kd = self.pid_gate(i_feat)
        f_pid = kp * p_gated + ki * i_proj + kd * d_comp

        # 4. Channel recalibration, asymmetric convolution, and residual shortcut
        fused = self.fuse(self.se(f_pid))
        return fused + i_proj


# ---------------------------------------------------------------------------
# 5-Scale Semantic Boundary Branch (e1, e2, e3, e4, i_feat)
# ---------------------------------------------------------------------------

class SemanticBoundaryBranch(nn.Module):
    """
    5-Scale Derivative Boundary Branch with Semantic Filtering:
    Combines multi-scale encoder cues (e1, e2, e3, e4) with deep semantic guidance (i_feat).
    Ensures that boundary detection fires at true lesion edges, NOT normal mucosal folds.
    """
    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int, d_channels: int = 24):
        super().__init__()
        self.r1 = nn.Conv2d(c1, d_channels, kernel_size=1, bias=False)
        self.r2 = nn.Conv2d(c2, d_channels, kernel_size=1, bias=False)
        self.r3 = nn.Conv2d(c3, d_channels, kernel_size=1, bias=False)
        self.r4 = nn.Conv2d(c4, d_channels, kernel_size=1, bias=False)
        self.r_sem = nn.Conv2d(c5, d_channels, kernel_size=1, bias=False)

        self.fuse = nn.Sequential(
            AsymmetricConv(d_channels * 5, d_channels, dilation=1),
            nn.Conv2d(d_channels, 1, kernel_size=1)  # 1-channel boundary logits
        )

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
        i_feat: torch.Tensor
    ) -> torch.Tensor:
        target_size = e2.shape[2:]  # H/2, W/2
        f1 = F.interpolate(self.r1(e1), size=target_size, mode='bilinear', align_corners=True)
        f2 = self.r2(e2)
        f3 = F.interpolate(self.r3(e3), size=target_size, mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.r4(e4), size=target_size, mode='bilinear', align_corners=True)
        f_sem = F.interpolate(self.r_sem(i_feat), size=target_size, mode='bilinear', align_corners=True)

        return self.fuse(torch.cat([f1, f2, f3, f4, f_sem], dim=1))


# ---------------------------------------------------------------------------
# Full Network: Enhanced Tiny PID-UNet (< 1M Parameters, Flagship Version 5)
# ---------------------------------------------------------------------------

class EnhancedTinyPIDUNet(nn.Module):
    """
    Enhanced Tiny PID-UNet for High-Precision Polyp Segmentation (< 1M Parameters).
    Flagship Version 5:
    - SoftPIDBAG v5: Soft-gated derivative prevents silencing diminutive lesions in ETIS.
    - Hybrid RFB Bottleneck: Multi-scale dilations (1, 2, 4, 8) + Global Context Pooling.
    - 5-Scale Boundary Branch: Multi-resolution lesion boundary detection.
    Total Parameters: ~0.805M (< 1.0M).
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 32):
        super().__init__()
        c1 = base_c          # 32
        c2 = base_c * 2      # 64
        c3 = base_c * 3      # 96
        c4 = base_c * 4      # 128
        c5 = base_c * 6      # 192

        # ----------------- Stem & Encoder with SE-Asymmetric RFB -----------------
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.enc1 = AsymmetricRFB(c1, c1, dilations=(1, 2))
        self.down1 = InfoPreservingDownsample(c1, c2)

        self.enc2 = AsymmetricRFB(c2, c2, dilations=(1, 2, 4))
        self.down2 = InfoPreservingDownsample(c2, c3)

        self.enc3 = AsymmetricRFB(c3, c3, dilations=(1, 2, 4))
        self.down3 = InfoPreservingDownsample(c3, c4)

        self.enc4 = AsymmetricRFB(c4, c4, dilations=(1, 2, 4))
        self.down4 = InfoPreservingDownsample(c4, c5)

        # ----------------- Hybrid Bottleneck (Dilations 1, 2, 4, 8 + Global Context) ------
        self.bottleneck = AsymmetricRFB(c5, c5, dilations=(1, 2, 4, 8), dropout_p=0.05, use_global=True)

        # ----------------- 5-Scale Derivative Boundary Stream (e1, e2, e3, e4, i_feat) -----
        self.boundary_branch = SemanticBoundaryBranch(c1, c2, c3, c4, c5, d_channels=24)

        # ----------------- Decoder with Soft Semantic PID Attention (SoftPIDBAG v5) --------
        self.bag4 = SoftPIDBAG(p_channels=c4, i_channels=c5, out_channels=c4)
        self.bag3 = SoftPIDBAG(p_channels=c3, i_channels=c4, out_channels=c3)
        self.bag2 = SoftPIDBAG(p_channels=c2, i_channels=c3, out_channels=c2)
        self.bag1 = SoftPIDBAG(p_channels=c1, i_channels=c2, out_channels=c1)

        # ----------------- Deep Supervision Heads ------------------------------------------
        self.head4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_aux=False):
        sz = x.shape[2:]

        # 1. Multi-scale asymmetric encoder forward
        x0 = self.inc(x)
        e1 = self.enc1(x0)             # (B, 32, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 64, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 96, H/4, W/4)
        e4 = self.enc4(self.down3(e3)) # (B, 128, H/8, W/8)

        # 2. Hybrid Context Bottleneck (Dilations 1, 2, 4, 8 + Global Image Pooling)
        i_feat = self.bottleneck(self.down4(e4)) # (B, 192, H/16, W/16)

        # 3. 5-Scale Semantic Boundary Map (e1, e2, e3, e4, i_feat)
        b_map = self.boundary_branch(e1, e2, e3, e4, i_feat) # (B, 1, H/2, W/2)

        # 4. Decoder with Soft Semantic PID Attention
        d4 = self.bag4(e4, i_feat, b_map)  # (B, 128, H/8, W/8)
        d3 = self.bag3(e3, d4, b_map)      # (B, 96, H/4, W/4)
        d2 = self.bag2(e2, d3, b_map)      # (B, 64, H/2, W/2)
        d1 = self.bag1(e1, d2, b_map)      # (B, 32, H, W)

        # 5. Multi-Scale Deep Supervision (PraNet 4-scale convention)
        out4 = F.interpolate(self.head4(d4), size=sz, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head3(d3), size=sz, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), size=sz, mode='bilinear', align_corners=True)
        out1 = self.head1(d1)  # already (H, W)

        return ((out4, out3, out2, out1), b_map) if return_aux else (out4, out3, out2, out1)


class BoundedPIDBlock(SoftPIDBAG):
    def forward(self, p_feat, i_feat, boundary_map):
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


# Only the public name changes; constructor and forward are inherited unchanged.
try:
    from models import register_model
except ImportError:
    def register_model(name):
        return lambda cls: cls

@register_model("pid_unet")
@register_model("pid_unet_v2_hardgate")
class PIDUnet(PIDUNetV2_HardGate):
    pass

PIDUNet = PIDUnet

# Compatibility with the previous standalone import name.
PIDUNet_HardGate = PIDUnet
