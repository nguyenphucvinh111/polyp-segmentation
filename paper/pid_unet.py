"""
Standalone Implementation of PID-UNet V2 Hardgate (Flagship Architecture).

Total Parameters: 814,573 (~0.815M, strictly < 1.0M)
FLOPs: 6.63 GFLOPs at 352x352
FPS: 46.93 FPS on NVIDIA Tesla T4 (Real-time >= 30 FPS)

Key Components:
1. AsymmetricRFB Encoder (factorized 3x1 and 1x3 convolutions with dilations and SE).
2. Info-Preserving Downsampler (IPD: dual-path strided Conv || MaxPool concat).
3. Hybrid Bottleneck (HARF-GC: dilations 1, 2, 4, 8 + Global Image-level Context).
4. 5-Scale Semantic Boundary Branch (SBS: e1, e2, e3, e4, i_feat -> boundary map).
5. Saliency-Gated PID Decoder Blocks (SG-PID: P, I, D synthesis with Hard Saliency Gate).
6. Dynamic PID Controller (image-level adaptive gains Kp, Ki, Kd).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models import register_model
except ImportError:
    def register_model(name: str):
        def decorator(cls):
            return cls
        return decorator


# ---------------------------------------------------------------------------
# 1. Squeeze-and-Excitation (SE) Channel Recalibration
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel-wise feature recalibration to suppress normal mucosa channels."""
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
# 2. Asymmetric Factorized Convolution (3x1 followed by 1x3)
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
# 3. Information-Preserving Downsampler (Dual-Path Conv + MaxPool Concat)
# ---------------------------------------------------------------------------

class InfoPreservingDownsample(nn.Module):
    """
    Prevents spatial annihilation of diminutive lesions (< 20px) during downsampling.
    Combines learned strided convolution with localized max-pooling.
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
# 4. Multi-Scale Asymmetric Receptive Field Block (AsymRFB)
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
# 5. Global Context Bottleneck Residual Module
# ---------------------------------------------------------------------------

class GlobalContext(nn.Module):
    """Image-level illumination context enhancement at bottleneck."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 24, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# 6. Dynamic Self-Tuning PID Controller Module
# ---------------------------------------------------------------------------

class AdaptivePIDGating(nn.Module):
    """
    Dynamically predicts per-frame gains (Kp, Ki, Kd) using global feature statistics.
    Ensures Kp + Ki + Kd = 1.0 via Softmax.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3, bias=False),
            nn.Softmax(dim=1)
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
# 7. Saliency-Gated PID Attention Block (HardGatePIDBlock)
# ---------------------------------------------------------------------------

class HardGatePIDBlock(nn.Module):
    """
    The Core Proposed Module: Saliency-Gated PID (SG-PID) Block with Hard Gate.
    - Proportional (P): p_gated = Proj_P(P) * gate
    - Derivative (D):   d = |Proj_P(P) - AvgPool(Proj_P(P))| * sigmoid(boundary) * gate
    - Integral (I):     deep context with residual highway (+ Proj_I(I))
    - Hard Saliency Gate: gate = sal = sigmoid(Conv(I)) without learned soft baseline floor.
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
        mid_c = max(16, out_channels // 2)
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(i_channels, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Kept for strict checkpoint compatibility (unused in forward)
        self.alpha = nn.Parameter(torch.tensor(0.35))
        self.diff_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pid_gate = AdaptivePIDGating(i_channels)
        self.se = SqueezeExcitation(out_channels)
        self.fuse = AsymmetricConv(out_channels, out_channels, dilation=1)

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        size = p_feat.shape[2:]
        if i_feat.shape[2:] != size:
            i_feat = F.interpolate(i_feat, size=size, mode="bilinear", align_corners=True)
        if boundary_map.shape[2:] != size:
            boundary_map = F.interpolate(boundary_map, size=size, mode="bilinear", align_corners=True)

        sal = self.saliency_gate(i_feat)
        p = self.proj_p(p_feat)
        i = self.proj_i(i_feat)

        # Hard Saliency Gate directly from predicted foreground probability
        gate = sal
        # Derivative (D): Discrete Laplacian modulated by boundary and hard saliency gate
        d = (p - self.diff_pool(p)).abs() * boundary_map.sigmoid() * gate
        # Dynamic Gain Allocation
        kp, ki, kd = self.pid_gate(i_feat)

        # Fused PID features + Residual semantic highway
        return self.fuse(self.se(kp * p * gate + ki * i + kd * d)) + i


# ---------------------------------------------------------------------------
# 8. 5-Scale Semantic Boundary Branch (SBS)
# ---------------------------------------------------------------------------

class SemanticBoundaryBranch(nn.Module):
    """
    Aggregates multi-scale encoder cues (e1, e2, e3, e4) with deep semantic context (i_feat).
    Generates auxiliary boundary probability map b.
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
            nn.Conv2d(d_channels, 1, kernel_size=1)
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
# 9. Complete Standalone Architecture: PID-UNet (Flagship Model)
# ---------------------------------------------------------------------------

@register_model("pid_unet")
@register_model("pid_unet_v2_hardgate")
class PIDUNet(nn.Module):
    """
    Flagship Architecture: PID-UNet (formerly pid_unet_v2_hardgate).
    Sub-1M parameters (814,573 params), Real-Time (46.93 FPS on Tesla T4).
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 32):
        super().__init__()
        c1 = base_c          # 32
        c2 = base_c * 2      # 64
        c3 = base_c * 3      # 96
        c4 = base_c * 4      # 128
        c5 = base_c * 6      # 192

        # ----------------- Encoder -----------------
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

        # ----------------- Bottleneck (HARF-GC) -----------------
        self.bottleneck = nn.Sequential(
            AsymmetricRFB(c5, c5, dilations=(1, 2, 4, 8), dropout_p=0.05, use_global=True),
            GlobalContext(c5)
        )

        # ----------------- 5-Scale Boundary Branch -----------------
        self.boundary_branch = SemanticBoundaryBranch(c1, c2, c3, c4, c5, d_channels=24)

        # ----------------- Decoder (SG-PID Blocks) -----------------
        self.bag4 = HardGatePIDBlock(p_channels=c4, i_channels=c5, out_channels=c4)
        self.bag3 = HardGatePIDBlock(p_channels=c3, i_channels=c4, out_channels=c3)
        self.bag2 = HardGatePIDBlock(p_channels=c2, i_channels=c3, out_channels=c2)
        self.bag1 = HardGatePIDBlock(p_channels=c1, i_channels=c2, out_channels=c1)

        # ----------------- Deep Supervision Heads -----------------
        self.head4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, n_classes, kernel_size=1)

        assert sum(p.numel() for p in self.parameters()) < 1_000_000

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        sz = x.shape[2:]

        # 1. Multi-scale asymmetric encoder forward
        x0 = self.inc(x)
        e1 = self.enc1(x0)             # (B, 32, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 64, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 96, H/4, W/4)
        e4 = self.enc4(self.down3(e3)) # (B, 128, H/8, W/8)

        # 2. Hybrid Context Bottleneck with Global Context
        i_feat = self.bottleneck(self.down4(e4)) # (B, 192, H/16, W/16)

        # 3. 5-Scale Semantic Boundary Map
        b_map = self.boundary_branch(e1, e2, e3, e4, i_feat) # (B, 1, H/2, W/2)

        # 4. Decoder with Hard Saliency-Gated PID Attention
        d4 = self.bag4(e4, i_feat, b_map)  # (B, 128, H/8, W/8)
        d3 = self.bag3(e3, d4, b_map)      # (B, 96, H/4, W/4)
        d2 = self.bag2(e2, d3, b_map)      # (B, 64, H/2, W/2)
        d1 = self.bag1(e1, d2, b_map)      # (B, 32, H, W)

        # 5. Multi-Scale Deep Supervision (PraNet convention)
        out4 = F.interpolate(self.head4(d4), size=sz, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head3(d3), size=sz, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), size=sz, mode='bilinear', align_corners=True)
        out1 = self.head1(d1)  # already (H, W)

        return ((out4, out3, out2, out1), b_map) if return_aux else (out4, out3, out2, out1)


# Backward compatibility aliases
PIDUNetV2_HardGate = PIDUNet
PIDUNet_HardGate = PIDUNet


# Quick verification when executed directly
if __name__ == "__main__":
    model = PIDUNet()
    x = torch.randn(2, 3, 352, 352)
    outs, b = model(x, return_aux=True)
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print(f"PID-UNet (pid_unet / pid_unet_v2_hardgate) Standalone Verification:")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"Outputs: {[o.shape for o in outs]}")
    print(f"Boundary map: {b.shape}")
    print("Verification SUCCESSFUL!")
    print("=" * 60)
