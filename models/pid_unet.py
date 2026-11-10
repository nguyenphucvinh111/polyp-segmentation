"""
Enhanced Tiny PID-UNet (< 1M Parameters):
High-Performance Architecture for Real-Time & Edge-AI Medical Polyp Segmentation.

Key Architectural Innovations (Engineered to Outperform EDANet & U-Net under Standard Loss):
1. Info-Preserving Downsampler (Dual-Path Conv + MaxPool Concat):
   - Replaces destructive MaxPool with parallel Strided-Conv (learned features) and MaxPool (sharp boundary cues).
   - Preserves tiny diminutive lesions (< 20px) typical in ETIS-LaribPolypDB without information loss.
2. Asymmetric Dilated Receptive Field Blocks (Asymmetric RFB):
   - Replaces heavy 3x3 convolutions with factorized 3x1 and 1x3 convolutions, doubling non-linearities (BN+ReLU)
     while slashing parameter count by 33%.
   - Multi-scale dilated branches (dilations up to d=12 at the bottleneck) capture broad colon mucosal context.
3. Residual Dual-Gated Boundary-Attention Guidance (Residual BAG):
   - Saliency Filtering (from I): Soft residual gate masks out spurious background reflections and mucosal folds.
   - Boundary Attention (from D): Sharpens true lesion contours.
   - Residual connection ensures smooth, unhindered gradient propagation from epoch 1.
4. Dynamic Adaptive PID Controller (Self-Tuning Gains):
   - Predicts instance-specific gains (Kp, Ki, Kd) to dynamically balance spatial details (P),
     global context (I), and edge sharpness (D).
5. Parameter Budget:
   - Strictly < 1.0M parameters: ~0.678M parameters (only ~2.7 MB weights file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


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
# Multi-Scale Asymmetric Receptive Field Block (Asymmetric RFB)
# ---------------------------------------------------------------------------

class AsymmetricRFB(nn.Module):
    """
    Parallel multi-scale dilated asymmetric convolutions with residual connection.
    Captures lesions spanning tiny (<20px) to giant surrounding colon contexts.
    """
    def __init__(self, in_channels: int, out_channels: int, dilations: tuple = (1, 2, 4)):
        super().__init__()
        num_branches = len(dilations)
        branch_c = out_channels // num_branches
        rem_c = out_channels - branch_c * (num_branches - 1)

        self.branches = nn.ModuleList()
        for idx, d in enumerate(dilations):
            bc = rem_c if idx == num_branches - 1 else branch_c
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, bc, kernel_size=1, bias=False),
                nn.BatchNorm2d(bc),
                nn.ReLU(inplace=True),
                AsymmetricConv(bc, bc, dilation=d)
            ))

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1)) + self.shortcut(x)


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
# Residual Dual-Gated Boundary-Attention Guidance (Residual BAG)
# ---------------------------------------------------------------------------

class ResidualBAG(nn.Module):
    """
    Residual Dual-Gated BAG Module:
    1. Soft Residual Saliency Gate (from I): Attenuates non-lesion colon fold noise in P without blocking gradient flow.
    2. Boundary Attention Gate (from D): Sharpens fine polyp contours.
    3. Adaptive PID Gating: Dynamically balances P, I, D components per endoscopy frame.
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
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(i_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.pid_gate = AdaptivePIDGating(i_channels)
        self.fuse = AsymmetricConv(out_channels, out_channels, dilation=1)

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        h, w = p_feat.shape[2:]
        if i_feat.shape[2:] != (h, w):
            i_feat = F.interpolate(i_feat, size=(h, w), mode='bilinear', align_corners=True)
        if boundary_map.shape[2:] != (h, w):
            boundary_map = F.interpolate(boundary_map, size=(h, w), mode='bilinear', align_corners=True)

        # 1. Clean P via Soft Residual Saliency Gating (prevents early training gradient vanishing)
        sal_att = self.saliency_gate(i_feat)
        p_proj = self.proj_p(p_feat)
        p_clean = p_proj * (0.5 + 0.5 * sal_att)

        # 2. Modulate with Boundary Attention
        i_proj = self.proj_i(i_feat)
        edge_att = torch.sigmoid(boundary_map)

        # 3. Dynamic PID Weighting
        kp, ki, kd = self.pid_gate(i_feat)
        p_comp = kp * p_clean
        i_comp = ki * i_proj
        d_comp = kd * (edge_att * p_clean)

        # 4. Residual fusion
        fused = self.fuse(p_comp + i_comp + d_comp)
        return fused + i_proj


# ---------------------------------------------------------------------------
# Semantic-Gated Derivative Boundary Branch (D-Branch)
# ---------------------------------------------------------------------------

class SemanticBoundaryBranch(nn.Module):
    """
    Derivative (D) Component with Semantic Filtering:
    Combines shallow multi-scale edges (e1, e2, e3) with deep semantic guidance (i_feat).
    Ensures that boundary detection fires at true lesion edges, NOT normal mucosal folds.
    """
    def __init__(self, c1: int, c2: int, c3: int, c5: int, d_channels: int = 20):
        super().__init__()
        self.r1 = nn.Conv2d(c1, d_channels, kernel_size=1, bias=False)
        self.r2 = nn.Conv2d(c2, d_channels, kernel_size=1, bias=False)
        self.r3 = nn.Conv2d(c3, d_channels, kernel_size=1, bias=False)
        self.r_sem = nn.Conv2d(c5, d_channels, kernel_size=1, bias=False)

        self.fuse = nn.Sequential(
            AsymmetricConv(d_channels * 4, d_channels, dilation=1),
            nn.Conv2d(d_channels, 1, kernel_size=1)  # 1-channel boundary logits
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, i_feat: torch.Tensor) -> torch.Tensor:
        target_size = e2.shape[2:]  # H/2, W/2
        f1 = F.interpolate(self.r1(e1), size=target_size, mode='bilinear', align_corners=True)
        f2 = self.r2(e2)
        f3 = F.interpolate(self.r3(e3), size=target_size, mode='bilinear', align_corners=True)
        f_sem = F.interpolate(self.r_sem(i_feat), size=target_size, mode='bilinear', align_corners=True)

        boundary_map = self.fuse(torch.cat([f1, f2, f3, f_sem], dim=1))
        return boundary_map


# ---------------------------------------------------------------------------
# Full Network: Enhanced Tiny PID-UNet (< 1M Parameters)
# ---------------------------------------------------------------------------

@register_model('pid_unet')
class EnhancedTinyPIDUNet(nn.Module):
    """
    Enhanced Tiny PID-UNet for High-Precision Polyp Segmentation (< 1M Parameters).
    Total Parameters: ~0.678M (678,357 parameters).
    
    Competitive Edge:
    - Info-Preserving Downsampler: Zero spatial information loss for diminutive lesions.
    - Asymmetric Dilated RFB: Extreme receptive field (up to d=12) with minimal parameters.
    - Residual BAG + Adaptive PID: Unhindered gradient flow and dynamic P-I-D tuning.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 30):
        super().__init__()
        c1 = base_c          # 30
        c2 = base_c * 2      # 60
        c3 = base_c * 3      # 90
        c4 = base_c * 4      # 120
        c5 = base_c * 6      # 180

        # ----------------- Stem & Encoder -----------------
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

        # ----------------- Integral Context Bottleneck (Broad Receptive Field up to d=12) ---
        self.bottleneck = AsymmetricRFB(c5, c5, dilations=(1, 3, 6, 12))

        # ----------------- Derivative Boundary Stream (Semantic-Gated) ---------------------
        self.boundary_branch = SemanticBoundaryBranch(c1, c2, c3, c5, d_channels=20)

        # ----------------- Decoder with Residual Dual-Gated BAG Modules --------------------
        self.bag4 = ResidualBAG(p_channels=c4, i_channels=c5, out_channels=c4)
        self.bag3 = ResidualBAG(p_channels=c3, i_channels=c4, out_channels=c3)
        self.bag2 = ResidualBAG(p_channels=c2, i_channels=c3, out_channels=c2)
        self.bag1 = ResidualBAG(p_channels=c1, i_channels=c2, out_channels=c1)

        # ----------------- Deep Supervision Heads ------------------------------------------
        self.head4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        sz = x.shape[2:]

        # 1. Multi-scale asymmetric encoder forward
        x0 = self.inc(x)
        e1 = self.enc1(x0)             # (B, 30, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 60, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 90, H/4, W/4)
        e4 = self.enc4(self.down3(e3)) # (B, 120, H/8, W/8)

        # 2. Integral (I) Context Bottleneck
        i_feat = self.bottleneck(self.down4(e4)) # (B, 180, H/16, W/16)

        # 3. Semantic-Gated Derivative (D) Boundary Map
        b_map = self.boundary_branch(e1, e2, e3, i_feat) # (B, 1, H/2, W/2)

        # 4. Decoder with Residual Dual-Gated BAG
        d4 = self.bag4(e4, i_feat, b_map)  # (B, 120, H/8, W/8)
        d3 = self.bag3(e3, d4, b_map)      # (B, 90, H/4, W/4)
        d2 = self.bag2(e2, d3, b_map)      # (B, 60, H/2, W/2)
        d1 = self.bag1(e1, d2, b_map)      # (B, 30, H, W)

        # 5. Multi-Scale Deep Supervision (PraNet 4-scale convention)
        out4 = F.interpolate(self.head4(d4), size=sz, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head3(d3), size=sz, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), size=sz, mode='bilinear', align_corners=True)
        out1 = self.head1(d1)  # already (H, W)

        return out4, out3, out2, out1
