"""
DGANet: Dilated Gated Asymmetric Network for Lightweight Polyp Segmentation.

A novel lightweight architecture designed to surpass all <1M-parameter models
(ENet, EDANet, MALUNet, ESPNetv2) on 5 polyp segmentation benchmarks.

Key Innovations:
1. DGAS Block (Dilated Gated Asymmetric Separable Block):
   - Asymmetric factorized conv (3x1 + 1x3) from EDANet (33% param reduction vs 3x3)
   - Depthwise Separable inner path for further efficiency
   - Multi-scale dilated receptive field for polyps of all sizes
   - Gated GELU activation (learnable adaptive feature selection, from MALUNet DGA)
   - Squeeze-and-Excitation channel recalibration (from PID-UNet)

2. InfoDown Downsampler (from PID-UNet):
   - Dual-path: Strided Conv (learned) + MaxPool (sharp edges) → Concat
   - Prevents information loss of diminutive polyps (<20px)

3. LightBAG (Lightweight Boundary-Aware Guidance):
   - Depthwise Separable version of PID-UNet's ResidualBAG
   - Soft Saliency Gate from integral context (I)
   - Boundary Attention Gate from D-branch
   - Residual projection for unhindered gradient flow

4. Semantic Boundary Branch (D-path):
   - Multi-scale edge fusion from enc1, enc2, enc3
   - Semantic filtering via bottleneck features (reduces false positives)

Architecture:
    Stem:  3→32  (stride 2, 352→176)
    Enc1:  32→32  (DGAS, d=1,2)      + InfoDown 32→48  (176→88)
    Enc2:  48→48  (DGAS, d=1,2,4)    + InfoDown 48→80  (88→44)
    Enc3:  80→80  (DGAS, d=1,3,5)    + InfoDown 80→112 (44→22)
    BN:   112→112 (DGAS, d=1,3,6,10)
    Dec3:  LightBAG(enc3=80, ctx=112) → 80  (22→44)
    Dec2:  LightBAG(enc2=48, ctx=80)  → 48  (44→88)
    Dec1:  LightBAG(enc1=32, ctx=48)  → 32  (88→176)
    Heads: 4× deep supervision (1ch)

Parameters: ~0.72M (< 1M)
Registered as: 'dganet'

Cite:
    If used in research, please cite this paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Utility: Squeeze-and-Excitation (SE)
# ---------------------------------------------------------------------------

class SE(nn.Module):
    """Lightweight Squeeze-and-Excitation for channel recalibration."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ---------------------------------------------------------------------------
# Module 1: DGAS Block
# Dilated Gated Asymmetric Separable Block
# ---------------------------------------------------------------------------

class _AsymDWBranch(nn.Module):
    """
    One dilated asymmetric-separable branch:
      1x1 PW reduce → DW 3x1 → DW 1x3 → 1x1 PW expand
    All with BatchNorm. Output has same spatial size as input.
    """
    def __init__(self, in_c: int, branch_c: int, dilation: int = 1):
        super().__init__()
        self.pw_in = nn.Sequential(
            nn.Conv2d(in_c, branch_c, 1, bias=False),
            nn.BatchNorm2d(branch_c),
            nn.ReLU(inplace=True)
        )
        self.dw_v = nn.Sequential(
            nn.Conv2d(branch_c, branch_c, (3, 1),
                      padding=(dilation, 0), dilation=(dilation, 1),
                      groups=branch_c, bias=False),
            nn.BatchNorm2d(branch_c),
        )
        self.dw_h = nn.Sequential(
            nn.Conv2d(branch_c, branch_c, (1, 3),
                      padding=(0, dilation), dilation=(1, dilation),
                      groups=branch_c, bias=False),
            nn.BatchNorm2d(branch_c),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        x = self.dw_v(x)
        x = self.dw_h(x)
        return self.act(x)


class DGASBlock(nn.Module):
    """
    Dilated Gated Asymmetric Separable Block.

    - Multi-scale dilated asymmetric-DW branches (parallel)
    - Gated fusion: sigmoid gate on half channels * GELU on other half
    - SE channel recalibration
    - Residual connection
    """
    def __init__(self, in_c: int, out_c: int, dilations: tuple = (1, 2)):
        super().__init__()
        n_branches = len(dilations)
        branch_c = out_c // n_branches

        self.branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            bc = out_c - branch_c * (n_branches - 1) if i == n_branches - 1 else branch_c
            self.branches.append(_AsymDWBranch(in_c, bc, dilation=d))

        # Gated fusion: split concat into gate & value halves
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        # SE recalibration
        self.se = SE(out_c, reduction=8)

        # Shortcut
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            ) if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        gate = torch.sigmoid(self.gate_conv(feats))
        value = F.gelu(self.value_conv(feats))
        out = self.se(gate * value)
        return out + self.shortcut(x)


# ---------------------------------------------------------------------------
# Module 2: InfoDown Downsampler (from PID-UNet, keeps diminutive polyps)
# ---------------------------------------------------------------------------

class InfoDown(nn.Module):
    """
    Information-Preserving Downsampler.
    Dual-path: Strided Conv (learned boundaries) + MaxPool (structural sharp edges).
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        c_conv = out_c // 2
        c_pool = out_c - c_conv
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_c, c_conv, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_conv)
        )
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_c, c_pool, 1, bias=False),
            nn.BatchNorm2d(c_pool)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([self.conv_path(x), self.pool_path(x)], dim=1))


# ---------------------------------------------------------------------------
# Module 3: Semantic Boundary Branch (D-path)
# ---------------------------------------------------------------------------

class SemanticBoundaryBranch(nn.Module):
    """
    Semantic-guided boundary detection.
    Fuses multi-scale encoder features with bottleneck semantic context
    to produce a 1-channel boundary logit map.
    Uses DW separable 1x1 projections for efficiency.
    """
    def __init__(self, c1: int, c2: int, c3: int, c_bn: int, d_c: int = 16):
        super().__init__()
        self.r1 = nn.Conv2d(c1, d_c, 1, bias=False)
        self.r2 = nn.Conv2d(c2, d_c, 1, bias=False)
        self.r3 = nn.Conv2d(c3, d_c, 1, bias=False)
        self.r_sem = nn.Conv2d(c_bn, d_c, 1, bias=False)

        # Lightweight fuse: DW separable 3x3 on 4*d_c channels
        self.fuse = nn.Sequential(
            nn.Conv2d(d_c * 4, d_c * 4, 3, padding=1, groups=d_c * 4, bias=False),
            nn.Conv2d(d_c * 4, d_c, 1, bias=False),
            nn.BatchNorm2d(d_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_c, 1, 1)   # boundary logit
        )

    def forward(self, e1, e2, e3, bn):
        target = e2.shape[2:]  # H/4, W/4
        f1 = F.interpolate(self.r1(e1), size=target, mode='bilinear', align_corners=True)
        f2 = self.r2(e2)
        f3 = F.interpolate(self.r3(e3), size=target, mode='bilinear', align_corners=True)
        f_sem = F.interpolate(self.r_sem(bn), size=target, mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([f1, f2, f3, f_sem], dim=1))


# ---------------------------------------------------------------------------
# Module 4: LightBAG — Lightweight Boundary-Aware Guidance
# ---------------------------------------------------------------------------

class LightBAG(nn.Module):
    """
    Lightweight Boundary-Aware Guidance Decoder Module.
    A depthwise-separable, parameter-efficient version of PID-UNet's ResidualBAG.

    Components:
    - Soft Saliency Gate from I (integral/context features): suppresses background
    - Boundary Attention from D (boundary map): sharpens polyp contours
    - Residual projection for gradient flow
    """
    def __init__(self, p_c: int, i_c: int, out_c: int):
        super().__init__()
        # Project skip (P) and context (I)
        self.proj_p = nn.Sequential(
            nn.Conv2d(p_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.proj_i = nn.Sequential(
            nn.Conv2d(i_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        # Soft saliency gate from I
        self.sal_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(i_c, i_c // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(i_c // 4, out_c, 1, bias=False),
            nn.Sigmoid()
        )
        # DW separable refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
        self.se = SE(out_c)

    def forward(self, p: torch.Tensor, i_feat: torch.Tensor,
                boundary_map: torch.Tensor) -> torch.Tensor:
        h, w = p.shape[2:]
        if i_feat.shape[2:] != (h, w):
            i_feat = F.interpolate(i_feat, (h, w), mode='bilinear', align_corners=True)
        if boundary_map.shape[2:] != (h, w):
            boundary_map = F.interpolate(boundary_map, (h, w), mode='bilinear', align_corners=True)

        p_proj = self.proj_p(p)
        i_proj = self.proj_i(i_feat)

        # Saliency gate from context (channel-wise)
        sal = self.sal_gate(i_feat)             # (B, out_c, 1, 1)
        p_gated = p_proj * (0.5 + 0.5 * sal)   # soft gate: preserve gradient

        # Boundary sharpening
        edge = torch.sigmoid(boundary_map)      # (B, 1, H, W)
        d_comp = edge * p_gated

        fused = self.se(self.refine(p_gated + i_proj + d_comp))
        return fused + i_proj   # residual to i_proj for gradient flow


# ---------------------------------------------------------------------------
# Full Network: DGANet
# ---------------------------------------------------------------------------

@register_model('dganet')
class DGANet(nn.Module):
    """
    DGANet: Dilated Gated Asymmetric Network for Lightweight Polyp Segmentation.

    Parameters: ~0.72M  (<1M)
    Input:  (B, 3, H, W) — recommended 352×352
    Output: (out_d3, out_d2, out_d1, out_stem) — 4 logit maps at input resolution
            (deep supervision; primary output = out_stem during inference)

    Training: Use all 4 outputs with structure_loss + boundary aux loss.
    Inference: Use out_stem (the finest-scale output).
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1,
                 base_c: int = 64):
        """
        Args:
            base_c: Base channel width. Controls model capacity:
                base_c=36 → ~0.20M (ultra-light)
                base_c=48 → ~0.42M (light)
                base_c=64 → ~0.74M (default, balanced) ← recommended
                base_c=72 → ~0.93M (max under 1M)
        """
        super().__init__()

        # Channel config — scaled from base_c
        c1  = base_c           # e.g. 64
        c2  = int(base_c * 1.5)  # e.g. 96
        c3  = base_c * 2       # e.g. 128
        cbn = int(base_c * 2.5)  # e.g. 160

        # --- Stem: fast initial downsampling ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )   # (B, 32, H/2, W/2)

        # --- Encoder ---
        self.enc1 = DGASBlock(c1, c1, dilations=(1, 2))
        self.down1 = InfoDown(c1, c2)              # (B, 48, H/4, W/4)

        self.enc2 = DGASBlock(c2, c2, dilations=(1, 2, 4))
        self.down2 = InfoDown(c2, c3)              # (B, 80, H/8, W/8)

        self.enc3 = DGASBlock(c3, c3, dilations=(1, 3, 5))
        self.down3 = InfoDown(c3, cbn)             # (B, 112, H/16, W/16)

        # --- Bottleneck (Integral context, wide receptive field) ---
        self.bottleneck = DGASBlock(cbn, cbn, dilations=(1, 3, 6, 10))

        # --- Derivative Boundary Branch ---
        self.boundary = SemanticBoundaryBranch(c1, c2, c3, cbn, d_c=16)

        # --- Decoder with LightBAG ---
        self.dec3 = LightBAG(p_c=c3, i_c=cbn, out_c=c3)
        self.dec2 = LightBAG(p_c=c2, i_c=c3, out_c=c2)
        self.dec1 = LightBAG(p_c=c1, i_c=c2, out_c=c1)

        # --- Deep Supervision Heads ---
        self.head3 = nn.Conv2d(c3, n_classes, 1)   # coarsest
        self.head2 = nn.Conv2d(c2, n_classes, 1)
        self.head1 = nn.Conv2d(c1, n_classes, 1)   # finest

        # --- Boundary auxiliary head (for training only) ---
        self.head_boundary = nn.Identity()          # boundary already 1ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        orig_size = x.shape[2:]

        # 1. Encoder
        s = self.stem(x)                    # (B, 32, H/2, W/2)
        e1 = self.enc1(s)                   # (B, 32, H/2, W/2)
        e2 = self.enc2(self.down1(e1))      # (B, 48, H/4, W/4)
        e3 = self.enc3(self.down2(e2))      # (B, 80, H/8, W/8)

        # 2. Bottleneck (Integral context)
        bn = self.bottleneck(self.down3(e3))  # (B, 112, H/16, W/16)

        # 3. Derivative Boundary Branch
        b_map = self.boundary(e1, e2, e3, bn)  # (B, 1, H/4, W/4)

        # 4. Decoder
        d3 = self.dec3(e3, bn, b_map)       # (B, 80, H/8, W/8)
        d2 = self.dec2(e2, d3, b_map)       # (B, 48, H/4, W/4)
        d1 = self.dec1(e1, d2, b_map)       # (B, 32, H/2, W/2)

        # 5. Deep supervision (upsample all to input resolution)
        out3 = F.interpolate(self.head3(d3), orig_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), orig_size, mode='bilinear', align_corners=True)
        out1 = F.interpolate(self.head1(d1), orig_size, mode='bilinear', align_corners=True)
        # Boundary map auxiliary output — also upsampled to input resolution for
        # consistent loss computation with ground-truth masks (structure loss).
        # At inference, boundary_loss contribution is from this same prediction.
        b_up = F.interpolate(b_map, orig_size, mode='bilinear', align_corners=True)

        # Return order: (coarsest, mid, finest, boundary_aux)
        # Trainer applies structure_loss to all 4; boundary head acts as an aux segmentor.
        # Primary inference output = out1 (finest, index [-1] in tuple).
        return out3, out2, out1, b_up
