"""
DGANet (v2 Flagship): Dilated Gated Asymmetric & Reverse Attention Network
Engineered for SOTA Performance in Lightweight Medical Polyp Segmentation (< 1M Parameters).

Key Architectural Pillars (Designed to surpass EDANet, ESPNetv2, and CMUNeXt):
1. Info-Preserving Downsampler (InfoDown):
   - Dual-path strided conv + maxpool concat.
   - Prevents loss of diminutive lesions (<20px in ETIS-LaribPolypDB) during downsampling.
2. Multi-Scale Dilated Asymmetric Receptive Field Block with SE (DilatedAsymRFB):
   - Factorized 3x1 and 1x3 convolutions save 33% parameters while maintaining full cross-channel capacity.
   - Dilations spanning up to d=12 provide receptive fields > 240px to capture large colon folds and mucosal morphology.
   - Squeeze-and-Excitation (SE) channel recalibration dynamically suppresses colonoscopy specular reflection noise.
3. Partial Decoder (PPD):
   - Aggregates multi-scale semantic representations from high-level features (e3, e4, e5).
   - Generates an accurate global semantic prior map (S_g) as coarse guidance.
4. Boundary-Enhanced Reverse Attention (BERA):
   - Progressive erasing mechanism: A_i = 1 - sigmoid(P(S_{i+1})).
   - Forces decoder to focus precisely on ambiguous polyp margins and blurry mucosa boundaries.
   - Injects high-resolution edge priors from shallow encoder layers to achieve razor-sharp contours.
5. 4-Scale Deep Supervision:
   - Returns (out_5, out_4, out_3, out_2) where out_2 is the finest-resolution map at outputs[-1].
   - Compatible with PraNet Structure Loss (Weighted BCE + Weighted IoU).

Parameter Budget:
   - base_c=44 -> ~0.745M parameters (strictly < 1.0M parameters, only ~3.0 MB weights file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Basic Building Blocks
# ---------------------------------------------------------------------------

class BasicConv(nn.Module):
    """Standard Conv2d -> BatchNorm -> ReLU."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class AsymConv(nn.Module):
    """
    Factorized 3x1 and 1x3 convolution with dilation.
    Saves 33% parameters compared to standard 3x3 while doubling activation depth.
    """
    def __init__(self, in_c: int, out_c: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU(inplace=True)

        self.c2 = nn.Conv2d(out_c, out_c, (1, 3), padding=(0, dilation), dilation=(1, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.bn2(self.c2(self.act1(self.bn1(self.c1(x))))))


# ---------------------------------------------------------------------------
# Module 1: Info-Preserving Downsampler (InfoDown)
# ---------------------------------------------------------------------------

class InfoDown(nn.Module):
    """
    Prevents spatial loss of diminutive polyps during downsampling.
    Combines learned strided convolution with localized max pooling.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        c_conv = out_c // 2
        c_pool = out_c - c_conv

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, c_conv, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_conv)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_c, c_pool, 1, bias=False),
            nn.BatchNorm2d(c_pool)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([self.conv(x), self.pool(x)], dim=1))


# ---------------------------------------------------------------------------
# Module 2: Dilated Asymmetric RFB with Squeeze-and-Excitation
# ---------------------------------------------------------------------------

class DilatedAsymRFB(nn.Module):
    """
    Parallel dilated asymmetric factorized convolutions + Squeeze-and-Excitation.
    Captures polyp scales spanning tiny diminutive to giant mucosal context.
    """
    def __init__(self, in_c: int, out_c: int, dilations: tuple = (1, 2, 4)):
        super().__init__()
        n = len(dilations)
        bc = out_c // n

        self.branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            c_out = out_c - bc * (n - 1) if i == n - 1 else bc
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_c, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                AsymConv(c_out, c_out, dilation=d)
            ))

        self.proj = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-Excitation channel recalibration
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, max(8, out_c // 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, out_c // 8), out_c, 1, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            ) if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.proj(feats)
        out = out * self.se(out)
        return out + self.shortcut(x)


# ---------------------------------------------------------------------------
# Module 3: High-Level Partial Decoder (PPD)
# ---------------------------------------------------------------------------

class PartialDecoder(nn.Module):
    """
    Aggregates multi-scale high-level semantic features (e3: H/8, e4: H/16, e5: H/32).
    Generates an initial global localization map (S_g).
    """
    def __init__(self, c3: int, c4: int, c5: int, out_c: int = 64):
        super().__init__()
        self.p3 = nn.Sequential(nn.Conv2d(c3, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.p4 = nn.Sequential(nn.Conv2d(c4, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.p5 = nn.Sequential(nn.Conv2d(c5, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            AsymConv(out_c * 3, out_c, dilation=1),
            AsymConv(out_c, out_c, dilation=2)
        )
        self.global_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, e3: torch.Tensor, e4: torch.Tensor, e5: torch.Tensor):
        h, w = e3.shape[2:]
        f3 = self.p3(e3)
        f4 = F.interpolate(self.p4(e4), size=(h, w), mode='bilinear', align_corners=True)
        f5 = F.interpolate(self.p5(e5), size=(h, w), mode='bilinear', align_corners=True)

        fused = self.fuse(torch.cat([f3, f4, f5], dim=1))
        sg = self.global_head(fused)
        return fused, sg


# ---------------------------------------------------------------------------
# Module 4: Boundary-Enhanced Reverse Attention (BERA)
# ---------------------------------------------------------------------------

class BoundaryEnhancedReverseAttention(nn.Module):
    """
    Boundary-Enhanced Reverse Attention Module:
    1. Erases currently identified coarse polyp region: RA = 1 - sigmoid(prev_mask).
    2. Forces feature representation to focus specifically on ambiguous polyp boundaries.
    3. Fuses edge priors from shallow encoder layers to sharpen fine lesion contours.
    """
    def __init__(self, in_c: int, edge_c: int, out_c: int):
        super().__init__()
        self.conv_in = BasicConv(in_c, out_c, k=3, p=1)
        self.edge_proj = (
            nn.Sequential(
                nn.Conv2d(edge_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ) if edge_c > 0 else None
        )

        fuse_in = out_c * 2 if edge_c > 0 else out_c
        self.conv_refine = nn.Sequential(
            AsymConv(fuse_in, out_c, dilation=1),
            AsymConv(out_c, out_c, dilation=2)
        )
        self.head = nn.Conv2d(out_c, 1, 1)

    def forward(self, feat: torch.Tensor, prev_mask: torch.Tensor, edge_feat: torch.Tensor = None):
        h, w = feat.shape[2:]
        if prev_mask.shape[2:] != (h, w):
            prev_mask_up = F.interpolate(prev_mask, size=(h, w), mode='bilinear', align_corners=True)
        else:
            prev_mask_up = prev_mask

        # Reverse attention erasing weight
        ra_weight = 1.0 - torch.sigmoid(prev_mask_up)
        x = feat * ra_weight
        x = self.conv_in(x)

        if self.edge_proj is not None and edge_feat is not None:
            if edge_feat.shape[2:] != (h, w):
                edge_feat = F.interpolate(edge_feat, size=(h, w), mode='bilinear', align_corners=True)
            e_p = self.edge_proj(edge_feat)
            x = torch.cat([x, e_p], dim=1)

        x = self.conv_refine(x)
        out_mask = self.head(x) + prev_mask_up
        return x, out_mask


# ---------------------------------------------------------------------------
# Full Network: DGANet (v2 Flagship)
# ---------------------------------------------------------------------------

@register_model('dganet')
class DGANet(nn.Module):
    """
    DGANet (v2 Flagship): Dilated Gated Asymmetric & Reverse Attention Network.
    Parameters: ~0.745M (strictly < 1.0M parameters).

    Outputs: (out_5, out_4, out_3, out_2)
    - out_5: Global semantic initial map (coarse, S_g)
    - out_4: RA stage 4 map (S_4)
    - out_3: RA stage 3 map (S_3)
    - out_2: RA stage 2 map (S_2, finest resolution prediction at outputs[-1])
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 44):
        super().__init__()
        c1 = base_c            # 44
        c2 = int(base_c * 1.5) # 66
        c3 = base_c * 2        # 88
        c4 = base_c * 3        # 132
        c5 = base_c * 4        # 176

        # --- Stem & Multi-scale Encoder ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.enc1 = DilatedAsymRFB(c1, c1, dilations=(1, 2))
        self.down1 = InfoDown(c1, c2)  # 176 -> 88 (H/4)

        self.enc2 = DilatedAsymRFB(c2, c2, dilations=(1, 2, 4))
        self.down2 = InfoDown(c2, c3)  # 88 -> 44 (H/8)

        self.enc3 = DilatedAsymRFB(c3, c3, dilations=(1, 2, 4))
        self.down3 = InfoDown(c3, c4)  # 44 -> 22 (H/16)

        self.enc4 = DilatedAsymRFB(c4, c4, dilations=(1, 3, 6))
        self.down4 = InfoDown(c4, c5)  # 22 -> 11 (H/32)

        self.bottleneck = DilatedAsymRFB(c5, c5, dilations=(1, 3, 6, 12))

        # --- Low-level Edge Guidance Extractor (from enc1) ---
        self.edge_extractor = BasicConv(c1, 24, k=3, p=1)

        # --- High-level Partial Decoder (PPD) ---
        self.pd = PartialDecoder(c3, c4, c5, out_c=64)

        # --- Boundary-Enhanced Reverse Attention Decoders ---
        self.ra4 = BoundaryEnhancedReverseAttention(in_c=c4, edge_c=0, out_c=48)
        self.ra3 = BoundaryEnhancedReverseAttention(in_c=c3, edge_c=0, out_c=36)
        self.ra2 = BoundaryEnhancedReverseAttention(in_c=c2, edge_c=24, out_c=28)

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
        orig_sz = x.shape[2:]

        # 1. Multi-scale Encoder
        s = self.stem(x)               # (B, 44, H/2)
        e1 = self.enc1(s)              # (B, 44, H/2)
        e2 = self.enc2(self.down1(e1)) # (B, 66, H/4)
        e3 = self.enc3(self.down2(e2)) # (B, 88, H/8)
        e4 = self.enc4(self.down3(e3)) # (B, 132, H/16)
        e5 = self.bottleneck(self.down4(e4)) # (B, 176, H/32)

        # 2. Extract edge cues from high-res feature
        edge_feat = self.edge_extractor(e1)

        # 3. High-level Partial Decoder (Coarse Global Map S_g)
        _, sg = self.pd(e3, e4, e5)   # (B, 1, H/8)

        # 4. Progressive Reverse Attention Cascade
        _, s4 = self.ra4(e4, sg)              # (B, 1, H/16)
        _, s3 = self.ra3(e3, s4)              # (B, 1, H/8)
        _, s2 = self.ra2(e2, s3, edge_feat)   # (B, 1, H/4) finest

        # 5. Deep Supervision maps (all interpolated to input resolution)
        out_5 = F.interpolate(sg, size=orig_sz, mode='bilinear', align_corners=False) # Coarse global
        out_4 = F.interpolate(s4, size=orig_sz, mode='bilinear', align_corners=False)
        out_3 = F.interpolate(s3, size=orig_sz, mode='bilinear', align_corners=False)
        out_2 = F.interpolate(s2, size=orig_sz, mode='bilinear', align_corners=False) # Finest prediction

        # PraNet standard: return (coarse -> fine)
        # outputs[-1] is out_2 (primary prediction evaluated during test & validation)
        return out_5, out_4, out_3, out_2
