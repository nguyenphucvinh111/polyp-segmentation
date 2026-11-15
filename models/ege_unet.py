"""
EGE-UNet: An Efficient Group Enhanced UNet for Skin Lesion and Polyp Segmentation.

Reference:
    Ruan, J., Xiang, S., Xie, M., Liu, T., & Fu, Y. (2023).
    EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation.
    International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2023).
    arXiv:2307.08041
    Official repository: https://github.com/JCruan519/EGE-UNet

Architectural Innovations:
    1. Group multi-axis Hadamard Product Attention (GHPA):
       - Replaces heavy multi-head self-attention with linear-complexity element-wise Hadamard products.
       - Splits channel representations into 4 groups operating across:
         * Spatial Height-Width plane (P_xy)
         * Channel-Height plane (P_zx)
         * Channel-Width plane (P_zy)
         * Local Depthwise Convolution
    2. Group Aggregation Bridge (GAB):
       - Bridges encoder and decoder features while incorporating auxiliary guidance masks.
       - Employs multi-rate dilated grouped convolutions (d=[1, 2, 5, 7]) for lightweight multi-scale aggregation.
    3. Ultra-Lightweight Footprint:
       - Default channels c_list=[8, 16, 24, 32, 48, 64] -> ~0.159M parameters (strictly < 1.0M parameters).

Registered as: 'ege_unet', 'egeunet'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Layer Normalization (Channels-First)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm supporting channels_first tensor layout (B, C, H, W)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ---------------------------------------------------------------------------
# Module 1: Group Multi-axis Hadamard Product Attention (GHPA)
# ---------------------------------------------------------------------------

class Group_multi_axis_Hadamard_Product_Attention(nn.Module):
    """
    GHPA: Group Multi-axis Hadamard Product Attention.
    Splits incoming feature channels into 4 orthogonal groups:
    - Group 1: Height-Width axis (P_xy) spatial Hadamard attention
    - Group 2: Channel-Height axis (P_zx) cross-axis attention
    - Group 3: Channel-Width axis (P_zy) cross-axis attention
    - Group 4: Local Depthwise Convolution
    Linear complexity O(N) replaces quadratic O(N^2) multi-head self-attention.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, f"Channel dimension ({dim}) must be divisible by 4 groups."
        self.group_dim = dim // 4

        # P_xy: Spatial Attention (1, 1, H, W)
        self.p_xy = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.dw_xy = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # P_zx: Channel-Height Attention (1, C, H, 1)
        self.p_zx = nn.Parameter(torch.ones(1, self.group_dim, 1, 1))
        self.dw_zx = nn.Conv2d(
            self.group_dim, self.group_dim, kernel_size=(3, 1), padding=(1, 0),
            groups=self.group_dim, bias=False
        )

        # P_zy: Channel-Width Attention (1, C, 1, W)
        self.p_zy = nn.Parameter(torch.ones(1, self.group_dim, 1, 1))
        self.dw_zy = nn.Conv2d(
            self.group_dim, self.group_dim, kernel_size=(1, 3), padding=(0, 1),
            groups=self.group_dim, bias=False
        )

        # Local Depthwise Convolution for local detail retention
        self.dw_local = nn.Conv2d(
            self.group_dim, self.group_dim, kernel_size=3, padding=1,
            groups=self.group_dim, bias=False
        )

        self.norm = LayerNorm(dim, data_format="channels_first")
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x1, x2, x3, x4 = torch.split(x, self.group_dim, dim=1)

        # 1. P_xy: Spatial Attention
        p_xy = F.interpolate(self.p_xy, size=(H, W), mode='bilinear', align_corners=False)
        p_xy = self.dw_xy(p_xy)
        x1 = x1 * p_xy

        # 2. P_zx: Channel-Height Attention
        p_zx = F.interpolate(self.p_zx, size=(H, 1), mode='bilinear', align_corners=False)
        p_zx = self.dw_zx(p_zx)
        x2 = x2 * p_zx

        # 3. P_zy: Channel-Width Attention
        p_zy = F.interpolate(self.p_zy, size=(1, W), mode='bilinear', align_corners=False)
        p_zy = self.dw_zy(p_zy)
        x3 = x3 * p_zy

        # 4. Local DWConv
        x4 = self.dw_local(x4)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.norm(out)
        out = self.proj(out)
        return out + residual


# ---------------------------------------------------------------------------
# Module 2: Group Aggregation Bridge (GAB)
# ---------------------------------------------------------------------------

class group_aggregation_bridge(nn.Module):
    """
    GAB: Group Aggregation Bridge.
    Fuses high-level features, low-level features, and auxiliary masks across
    4 grouped branches with multi-rate dilated convolutions (d=[1, 2, 5, 7]).
    """
    def __init__(self, dim_xh: int, dim_xl: int, k_size: int = 3, d_list: list = [1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, kernel_size=1, bias=False)
        group_size = dim_xl // 2  # (dim_xl//4 + dim_xl//4 + 1) = group_size + 1

        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + 1, bias=False),
            nn.Conv2d(group_size + 1, group_size, kernel_size=1, bias=False)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + 1, bias=False),
            nn.Conv2d(group_size + 1, group_size, kernel_size=1, bias=False)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + 1, bias=False),
            nn.Conv2d(group_size + 1, group_size, kernel_size=1, bias=False)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + 1, bias=False),
            nn.Conv2d(group_size + 1, group_size, kernel_size=1, bias=False)
        )

        self.post_project = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim_xl * 2, dim_xl, kernel_size=1, bias=False)
        )

    def forward(self, xh: torch.Tensor, xl: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        xh = self.pre_project(xh)
        if xh.shape[2:] != xl.shape[2:]:
            xh = F.interpolate(xh, size=xl.shape[2:], mode='bilinear', align_corners=True)

        if mask is None:
            mask = torch.zeros((xl.size(0), 1, xl.size(2), xl.size(3)), device=xl.device, dtype=xl.dtype)
        elif mask.shape[2:] != xl.shape[2:]:
            mask = F.interpolate(mask, size=xl.shape[2:], mode='bilinear', align_corners=True)

        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh_chunks[0], xl_chunks[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh_chunks[1], xl_chunks[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh_chunks[2], xl_chunks[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh_chunks[3], xl_chunks[3], mask), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.post_project(x)
        return x + xl


# ---------------------------------------------------------------------------
# Full EGE-UNet Architecture
# ---------------------------------------------------------------------------

@register_model('ege_unet')
@register_model('egeunet')
class EGEUNet(nn.Module):
    """
    EGE-UNet: An Efficient Group Enhanced UNet for Skin Lesion and Polyp Segmentation.
    Parameters: ~0.159M (ultra-lightweight < 1.0M budget).
    """
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list: list = None,
        bridge: bool = True
    ):
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]
        self.bridge = bridge

        # 6-Stage Encoder
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[2]),
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1)
        )
        self.encoder5 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[3]),
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1)
        )
        self.encoder6 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[4]),
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1)
        )

        # Group Aggregation Bridges
        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])

        # 5-Stage Decoder
        self.decoder1 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[5]),
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1)
        )
        self.decoder2 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[4]),
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1)
        )
        self.decoder3 = nn.Sequential(
            Group_multi_axis_Hadamard_Product_Attention(c_list[3]),
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1)
        )
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)

        # Normalization Layers
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # Deep Supervision Heads
        self.head5 = nn.Conv2d(c_list[4], num_classes, 1)
        self.head4 = nn.Conv2d(c_list[3], num_classes, 1)
        self.head3 = nn.Conv2d(c_list[2], num_classes, 1)
        self.head2 = nn.Conv2d(c_list[1], num_classes, 1)
        self.final = nn.Conv2d(c_list[0], num_classes, 1)

    def forward(self, x: torch.Tensor):
        orig_size = x.shape[2:]

        # Encoder forward pass
        out1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))      # 1/2, c[0]
        out2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out1)), 2, 2))   # 1/4, c[1]
        out3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out2)), 2, 2))   # 1/8, c[2]
        out4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out3)), 2, 2))   # 1/16, c[3]
        out5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out4)), 2, 2))   # 1/32, c[4]
        bottleneck = F.gelu(self.encoder6(out5))                            # 1/32, c[5]

        # Decoder Stage 5
        d5 = F.gelu(self.dbn1(self.decoder1(bottleneck)))
        mask5 = torch.sigmoid(self.head5(d5))
        t5 = self.GAB5(bottleneck, out5, mask5) if self.bridge else out5
        d5 = d5 + t5

        # Decoder Stage 4
        d4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(d5)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        mask4 = torch.sigmoid(self.head4(d4))
        t4 = self.GAB4(d5, out4, mask4) if self.bridge else out4
        d4 = d4 + t4

        # Decoder Stage 3
        d3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(d4)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        mask3 = torch.sigmoid(self.head3(d3))
        t3 = self.GAB3(d4, out3, mask3) if self.bridge else out3
        d3 = d3 + t3

        # Decoder Stage 2
        d2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(d3)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        mask2 = torch.sigmoid(self.head2(d2))
        t2 = self.GAB2(d3, out2, mask2) if self.bridge else out2
        d2 = d2 + t2

        # Decoder Stage 1
        d1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(d2)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        t1 = self.GAB1(d2, out1, None) if self.bridge else out1
        d1 = d1 + t1

        # Final prediction map
        out0 = F.interpolate(self.final(d1), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        if out0.shape[2:] != orig_size:
            out0 = F.interpolate(out0, size=orig_size, mode='bilinear', align_corners=True)

        if self.training:
            # 4-scale deep supervision (outputs[-1] is out0, evaluated by benchmark)
            p4 = F.interpolate(self.head4(d4), size=orig_size, mode='bilinear', align_corners=True)
            p3 = F.interpolate(self.head3(d3), size=orig_size, mode='bilinear', align_corners=True)
            p2 = F.interpolate(self.head2(d2), size=orig_size, mode='bilinear', align_corners=True)
            return p4, p3, p2, out0
        return out0
