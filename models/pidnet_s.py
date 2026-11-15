"""
PIDNet-S: A Real-time Semantic Segmentation Network Inspired by PID Controller.

Reference:
    Xu, J., Xiong, Z., & Bhattacharyya, S. P. (2023).
    PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controller.
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023).
    Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controller_CVPR_2023_paper.html
    Official repository: https://github.com/XuJiacong/PIDNet
    Target Parameter count: ~7.6M - 7.9M parameters.

Registered as: 'pidnet_s', 'pidnet'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding, no bias for BN pairing."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None, no_relu: bool = False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class PagFM(nn.Module):
    """
    Pixel-attention-guided Fusion Module (PagFM):
    Dynamically fuses context (I) and detail (P) features using learned attention.
    """
    def __init__(self, in_channels: int, mid_channels: int, after_relu: bool = False, with_channel: bool = False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = F.interpolate(self.f_y(y), size=[h, w], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y_up = F.interpolate(y, size=[h, w], mode='bilinear', align_corners=False)
        return (1.0 - sim_map) * x + sim_map * y_up


class Bag(nn.Module):
    """
    Boundary-attention-guided Fusion Module (Bag):
    Fuses Detail (P) and Context (I) features guided by the Derivative boundary (D).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if d.shape[1] != 1 and d.shape[1] != p.shape[1]:
            edge_att = torch.sigmoid(d.mean(dim=1, keepdim=True))
        else:
            edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1.0 - edge_att) * i)


class PAPPM(nn.Module):
    """
    Parallel Aggregated Pyramid Pooling Module (PAPPM):
    Captures multi-scale receptive fields in the deep context branch.
    """
    def __init__(self, in_channels: int, branch_planes: int, out_channels: int):
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, branch_planes, kernel_size=1, bias=False)
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, branch_planes, kernel_size=1, bias=False)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, branch_planes, kernel_size=1, bias=False)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, branch_planes, kernel_size=1, bias=False)
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, branch_planes, kernel_size=1, bias=False)
        )

        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes * 5, out_channels, kernel_size=1, bias=False)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x_list = [self.scale0(x)]
        x_list.append(self.process1(F.interpolate(self.scale1(x), size=[h, w], mode='bilinear', align_corners=False) + x_list[0]))
        x_list.append(self.process2(F.interpolate(self.scale2(x), size=[h, w], mode='bilinear', align_corners=False) + x_list[1]))
        x_list.append(self.process3(F.interpolate(self.scale3(x), size=[h, w], mode='bilinear', align_corners=False) + x_list[2]))
        x_list.append(self.process4(F.interpolate(self.scale4(x), size=[h, w], mode='bilinear', align_corners=False) + x_list[3]))
        return self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)


class SegmentHead(nn.Module):
    """Segmentation output head."""
    def __init__(self, inplanes: int, interplanes: int, outplanes: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


@register_model('pidnet_s')
@register_model('pidnet')
class PIDNet_S(nn.Module):
    """
    PIDNet-S: Small variant of PIDNet for Real-Time Semantic Segmentation (~7.6M - 7.9M params).
    Uses 3 parallel branches:
      - P-Branch (Proportional): Spatial detail preservation.
      - I-Branch (Integral): Global context embedding.
      - D-Branch (Derivative): High-frequency boundary delineation.
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        m: int = 2,
        n: int = 3,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128
    ):
        super().__init__()
        # Stem: 2x stride-2 convolutions -> 1/4 resolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=False)

        # I-Branch (Context backbone)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(BasicBlock, planes * 8, planes * 8, 2, stride=2)

        self.ppm = PAPPM(planes * 8, ppm_planes, planes * 4)

        # Dimension compression for PagFM fusion
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        self.compression_i = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )

        # PagFM fusion modules (I -> P)
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        # P-Branch (Detail processing)
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)

        # D-Branch (Boundary processing)
        self.conv_d = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
        )
        self.layer3_d = self._make_layer(BasicBlock, planes, planes, 2)
        self.layer4_d = self._make_layer(BasicBlock, planes, planes, 2)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        # Bag module for P, I, D integration
        self.bag = Bag(planes * 2, planes * 2)

        # Segmentation Head
        self.final_head = SegmentHead(planes * 2, head_planes, num_classes)

    def _make_layer(self, block, inplanes: int, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        x = self.conv1(x)              # 1/4, 32
        x = self.layer1(x)             # 1/4, 32
        x = self.layer2(self.relu(x))  # 1/8, 64

        x_p = x                        # P-branch at 1/8, 64
        x_d = self.conv_d(x)           # D-branch at 1/8, 32

        # Level 3
        x_i = self.layer3(self.relu(x))  # 1/16, 128
        x_p = self.pag3(self.layer3_p(x_p), self.compression3(x_i))
        x_d = self.layer3_d(x_d) + F.interpolate(self.diff3(x_i), size=x_d.shape[2:], mode='bilinear', align_corners=False)

        # Level 4
        x_i = self.layer4(self.relu(x_i))  # 1/32, 256
        x_p = self.pag4(self.layer4_p(x_p), self.compression4(x_i))
        x_d = self.layer4_d(x_d) + F.interpolate(self.diff4(x_i), size=x_d.shape[2:], mode='bilinear', align_corners=False)

        # Level 5
        x_i = self.layer5(self.relu(x_i))  # 1/64, 256
        x_i = self.ppm(x_i)                # 1/64, 128
        x_i = self.compression_i(x_i)      # 1/64, 64
        x_i = F.interpolate(x_i, size=x_p.shape[2:], mode='bilinear', align_corners=False)

        # Final BAG Fusion of P, I, D
        out = self.bag(x_p, x_i, x_d)
        logits = self.final_head(out)

        return F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
