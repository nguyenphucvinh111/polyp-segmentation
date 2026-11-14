"""V2 with partial instance normalization at shallow RFB branch entrances.

Related to IBN-Net: https://arxiv.org/abs/1807.09441
This is an adaptation to PID-UNet, not a reproduction of a pretrained IBN model.
"""

import torch
from torch import nn

from models import register_model
from models.pid_unet_v2 import PIDUNetV2


class InstanceBatchNorm2d(nn.Module):
    """Normalize the first half of channels per image and the remainder with BN."""

    def __init__(self, channels, eps=1e-5, momentum=.1):
        super().__init__()
        if channels < 2:
            raise ValueError('Instance/Batch split requires at least two channels')
        self.instance_channels = channels // 2
        self.instance = nn.InstanceNorm2d(
            self.instance_channels, eps=eps, affine=True, track_running_stats=False,
        )
        self.batch = nn.BatchNorm2d(channels - self.instance_channels, eps=eps, momentum=momentum)

    @classmethod
    def from_batchnorm(cls, source):
        layer = cls(source.num_features, eps=source.eps, momentum=source.momentum)
        split = layer.instance_channels
        with torch.no_grad():
            layer.instance.weight.copy_(source.weight[:split])
            layer.instance.bias.copy_(source.bias[:split])
            layer.batch.weight.copy_(source.weight[split:])
            layer.batch.bias.copy_(source.bias[split:])
            layer.batch.running_mean.copy_(source.running_mean[split:])
            layer.batch.running_var.copy_(source.running_var[split:])
            layer.batch.num_batches_tracked.copy_(source.num_batches_tracked)
        return layer

    def forward(self, features):
        split = self.instance_channels
        appearance = self.instance(features[:, :split].contiguous())
        content = self.batch(features[:, split:].contiguous())
        return torch.cat((appearance, content), dim=1)


@register_model('pid_unet_v9')
class PIDUNetV9(PIDUNetV2):
    """Scratch-trained v2 with IBN only at five enc1/enc2 branch entrances.

    Stem, downsamplers, later normalization layers and all decoder blocks stay v2.
    use_ibn=False restores the original v2 architecture for ablation.
    """

    strict_checkpoint_loading = True

    def __init__(self, use_ibn=True):
        super().__init__()
        if use_ibn:
            for encoder in (self.enc1, self.enc2):
                for branch in encoder.branches:
                    branch[1] = InstanceBatchNorm2d.from_batchnorm(branch[1])
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise ValueError('PID-UNet v9 exceeds the 1M parameter budget')
