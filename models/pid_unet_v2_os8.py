"""Scratch-trained v2 with its deepest feature map kept at output stride 8.

Retains v2's learned layers, PID decoder, four heads and boundary branch. Only
the final spatial transition and bottleneck dilation change. All training
defaults and losses are provided by the existing PID v2 family recipe.

Dense feature extraction with compensated dilation follows the established
atrous-convolution approach: https://arxiv.org/abs/1706.05587.
Improved segmentation accuracy is a hypothesis, not a measured result.
"""

from torch import nn

from models import register_model
from models.pid_unet_v2 import PIDUNetV2


@register_model('pid_unet_v2_os8')
class PIDUNetV2OS8(PIDUNetV2):
    """Keep 192-channel semantic features at H/8 instead of H/16.

    Stride/dilation are not stored in PyTorch parameter tensors. This explicit
    architecture identifier prevents a v2 checkpoint from silently loading
    even though all state_dict keys and tensor shapes match.
    """

    strict_checkpoint_loading = True
    checkpoint_architecture = 'pid_unet_v2_os8.v1.stride8.dilation2-6-12-24'

    def __init__(self):
        super().__init__()
        self.down4.conv_path[0].stride = (1, 1)
        self.down4.pool_path[0] = nn.MaxPool2d(3, stride=1, padding=1)
        for branch in self.bottleneck[0].branches:
            asymmetric = branch[3]
            for conv in (asymmetric.conv1, asymmetric.conv2):
                conv.dilation = tuple(2 * value if kernel > 1 else value
                                      for value, kernel in zip(conv.dilation, conv.kernel_size))
                conv.padding = tuple(2 * value for value in conv.padding)
