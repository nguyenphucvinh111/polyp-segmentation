"""
CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion.

Reference:
    Tang, F., Ding, J., Wang, L., Ning, C., & Zhou, S. K. (2024).
    CMUNeXt: An Efficient Medical Image Segmentation Network based on
    Large Kernel and Skip Fusion.
    arXiv:2308.01239
    https://github.com/FengheTan9/CMUNeXt

Architecture (faithfully from network/CMUNeXt.py):
    CMUNeXtBlock: depthwise large-kernel conv (k×k) + inverted bottleneck (×4) + residual
    Encoder: stem (conv_block) + 5 CMUNeXtBlock stages with MaxPool2d between them
    Decoder: 4 upsampling stages with Skip-Fusion blocks (grouped conv + inverted bottleneck)
    Output: 1×1 conv head

Variants (from source code factory functions):
    cmunext   (base): dims=[16,32,128,160,256], depths=[1,1,1,3,1], kernels=[3,3,7,7,7]
    cmunext_s (small): dims=[8,16,32,64,128],   depths=[1,1,1,1,1], kernels=[3,3,7,7,9]
    cmunext_l (large): dims=[32,64,128,256,512], depths=[1,1,1,6,3], kernels=[3,3,7,7,7]
"""

import torch
import torch.nn as nn
from models import register_model


# ---------------------------------------------------------------------------
# Building blocks (faithful to https://github.com/FengheTan9/CMUNeXt)
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """
    CMUNeXt encoder block.
    Depthwise large-kernel conv (k×k) with residual, followed by
    inverted bottleneck (×4 expand) pointwise convolutions.
    Ends with a conv_block to change channel dimension.
    """

    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # depthwise conv
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in,
                              padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    """3×3 Conv → BN → ReLU."""

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """2× bilinear upsample + Conv3×3 + BN + ReLU."""

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    """
    Skip-Fusion block.
    Grouped conv (groups=2) to separately adapt encoder & decoder features,
    followed by inverted bottleneck (×4) pointwise convolutions with GELU + BN.
    """

    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# CMUNeXt network
# ---------------------------------------------------------------------------

class CMUNeXt(nn.Module):
    """
    CMUNeXt: U-shaped encoder-decoder with CMUNeXt blocks and Skip-Fusion.

    Args:
        input_channel: Number of input image channels (default: 3).
        num_classes: Number of output segmentation classes (default: 1).
        dims: Channel dimensions for each of the 5 encoder levels.
        depths: Number of CMUNeXt block repetitions at each level.
        kernels: Depthwise conv kernel size at each level.
    """

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=None,
        depths=None,
        kernels=None,
    ):
        super(CMUNeXt, self).__init__()
        if dims is None:
            dims = [16, 32, 128, 160, 256]
        if depths is None:
            depths = [1, 1, 1, 3, 1]
        if kernels is None:
            kernels = [3, 3, 7, 7, 7]

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        # Output head
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # Decoder with Skip-Fusion
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


# ---------------------------------------------------------------------------
# Registered variants
# ---------------------------------------------------------------------------

@register_model('cmunext')
def _cmunext_base(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt base variant: dims=[16,32,128,160,256], depths=[1,1,1,3,1], kernels=[3,3,7,7,7]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],
    )


@register_model('cmunext_s')
def _cmunext_small(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-S (small) variant: dims=[8,16,32,64,128], depths=[1,1,1,1,1], kernels=[3,3,7,7,9]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9],
    )


@register_model('cmunext_l')
def _cmunext_large(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-L (large) variant: dims=[32,64,128,256,512], depths=[1,1,1,6,3], kernels=[3,3,7,7,7]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7],
    )


# ============================================================================
# IMPROVEMENT (2026): TRAINING-ONLY HIGHLIGHT-AWARE TARGET REGULARIZATION (HAT)
# ============================================================================
# Based on:
# Li & Masood (2026), "HAT-SAM3: Endoscopy-Aware Adaptation of a Foundation
# Segmentation Model for Generalizable Polyp Segmentation", The Visual Computer.
# https://doi.org/10.1007/s00371-026-04562-3
#
# The complete CMUNeXt network above is left UNCHANGED. This addition has
# zero learnable parameters and is not executed during validation/inference.
# Using the supplied model file by itself as models/cmunext.py does NOT enable
# HAT. Use this file as a TRAINING ENTRY POINT from the repository root:
#     python cmunext_hat.py --check-only
#     python cmunext_hat.py --train --seed 42 --epochs 100 --batchsize 8
# ============================================================================


def highlight_aware_target(image, mask, rng, probability=0.5):
    """Return a temporary HAT training mask from a paired RGB image and GT.

    The original HAT paper uses V > 0.90, saturation < 0.25, p=0.5, and a
    10-pixel dilation radius at 1008x1008. Since the CMUNeXt pipeline resizes
    input to 352x352, we scale that radius at the *pre-resize* image size.

    The image and the input mask objects are NOT mutated. Only specular pixels
    in a thin exterior band around an annotated foreground are added.

    Args:
        image: PIL RGB image, after spatial flips/rotations but before jitter.
        mask: PIL L image, spatially aligned with image.
        rng: Dedicated random.Random instance; does not perturb baseline RNG.
        probability: Chance to apply the regularizer to an individual image.

    Returns:
        PIL L mask. No inference-time transformations are introduced.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError('HAT probability must be within [0, 1].')
    if probability == 0.0 or rng.random() >= probability:
        return mask

    import numpy as np
    from PIL import Image, ImageFilter

    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    value = rgb.max(axis=2)
    saturation = (value - rgb.min(axis=2)) / (value + 1e-6)
    bright_specular = (value > 0.90) & (saturation < 0.25)

    original = mask.convert('L')
    mask_array = np.asarray(original)
    foreground = mask_array > 127
    if not bright_specular.any() or not foreground.any():
        return mask

    height, width = mask_array.shape
    radius = max(1, round(10 * min(height, width) / 1008))
    dilated = np.asarray(
        original.filter(ImageFilter.MaxFilter(2 * radius + 1))
    ) > 127
    exterior_band = dilated & ~foreground
    target_pixels = bright_specular & exterior_band

    modified = mask_array.copy()
    modified[target_pixels] = 255
    return Image.fromarray(modified, mode='L')


def enable_highlight_aware_training(probability=0.5):
    """Install HAT into the repo's *training* augmentation, not the model.

    Called before train_single_run() builds loaders. The standard loader uses
    PolypAugmentation only for is_train=True, so validation/test are unchanged.
    Google Colab/Linux data workers use fork, preserving this substitution.
    """
    import random
    from utils import dataloader
    from utils.augmentation import PolypAugmentation

    class HATPolypAugmentation(PolypAugmentation):
        def __init__(self):
            super().__init__()  # Preserve original flip/rotation configuration.
            self._hat_random = None

        def __call__(self, image, mask):
            # Preserve original paired spatial augmentation unchanged.
            image, mask = super().__call__(image, mask)
            if self._hat_random is None:
                # Own RNG stream, seeded per DataLoader worker; baseline
                # flip/rotate calls still consume their original RNG stream.
                self._hat_random = random.Random(torch.initial_seed() ^ 0x4841542026)
            return image, highlight_aware_target(
                image, mask, rng=self._hat_random, probability=probability
            )

    dataloader.PolypAugmentation = HATPolypAugmentation
    return HATPolypAugmentation


def _verify_base_architecture():
    """Check every state-dict key/shape against the repo's original CMUNeXt.

    This is stronger than merely comparing two rounded parameter counts.
    """
    from models.cmunext import CMUNeXt as RepositoryCMUNeXt

    baseline = RepositoryCMUNeXt()
    proposed = _cmunext_base()
    before = {name: tuple(t.shape) for name, t in baseline.state_dict().items()}
    after = {name: tuple(t.shape) for name, t in proposed.state_dict().items()}
    if before != after:
        raise AssertionError('CMUNeXt state_dict keys/shapes differ from baseline!')
    n_before = sum(p.numel() for p in baseline.parameters())
    n_after = sum(p.numel() for p in proposed.parameters())
    if n_before != n_after:
        raise AssertionError('CMUNeXt parameter count changed!')

    # Importing models.cmunext may replace the registry entry. Restore the
    # uploaded architecture's model factory for this training process.
    register_model('cmunext')(_cmunext_base)
    print(f'[Architecture verified] Original: {n_before:,} parameters')
    print(f'[Architecture verified] Proposed: {n_after:,} parameters')
    print(f'[Architecture verified] Matched state_dict entries: {len(before)}')
    return n_after


def _main():
    import argparse
    from argparse import Namespace
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='CMUNeXt Base + training-only HAT (same architecture/params)'
    )
    parser.add_argument('--train', action='store_true', help='Actually run training')
    parser.add_argument('--check-only', action='store_true', help='Only verify model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    if not Path('train.py').is_file() or not Path('utils/dataloader.py').is_file():
        parser.error('Run from the polyp-segmentation repository root, next to train.py')
    if args.train and args.check_only:
        parser.error('Choose --train OR --check-only, not both.')

    _verify_base_architecture()
    if not args.train:
        print('For the 100-epoch HAT experiment: python cmunext_hat.py --train')
        return

    # The original train.py routine remains the sole training/evaluation code.
    # Only the dataset augmentation class is changed (train set only).
    enable_highlight_aware_training(probability=0.5)
    from train import train_single_run

    cfg = Namespace(
        model='cmunext', loss='structure_loss', seed=args.seed,
        epochs=args.epochs, batchsize=args.batchsize, data_root=args.data_root,
        resume=None, fast=False, amp=True, lr=1e-4, trainsize=352,
        clip=0.5, decay_rate=0.1, decay_epoch=50,
    )
    train_single_run(cfg, args.seed, run_name=f'cmunext_hat_seed{args.seed}')


if __name__ == '__main__':
    _main()

