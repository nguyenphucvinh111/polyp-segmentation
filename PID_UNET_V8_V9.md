# PID-UNet v8 and v9: independent experiments on v2

Both models train entirely from scratch with the original v2 loss and training
recipe. They test two separate hypotheses about the CVC-300 weakness. Neither
has a measured accuracy improvement yet; original v2 and v7 remain available.

| Model | One architecture change | Parameters |
|---|---|---:|
| `pid_unet_v2` | Baseline | 835,373 |
| `pid_unet_v8` | Spatial control of both integral routes in bag1/bag2 | 840,463 |
| `pid_unet_v9` | Partial IN/BN at five shallow RFB branch entrances | 835,373 |

## v8: control the integral path

The original v2 gate modulates P and D, while I enters fusion and also bypasses
fusion through the direct residual. v8 multiplies I by the same learned spatial
gate on **both** of those routes. Only bag1 and bag2, the full- and half-resolution
decoder stages, are changed. The original asymmetric convolution, SE, P/D gates,
PID gains, encoder, boundary branch and four heads are retained.

The new gate combines projected skip features, projected integral features,
their absolute difference and the existing boundary probability. A small local
network produces `g = 1 + 0.5 * tanh(correction)`. Thus `g` is in [0.5, 1.5].
Its final convolution starts at zero, so the initialized v8 gives exactly the v2
evaluation outputs when the shared randomly initialized weights match. This is an initial
condition, not a guarantee that trained predictions stay the same. No external
checkpoint is loaded. The tanh calculation uses FP32 under autocast.

The hypothesis is that spatial control can reduce unwanted integral activations
in background while keeping a nonzero signal path. It may also reduce recall;
check all five datasets. The direct residual in v2 is not a proven bug.

## v9: partial instance normalization in the shallow encoder

v9 replaces the first BatchNorm after the 1x1 projection in each branch of enc1
and enc2: two branches in enc1 and three in enc2. In each replaced layer, the
first `floor(C/2)` channels use affine InstanceNorm and the remaining channels
use BatchNorm. InstanceNorm uses per-image statistics in both training and
evaluation; BatchNorm keeps its usual running statistics. Odd channel counts
are supported. The total number of learned affine parameters is unchanged.

Stem, downsampling paths, later normalizations within the RFBs, deep encoder,
decoder, boundary branch and heads remain v2. Limiting IN to branch entrances
keeps the existing residual paths and later normalization available. The
hypothesis is reduced sensitivity to appearance changes between datasets;
normalizing useful color cues away remains a possible drawback.

This adapts the mixed-normalization idea from
[IBN-Net](https://arxiv.org/abs/1807.09441) to the v2 RFB structure. It is not a
pretrained IBN backbone or a claim that mixed normalization itself is novel.

## Train in separate Colab GPU runtimes

- [Open v8 notebook](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_unet_v8_colab.ipynb)
- [Open v9 notebook](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_unet_v9_colab.ipynb)

Each notebook mounts Drive, installs dependencies and uses its own output tag.
For an existing checkout, mount Drive, pull main, and run **one model per GPU
runtime** for concurrent experiments:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Run from /content/polyp-segmentation in each runtime.
!git pull --ff-only origin main
```

```python
# Runtime A
!python train.py --model pid_unet_v8 --loss structure_loss --epochs 100 --batchsize 8 --seed 42 --output_root /content/drive/MyDrive/polyp_runs
```

```python
# Runtime B
!python train.py --model pid_unet_v9 --loss structure_loss --epochs 100 --batchsize 8 --seed 42 --output_root /content/drive/MyDrive/polyp_runs
```

The recipe stays v2: four structure losses plus the existing boundary loss at
weight .2, AdamW at 3e-4 with weight decay 1e-4, five warmup epochs, cosine decay,
352 base resolution, stochastic multi-scale training, CUDA AMP and copy-paste
disabled. The same fixed train/validation split selects checkpoints. No new
loss, augmentation, threshold or test-set checkpoint selection is introduced.

Drive outputs use `snapshots/pid_unet_v8/{best,last}.pth` and
`snapshots/pid_unet_v9/{best,last}.pth`, plus matching logs/results directories.
After interruption, set `RESUME = True` in the corresponding notebook and keep
the same seed, total epoch count and run name. For CLI resume, append
`--resume /content/drive/MyDrive/polyp_runs/snapshots/<model>/last.pth`.
v8 and v9 reject checkpoints from each other or from v2/v7. A fresh run needs
a new `--run_name` if the default destination already contains training output.

## Verification and comparison

```bash
python -m unittest discover -s tests -p test_pid_unet_v8_v9.py -v
```

Tests cover exact v8 initial equivalence to v2, control of both I routes, v9's
per-image normalization behavior and exact replacement locations, parameter
budgets, 256/352/448 and rectangular outputs, unchanged Trainer loss with actual
optimizer updates, mixed-precision backward, and checkpoint round-trips with
architecture mismatch rejection. Local execution uses CPU; CUDA precision,
latency, peak training memory and full-dataset accuracy still require Colab.

Python ablations are `PIDUNetV8(integral_gating=False)` and
`PIDUNetV9(use_ibn=False)`, both recovering v2 with matching weights. Train the
default variants separately so their effects can be compared. Choose on held-out
validation, report all five test sets and repeat multiple seeds before a paper
claim. The reported v7 CVC-300 result (.8195 mDice) does not establish a cause
for its errors or guarantee either new hypothesis will improve it.
