# PID-UNet v7: scratch-trained semantic PID decoder

Train **`pid_unet_v7`**. All parameters are initialized from scratch; no ImageNet
weights or recovered checkpoint are needed. This is a new architecture experiment
based on v2, not a measured accuracy improvement. Original v2 model code remains
available as `pid_unet_v2`.

## Why this experiment

The supplied v2 Plus run reached CVC-300 mDice .8340, Recall .9607 and Precision
.7804. These aggregate metrics suggest investigating excess foreground predictions;
without saved masks, they do not identify the particular failure cases. Prior
refinement variants have not consistently improved all five test sets.

v7 changes the decoder at each stage while retaining the v2 encoder, global
context bottleneck, boundary branch, four segmentation heads, PID gains and soft
gate floor:

1. **Supervised coarse guidance.** The next coarser segmentation head supplies a
   foreground probability to the finer skip gate. It is averaged with the existing
   saliency probability. A local gate uses projected skip features, decoder
   features and their absolute difference to adjust that probability in log-odds
   space. Its correction starts at zero and is bounded to one log-odds unit.
   The coarsest block uses its existing saliency prior. The soft floor remains
   available so an uncertain coarse prediction cannot completely erase a skip.
2. **Multi-scale spatial refinement.** Each fusion block uses parallel depthwise
   3x3, 5x5 and 7x7 kernels, pointwise channel mixing and a residual connection.
   This supplies contiguous neighborhoods at several scales within the parameter
   budget. It replaces the original asymmetric convolution in decoder fusion.

The coarse heads already receive deep supervision. No additional target or loss
is introduced. Gradients also flow from the final head through the coarse guidance.
The log-odds calculation uses FP32 under autocast to avoid endpoint instability.

Joint skip gating is related to [Attention U-Net](https://arxiv.org/abs/1804.03999);
multi-scale depthwise decoding is related to
[EMCAD](https://arxiv.org/abs/2405.06880). This is a PID-specific implementation,
not a reproduction of either model or a claim that these primitives are new.
Performance reported in those papers does not establish performance for v7.

| Model | Parameters | THOP GMACs at 352 x 352 |
|---|---:|---:|
| v2 | 835,373 | 6.6414 |
| v2 Plus | 858,245 | 7.0540 |
| v7 | 758,257 | 6.2958 |

THOP counts do not include every functional operation, interpolation or
elementwise gate. MACs are not a GPU latency measurement; v7 T4 FPS, training
memory and accuracy still need to be measured. The local machine has CPU only.

## Train and keep outputs on Drive

Use [the Colab notebook](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_unet_v7_colab.ipynb)
in a GPU runtime. It mounts Drive, installs requirements, trains and compares the
completed results with the supplied EDANet reference. For an existing checkout:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run from `/content/polyp-segmentation`:

```python
!git pull --ff-only origin main
!python train.py --model pid_unet_v7 --loss structure_loss --epochs 100 --batchsize 8 --seed 42 --output_root /content/drive/MyDrive/polyp_runs
```

The default recipe matches v2: four structure losses plus the existing boundary
loss with weight .2, AdamW at 3e-4, weight decay 1e-4, five warmup epochs, cosine
decay, base resolution 352, stochastic multi-scale training, AMP on CUDA and
copy-paste disabled. The fixed 1305/145 train/validation split for the provided
1450-image training set, validation checkpoint selection and evaluation code are
unchanged. All five test sets remain outside training and checkpoint selection.

Under `/content/drive/MyDrive/polyp_runs`:

- `snapshots/pid_unet_v7/best.pth`: best validation mDice checkpoint.
- `snapshots/pid_unet_v7/last.pth`: latest completed epoch, saved every epoch.
- `logs/pid_unet_v7/train_log.csv`: epoch metrics.
- `results/pid_unet_v7/metrics.csv`: final five-dataset benchmark.

Use a new `--run_name` for another fresh experiment. A run refuses to overwrite
existing training output without `--resume`. Reconnect, mount Drive, restore the
checkout/dependencies and resume the **same seed and configuration** with:

```python
!python train.py --model pid_unet_v7 --loss structure_loss --epochs 100 --batchsize 8 --seed 42 --output_root /content/drive/MyDrive/polyp_runs --resume /content/drive/MyDrive/polyp_runs/snapshots/pid_unet_v7/last.pth
```

`--epochs 100` is the total epoch count, not 100 additional epochs. The checkpoint
stores optimizer state, AMP scaler, RNG state and the best validation score.
Resume starts after the saved epoch; an interrupted partial epoch is repeated.
Saving uses a temporary file before replacement. Mounted Drive preserves completed
files independently of the Colab VM, but no file can be recovered if it was only
stored in a deleted VM. GPU nondeterminism still prevents a bitwise guarantee.
Checkpoints from other architectures are rejected by v7.

## What the latest v2 Plus result actually beats

The user-supplied EDANet PDF provides mDice, mIoU, Sm, Em and MAE. Comparing the
supplied v2 Plus run numerically with those rounded reference values:

| Dataset | Better entries out of 5 | Remaining difference |
|---|---:|---|
| CVC-300 | 0 | .8340/.7548/.8748/.9173/.0162 vs .848/.758/.878/.922/.014 |
| CVC-ClinicDB | 5 | None |
| CVC-ColonDB | 4 | MAE .0552 vs EDANet .049; lower is better |
| ETIS-LaribPolypDB | 5 | None |
| Kvasir | 5 | None |

That is **19/25**, not all metrics outside CVC-300. Recall, Precision, Specificity,
HD95 and wFb have no EDANet reference in that PDF. The underlying data split and
metric definitions still need to match before a paper claim; validation of the
repository's Sm/Em/wFb implementations against reference code remains outstanding.

```python
!python compare_edanet.py --metrics /content/drive/MyDrive/polyp_runs/results/pid_unet_v7/metrics.csv
```

Do not select thresholds or checkpoints using CVC-300 test scores. Compare
multiple seeds for the locked configuration before claiming superiority. Given
the already inspected test scores across versions, use an additional untouched
evaluation set for a stronger final generalization claim.

## Verification and ablations

```bash
python -m unittest discover -s tests -v
```

CPU verification covers 256/352/448 and rectangular output shapes, saturated
coarse logits, bfloat16 forward/backward, actual updates using the unchanged
Trainer loss, gradients through all new paths, checkpoint round-trips and
architecture mismatch rejection. Persistence tests check failed-write protection,
best/last separation, RNG restoration and the missing Drive mount guard.
No full dataset training or CUDA benchmark has been performed locally.

Python ablations are `PIDUNetV7(multiscale=False)` and
`PIDUNetV7(semantic_guidance=False)`. Disabling both and loading matching v2
weights reproduces the v2 function exactly; the default v7 fusion is different
from v2 from initialization. Use the unmodified `pid_unet_v2` for the baseline.
All ablations should use the same loss and training split.
