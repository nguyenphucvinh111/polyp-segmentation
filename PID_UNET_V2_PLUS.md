# PID-UNet v2 Plus

This is an architecture experiment based directly on the measured v2 model.
It targets foreground/background discrimination across all five datasets,
especially CVC-300. It has not been trained on the polyp dataset locally;
exceeding EDANet is the experimental objective, not an achieved result.

## Architecture

The encoder, global context bottleneck, four PID blocks, segmentation heads
and boundary branch are inherited from v2. At each of the three finer decoder
stages, the next coarser head supplies a soft foreground/background mask.
Those two regions aggregate decoder features into two context tokens. Local
features query the tokens, then a small residual block combines local features,
context and their difference. The coarse segmentation heads already have deep
supervision, so this requires no additional target or loss.

The last residual is computed at half resolution to control memory and compute.
The full-resolution v2 detail path remains available. Residual output projections
start at zero: with matching base weights, the initialized model gives exactly
v2's predictions. This is not a guarantee that training will preserve accuracy.
There is no raw-RGB refinement branch, ensemble or dataset-specific threshold.

Region aggregation follows the idea of [Object-Contextual Representations
(OCR)](https://arxiv.org/abs/1909.11065). This implementation is a binary,
cross-scale adaptation written for PID-UNet, not a reproduction of OCR. Cite
the underlying work when describing the context module in a paper.

| Model | Parameters | THOP GMACs at 352 x 352 |
|---|---:|---:|
| v2 | 835,373 | 6.6414 |
| v2 Plus | 858,245 | 7.0540 |

THOP does not include the functional context matrix products (approximately
0.009 additional GMACs) or all elementwise operations. GPU latency, peak training
memory and accuracy must be measured on Colab; the local machine has CPU only.

## Train on Colab

```python
!git pull origin main
!python train.py --model pid_unet_v2_plus --loss structure_loss --epochs 100 --batchsize 8 --seed 42
```

Start from scratch for the direct comparison. `--resume` requires a checkpoint
from `pid_unet_v2_plus`; checkpoints from v2 or older variants are rejected.
Results, logs and snapshots are stored under `pid_unet_v2_plus`. Existing v2,
v5 and v6 files remain available under their original model names.

Training settings match v2: four structure losses, the existing boundary loss
with weight 0.2, AdamW at 3e-4, five warmup epochs, cosine decay, 352 base size,
stochastic multi-scale training, and copy-paste disabled. The existing seed-42
1305/145 split and validation checkpoint selection are unchanged. No new loss,
augmentation or metric normalization is introduced.

For a final multi-seed comparison, run the same recipe with `--seeds 42 3407 1234`
for both v2 and v2 Plus. The Python constructor supports `context_stages=()` or
subsets of `(3, 2, 1)` for ablation. Choose the configuration on validation.

## EDANet reference and acceptance

The user-supplied PDF reports five metrics, rounded to three decimals:

| Dataset | mDice > | mIoU > | Sm > | Em > | MAE < |
|---|---:|---:|---:|---:|---:|
| CVC-300 | .848 | .758 | .878 | .922 | .014 |
| ClinicDB | .815 | .736 | .852 | .899 | .031 |
| ColonDB | .719 | .625 | .759 | .829 | .049 |
| ETIS | .557 | .463 | .603 | .719 | .053 |
| Kvasir | .875 | .802 | .900 | .917 | .045 |

Numerically, the supplied v2 run exceeds 19/25 entries. The six remaining entries
are all five CVC-300 metrics and ColonDB MAE. MAE should decrease, not increase.
The PDF does not provide EDANet Recall, Precision, Specificity, HD95 or wFb;
there is no basis for an EDANet comparison on those metrics yet.

After the fixed run completes:

```python
!python compare_edanet.py --metrics results/pid_unet_v2_plus/metrics.csv
```

This checks every dataset/metric pair, reports ties as ties rather than wins,
and rejects missing datasets/invalid scores. It only reads completed results;
it is not a checkpoint selector. Match data splits, inference and metric
definitions before claiming superiority. Existing Sm/Em/wFb implementation
validation is still outstanding; a numeric comparison with the PDF alone is
not proof of comparable evaluation or statistical significance.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover exact initial equivalence to v2, four output resolutions including
256/352/448 training sizes, foreground/background discrimination, saturated
coarse predictions, unchanged training loss with actual optimizer updates,
gradients through all new context stages, checkpoint round-trip and architecture
mismatch rejection. CPU bfloat16 autocast is covered; the CUDA FP16 test skips
when no GPU is available. No real-dataset accuracy is implied by these tests.
