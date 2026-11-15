# PID-UNet v2 OS8

Model name: **`pid_unet_v2_os8`**. A controlled architecture experiment based
directly on the original v2, trained entirely from scratch. No pretrained
encoder, new attention block, loss change, or additional training images.
The original `pid_unet_v2` remains available and unchanged.

## Motivation and limits

The supplied v10 CVC-300 results show recall 0.9688 but precision 0.7397,
versus v2 recall 0.9455 and precision 0.7807. v10 Dice is 0.8132, below v2
0.8294 and the supplied EDANet reference 0.848. Higher recall alone did not
improve overlap.

The earlier v8/v9 checkpoint analysis found small lesions confused with
adjacent bright background, including errors already present in coarse heads.
This motivates testing denser semantic features. It does not establish that
downsampling caused those errors, or that OS8 will solve them. No v10 checkpoint
was available for per-image analysis. OS8 has no completed full training run or
measured CVC-300 improvement yet.

## Architecture

At input size 352 x 352:

| Component | Original v2 | v2 OS8 |
|---|---|---|
| First three downsampling stages | Unchanged | Unchanged |
| Last convolution stride | 2 | 1 |
| Last pooling branch | MaxPool 2, stride 2 | MaxPool 3, stride 1, padding 1 |
| Deepest features | 192 x 22 x 22 | 192 x 44 x 44 |
| Bottleneck dilation rates | 1, 3, 6, 12 | 2, 6, 12, 24 |
| PID decoder, global context, boundary branch | Original v2 | Original v2 |
| Segmentation outputs | Four full-resolution logit maps | Same interface |
| Trainable parameters | 835,373 | 835,373 |
| THOP MACs at 352 x 352 | 6.6414 G | 7.0200 G |

The bottleneck now retains four times as many spatial positions. Doubling the
dilations compensates for the smaller feature stride to approximately preserve
the context range. The first decoder stage receives native /8 semantic features
instead of upsampling /16 features. v2 already has fine-resolution skip features;
this experiment specifically changes its deepest semantic path.

This uses the established output-stride/atrous-convolution approach described in
[DeepLabv3](https://arxiv.org/abs/1706.05587), not a claim of a new attention
mechanism. Pooling alignment, dilation sampling and padding also change; accuracy
may decrease. MACs increase by about 5.7%, but GPU latency and training memory
must be measured on the target GPU. MAC counts are not a measured FPS estimate.

## Training

Use the [Colab notebook](https://colab.research.google.com/github/nguyenphucvinh111/polyp-segmentation/blob/main/pid_unet_v2_os8_colab.ipynb)
to mount Drive and keep outputs across runtime resets. Select a GPU runtime.

```bash
python train.py --model pid_unet_v2_os8 --loss structure_loss --seeds 42 \
  --epochs 100 --batchsize 8 --output_root /content/drive/MyDrive/polyp_runs
```

The Drive directory must be mounted before running that command. For a local
run, omit `--output_root` or choose an existing local output root.

The v2 recipe is retained: AdamW, lr 0.0003, weight decay 0.0001, five warmup
epochs, cosine schedule, 352 input with stochastic multiscale, AMP on CUDA,
existing augmentations with copy-paste disabled, four structure-loss terms and
the existing boundary objective weighted by 0.2. The same fixed seed-42 90/10
TrainDataset split selects the checkpoint; test datasets do not select epochs.

Start a **new** run. Do not initialize or resume OS8 from a v2/v8/v9/v10 checkpoint.
To continue this OS8 run after disconnection:

```bash
python train.py --model pid_unet_v2_os8 --loss structure_loss --seeds 42 \
  --epochs 100 --batchsize 8 --output_root /content/drive/MyDrive/polyp_runs \
  --resume /content/drive/MyDrive/polyp_runs/snapshots/pid_unet_v2_os8/last.pth
```

Keep the same source revision and configuration; the notebook records these in
a run manifest and checks them before resuming. `--epochs 100` is the total
epoch count, including completed epochs. Use a different `--run_name` for a new
experiment so existing outputs are preserved.

## Checkpoints and comparison

Stride and dilation are not stored in PyTorch parameter tensors. v2 and OS8 have
identical parameter names and shapes, so shape checking alone cannot distinguish
them. OS8 saves a `model_architecture` identifier in the checkpoint; the loader
checks it before modifying weights or optimizer state and requires all tensors
to match. Keep the **complete `best.pth` / `last.pth` files**, not an extracted raw
state dictionary. Legacy checkpoints for existing models remain supported.

After training, keep `best.pth`, `last.pth`, `train_log.csv`, `metrics.csv`,
`efficiency.csv`, the split/config JSON and the run manifest. The notebook saves
these under `/content/drive/MyDrive/polyp_runs` and runs:

```bash
python compare_edanet.py --metrics /content/drive/MyDrive/polyp_runs/results/pid_unet_v2_os8/metrics.csv
```

Inspect CVC-300 Dice/IoU and precision/recall together, and report all five
datasets. The EDANet reference contains only Dice, IoU, Sm, Em and MAE; superiority
on the other metrics cannot be inferred from it. Full training, repeated seeds
and matched evaluation protocols are required for a paper claim. Code and
gradient checks do not establish segmentation accuracy.

## Local verification

The full suite ran 47 tests: 46 passed, one CUDA-only test skipped on this CPU
machine. OS8 checks cover multiscale outputs, finite gradients, the existing
boundary objective, CPU bfloat16 autocast, checkpoint compatibility and exact
next-step reproduction after restoring optimizer/RNG state. The training CLI
discovers the model and all four notebook code cells parse.

A separate CPU optimization check used 16 cached TrainDataset image/mask pairs
at 96 x 96 for 32 steps, with augmentation and multiscale disabled for this
check only. Training loss decreased from 6.5449 to 4.5340. This verifies learning
on that small training sample; it is not a validation or CVC-300 result. CUDA
training memory, FP16 behavior and full-run accuracy still require the Colab run.
