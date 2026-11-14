# PID-UNet v2 experiments

Current architecture experiments: [PID-UNet v8 and v9](PID_UNET_V8_V9.md),
independent v2 variants for integral gating and shallow mixed normalization.
Both train from scratch with the same loss and training settings.
[v7](PID_UNET_V7.md) and [v2 Plus](PID_UNET_V2_PLUS.md) remain available. Earlier variants below
are retained for comparison; their names and checkpoints are unchanged.

This is an experimental architecture, not a demonstrated accuracy improvement.
835,373 parameters; no pretrained backbone. Output remains four logits; optional
`return_aux=True` returns `(segmentation_logits, boundary_logits)`.

## Colab T4

```bash
pip install -r requirements.txt
python train.py --model pid_unet_v2 --loss structure_loss --epochs 100 --batchsize 8 --seed 42
python train.py --model pid_unet_v2 --loss structure_loss --epochs 100 --batchsize 8 --seeds 42 3407 1234
```

v2 defaults: AdamW, lr 0.0003, weight decay 0.0001, five warmup epochs,
cosine decay, boundary loss weight 0.2, no copy-paste. AMP and stochastic
multiscale remain enabled. Override with --lr, --scheduler, --warmup_epochs,
--weight_decay and --boundary_weight. Use --boundary_weight 0 for boundary
supervision ablation; Python constructor accepts bounded_gates=False and
global_context=False for architecture ablations.

A fixed seed-42 90/10 image split is taken from TrainDataset for all models.
Test ClinicDB no longer selects checkpoints. This split is NOT patient/video
grouped or source-stratified: recover metadata and check duplicates before
paper experiments. Logs include split filenames and config. A fresh run refuses
to overwrite existing training output; use a new --run_name or --resume.
Use --output_root with a mounted Drive directory to keep Colab outputs.

Compare the original model under the same validation protocol:

```bash
python train.py --model pid_unet --loss structure_loss --epochs 100 --batchsize 8 --seeds 42 3407 1234
```

Report all seeds, dataset-level results and efficiency with TTA disabled.
Current wFb/Sm/Em implementations still require reference validation before
publication. Validation uses resized masks; final benchmark uses original
resolution, so the two scores are not directly interchangeable. No dataset or
trained checkpoints were available during local implementation.

## v3: uncertainty-guided residual detail refinement

v3 adds a small RGB detail branch conditioned on coarse probability,
uncertainty and supervised boundary probability. Its signed residual can add
or remove foreground; attention has a 0.25 floor. Training settings match v2
for a direct architecture comparison. Original v2 behavior is unchanged.

```bash
python train.py --model pid_unet_v3 --loss structure_loss --epochs 100 --batchsize 8 --seed 42
```

Use a separate Colab GPU runtime for concurrent v2/v3 runs. Running both on
one T4 competes for VRAM and compute. Outputs use separate model directories.
Do not reuse a v2 checkpoint with --resume for v3. The Python constructor
supports refinement=False for ablation. v3 has not yet been trained on real
data; compare validation before choosing it, then evaluate the locked model.

## v4: decoder feature context refinement

The v3 run improved CVC-300 (0.8370 versus v2 0.8294), but its ETIS
Dice fell to 0.5202. v4 returns to v2 and adds local/dilated decoder-feature
refinement instead of direct RGB refinement. A signed tanh correction is
bounded to one logit and initialized to zero. This is a hypothesis about
foreground/background discrimination, not a demonstrated improvement.
Training and validation match v2. No test-set threshold tuning is used.

```bash
python train.py --model pid_unet_v4 --loss structure_loss --epochs 100 --batchsize 8 --seed 42
```

Train from scratch; do not resume a v2/v3 checkpoint as v4. Select checkpoint
on held-out validation and report all five test sets once configuration is
locked. Target CVC-300 Dice >0.848 is a reference to the supplied EDANet table;
matching its protocol remains necessary. Archive prior runs and compare three
seeds before making a paper claim. Raw per-image masks are needed to determine
whether excess predictions or missed lesions explain CVC-300 errors.

## Preserve v2: precision-oriented loss experiment

Keep architecture, augmentation, boundary supervision, scheduler and resolution
identical to v2. Only the segmentation objective changes: original structure
loss plus 0.3 * Tversky loss (FP weight 0.6, FN weight 0.4). These constants
are a fixed hypothesis, not selected on CVC-300 test scores. Parameter count
and inference cost are unchanged. There is no guarantee of improvement;
recall may fall. Select on the existing validation split and report all sets.

```bash
python train.py --model pid_unet_v2 --loss structure_precision --run_name pid_unet_v2_precision --epochs 100 --batchsize 8 --seed 42
```

Train from scratch. Outputs go to snapshots/pid_unet_v2_precision,
logs/pid_unet_v2_precision and results/pid_unet_v2_precision; original v2
checkpoint and model code are preserved. --run_name also prefixes seed tags
for multi-seed runs. This is a training ablation, not a new architecture.

## v5: architecture-only spatial PID experiment

Uses structure_loss and the SAME boundary supervision weight, optimizer,
scheduler, augmentation and resolution as v2. No precision loss is used.
v2 model code and checkpoints remain unchanged. v5 replaces global-only PID
gains by global priors plus bounded spatial corrections at each decoder scale.
Corrections start at zero, retaining the initialized global gains. A spatial
allocation can weight detail/context/edges differently in foreground and
background; whether this improves CVC-300 is an untested hypothesis.

```bash
python train.py --model pid_unet_v5 --loss structure_loss --epochs 100 --batchsize 8 --seed 42
```

Train from scratch, not --resume from another architecture. All outputs use
pid_unet_v5 directories. Python spatial_pid=False disables the new controller
for ablation. Keep validation checkpoint selection unchanged; do not tune
thresholds or select epochs against CVC-300. Report all five test sets and
multiple seeds. Baseline v2 CVC-300 Dice 0.8294; EDANet PDF reference 0.848.
The earlier optional structure_precision experiment is not part of v5.

## v6: independent channel-fusion architecture ablation

v6 starts from v2, not v5. Each decoder block concatenates weighted P/I/D
features and learns a 1x1 channel projection before the existing SE/refinement.
The projection starts as three identity matrices, reproducing v2 addition
at initialization. This retains branch identity for learned fusion rather
than irrevocably summing channels. The new interaction may improve foreground
and background discrimination, but superiority over EDANet is unverified.
All losses (structure_loss plus existing boundary supervision), data and
training settings match v2/v5; no CVC-300 threshold tuning is introduced.

```bash
python train.py --model pid_unet_v6 --loss structure_loss --epochs 100 --batchsize 8 --seed 42
```

Use a separate GPU runtime for concurrent runs. Train from scratch; outputs
use pid_unet_v6. Compare validation and report all datasets and multiple seeds.
Parameter budget remains below 1M; measure latency rather than assuming it
matches v2. v2 and v5 source and checkpoints are unchanged.
