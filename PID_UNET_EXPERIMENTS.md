# PID-UNet v2 experiments

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
paper experiments. Logs include split filenames and config. Existing output
names are reused: archive results before rerunning the same model and seed.

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
