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
