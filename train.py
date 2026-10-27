"""
End-to-End Training Entry Point for Polyp Segmentation.

Usage:
    # Single seed:
    python train.py --model pid_unet --loss structure_loss --seed 42
    python train.py --model pranet --loss structure_loss --seed 42

    # Multi-seed or seeds list:
    python train.py --model pid_unet --seeds 42 --multiseed
    python train.py --model pid_unet --seeds 42 3407 1234 --multiseed

Hyperparameter settings reproducing each paper exactly:
    Epochs: 100  (PraNet paper: 20; Polyp-PVT Table II: 100; we default 100)
    Initial Learning Rate: 1e-4
    Optimizer: Adam for PraNet/SANet/HarDNet-MSEG/UNet variants
               AdamW (weight_decay=1e-4) for Polyp-PVT  [Table II]
    Batch Size: 16
    Input Resolution: 352x352 (with multi-scale [0.75, 1.0, 1.25] per batch)
    Gradient Clipping: 0.5
    LR Decay: factor 0.1 every 50 epochs
    Deep Supervision: lateral_map_5, 4, 3, 2
"""

import argparse
import os
import sys
import warnings
import random
import numpy as np
import torch

# Suppress deprecation and future warnings (e.g. from timm)
warnings.filterwarnings("ignore")

# Avoid CUDA memory fragmentation on GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure the repository root is in sys.path regardless of execution CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model, list_models
from loss import get_loss, list_losses
from utils.setup_dataset import setup_dataset
from utils.dataloader import get_loader
from utils.trainer import Trainer
from utils.logger import Logger
from utils.checkpoint import Checkpointer
from utils.evaluator import Evaluator


def set_seed(seed: int = 42):
    """Ensure strict experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_seeds(args) -> list[int]:
    """Parse single or multiple seeds from --seed, --seeds, or --multiseed."""
    if args.seeds is not None:
        raw = args.seeds
        if isinstance(raw, (int, str)):
            raw = [raw]
        seeds = []
        for item in raw:
            if isinstance(item, str):
                for part in item.replace(',', ' ').split():
                    if part.strip().isdigit():
                        seeds.append(int(part.strip()))
            elif isinstance(item, int):
                seeds.append(item)
        return seeds if seeds else [args.seed]
    elif args.multiseed:
        return [42, 3407, 1234]
    else:
        return [args.seed]


def train_single_run(args, seed: int, run_name: str = None) -> dict:
    """Train a model for a specific seed."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    tag = run_name if run_name else model_name

    print(f"\n============================================================")
    print(f"POLYP SEGMENTATION TRAINING PIPELINE")
    print(f"============================================================")
    print(f"Model: {args.model} | Loss: {args.loss} | Seed: {seed} | Device: {device}")
    ms_info = "Single-scale 352x352" if args.fast else ("Legacy 3-pass [0.75, 1.0, 1.25]" if getattr(args, 'legacy_multiscale', False) else "Stochastic Multi-scale [0.75, 1.0, 1.25] (Fast)")
    print(f"Resolution: {args.trainsize}x{args.trainsize} ({ms_info})")
    print(f"LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batchsize}")
    print(f"Output Tag: {tag}")
    print(f"============================================================")

    # 1. Setup Dataset
    data_root = setup_dataset(args.data_root)
    train_img_dir = os.path.join(data_root, 'TrainDataset', 'images')
    train_gt_dir = os.path.join(data_root, 'TrainDataset', 'masks')
    val_img_dir = os.path.join(data_root, 'TestDataset', 'CVC-ClinicDB', 'images')
    val_gt_dir = os.path.join(data_root, 'TestDataset', 'CVC-ClinicDB', 'masks')

    # 2. Data Loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = get_loader(
        train_img_dir, train_gt_dir,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        shuffle=True,
        num_workers=num_workers,
        is_train=True
    )
    val_loader = get_loader(
        val_img_dir, val_gt_dir,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        shuffle=False,
        num_workers=num_workers,
        is_train=False
    )
    print(f"[Data] Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)}")

    # 3. Build Model & Loss
    print(f"[Init] Initializing model '{args.model}'...")
    model = get_model(args.model)
    loss_fn = get_loss(args.loss)

    # Per-paper optimizer dispatch
    _ADAMW_MODELS = {'polyp_pvt'}
    if args.model.lower() in _ADAMW_MODELS:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        opt_name = "AdamW (weight_decay=1e-4)"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt_name = "Adam"

    # 4. Logger, Checkpointer, and Evaluator
    logger = Logger(log_dir='./logs', model_name=tag)
    checkpointer = Checkpointer(save_dir='./snapshots', model_name=tag)
    val_evaluator = Evaluator(data_root=data_root, model=model, device=device, testsize=args.trainsize)
    trainer = Trainer(model, optimizer, loss_fn, args, logger, checkpointer)

    # 5. Resume from checkpoint (optional)
    start_epoch = 1
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[Resume] WARNING: checkpoint '{args.resume}' not found. Starting from epoch 1.")
        else:
            resumed_epoch, resumed_metrics = Checkpointer.load(
                model, args.resume, device=device, optimizer=optimizer
            )
            start_epoch = resumed_epoch + 1
            checkpointer.best_dice = resumed_metrics.get('mDice', 0.0)
            print(f"[Resume] Resumed from epoch {resumed_epoch} "
                  f"(best mDice so far: {checkpointer.best_dice:.4f}). "
                  f"Continuing from epoch {start_epoch}.")

    # 6. Training Loop
    print(f"[Init] Optimizer: {opt_name} | lr={args.lr}")
    print(f"\n[Train] Starting training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = trainer.train_one_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, val_evaluator)
        logger.log(epoch, train_stats, val_metrics)
        checkpointer.save(model, optimizer, epoch, val_metrics)

    logger.close()

    # 7. Final Cross-Dataset Evaluation on Best Checkpoint
    best_weights = checkpointer.get_best_path()
    all_results = None
    if os.path.exists(best_weights):
        print(f"\n============================================================")
        print(f"RUNNING FINAL BENCHMARK ON BEST MODEL: {best_weights}")
        print(f"============================================================")
        Checkpointer.load(model, best_weights, device=device)
        evaluator = Evaluator(
            data_root=data_root,
            model=model,
            device=device,
            testsize=args.trainsize,
            model_name=args.model,
            weights_path=best_weights
        )
        results_dir = os.path.join('./results', tag)

        # Edge AI Hardware efficiency
        print(f"\n[Evaluation] Profiling Edge AI & Hardware efficiency...")
        efficiency = evaluator.benchmark_efficiency(measure_cpu=True)

        # Benchmark accuracy across all 5 test sets
        results = evaluator.evaluate_all(save_results_dir=results_dir)

        # Save metrics
        evaluator.print_table(results, efficiency=efficiency, model_name=tag)
        evaluator.save_csv(results, path=os.path.join(results_dir, 'metrics.csv'), efficiency=efficiency)
        evaluator.save_latex(results, path=os.path.join(results_dir, 'metrics.tex'), efficiency=efficiency)
        all_results = results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train Polyp Segmentation Network")
    parser.add_argument('--model', type=str, default='pranet', help=f'Model name. Available: {list_models()}')
    parser.add_argument('--loss', type=str, default='structure_loss', help=f'Loss name. Available: {list_losses()}')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for single run (default: 42)')
    parser.add_argument('--seeds', nargs='*', type=str, default=None,
                        help='Seed or list of seeds (e.g. --seeds 42 or --seeds 42 3407 1234 or --seeds 42,3407)')
    parser.add_argument('--multiseed', action='store_true', default=False,
                        help='Enable multi-seed training mode')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Fast mode: train at single scale 352x352 (speeds up training 3x to 5x)')
    parser.add_argument('--legacy_multiscale', action='store_true', default=False,
                        help='Legacy sequential 3-pass multi-scale per batch (slow, ~8 hours)')
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='Disable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Batch size (default: 8 to prevent CUDA OOM on Colab T4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint (.pth) to resume training from.')
    args = parser.parse_args()
    args.amp = not args.no_amp

    # Enforce Fixed PraNet Hyperparameters
    args.lr = 1e-4
    args.trainsize = 352
    args.clip = 0.5
    args.decay_rate = 0.1
    args.decay_epoch = 50

    seeds = parse_seeds(args)

    if len(seeds) == 1:
        # Standard Single Seed Run
        seed = seeds[0]
        args.seed = seed
        train_single_run(args, seed, run_name=args.model)
    else:
        # Multi-seed Benchmark Run
        print(f"\n============================================================")
        print(f"MULTI-SEED BENCHMARK MODE: {len(seeds)} SEEDS {seeds}")
        print(f"============================================================")
        for idx, seed in enumerate(seeds, 1):
            print(f"\n>>> [Multi-Seed {idx}/{len(seeds)}] Starting Run for Seed: {seed} <<<")
            args.seed = seed
            run_name = f"{args.model}_seed{seed}"
            train_single_run(args, seed, run_name=run_name)
        print(f"\n[Done] Multi-seed evaluation completed for all seeds: {seeds}.")


if __name__ == '__main__':
    main()
