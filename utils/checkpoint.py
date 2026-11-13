"""
Checkpoint Manager for Polyp Segmentation.

Saves regular snapshots and tracks the best model based on validation mDice.
Robust loading handles diverse checkpoint formats (raw weights, wrapped dict, DataParallel prefixes).
"""

import os
import torch


class Checkpointer:
    """Manages saving and loading model checkpoints."""

    def __init__(self, save_dir: str = './snapshots', model_name: str = 'pranet'):
        self.save_dir = os.path.join(save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_name = model_name
        self.best_dice = -1.0
        self.best_epoch = 0

    def save(self, model, optimizer, epoch: int, metrics: dict):
        current_dice = metrics.get('mDice', 0.0)
        state = {
            'epoch': epoch,
            'model_name': self.model_name,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'best_dice': max(self.best_dice, current_dice)
        }

        # Save periodic snapshot (every 10 epochs or last epoch)
        if epoch % 10 == 0:
            periodic_path = os.path.join(self.save_dir, f"{self.model_name}-{epoch}.pth")
            torch.save(state, periodic_path)
            print(f"[Checkpointer] Saved periodic snapshot to '{periodic_path}'.")

        # Save best model
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.best_epoch = epoch
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(state, best_path)
            print(f"[Checkpointer] New best model at epoch {epoch} with mDice={current_dice:.4f} -> Saved to '{best_path}'.")

    @staticmethod
    def load(model, path: str, device=None, optimizer=None):
        """
        Load weights safely handling raw state_dict, wrapped state_dict, or 'module.' prefixes.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            sd = checkpoint['model']
        elif isinstance(checkpoint, dict):
            sd = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format at '{path}'.")

        # Strip 'module.' prefix if saved via nn.DataParallel
        clean_sd = {}
        for k, v in sd.items():
            clean_k = k[7:] if k.startswith('module.') else k
            clean_sd[clean_k] = v

        model_sd = model.state_dict()
        matched_sd = {}
        for k, v in clean_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                matched_sd[k] = v

        missing = set(model_sd.keys()) - set(matched_sd.keys())
        unexpected = set(clean_sd.keys()) - set(model_sd.keys())

        if getattr(model, 'strict_checkpoint_loading', False) and (missing or unexpected):
            raise ValueError(
                f"Checkpoint does not match {model.__class__.__name__}: "
                f"{len(missing)} missing/mismatched keys, {len(unexpected)} unexpected keys. "
                "Use a checkpoint from the same architecture. Train a new model without --resume."
            )

        model.load_state_dict(matched_sd, strict=False)
        print(f"[Checkpointer] Loaded {len(matched_sd)}/{len(model_sd)} layers from '{path}'.")
        if missing:
            print(f"[Checkpointer] Note: {len(missing)} missing keys (e.g. {list(missing)[:3]}...)")
        if unexpected:
            print(f"[Checkpointer] Note: {len(unexpected)} unexpected keys in checkpoint.")

        if optimizer and isinstance(checkpoint, dict) and checkpoint.get('optimizer'):
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"[Checkpointer] Optimizer state could not be restored ({e}).")

        epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
        metrics = checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
        return epoch, metrics

    def get_best_path(self) -> str:
        return os.path.join(self.save_dir, "best.pth")
