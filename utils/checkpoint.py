"""
Checkpoint Manager for Polyp Segmentation.

Saves regular snapshots and tracks the best model based on validation mDice.
Robust loading handles diverse checkpoint formats (raw weights, wrapped dict, DataParallel prefixes).
"""

import os
import shutil
import torch


class Checkpointer:
    """Manages saving and loading model checkpoints."""

    def __init__(self, save_dir: str = './snapshots', model_name: str = 'pranet'):
        self.save_dir = os.path.join(save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_name = model_name
        self.best_dice = -1.0
        self.best_epoch = 0

    @staticmethod
    def _save_file(state, path):
        path = os.fspath(path)
        temporary = path + '.tmp'
        torch.save(state, temporary)
        os.replace(temporary, path)

    def save(self, model, optimizer, epoch: int, metrics: dict, training_state=None):
        current_dice = metrics.get('mDice', 0.0)
        improved = current_dice > self.best_dice
        state = {
            'epoch': epoch,
            'model_name': self.model_name,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'best_dice': max(self.best_dice, current_dice),
            'best_epoch': epoch if improved else self.best_epoch,
            'training_state': training_state,
        }

        # Save best model
        if improved:
            best_path = os.path.join(self.save_dir, "best.pth")
            self._save_file(state, best_path)
            self.best_dice = current_dice
            self.best_epoch = epoch
            print(f"[Checkpointer] New best model at epoch {epoch} with mDice={current_dice:.4f} -> Saved to '{best_path}'.")

        self._save_file(state, os.path.join(self.save_dir, 'last.pth'))
        if epoch % 10 == 0:
            periodic_path = os.path.join(self.save_dir, f"{self.model_name}-{epoch}.pth")
            self._save_file(state, periodic_path)
            print(f"[Checkpointer] Saved periodic snapshot to '{periodic_path}'.")

    @staticmethod
    def load(model, path: str, device=None, optimizer=None, trainer=None):
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
                if trainer is not None:
                    raise ValueError('Cannot resume with an incompatible optimizer state') from e
                print(f"[Checkpointer] Optimizer state could not be restored ({e}).")

        epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
        metrics = dict(checkpoint.get('metrics', {})) if isinstance(checkpoint, dict) else {}
        if isinstance(checkpoint, dict):
            metrics['best_dice'] = checkpoint.get('best_dice', metrics.get('mDice', 0.0))
            metrics['best_epoch'] = checkpoint.get('best_epoch', epoch)
            if trainer is not None and checkpoint.get('training_state'):
                trainer.load_training_state(checkpoint['training_state'])
        return epoch, metrics

    def get_best_path(self) -> str:
        return os.path.join(self.save_dir, "best.pth")

    def restore_best(self, resume_path, resumed_epoch, resumed_metrics):
        """Track the best file actually available, including after an interrupted save."""
        best_path = self.get_best_path()
        if not os.path.isfile(best_path):
            source_best = os.path.join(os.path.dirname(os.path.abspath(resume_path)), 'best.pth')
            if os.path.isfile(source_best):
                shutil.copy2(source_best, best_path)
            else:
                shutil.copy2(resume_path, best_path)
                print('[Resume] Historical best unavailable; using resumed weights as the initial best candidate.')
        # best.pth may be newer than last.pth if writing last was interrupted.
        # Trust the metrics of the stored best weights, not last's cached best_dice.
        best = torch.load(best_path, map_location='cpu')
        self.best_dice = best.get('metrics', {}).get('mDice', resumed_metrics.get('mDice', 0.0))
        self.best_epoch = best.get('epoch', resumed_epoch)
