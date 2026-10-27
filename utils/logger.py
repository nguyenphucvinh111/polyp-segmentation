"""
Logging & Plotting Utilities for Polyp Segmentation.

Logs per-epoch metrics to CSV and automatically renders publication-grade training loss curves.
"""

import os
import csv

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Logger:
    """Manages CSV logging and loss/metric curve generation."""

    def __init__(self, log_dir: str = './logs', model_name: str = 'pranet'):
        self.model_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.csv_path = os.path.join(self.model_dir, 'train_log.csv')
        self.curve_path = os.path.join(self.model_dir, 'training_curves.png')

        self.fieldnames = [
            'epoch', 'lr', 'loss', 'loss2', 'loss3', 'loss4', 'loss5',
            'mDice', 'mIoU', 'wFb', 'Sm', 'Em', 'MAE'
        ]

        # Initialize CSV with header if new
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        self.history = []

    def log(self, epoch: int, train_loss_dict: dict, val_metrics_dict: dict):
        """Record epoch statistics and update plots."""
        row = {'epoch': epoch}
        row['lr'] = train_loss_dict.get('lr', 0.0)
        row['loss'] = train_loss_dict.get('loss', 0.0)
        row['loss2'] = train_loss_dict.get('loss2', 0.0)
        row['loss3'] = train_loss_dict.get('loss3', 0.0)
        row['loss4'] = train_loss_dict.get('loss4', 0.0)
        row['loss5'] = train_loss_dict.get('loss5', 0.0)

        for k in ['mDice', 'mIoU', 'wFb', 'Sm', 'Em', 'MAE']:
            row[k] = val_metrics_dict.get(k, 0.0)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        self.history.append(row)

        # Print summary line
        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {row['loss']:.4f} | "
            f"Val mDice: {row['mDice']:.4f} | "
            f"Val mIoU: {row['mIoU']:.4f} | "
            f"Val MAE: {row['MAE']:.4f}"
        )

        # Plot curves
        self._plot_curves()

    def _plot_curves(self):
        if not HAS_MATPLOTLIB or len(self.history) < 2:
            return

        epochs = [r['epoch'] for r in self.history]
        losses = [r['loss'] for r in self.history]
        dices = [r['mDice'] for r in self.history]
        ious = [r['mIoU'] for r in self.history]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(epochs, losses, color=color, marker='o', label='Total Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Metrics', color='tab:blue')
        ax2.plot(epochs, dices, color='tab:blue', marker='s', label='Val mDice')
        ax2.plot(epochs, ious, color='tab:green', marker='^', label='Val mIoU')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.tight_layout()
        plt.title('Training & Validation Trajectory')
        plt.savefig(self.curve_path, dpi=200)
        plt.close(fig)

    def close(self):
        print(f"[Logger] All epoch logs saved to '{self.csv_path}'. Curves saved to '{self.curve_path}'.")
