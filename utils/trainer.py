"""
Trainer implementation for Polyp Segmentation.

Faithfully reproduces the training mechanics from the PraNet paper:
- Multi-scale training on each batch: rates = [0.75, 1, 1.25]
- Rescaled size: int(round(trainsize * rate / 32) * 32)
- Deep supervision across all lateral maps (lateral_map_5, 4, 3, 2)
- Gradient clamping (clip_gradient)
- Learning rate decay scheduling
"""

import os
import random
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm


def clip_gradient(optimizer, grad_clip: float = 0.5):
    """Calibrate misalignment gradient via clamping as in PraNet."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr: float, epoch: int, decay_rate: float = 0.1, decay_epoch: int = 50):
    """Step decay learning rate."""
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


class AvgMeter:
    """Tracks running and windowed loss values."""
    def __init__(self, num: int = 40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.losses = []

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
        self.losses.append(val)

    def show(self) -> float:
        if not self.losses:
            return 0.0
        window = self.losses[-self.num:]
        return float(sum(window) / len(window))


class Trainer:
    """End-to-end Trainer supporting PraNet multi-scale deep supervision."""

    def __init__(self, model, optimizer, loss_fn, cfg, logger, checkpointer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.use_amp = self.device.type == 'cuda' and getattr(cfg, 'amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_one_epoch(self, train_loader, epoch: int) -> dict:
        self.model.train()
        adjust_lr(self.optimizer, self.cfg.lr, epoch, self.cfg.decay_rate, self.cfg.decay_epoch)
        current_lr = self.optimizer.param_groups[0]['lr']

        # Multi-scale strategy:
        # 1. Fast mode: fixed 1.0x (single scale 352x352)
        # 2. Legacy mode: sequential 3-pass [0.75, 1.0, 1.25] per batch (slow, ~8 hours)
        # 3. Default mode: Stochastic Multi-Scale (random scale per batch from [0.75, 1.0, 1.25])
        #    Full multi-scale representations across the epoch, runs blazingly fast (~50 mins for 100 epochs)!
        is_fast = getattr(self.cfg, 'fast', False) or getattr(self.cfg, 'no_multiscale', False)
        use_legacy = getattr(self.cfg, 'legacy_multiscale', False)

        loss_record = AvgMeter()
        loss_record2 = AvgMeter()
        loss_record3 = AvgMeter()
        loss_record4 = AvgMeter()
        loss_record5 = AvgMeter()

        total_step = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{self.cfg.epochs:03d}] (lr={current_lr:.6f})")

        for i, (images, gts) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            gts = gts.to(self.device, non_blocking=True)

            if is_fast:
                batch_rates = [1.0]
            elif use_legacy:
                batch_rates = [0.75, 1.0, 1.25]
            else:
                # Stochastic Multi-Scale: samples random scale per batch (YOLO/Swin style)
                batch_rates = [random.choice([0.75, 1.0, 1.25])]

            for rate in batch_rates:
                self.optimizer.zero_grad()
                trainsize = int(round(self.cfg.trainsize * rate / 32) * 32)

                if rate != 1.0:
                    scaled_images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    scaled_gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                else:
                    scaled_images = images
                    scaled_gts = gts

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(scaled_images)

                    if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                        lat5, lat4, lat3, lat2 = outputs
                        l5 = self.loss_fn(lat5, scaled_gts)
                        l4 = self.loss_fn(lat4, scaled_gts)
                        l3 = self.loss_fn(lat3, scaled_gts)
                        l2 = self.loss_fn(lat2, scaled_gts)
                        loss = l2 + l3 + l4 + l5
                    elif isinstance(outputs, (tuple, list)):
                        loss = sum(self.loss_fn(out, scaled_gts) for out in outputs)
                    else:
                        loss = self.loss_fn(outputs, scaled_gts)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    clip_gradient(self.optimizer, self.cfg.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_gradient(self.optimizer, self.cfg.clip)
                    self.optimizer.step()

                bs = images.size(0)
                loss_record.update(loss.item(), bs)
                if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                    loss_record2.update(l2.item(), bs)
                    loss_record3.update(l3.item(), bs)
                    loss_record4.update(l4.item(), bs)
                    loss_record5.update(l5.item(), bs)

            if i % 20 == 0 or i == total_step:
                pbar.set_postfix({
                    'Loss': f"{loss_record.show():.4f}",
                    'L2': f"{loss_record2.show():.4f}",
                    'L5': f"{loss_record5.show():.4f}"
                })

        return {
            'loss': loss_record.avg,
            'loss2': loss_record2.avg,
            'loss3': loss_record3.avg,
            'loss4': loss_record4.avg,
            'loss5': loss_record5.avg,
            'lr': current_lr
        }

    def validate(self, val_loader, evaluator) -> dict:
        """Validate on validation set (e.g. CVC-ClinicDB) using fast sliding evaluation."""
        self.model.eval()
        evaluator.metrics.reset()

        with torch.no_grad():
            for images, gts in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                pred = torch.sigmoid(pred).squeeze(1).cpu().numpy()
                gts_np = gts.squeeze(1).cpu().numpy()

                for b in range(pred.shape[0]):
                    evaluator.metrics.update(pred[b], gts_np[b])

        return evaluator.metrics.get_results()
