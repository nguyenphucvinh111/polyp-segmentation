"""Experimental precision-oriented training objective; no inference changes."""
import torch
from loss import register_loss
from loss.structure_loss import structure_loss

@register_loss("structure_precision")
def structure_precision(logits, target):
    # Float32 reductions remain stable under CUDA autocast.
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    dims = (2, 3)
    tp = (probability * target).sum(dims)
    fp = (probability * (1 - target)).sum(dims)
    fn = ((1 - probability) * target).sum(dims)
    # Mild FP emphasis; retain the original objective to preserve coverage.
    tversky = 1 - (tp + 1) / (tp + .6 * fp + .4 * fn + 1)
    return structure_loss(logits, target) + .3 * tversky.mean()
