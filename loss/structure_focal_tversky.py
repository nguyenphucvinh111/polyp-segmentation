"""
Compound Loss: Boundary Structure Loss + Small-Lesion Focal Tversky Loss.

Specifically engineered to solve:
1. ETIS-LaribPolypDB diminutive lesions & false negatives.
2. Boundary sharpness & low Hausdorff distance (HD95).
3. Foreground-background extreme imbalance in diminutive polyps (< 20px).

Formulation:
    Loss = Structure_Loss(pred, mask) + 0.5 * Focal_Tversky_Loss(pred, mask, alpha=0.3, beta=0.7, gamma=0.75)
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('structure_focal_tversky')
@register_loss('pid_loss')
def structure_focal_tversky_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    ftl_weight: float = 0.5
) -> torch.Tensor:
    """
    Compound loss combining PraNet weighted structure loss with Small-Object Focal Tversky.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    # 1. PraNet Structure Loss (Boundary-Weighted BCE + Weighted IoU)
    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-8)

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    s_loss = (wbce + wiou).mean()

    # 2. Focal Tversky Loss (Heavy penalty on False Negatives for diminutive lesions)
    b = pred.size(0)
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)

    true_pos = (p_flat * g_flat).sum(dim=1)
    false_pos = (p_flat * (1.0 - g_flat)).sum(dim=1)
    false_neg = ((1.0 - p_flat) * g_flat).sum(dim=1)

    smooth = 1.0
    ti = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    ftl = torch.pow((1.0 - ti).clamp(min=1e-7, max=1.0), gamma).mean()

    return s_loss + ftl_weight * ftl
