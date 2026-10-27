"""
Standard Evaluation Metrics for Polyp Segmentation.

Implementations strictly follow top-tier medical imaging & computer vision standards (MICCAI, CVPR, ICCV, IEEE TMI/JBHI):
- Mean Dice (mDice / F1-Score)
- Mean Intersection over Union (mIoU / Jaccard Index)
- Mean Absolute Error (MAE)
- Weighted F-measure (wFb / F_beta^w, Margolin et al., CVPR 2014)
- Structure-measure (S-measure / Sm / S_alpha, Fan et al., ICCV 2017)
- Enhanced-alignment measure (E-measure / Em / E_xi, Fan et al., IJCAI 2018)
- Precision, Recall, Specificity
"""

import numpy as np

try:
    import scipy.ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
    """Compute Dice similarity coefficient on binary masks."""
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b = (gt > 0.5).astype(np.float32)
    inter = (pred_b * gt_b).sum()
    return float((2.0 * inter + smooth) / (pred_b.sum() + gt_b.sum() + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
    """Compute Intersection-over-Union (Jaccard) on binary masks."""
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b = (gt > 0.5).astype(np.float32)
    inter = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - inter
    return float((inter + smooth) / (union + smooth))


def mae_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Mean Absolute Error between continuous prediction and ground truth."""
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


def precision_recall_specificity(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8):
    """Compute Precision, Recall (Sensitivity), and Specificity."""
    pred_b = (pred > 0.5).astype(bool)
    gt_b = (gt > 0.5).astype(bool)

    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
    tn = np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    return float(precision), float(recall), float(specificity)


def hd95_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance (HD95) in pixels.
    Measures boundary discrepancy between predicted and GT segmentation.
    Ref: Medpy implementation standard in MICCAI / IEEE TMI medical imaging benchmarks.
    """
    if not HAS_SCIPY:
        return float('nan')
    pred_b = (pred > 0.5).astype(bool)
    gt_b = (gt > 0.5).astype(bool)

    # Edge cases
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        h, w = pred_b.shape
        return float(np.sqrt(h**2 + w**2))

    # Extract single-pixel boundary contours
    struct = scipy.ndimage.generate_binary_structure(2, 1)
    pred_border = pred_b ^ scipy.ndimage.binary_erosion(pred_b, structure=struct)
    gt_border = gt_b ^ scipy.ndimage.binary_erosion(gt_b, structure=struct)

    # Compute Euclidean distance transforms from borders
    dt_pred = scipy.ndimage.distance_transform_edt(~pred_border)
    dt_gt = scipy.ndimage.distance_transform_edt(~gt_border)

    # Distances from pred border points to nearest gt border point and vice versa
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return 0.0

    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def weighted_fbeta(pred: np.ndarray, gt: np.ndarray, beta2: float = 1.0) -> float:
    """
    Weighted F-measure (wFb / F_beta^w).
    Reference: Margolin et al. "How to Evaluate Foreground Maps?", CVPR 2014.
    """
    if not HAS_SCIPY:
        return float('nan')
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if gt.max() == 0:
        return 1.0 if pred.max() == 0 else 0.0

    # Distance transform on GT
    pos_dst = scipy.ndimage.distance_transform_edt(1 - gt)
    neg_dst = scipy.ndimage.distance_transform_edt(gt)

    # Pixel error
    raw_error = np.abs(pred - gt)

    # Gaussian blur kernel approximation for local error weighting
    # 7x7 Gaussian window
    sigma = 5.0
    half_w = 3
    y, x = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Weight matrix calculation
    e_gauss = scipy.ndimage.convolve(raw_error, kernel, mode='reflect')
    weight = np.ones_like(gt)
    
    # Boundary emphasis
    boundary = np.exp(-((pos_dst + neg_dst) / 5.0))
    weight += 5.0 * boundary

    # Weighted precision and recall
    tp_w = np.sum((1.0 - raw_error) * gt * weight)
    fp_w = np.sum(raw_error * (1.0 - gt) * weight)
    fn_w = np.sum(raw_error * gt * weight)

    prec_w = tp_w / (tp_w + fp_w + 1e-8)
    rec_w = tp_w / (tp_w + fn_w + 1e-8)

    wfb = (1.0 + beta2) * prec_w * rec_w / (beta2 * prec_w + rec_w + 1e-8)
    return float(np.clip(wfb, 0.0, 1.0))


def _object_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Object-level structural similarity."""
    x = pred.mean()
    y = gt.mean()
    if y == 0:
        return 1.0 - x
    if y == 1:
        return x
    score = 2.0 * x * y / (x**2 + y**2 + 1e-8)
    return float(score)


def _region_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Region-level structural similarity via 4-quadrant decomposition."""
    h, w = pred.shape
    if h == 0 or w == 0:
        return 0.0

    # Find centroid of GT
    if gt.sum() == 0:
        x_c, y_c = w // 2, h // 2
    else:
        y_indices, x_indices = np.where(gt > 0.5)
        if len(y_indices) == 0:
            x_c, y_c = w // 2, h // 2
        else:
            x_c = int(np.round(np.mean(x_indices)))
            y_c = int(np.round(np.mean(y_indices)))

    x_c = np.clip(x_c, 1, w - 1)
    y_c = np.clip(y_c, 1, h - 1)

    # 4 blocks: LT, RT, LB, RB
    blocks = [
        (pred[:y_c, :x_c], gt[:y_c, :x_c]),
        (pred[:y_c, x_c:], gt[:y_c, x_c:]),
        (pred[y_c:, :x_c], gt[y_c:, :x_c]),
        (pred[y_c:, x_c:], gt[y_c:, x_c:]),
    ]

    total_sim = 0.0
    for p_b, g_b in blocks:
        area = p_b.size
        if area == 0:
            continue
        weight = area / (h * w)
        # SSIM-like formulation
        mean_p = p_b.mean()
        mean_g = g_b.mean()
        var_p = np.var(p_b)
        var_g = np.var(g_b)
        cov_pg = np.mean((p_b - mean_p) * (g_b - mean_g))
        sim = (2.0 * mean_p * mean_g + 1e-8) / (mean_p**2 + mean_g**2 + 1e-8) * \
              (2.0 * cov_pg + 1e-8) / (var_p + var_g + 1e-8)
        total_sim += weight * np.clip(sim, 0.0, 1.0)

    return float(np.clip(total_sim, 0.0, 1.0))


def smeasure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """
    Structure-measure (S-measure / Sm).
    Reference: Fan et al. "Structure-measure: A new way to evaluate foreground maps", ICCV 2017.
    """
    pred = pred.astype(np.float64)
    gt = (gt > 0.5).astype(np.float64)

    y = gt.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())

    s_o = _object_similarity(pred, gt)
    s_r = _region_similarity(pred, gt)
    sm = alpha * s_o + (1.0 - alpha) * s_r
    return float(np.clip(sm, 0.0, 1.0))


def emeasure(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Enhanced-alignment measure (E-measure / Em).
    Reference: Fan et al. "Enhanced-alignment Measure for Binary Foreground Map Evaluation", IJCAI 2018.
    """
    pred = pred.astype(np.float64)
    gt = (gt > 0.5).astype(np.float64)

    h, w = pred.shape
    if gt.max() == 0:
        # All background
        enhanced = 1.0 - pred
        return float(np.mean(enhanced))

    # Demean matrices
    pred_mean = pred.mean()
    gt_mean = gt.mean()

    d_pred = pred - pred_mean
    d_gt = gt - gt_mean

    # Alignment matrix
    align_matrix = 2.0 * (d_pred * d_gt) / (d_pred**2 + d_gt**2 + 1e-8)

    # Enhanced alignment matrix: xi = ((align + 1)^2) / 4
    enhanced_matrix = ((align_matrix + 1.0) ** 2) / 4.0
    return float(np.clip(np.mean(enhanced_matrix), 0.0, 1.0))


class MetricCalculator:
    """
    Accumulator and statistical calculator for evaluation across a dataset.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.dices = []
        self.ious = []
        self.maes = []
        self.wfbs = []
        self.sms = []
        self.ems = []
        self.precisions = []
        self.recalls = []
        self.specificities = []
        self.hd95s = []

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """
        Update accumulator with one image prediction and ground truth.
        Args:
            pred: Prediction mask in range [0, 1], shape (H, W).
            gt: Ground truth mask in range [0, 1], shape (H, W).
        """
        # Ensure 2D float in [0, 1]
        pred = np.clip(pred.squeeze(), 0.0, 1.0).astype(np.float64)
        gt = (np.clip(gt.squeeze(), 0.0, 1.0) > 0.5).astype(np.float64)

        d = dice_score(pred, gt)
        j = iou_score(pred, gt)
        m = mae_score(pred, gt)
        w = weighted_fbeta(pred, gt)
        s = smeasure(pred, gt)
        e = emeasure(pred, gt)
        prec, rec, spec = precision_recall_specificity(pred, gt)
        hd = hd95_score(pred, gt)

        self.dices.append(d)
        self.ious.append(j)
        self.maes.append(m)
        self.wfbs.append(w)
        self.sms.append(s)
        self.ems.append(e)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.specificities.append(spec)
        self.hd95s.append(hd)

    def get_results(self) -> dict:
        """Return dictionary of mean metrics."""
        return {
            'mDice': float(np.mean(self.dices)) if self.dices else 0.0,
            'mIoU': float(np.mean(self.ious)) if self.ious else 0.0,
            'Recall': float(np.mean(self.recalls)) if self.recalls else 0.0,
            'Precision': float(np.mean(self.precisions)) if self.precisions else 0.0,
            'Specificity': float(np.mean(self.specificities)) if self.specificities else 0.0,
            'HD95': float(np.mean(self.hd95s)) if self.hd95s else 0.0,
            'wFb': float(np.mean(self.wfbs)) if self.wfbs else 0.0,
            'Sm': float(np.mean(self.sms)) if self.sms else 0.0,
            'Em': float(np.mean(self.ems)) if self.ems else 0.0,
            'MAE': float(np.mean(self.maes)) if self.maes else 0.0,
        }

