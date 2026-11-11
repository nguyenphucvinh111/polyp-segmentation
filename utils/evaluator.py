"""
Comprehensive Evaluator & Edge AI Benchmark for Polyp Segmentation.

Evaluates on:
1. CVC-300 (EndoScene test subset)
2. CVC-ClinicDB
3. CVC-ColonDB
4. ETIS-LaribPolypDB
5. Kvasir-SEG

Calculates:
- Segmentation Benchmarks: mDice, mIoU, wFb, Sm, Em, MAE.
- Edge AI & Efficiency Metrics: Parameters (M), FLOPs (G), Model Size (MB),
  GPU Latency (ms), GPU FPS, GPU Peak Mem (MB), CPU Latency (ms), CPU FPS.

Outputs formatted tables in ASCII/PSQL, CSV, and publication LaTeX formats.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from utils.metrics import MetricCalculator
from utils.dataloader import TestDataset
from utils.edge_profiler import EdgeProfiler

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class Evaluator:
    """Evaluates segmentation models across standard polyp benchmarks and Edge AI efficiency."""

    TEST_SETS = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']

    def __init__(
        self,
        data_root: str,
        model=None,
        device=None,
        testsize: int = 352,
        model_name: str = None,
        weights_path: str = None
    ):
        self.data_root = os.path.join(data_root, 'TestDataset') if not data_root.endswith('TestDataset') else data_root
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.testsize = testsize
        self.model_name = model_name or (model.__class__.__name__ if model else "Model")
        self.weights_path = weights_path
        self.metrics = MetricCalculator()

    def benchmark_efficiency(self, measure_cpu: bool = True) -> dict:
        """Run hardware profiling for Edge AI: Params, FLOPs, Latency (GPU & CPU), FPS, Memory."""
        assert self.model is not None, "Model must be provided for profiling."
        profiler = EdgeProfiler(
            model=self.model,
            testsize=self.testsize,
            device=self.device,
            weights_path=self.weights_path
        )
        return profiler.profile(measure_cpu=measure_cpu)

    def evaluate_all(self, save_results_dir: str = None, use_tta: bool = False) -> dict:
        """
        Run inference and compute metrics on all 5 test sets.
        Optionally save binary prediction masks to save_results_dir.
        Optionally apply Test-Time Augmentation (use_tta=True) via horizontal flip ensembling.
        """
        assert self.model is not None, "Model must be provided for evaluation."
        self.model.eval()
        self.model.to(self.device)

        all_results = {}

        for ds_name in self.TEST_SETS:
            img_dir = os.path.join(self.data_root, ds_name, 'images')
            gt_dir = os.path.join(self.data_root, ds_name, 'masks')

            if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
                print(f"[Evaluator] Skipping '{ds_name}': directory not found.")
                continue

            test_loader = TestDataset(img_dir, gt_dir, testsize=self.testsize)
            self.metrics.reset()

            save_mask_dir = None
            if save_results_dir:
                save_mask_dir = os.path.join(save_results_dir, ds_name)
                os.makedirs(save_mask_dir, exist_ok=True)

            with torch.no_grad():
                for _ in tqdm(range(test_loader.size), desc=f"Evaluating {ds_name}"):
                    img_tensor, gt_pil, name = test_loader.load_data()
                    gt_np = np.asarray(gt_pil, dtype=np.float32)
                    if gt_np.max() > 0:
                        gt_np /= gt_np.max()

                    orig_h, orig_w = gt_np.shape

                    img_tensor = img_tensor.to(self.device)
                    if use_tta:
                        outputs1 = self.model(img_tensor)
                        p1 = outputs1[-1] if isinstance(outputs1, (tuple, list)) else outputs1

                        # Test-Time Augmentation: Horizontal Flip
                        outputs2 = self.model(torch.flip(img_tensor, dims=[3]))
                        p2 = outputs2[-1] if isinstance(outputs2, (tuple, list)) else outputs2
                        p2 = torch.flip(p2, dims=[3])

                        pred = (p1 + p2) * 0.5
                    else:
                        outputs = self.model(img_tensor)
                        pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

                    # Upsample prediction back to original image size
                    pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    pred_sig = torch.sigmoid(pred).squeeze().cpu().numpy()

                    # Normalize
                    pred_norm = (pred_sig - pred_sig.min()) / (pred_sig.max() - pred_sig.min() + 1e-8)

                    # Accumulate metrics
                    self.metrics.update(pred_norm, gt_np)

                    # Save prediction mask if requested
                    if save_mask_dir:
                        out_img = (pred_norm * 255.0).astype(np.uint8)
                        Image.fromarray(out_img).save(os.path.join(save_mask_dir, name))

            res = self.metrics.get_results()
            all_results[ds_name] = res

        # Compute average across all datasets
        if all_results:
            metric_keys = list(next(iter(all_results.values())).keys())
            mean_row = {}
            for k in metric_keys:
                vals = [all_results[ds][k] for ds in all_results]
                mean_row[k] = float(np.mean(vals))
            all_results['Mean'] = mean_row

        return all_results

    def print_table(self, results: dict = None, efficiency: dict = None, model_name: str = None):
        """Format and print benchmark accuracy and Edge AI efficiency tables."""
        m_name = model_name or self.model_name

        # 1. Print Accuracy Table
        if results:
            df = pd.DataFrame(results).T
            cols = ['mDice', 'mIoU', 'Recall', 'Precision', 'Specificity', 'HD95', 'wFb', 'Sm', 'Em', 'MAE']
            display_cols = [c for c in cols if c in df.columns]
            df_disp = df[display_cols]

            print("\n" + "=" * 76)
            print(f"POLYP SEGMENTATION ACCURACY BENCHMARK: {m_name}")
            print("=" * 76)
            if HAS_TABULATE:
                print(tabulate(df_disp, headers='keys', tablefmt='psql', floatfmt=".4f"))
            else:
                print(df_disp.to_string())
            print("=" * 76 + "\n")

        # 2. Print Edge AI Efficiency Table
        if efficiency:
            EdgeProfiler.print_table(efficiency, model_name=m_name)

    def save_csv(self, results: dict = None, path: str = 'metrics.csv', efficiency: dict = None):
        """Save accuracy results and optional efficiency benchmark to CSV."""
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)

        if results:
            df = pd.DataFrame(results).T
            df.to_csv(path)
            print(f"[Evaluator] Saved Accuracy metrics to '{path}'.")

        if efficiency:
            eff_path = os.path.join(out_dir, 'efficiency.csv')
            EdgeProfiler.save_csv(efficiency, model_name=self.model_name, path=eff_path)

    def save_latex(self, results: dict = None, path: str = 'metrics.tex', efficiency: dict = None):
        """Generate and save publication LaTeX tables for accuracy and edge efficiency."""
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)

        if results:
            df = pd.DataFrame(results).T
            cols = ['mDice', 'mIoU', 'Recall', 'Precision', 'Specificity', 'HD95', 'wFb', 'Sm', 'Em', 'MAE']
            display_cols = [c for c in cols if c in df.columns]
            df_disp = df[display_cols]

            latex_str = df_disp.to_latex(float_format="%.4f")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            print(f"[Evaluator] Saved Accuracy LaTeX table to '{path}'.")

        if efficiency:
            eff_path = os.path.join(out_dir, 'efficiency.tex')
            EdgeProfiler.save_latex(efficiency, model_name=self.model_name, path=eff_path)
