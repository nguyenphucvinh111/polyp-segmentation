"""
Test & Edge AI Benchmark Script for Polyp Segmentation Models.

Runs:
1. Edge AI Deployment Profiling:
   - Parameters (M), FLOPs (G), Model Size (MB)
   - GPU Latency (ms), GPU FPS, GPU Memory (MB)
   - CPU Latency (ms), CPU FPS
2. Inference across all 5 benchmark test sets:
   - CVC-300
   - CVC-ClinicDB
   - CVC-ColonDB
   - ETIS-LaribPolypDB
   - Kvasir

Saves:
- Predicted masks to './results/{model}/{dataset}/'
- 'metrics.csv' and 'metrics.tex' (Segmentation Accuracy)
- 'efficiency.csv' and 'efficiency.tex' (Hardware & Edge AI Deployment)
"""

import argparse
import os
import sys
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils.checkpoint import Checkpointer
from utils.evaluator import Evaluator
from utils.setup_dataset import setup_dataset


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation & Edge AI Profiling on Polyp Segmentation Models")
    parser.add_argument('--model', type=str, default='pranet', help='Model name (registered in models/)')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file (.pth)')
    parser.add_argument('--testsize', type=int, default=352, help='Input evaluation resolution')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save output masks')
    parser.add_argument('--skip_profile', action='store_true', help='Skip Edge AI latency/FLOPs profiling')
    parser.add_argument('--skip_cpu', action='store_true', help='Skip CPU latency measurement')
    parser.add_argument('--profile_only', action='store_true', help='Only profile hardware metrics without running test sets')
    parser.add_argument('--tta', action='store_true', default=False, help='Enable Test-Time Augmentation (Horizontal Flip Ensembling)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n============================================================")
    print(f"BENCHMARK & EDGE AI SUITE: {args.model.upper()}")
    print(f"============================================================")
    print(f"Hardware platform: {device}")
    print(f"Resolution       : {args.testsize}x{args.testsize}")
    print(f"Weights          : {args.weights}")
    print(f"============================================================\n")

    # 1. Build model and load weights
    print(f"[Test] Building model '{args.model}'...")
    model = get_model(args.model)
    Checkpointer.load(model, args.weights, device=device)
    model.to(device)
    model.eval()

    # 2. Setup Evaluator
    results_dir = os.path.join(args.save_dir, args.model)
    evaluator = Evaluator(
        data_root=args.data_root,
        model=model,
        device=device,
        testsize=args.testsize,
        model_name=args.model,
        weights_path=args.weights
    )

    # 3. Edge AI Deployment Profiling (Params, FLOPs, Latency, FPS, Peak Memory)
    efficiency = None
    if not args.skip_profile:
        print(f"[Test] Profiling Edge AI & Hardware efficiency (Params, FLOPs, Latency, FPS, Memory)...")
        efficiency = evaluator.benchmark_efficiency(measure_cpu=not args.skip_cpu)

    # 4. Accuracy Evaluation across 5 test sets
    if not args.profile_only:
        data_root = setup_dataset(args.data_root)
        evaluator.data_root = os.path.join(data_root, 'TestDataset') if not data_root.endswith('TestDataset') else data_root

        print(f"[Test] Running dataset evaluation (TTA={args.tta}) and saving masks to '{results_dir}'...")
        results = evaluator.evaluate_all(save_results_dir=results_dir, use_tta=args.tta)

        # Print & save both Accuracy and Efficiency
        evaluator.print_table(results, efficiency=efficiency, model_name=args.model)
        evaluator.save_csv(results, path=os.path.join(results_dir, 'metrics.csv'), efficiency=efficiency)
        evaluator.save_latex(results, path=os.path.join(results_dir, 'metrics.tex'), efficiency=efficiency)
    else:
        evaluator.print_table(results=None, efficiency=efficiency, model_name=args.model)
        evaluator.save_csv(results=None, path=os.path.join(results_dir, 'metrics.csv'), efficiency=efficiency)
        evaluator.save_latex(results=None, path=os.path.join(results_dir, 'metrics.tex'), efficiency=efficiency)

    print("[Test] Completed successfully.\n")


if __name__ == '__main__':
    main()
