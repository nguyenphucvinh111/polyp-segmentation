"""
Edge AI & Efficiency Profiler for Lightweight Segmentation Models.

Computes comprehensive deployment metrics required for Q1/A* Edge AI publications:
1. Model Complexity:
   - Parameters (M)
   - FLOPs / MACs (GFLOPs) at input resolution
   - Model Disk Size (MB)
2. Latency & Throughput (Speed):
   - GPU Latency (mean ± std ms) with CUDA Event synchronization
   - GPU FPS (Frames Per Second)
   - CPU Latency (mean ± std ms)
   - CPU FPS
3. Memory Footprint:
   - Peak GPU Memory Allocation during inference (MB)
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

try:
    import thop
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class EdgeProfiler:
    """Profiles deep learning models for Edge AI deployment benchmarks."""

    def __init__(self, model: nn.Module, testsize: int = 352, device=None, weights_path: str = None):
        self.model = model
        self.testsize = testsize
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = weights_path

    def count_parameters(self) -> dict:
        """Count total and trainable parameters in Millions."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'total_params_raw': total_params
        }

    def measure_flops(self) -> float:
        """Measure FLOPs / MACs in GFLOPs at test resolution."""
        if not HAS_THOP:
            return float('nan')
        try:
            # Run on CPU to avoid allocating extra CUDA memory for profiling hooks
            cpu_model = self.model.to('cpu')
            cpu_model.eval()
            dummy_input = torch.randn(1, 3, self.testsize, self.testsize, device='cpu')
            flops, _ = thop.profile(cpu_model, inputs=(dummy_input,), verbose=False)
            return flops / 1e9
        except Exception as e:
            print(f"[Profiler] Warning: Failed to calculate FLOPs via thop ({e})")
            return float('nan')

    def measure_gpu_latency(self, warmup: int = 25, runs: int = 75) -> dict:
        """Measure GPU inference latency (ms), FPS, and peak memory (MB)."""
        if not torch.cuda.is_available():
            return {
                'gpu_latency_ms': float('nan'),
                'gpu_latency_std': float('nan'),
                'gpu_fps': float('nan'),
                'gpu_peak_mem_mb': float('nan'),
            }

        gpu_device = torch.device('cuda')
        self.model.to(gpu_device)
        self.model.eval()
        dummy_input = torch.randn(1, 3, self.testsize, self.testsize, device=gpu_device)

        # Warmup GPU
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
            torch.cuda.synchronize()

        # Track peak memory
        torch.cuda.reset_peak_memory_stats(gpu_device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timings = []

        with torch.no_grad():
            for _ in range(runs):
                start_event.record()
                _ = self.model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))

        peak_mem = torch.cuda.max_memory_allocated(gpu_device) / (1024 * 1024)
        mean_latency = float(np.mean(timings))
        std_latency = float(np.std(timings))
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

        return {
            'gpu_latency_ms': mean_latency,
            'gpu_latency_std': std_latency,
            'gpu_fps': fps,
            'gpu_peak_mem_mb': peak_mem,
        }

    def measure_cpu_latency(self, warmup: int = 5, runs: int = 20) -> dict:
        """Measure CPU inference latency (ms) and FPS for edge processors."""
        # Enable multi-threading for CPU to accurately reflect multi-core edge devices
        num_threads = max(1, min(8, os.cpu_count() or 4))
        orig_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)

        self.model.to('cpu')
        self.model.eval()
        dummy_input = torch.randn(1, 3, self.testsize, self.testsize, device='cpu')

        with torch.inference_mode():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        timings = []
        with torch.inference_mode():
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = self.model(dummy_input)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)

        # Restore original thread setting
        torch.set_num_threads(orig_threads)

        mean_latency = float(np.mean(timings))
        std_latency = float(np.std(timings))
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

        return {
            'cpu_latency_ms': mean_latency,
            'cpu_latency_std': std_latency,
            'cpu_fps': fps,
        }

    def profile(self, measure_cpu: bool = True) -> dict:
        """Run complete Edge AI profiling suite."""
        params_info = self.count_parameters()
        flops_G = self.measure_flops()
        gpu_info = self.measure_gpu_latency()
        cpu_info = self.measure_cpu_latency() if measure_cpu else {
            'cpu_latency_ms': float('nan'),
            'cpu_latency_std': float('nan'),
            'cpu_fps': float('nan'),
        }

        # Weight file size
        weight_size_mb = float('nan')
        if self.weights_path and os.path.exists(self.weights_path):
            weight_size_mb = os.path.getsize(self.weights_path) / (1024 * 1024)

        # Restore model to original device
        self.model.to(self.device)

        # Hardware and Real-time capability check
        hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_fps = gpu_info['gpu_fps']
        cpu_fps = cpu_info['cpu_fps']
        if not np.isnan(gpu_fps) and gpu_fps >= 30.0:
            rt_status = "YES (GPU)"
        elif not np.isnan(cpu_fps) and cpu_fps >= 30.0:
            rt_status = "YES (CPU)"
        else:
            rt_status = "NO (<30 FPS)"

        result = {
            'Hardware': hw_name,
            'Resolution': f"{self.testsize}x{self.testsize}",
            'Params (M)': params_info['total_params_M'],
            'FLOPs (G)': flops_G,
            'Model Size (MB)': weight_size_mb if not np.isnan(weight_size_mb) else (params_info['total_params_raw'] * 4 / (1024 * 1024)),
            'GPU Latency (ms)': gpu_info['gpu_latency_ms'],
            'GPU FPS': gpu_info['gpu_fps'],
            'GPU Peak Mem (MB)': gpu_info['gpu_peak_mem_mb'],
            'CPU Latency (ms)': cpu_info['cpu_latency_ms'],
            'CPU FPS': cpu_info['cpu_fps'],
            'Real-time (>=30 FPS)': rt_status,
        }
        return result


    @staticmethod
    def print_table(results: dict, model_name: str = "Model"):
        """Print formatted Edge AI benchmark table."""
        df = pd.DataFrame([results], index=[model_name])
        print("\n" + "=" * 88)
        print(f"EDGE AI & HARDWARE DEPLOYMENT BENCHMARK: {model_name}")
        print("=" * 88)
        if HAS_TABULATE:
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f"))
        else:
            print(df.to_string())
        print("=" * 88 + "\n")

    @staticmethod
    def save_csv(results: dict, model_name: str, path: str):
        """Save efficiency metrics to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = pd.DataFrame([results], index=[model_name])
        df.to_csv(path)
        print(f"[Profiler] Saved Edge AI benchmark to '{path}'.")

    @staticmethod
    def save_latex(results: dict, model_name: str, path: str):
        """Save LaTeX table for Edge AI benchmark."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = pd.DataFrame([results], index=[model_name])
        latex_str = df.to_latex(float_format="%.2f")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f"[Profiler] Saved Edge AI LaTeX table to '{path}'.")
