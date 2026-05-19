"""
Inference benchmark script: FPS, latency, ONNX vs PyTorch, CPU vs GPU.

Usage:
    python scripts/benchmark.py --model model_train/outputs/run_XXXX/weights/best.pt
    python scripts/benchmark.py --model model_train/outputs/run_XXXX/weights/best.pt --onnx model_train/outputs/run_XXXX/export/best.onnx
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils import get_gpu_info


def benchmark_pytorch(model_path: str, imgsz: int = 832, warmup: int = 5, runs: int = 50, device: str = "") -> dict:
    """Benchmark PyTorch model inference."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    test_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model(test_img, device=device, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model(test_img, device=device, verbose=False)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies) * 1000  # Convert to ms

    return {
        "framework": "pytorch",
        "device": device or "auto",
        "imgsz": imgsz,
        "warmup": warmup,
        "runs": runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": 1000.0 / float(np.mean(latencies)),
    }


def benchmark_onnx(onnx_path: str, imgsz: int = 832, warmup: int = 5, runs: int = 50) -> dict:
    """Benchmark ONNX model inference."""
    from ultralytics import YOLO

    model = YOLO(onnx_path)
    test_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model(test_img, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model(test_img, verbose=False)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies) * 1000

    return {
        "framework": "onnx",
        "device": "cpu",
        "imgsz": imgsz,
        "warmup": warmup,
        "runs": runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": 1000.0 / float(np.mean(latencies)),
    }


def measure_memory(model_path: str, imgsz: int = 832, device: str = "") -> dict:
    """Measure peak memory usage during inference."""
    import torch

    from ultralytics import YOLO

    memory_info = {"peak_gpu_mb": 0, "peak_cpu_mb": 0}

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()
        model = YOLO(model_path)
        test_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(test_img, device="cuda", verbose=False)
        memory_info["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return memory_info


def print_results(results: dict) -> None:
    """Print benchmark results in a readable format."""
    print(f"\n  Framework: {results['framework']}")
    print(f"  Device:    {results['device']}")
    print(f"  Image size: {results['imgsz']}x{results['imgsz']}")
    print(f"  Runs:      {results['runs']}")
    print(f"  Mean:      {results['mean_ms']:.2f} ms")
    print(f"  Median:    {results['median_ms']:.2f} ms")
    print(f"  P95:       {results['p95_ms']:.2f} ms")
    print(f"  P99:       {results['p99_ms']:.2f} ms")
    print(f"  Min:       {results['min_ms']:.2f} ms")
    print(f"  Max:       {results['max_ms']:.2f} ms")
    print(f"  FPS:       {results['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--imgsz", type=int, default=832, help="Input image size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  INFERENCE BENCHMARK")
    print("=" * 60)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['vram_mb']} MB)")
    print(f"CUDA: {gpu_info['cuda_version']}")

    results = {"gpu_info": gpu_info, "benchmarks": []}

    # PyTorch benchmark
    print("\n--- PyTorch Benchmark ---")
    pt_results = benchmark_pytorch(args.model, args.imgsz, args.warmup, args.runs, args.device)
    print_results(pt_results)
    results["benchmarks"].append(pt_results)

    # ONNX benchmark
    if args.onnx:
        print("\n--- ONNX Benchmark ---")
        onnx_results = benchmark_onnx(args.onnx, args.imgsz, args.warmup, args.runs)
        print_results(onnx_results)
        results["benchmarks"].append(onnx_results)

        # Comparison
        print("\n--- Comparison ---")
        speedup = pt_results["mean_ms"] / onnx_results["mean_ms"]
        print(f"  PT vs ONNX speedup: {speedup:.2f}x")
        results["comparison"] = {
            "pt_vs_onnx_speedup": speedup,
            "pt_fps": pt_results["fps"],
            "onnx_fps": onnx_results["fps"],
        }

    # Memory profiling
    if torch_is_available():
        print("\n--- Memory Profiling ---")
        mem = measure_memory(args.model, args.imgsz, args.device)
        if mem["peak_gpu_mb"] > 0:
            print(f"  Peak GPU memory: {mem['peak_gpu_mb']:.1f} MB")
        results["memory"] = mem

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


def torch_is_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
