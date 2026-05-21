"""
Model evaluation script with per-class analysis, confusion matrix,
confidence threshold sweep, PR curves, and TTA evaluation.

Usage:
    python scripts/evaluate.py --model model_train/outputs/run_XXXX/weights/best.pt --data model_train/config/data.yaml --output-dir model_train/outputs/run_XXXX/evaluation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    load_yaml,
    plot_class_distribution,
)


def run_evaluation(model_path: str, data_yaml: str, output_dir: Path) -> dict:
    """Run full model evaluation."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    results = {}

    # Standard evaluation on test set
    print("\n--- Standard Evaluation (test set) ---")
    val_results = model.val(data=data_yaml, split="test", verbose=True)

    results["overall"] = {
        "mAP50": float(val_results.box.map50),
        "mAP50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
    }

    print(f"  mAP50:    {results['overall']['mAP50']:.4f}")
    print(f"  mAP50-95: {results['overall']['mAP50_95']:.4f}")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall:   {results['overall']['recall']:.4f}")

    # Per-class AP
    print("\n--- Per-Class AP ---")
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        ap50 = float(val_results.box.ap50[cls_id]) if cls_id < len(val_results.box.ap50) else 0
        ap50_95 = float(val_results.box.ap[cls_id]) if cls_id < len(val_results.box.ap) else 0
        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "mAP50": ap50,
            "mAP50_95": ap50_95,
        }
        print(f"  {cls_id}: {CLASS_NAMES[cls_id]:20s} mAP50={ap50:.4f}  mAP50-95={ap50_95:.4f}")

    results["per_class"] = per_class

    # Plot per-class AP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    ap50_vals = [per_class[i]["mAP50"] for i in range(NUM_CLASSES)]
    ap50_95_vals = [per_class[i]["mAP50_95"] for i in range(NUM_CLASSES)]

    ax1.barh(names, ap50_vals, color="steelblue")
    ax1.set_xlabel("mAP50")
    ax1.set_title("Per-Class mAP50")
    ax1.set_xlim(0, 1)

    ax2.barh(names, ap50_95_vals, color="coral")
    ax2.set_xlabel("mAP50-95")
    ax2.set_title("Per-Class mAP50-95")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_ap.png", dpi=150)
    plt.close()

    return results


def confidence_threshold_sweep(model_path: str, data_yaml: str, output_dir: Path) -> dict:
    """Sweep confidence thresholds to find optimal F1."""
    from ultralytics import YOLO

    print("\n--- Confidence Threshold Sweep ---")
    model = YOLO(model_path)
    data_config = load_yaml(data_yaml)

    thresholds = np.arange(0.1, 0.95, 0.05)
    sweep_results = []

    for conf in thresholds:
        val_results = model.val(data=data_yaml, split="test", conf=float(conf), verbose=False)
        p = float(val_results.box.mp)
        r = float(val_results.box.mr)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        sweep_results.append({
            "confidence": float(conf),
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP50": float(val_results.box.map50),
        })

    # Find best F1
    best = max(sweep_results, key=lambda x: x["f1"])
    print(f"  Best F1 threshold: {best['confidence']:.2f}")
    print(f"    Precision: {best['precision']:.4f}")
    print(f"    Recall:    {best['recall']:.4f}")
    print(f"    F1:        {best['f1']:.4f}")

    # Production threshold (0.4)
    prod = next((r for r in sweep_results if abs(r["confidence"] - 0.4) < 0.01), None)
    if prod:
        print(f"\n  Production threshold (0.4):")
        print(f"    Precision: {prod['precision']:.4f}")
        print(f"    Recall:    {prod['recall']:.4f}")
        print(f"    F1:        {prod['f1']:.4f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    confs = [r["confidence"] for r in sweep_results]
    ax.plot(confs, [r["precision"] for r in sweep_results], "b-", label="Precision")
    ax.plot(confs, [r["recall"] for r in sweep_results], "r-", label="Recall")
    ax.plot(confs, [r["f1"] for r in sweep_results], "g-", label="F1", linewidth=2)
    ax.axvline(x=best["confidence"], color="g", linestyle="--", alpha=0.5, label=f"Best F1 ({best['confidence']:.2f})")
    ax.axvline(x=0.4, color="orange", linestyle="--", alpha=0.5, label="Production (0.4)")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Confidence Threshold Sweep")
    ax.legend()
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=150)
    plt.close()

    return {"sweep": sweep_results, "best_f1": best, "production": prod}


def tta_evaluation(model_path: str, data_yaml: str) -> dict:
    """Test-Time Augmentation evaluation."""
    from ultralytics import YOLO

    print("\n--- TTA Evaluation (augment=True) ---")
    model = YOLO(model_path)
    val_results = model.val(data=data_yaml, split="test", augment=True, verbose=False)

    results = {
        "mAP50": float(val_results.box.map50),
        "mAP50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
    }

    print(f"  mAP50:    {results['mAP50']:.4f}")
    print(f"  mAP50-95: {results['mAP50_95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:   {results['recall']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tta", action="store_true", help="Run TTA evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run confidence threshold sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # Standard evaluation
    eval_results = run_evaluation(args.model, args.data, output_dir)

    # Confidence threshold sweep
    if args.sweep:
        sweep_results = confidence_threshold_sweep(args.model, args.data, output_dir)
        eval_results["threshold_sweep"] = sweep_results

    # TTA evaluation
    if args.tta:
        tta_results = tta_evaluation(args.model, args.data)
        eval_results["tta"] = tta_results

    # Save full results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Plots saved to: {output_dir}/")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
