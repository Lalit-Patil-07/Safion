"""
YOLO training script with deterministic training, checkpoint/resume,
and experiment metadata logging.

Usage:
    python scripts/train.py --config model_train/config/train_hyperparams.yaml --data model_train/config/data.yaml --output-dir model_train/outputs/run_TIMESTAMP
    python scripts/train.py --resume model_train/outputs/run_TIMESTAMP/weights/last.pt
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils import (
    compute_dataset_hash,
    get_experiment_metadata,
    load_yaml,
    save_yaml,
    set_seed,
)


def extract_metrics_from_csv(results_csv: str) -> dict:
    """Extract final epoch metrics from Ultralytics results.csv."""
    metrics = {}
    try:
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return metrics
            last_row = rows[-1]

            # Map CSV column names to our metric names
            column_map = {
                "metrics/mAP50(B)": "mAP50",
                "metrics/mAP50-95(B)": "mAP50_95",
                "metrics/precision(B)": "precision",
                "metrics/recall(B)": "recall",
            }
            for csv_name, metric_name in column_map.items():
                if csv_name in last_row:
                    try:
                        metrics[metric_name] = float(last_row[csv_name])
                    except (ValueError, TypeError):
                        metrics[metric_name] = None

            # Training loss
            if "train/box_loss" in last_row:
                metrics["train_box_loss"] = float(last_row["train/box_loss"])
            if "train/cls_loss" in last_row:
                metrics["train_cls_loss"] = float(last_row["train/cls_loss"])
            if "train/dfl_loss" in last_row:
                metrics["train_dfl_loss"] = float(last_row["train/dfl_loss"])
            if "val/box_loss" in last_row:
                metrics["val_box_loss"] = float(last_row["val/box_loss"])
            if "val/cls_loss" in last_row:
                metrics["val_cls_loss"] = float(last_row["val/cls_loss"])
            if "val/dfl_loss" in last_row:
                metrics["val_dfl_loss"] = float(last_row["val/dfl_loss"])

            metrics["epochs_completed"] = len(rows)
    except Exception as e:
        print(f"  Warning: Could not parse results.csv: {e}")
    return metrics


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint in an output directory."""
    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        return None

    # Check for last.pt first
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        return last_pt

    # Check for periodic checkpoints
    checkpoints = sorted(weights_dir.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]

    return None


def validate_resume_checkpoint(resume_path: Path, config: dict, data_config: dict) -> bool:
    """Validate that a checkpoint is compatible for resume."""
    print("\nValidating resume checkpoint...")

    if not resume_path.exists():
        print(f"  ERROR: Checkpoint not found: {resume_path}")
        return False

    # Check if experiment metadata exists
    run_dir = resume_path.parent.parent
    metadata_path = run_dir / "experiment_metadata.json"
    if metadata_path.exists():
        metadata = load_yaml(metadata_path)
        current_dataset_hash = compute_dataset_hash(data_config.get("path", ""))

        if metadata.get("dataset_hash") != current_dataset_hash:
            print("  WARNING: Dataset hash mismatch. Checkpoint was trained on a different dataset version.")
            print(f"    Checkpoint dataset: {metadata.get('dataset_hash', 'unknown')}")
            print(f"    Current dataset:    {current_dataset_hash}")

        if metadata.get("config", {}).get("model") != config.get("model"):
            print("  WARNING: Model architecture may differ.")
            print(f"    Checkpoint model: {metadata.get('config', {}).get('model', 'unknown')}")
            print(f"    Current model:    {config.get('model', 'unknown')}")

    print("  Checkpoint validation passed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--config", type=str, help="Training hyperparameters YAML")
    parser.add_argument("--data", type=str, help="Dataset config YAML")
    parser.add_argument("--output-dir", type=str, help="Output directory for training run")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Resume mode
    if args.resume:
        resume_path = Path(args.resume)
        if not args.config:
            # Try to find config from the run directory
            run_dir = resume_path.parent.parent
            metadata_path = run_dir / "experiment_metadata.json"
            if metadata_path.exists():
                metadata = load_yaml(metadata_path)
                config = metadata.get("config", {})
                print(f"Resuming with config from: {metadata_path}")
            else:
                print("ERROR: --config required for resume when no metadata found")
                sys.exit(1)
        else:
            config = load_yaml(args.config)

        if args.data:
            data_config = load_yaml(args.data)
        else:
            data_config = {}

        if not validate_resume_checkpoint(resume_path, config, data_config):
            sys.exit(1)

        output_dir = resume_path.parent.parent
    else:
        if not args.config or not args.data or not args.output_dir:
            print("ERROR: --config, --data, and --output-dir required for new training")
            sys.exit(1)

        config = load_yaml(args.config)
        data_config = load_yaml(args.data)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"\nRandom seed: {seed}")

    # Load model
    from ultralytics import YOLO

    model_name = config.get("model", "yolo11m.pt")
    print(f"Loading model: {model_name}")

    if args.resume:
        model = YOLO(str(resume_path))
        print(f"Resumed from: {resume_path}")
    else:
        model = YOLO(model_name)

    # Collect experiment metadata
    if not args.resume:
        data_dir = data_config.get("path", "")
        metadata = get_experiment_metadata(config, data_dir)
        metadata_path = output_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Experiment metadata saved to: {metadata_path}")

    # Prepare training arguments
    train_args = {
        "data": str(args.data) if args.data else None,
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch", -1),
        "imgsz": config.get("imgsz", 832),
        "patience": config.get("patience", 35),
        "seed": seed,
        "deterministic": config.get("deterministic", True),
        "cos_lr": config.get("cos_lr", True),
        "close_mosaic": config.get("close_mosaic", 10),
        "save_period": config.get("save_period", 10),
        "amp": config.get("amp", True),
        "optimizer": config.get("optimizer", "auto"),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 5),
        "warmup_momentum": config.get("warmup_momentum", 0.8),
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 0.5),
        "dfl": config.get("dfl", 1.5),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 10.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 2.0),
        "perspective": config.get("perspective", 0.0),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.0),
        "copy_paste": config.get("copy_paste", 0.0),
        "erasing": config.get("erasing", 0.4),
        "conf": config.get("conf", 0.001),
        "iou": config.get("iou", 0.7),
        "device": config.get("device", ""),
        "workers": config.get("workers", 8),
        "verbose": config.get("verbose", True),
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
    }

    # Remove None values
    train_args = {k: v for k, v in train_args.items() if v is not None}

    # Print training config
    print("\n" + "=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    for key, val in sorted(train_args.items()):
        if key not in ("data", "project", "name", "exist_ok"):
            print(f"  {key}: {val}")
    print("=" * 60)

    # Train
    print("\nStarting training...")
    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        sys.exit(1)

    # Extract metrics from results.csv
    print("\nExtracting metrics...")
    run_dir = output_dir / "train"
    results_csv = run_dir / "results.csv"

    if results_csv.exists():
        metrics = extract_metrics_from_csv(str(results_csv))
        print(f"  Epochs completed: {metrics.get('epochs_completed', 'unknown')}")
        print(f"  mAP50: {metrics.get('mAP50', 'N/A')}")
        print(f"  mAP50-95: {metrics.get('mAP50_95', 'N/A')}")
        print(f"  Precision: {metrics.get('precision', 'N/A')}")
        print(f"  Recall: {metrics.get('recall', 'N/A')}")

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to: {metrics_path}")
    else:
        print("  Warning: results.csv not found")

    # Copy best.pt to run root for easy access
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
        print(f"  Best model copied to: {output_dir / 'best.pt'}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Best weights: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
