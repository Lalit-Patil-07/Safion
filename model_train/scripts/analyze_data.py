"""
Exploratory Data Analysis for the Safion PPE detection dataset.

Generates class distribution, co-occurrence matrix, JSD between splits,
and per-class statistics.

Usage:
    python scripts/analyze_data.py --data-dir model_train/data --output-dir model_train/outputs/eda
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_cooccurrence,
    compute_distribution_metrics,
    compute_jsd,
    count_class_distribution,
    count_class_presence,
    parse_label_file,
    plot_class_distribution,
    plot_cooccurrence_matrix,
)


def analyze_split(data_dir: Path, split: str) -> dict:
    """Analyze a single data split."""
    label_dir = data_dir / split / "labels"
    instance_counts = count_class_distribution(label_dir)
    presence_counts = count_class_presence(label_dir)

    # Per-class box size statistics
    box_stats = {i: {"areas": [], "aspect_ratios": []} for i in range(NUM_CLASSES)}
    for label_file in label_dir.glob("*.txt"):
        for ann in parse_label_file(label_file):
            cls = ann["class_id"]
            if 0 <= cls < NUM_CLASSES:
                area = ann["w"] * ann["h"]
                ar = ann["w"] / ann["h"] if ann["h"] > 0 else 0
                box_stats[cls]["areas"].append(area)
                box_stats[cls]["aspect_ratios"].append(ar)

    per_class = {}
    for cls_id in range(NUM_CLASSES):
        areas = box_stats[cls_id]["areas"]
        ars = box_stats[cls_id]["aspect_ratios"]
        per_class[cls_id] = {
            "name": CLASS_NAMES[cls_id],
            "instances": instance_counts.get(cls_id, 0),
            "images_with_class": presence_counts.get(cls_id, 0),
            "mean_area": float(np.mean(areas)) if areas else 0,
            "median_area": float(np.median(areas)) if areas else 0,
            "min_area": float(np.min(areas)) if areas else 0,
            "max_area": float(np.max(areas)) if areas else 0,
            "mean_aspect_ratio": float(np.mean(ars)) if ars else 0,
        }

    return {
        "instance_counts": instance_counts,
        "presence_counts": presence_counts,
        "per_class": per_class,
        "total_images": len(list((data_dir / split / "images").glob("*.jpg"))),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze PPE detection dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots and reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DATASET ANALYSIS")
    print("=" * 60)

    # Analyze each split
    results = {}
    for split in ["train", "valid", "test"]:
        if not (data_dir / split / "labels").exists():
            print(f"  Skipping {split} - directory not found")
            continue
        print(f"\nAnalyzing {split}...")
        results[split] = analyze_split(data_dir, split)

        # Plot distribution
        plot_class_distribution(
            results[split]["instance_counts"],
            f"Class Distribution ({split})",
            output_dir / f"distribution_{split}.png",
        )

    # Cross-split analysis
    if len(results) >= 2:
        print("\n--- Cross-Split Analysis ---")
        splits = list(results.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                p = np.array([results[s1]["instance_counts"].get(k, 0) for k in range(NUM_CLASSES)])
                q = np.array([results[s2]["instance_counts"].get(k, 0) for k in range(NUM_CLASSES)])
                jsd = compute_jsd(p, q)
                print(f"  JSD({s1} vs {s2}): {jsd:.4f}")

    # Co-occurrence analysis (on train split)
    if "train" in results:
        print("\n--- Co-occurrence Analysis (train) ---")
        cooccurrence = compute_cooccurrence(data_dir / "train" / "labels")
        plot_cooccurrence_matrix(cooccurrence, output_dir / "cooccurrence_train.png")

        # Print top co-occurrences
        pairs = []
        for i in range(NUM_CLASSES):
            for j in range(i + 1, NUM_CLASSES):
                if cooccurrence[i][j] > 0:
                    pairs.append((cooccurrence[i][j], CLASS_NAMES[i], CLASS_NAMES[j]))
        pairs.sort(reverse=True)
        print("  Top co-occurring class pairs:")
        for count, n1, n2 in pairs[:10]:
            print(f"    {n1} + {n2}: {count} images")

    # Distribution quality metrics
    print("\n--- Distribution Quality ---")
    for split, data in results.items():
        metrics = compute_distribution_metrics(data["instance_counts"])
        print(f"  {split}: CV={metrics['cv']:.3f}, entropy={metrics['entropy']:.3f}, total={metrics['total']}")

    # Summary
    print("\n--- Summary ---")
    for split, data in results.items():
        print(f"\n  {split.upper()}:")
        print(f"    Images: {data['total_images']}")
        print(f"    Annotations: {sum(data['instance_counts'].values())}")
        print("    Per-class instances:")
        for cls_id in range(NUM_CLASSES):
            count = data["instance_counts"].get(cls_id, 0)
            presence = data["presence_counts"].get(cls_id, 0)
            print(f"      {cls_id}: {CLASS_NAMES[cls_id]:20s} = {count:5d} instances in {presence:4d} images")

    # Save report
    report = {}
    for split, data in results.items():
        report[split] = {
            "total_images": data["total_images"],
            "instance_counts": {str(k): v for k, v in data["instance_counts"].items()},
            "presence_counts": {str(k): v for k, v in data["presence_counts"].items()},
            "per_class": {str(k): v for k, v in data["per_class"].items()},
        }
    with open(output_dir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_dir / 'analysis_report.json'}")
    print(f"Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
