"""
Dataset validation script for YOLO label files.

Run this BEFORE any training to catch data quality issues early.
Exits with non-zero code if critical issues are found (fail-fast).

Usage:
    python scripts/validate_labels.py --data-dir model_train/data
    python scripts/validate_labels.py --data-dir model_train/data --strict
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_label_path_for_image,
    list_image_files,
    parse_label_file,
    validate_label_file,
)


def validate_dataset(data_dir: str, strict: bool = True) -> dict:
    """Validate entire dataset. Returns report dict."""
    data_dir = Path(data_dir)
    report = {
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "total_annotations": 0,
        "critical_issues": 0,
        "warnings": 0,
        "passed": True,
    }

    for split in ["train", "valid", "test"]:
        image_dir = data_dir / split / "images"
        label_dir = data_dir / split / "labels"

        if not image_dir.exists():
            report["splits"][split] = {"error": f"Image directory not found: {image_dir}"}
            report["critical_issues"] += 1
            report["passed"] = False
            continue

        split_report = {
            "images": 0,
            "labels": 0,
            "annotations": 0,
            "missing_labels": [],
            "missing_images": [],
            "corrupt_images": [],
            "invalid_labels": {},
            "empty_labels": [],
            "class_distribution": {},
        }

        # List files
        image_files = list_image_files(image_dir)
        label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {f.stem: f for f in image_files}
        label_stems = {f.stem: f for f in label_files}

        split_report["images"] = len(image_files)
        split_report["labels"] = len(label_files)

        # Check pairing
        for stem in sorted(image_stems.keys() - label_stems.keys()):
            split_report["missing_labels"].append(stem)
            report["critical_issues"] += 1
            report["passed"] = False

        for stem in sorted(label_stems.keys() - image_stems.keys()):
            split_report["missing_images"].append(stem)
            report["warnings"] += 1

        # Validate images
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                split_report["corrupt_images"].append(img_path.name)
                report["critical_issues"] += 1
                report["passed"] = False
                continue

            h, w = img.shape[:2]
            if h < 10 or w < 10:
                split_report["corrupt_images"].append(f"{img_path.name} ({w}x{h})")
                report["warnings"] += 1

        # Validate labels
        for label_path in label_files:
            issues = validate_label_file(label_path, NUM_CLASSES)
            if issues:
                split_report["invalid_labels"][label_path.name] = issues
                has_critical = any(
                    "Invalid class ID" in i or "outside [0,1]" in i
                    for i in issues
                )
                if has_critical:
                    report["critical_issues"] += len([i for i in issues if "Invalid class ID" in i or "outside [0,1]" in i])
                    report["passed"] = False
                else:
                    report["warnings"] += len(issues)

            # Check empty
            annotations = parse_label_file(label_path)
            if not annotations:
                split_report["empty_labels"].append(label_path.name)
            else:
                split_report["annotations"] += len(annotations)
                for ann in annotations:
                    cls = ann["class_id"]
                    split_report["class_distribution"][cls] = split_report["class_distribution"].get(cls, 0) + 1

        report["splits"][split] = split_report
        report["total_images"] += split_report["images"]
        report["total_labels"] += split_report["labels"]
        report["total_annotations"] += split_report["annotations"]

    return report


def print_report(report: dict) -> None:
    """Print human-readable validation report."""
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal: {report['total_images']} images, {report['total_labels']} labels, {report['total_annotations']} annotations")
    print(f"Critical issues: {report['critical_issues']}")
    print(f"Warnings: {report['warnings']}")
    print(f"Status: {'PASSED' if report['passed'] else 'FAILED'}")

    for split, data in report["splits"].items():
        print(f"\n--- {split.upper()} ---")
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            continue
        print(f"  Images: {data['images']}, Labels: {data['labels']}, Annotations: {data['annotations']}")

        if data["missing_labels"]:
            print(f"  Missing labels ({len(data['missing_labels'])}): {data['missing_labels'][:5]}...")
        if data["missing_images"]:
            print(f"  Missing images ({len(data['missing_images'])}): {data['missing_images'][:5]}...")
        if data["corrupt_images"]:
            print(f"  Corrupt images ({len(data['corrupt_images'])}): {data['corrupt_images'][:5]}...")
        if data["empty_labels"]:
            print(f"  Empty labels ({len(data['empty_labels'])}): {data['empty_labels'][:5]}...")
        if data["invalid_labels"]:
            print(f"  Invalid labels ({len(data['invalid_labels'])} files):")
            for fname, issues in list(data["invalid_labels"].items())[:3]:
                print(f"    {fname}: {issues[0]}")

        if data["class_distribution"]:
            print("  Class distribution:")
            for cls_id in sorted(data["class_distribution"].keys()):
                name = CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else f"Unknown({cls_id})"
                print(f"    {cls_id}: {name} = {data['class_distribution'][cls_id]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset labels")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--strict", action="store_true", help="Exit with error on any issue")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    report = validate_dataset(args.data_dir, strict=args.strict)
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    if not report["passed"]:
        print("\nFAILED: Critical data quality issues found. Fix before training.")
        sys.exit(1)

    if report["warnings"] > 0:
        print(f"\nWARNING: {report['warnings']} non-critical issues found.")

    print("\nPASSED: Dataset validation successful.")
    sys.exit(0)


if __name__ == "__main__":
    main()
