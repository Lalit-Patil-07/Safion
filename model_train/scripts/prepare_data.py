"""
Data preparation script: multilabel stratified split + augmentation.

Creates a new data_prepared/ directory with redistributed and augmented data.
The original data/ directory is NEVER modified.

Usage:
    python scripts/prepare_data.py --data-dir model_train/data --output-dir model_train/data_prepared --config model_train/config/augmentation.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_distribution_metrics,
    compute_jsd,
    count_class_distribution,
    count_class_presence,
    list_image_files,
    load_yaml,
    parse_label_file,
    plot_augmentation_preview,
    plot_class_distribution,
    write_label_file,
)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def get_multilabel_vector(label_dir: Path, image_stems: list[str]) -> np.ndarray:
    """Create binary presence matrix for multilabel stratification."""
    matrix = np.zeros((len(image_stems), NUM_CLASSES), dtype=np.int32)
    for idx, stem in enumerate(image_stems):
        label_path = label_dir / f"{stem}.txt"
        if label_path.exists():
            for ann in parse_label_file(label_path):
                cls = ann["class_id"]
                if 0 <= cls < NUM_CLASSES:
                    matrix[idx][cls] = 1
    return matrix


def stratified_split(
    image_stems: list[str],
    label_dir: Path,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Perform multilabel stratified split using iterative approach."""
    rng = np.random.RandomState(seed)
    n = len(image_stems)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Get multilabel matrix
    matrix = get_multilabel_vector(label_dir, image_stems)

    # Simple stratified split based on primary class (most frequent class per image)
    # This is a practical approximation for small datasets
    primary_classes = np.argmax(matrix, axis=1)

    from sklearn.model_selection import train_test_split

    # First split: train vs (valid + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=primary_classes[indices],
    )

    # Second split: valid vs test
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - valid_ratio_adjusted),
        random_state=seed,
        stratify=primary_classes[temp_idx],
    )

    train_stems = [image_stems[i] for i in train_idx]
    valid_stems = [image_stems[i] for i in valid_idx]
    test_stems = [image_stems[i] for i in test_idx]

    return train_stems, valid_stems, test_stems


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_image_horizontal_flip(img: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Flip image horizontally and transform bounding boxes."""
    flipped_img = cv2.flip(img, 1)
    flipped_anns = []
    for ann in annotations:
        new_ann = ann.copy()
        new_ann["cx"] = 1.0 - ann["cx"]  # Mirror x coordinate
        flipped_anns.append(new_ann)
    return flipped_img, flipped_anns


def augment_image_brightness(img: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
    """Adjust image brightness."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def augment_minority_classes(
    image_dir: Path,
    label_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    target_classes: list[dict],
    exclude_classes: list[int],
    transforms: dict,
) -> int:
    """Augment images containing minority classes."""
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    augmented_count = 0
    rng = random.Random(42)

    for target in target_classes:
        target_id = target["id"]
        min_augmented = target["min_augmented_images"]

        # Find images containing the target class
        candidate_images = []
        for img_path in list_image_files(image_dir):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            annotations = parse_label_file(label_path)
            has_target = any(a["class_id"] == target_id for a in annotations)
            has_exclude = any(a["class_id"] in exclude_classes for a in annotations)

            if has_target and not has_exclude:
                candidate_images.append((img_path, label_path, annotations))

        if not candidate_images:
            print(f"  Warning: No candidate images found for class {target['name']} (id={target_id})")
            continue

        print(f"  Augmenting class {target['name']}: {len(candidate_images)} source images, target {min_augmented}")

        aug_count = 0
        while aug_count < min_augmented:
            for img_path, label_path, annotations in candidate_images:
                if aug_count >= min_augmented:
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Horizontal flip
                if transforms.get("horizontal_flip", False):
                    flipped_img, flipped_anns = augment_image_horizontal_flip(img, annotations)
                    suffix = "_flipped"
                    cv2.imwrite(str(output_image_dir / f"{img_path.stem}{suffix}.jpg"), flipped_img)
                    write_label_file(output_label_dir / f"{img_path.stem}{suffix}.txt", flipped_anns)
                    aug_count += 1

                # Brightness
                if transforms.get("brightness", False) and aug_count < min_augmented:
                    alpha = rng.uniform(0.8, 1.2)
                    beta = rng.randint(-20, 20)
                    bright_img = augment_image_brightness(img, alpha, beta)
                    suffix = "_brightened"
                    cv2.imwrite(str(output_image_dir / f"{img_path.stem}{suffix}.jpg"), bright_img)
                    write_label_file(output_label_dir / f"{img_path.stem}{suffix}.txt", annotations)
                    aug_count += 1

        augmented_count += aug_count
        print(f"    Generated {aug_count} augmented images")

    return augmented_count


# ---------------------------------------------------------------------------
# Copy data
# ---------------------------------------------------------------------------

def copy_split(
    stems: list[str],
    src_image_dir: Path,
    src_label_dir: Path,
    dst_image_dir: Path,
    dst_label_dir: Path,
) -> None:
    """Copy images and labels for a set of stems."""
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        # Find the source image (could be .jpg, .jpeg, .png)
        for ext in [".jpg", ".jpeg", ".png"]:
            src_img = src_image_dir / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, dst_image_dir / src_img.name)
                break

        src_lbl = src_label_dir / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_label_dir / src_lbl.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset with stratified split and augmentation")
    parser.add_argument("--data-dir", type=str, required=True, help="Source data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for prepared data")
    parser.add_argument("--config", type=str, required=True, help="Augmentation config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    config = load_yaml(args.config)

    print("=" * 60)
    print("  DATA PREPARATION")
    print("=" * 60)

    # Collect all image stems across all splits
    print("\nCollecting all images...")
    all_stems = []
    src_image_dir = None
    src_label_dir = None
    for split in ["train", "valid", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if img_dir.exists():
            src_image_dir = img_dir
            src_label_dir = lbl_dir
            for img_path in list_image_files(img_dir):
                all_stems.append(img_path.stem)

    all_stems = sorted(set(all_stems))
    print(f"  Total unique images: {len(all_stems)}")

    # Original distribution
    print("\nOriginal class distribution:")
    all_label_dir = data_dir / "train" / "labels"
    original_counts = count_class_distribution(all_label_dir)
    for cls_id in range(NUM_CLASSES):
        print(f"  {cls_id}: {CLASS_NAMES[cls_id]:20s} = {original_counts.get(cls_id, 0)}")

    # Stratified split
    print("\nPerforming multilabel stratified split...")
    ratios = config.get("split_ratios", {"train": 0.80, "valid": 0.15, "test": 0.05})
    train_stems, valid_stems, test_stems = stratified_split(
        all_stems,
        all_label_dir,
        train_ratio=ratios["train"],
        valid_ratio=ratios["valid"],
        test_ratio=ratios["test"],
        seed=args.seed,
    )
    print(f"  Train: {len(train_stems)}, Valid: {len(valid_stems)}, Test: {len(test_stems)}")

    # Copy data to output
    print("\nCopying data...")
    for split_name, stems in [("train", train_stems), ("valid", valid_stems), ("test", test_stems)]:
        copy_split(
            stems,
            data_dir / "train" / "images",  # All images are in train originally
            data_dir / "train" / "labels",
            output_dir / split_name / "images",
            output_dir / split_name / "labels",
        )
        print(f"  {split_name}: {len(stems)} images copied")

    # Augment minority classes (train only)
    print("\nAugmenting minority classes...")
    target_classes = config.get("target_classes", [])
    exclude_classes = config.get("exclude_from_source", [])
    transforms = config.get("transforms", {})

    aug_count = augment_minority_classes(
        output_dir / "train" / "images",
        output_dir / "train" / "labels",
        output_dir / "train" / "images",  # Augment in-place
        output_dir / "train" / "labels",
        target_classes,
        exclude_classes,
        transforms,
    )
    print(f"  Total augmented: {aug_count}")

    # Post-split distribution
    print("\nPost-split class distribution:")
    for split_name in ["train", "valid", "test"]:
        counts = count_class_distribution(output_dir / split_name / "labels")
        total = sum(counts.values())
        print(f"  {split_name}: {total} annotations")

    # JSD between splits
    print("\nJensen-Shannon Divergence:")
    train_counts = count_class_distribution(output_dir / "train" / "labels")
    valid_counts = count_class_distribution(output_dir / "valid" / "labels")
    test_counts = count_class_distribution(output_dir / "test" / "labels")

    p_train = np.array([train_counts.get(i, 0) for i in range(NUM_CLASSES)])
    p_valid = np.array([valid_counts.get(i, 0) for i in range(NUM_CLASSES)])
    p_test = np.array([test_counts.get(i, 0) for i in range(NUM_CLASSES)])

    jsd_tv = compute_jsd(p_train, p_valid)
    jsd_tt = compute_jsd(p_train, p_test)
    thresholds = config.get("jsd_thresholds", {"acceptable": 0.10, "excellent": 0.05})

    print(f"  Train vs Valid: {jsd_tv:.4f} {'(excellent)' if jsd_tv < thresholds['excellent'] else '(acceptable)' if jsd_tv < thresholds['acceptable'] else '(WARNING)'}")
    print(f"  Train vs Test:  {jsd_tt:.4f} {'(excellent)' if jsd_tt < thresholds['excellent'] else '(acceptable)' if jsd_tt < thresholds['acceptable'] else '(WARNING)'}")

    # Plot distributions
    output_plots = output_dir / "plots"
    output_plots.mkdir(exist_ok=True)
    for split_name in ["train", "valid", "test"]:
        counts = count_class_distribution(output_dir / split_name / "labels")
        plot_class_distribution(counts, f"Class Distribution ({split_name})", output_plots / f"distribution_{split_name}.png")

    # Augmentation sanity visualization
    print("\nGenerating augmentation preview...")
    preview_images = []
    preview_bboxes = []
    preview_titles = []

    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    augmented_files = [f for f in list_image_files(train_img_dir) if "_flipped" in f.stem or "_brightened" in f.stem]

    if augmented_files:
        sample = random.sample(augmented_files, min(50, len(augmented_files)))
        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            label_path = train_lbl_dir / f"{img_path.stem}.txt"
            annotations = parse_label_file(label_path)
            preview_images.append(img)
            preview_bboxes.append(annotations)
            label_type = "flip" if "_flipped" in img_path.stem else "bright"
            preview_titles.append(f"{label_type}: {img_path.stem[:30]}")

        if preview_images:
            plot_augmentation_preview(
                preview_images, preview_bboxes, preview_titles,
                output_plots / "augmentation_preview.png",
            )
            print(f"  Saved preview with {len(preview_images)} samples")

    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Train: {len(list_image_files(output_dir / 'train' / 'images'))} images")
    print(f"  Valid: {len(list_image_files(output_dir / 'valid' / 'images'))} images")
    print(f"  Test:  {len(list_image_files(output_dir / 'test' / 'images'))} images")


if __name__ == "__main__":
    main()
