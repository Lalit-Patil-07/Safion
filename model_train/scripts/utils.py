"""
Shared utilities for the Safion ML training pipeline.
"""

import hashlib
import json
import logging
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle",
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all random number generators for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label_file(path: str | Path) -> list[dict]:
    """Parse a YOLO label file. Returns list of {class_id, cx, cy, w, h}."""
    annotations = []
    if not os.path.exists(path):
        return annotations
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                logger.warning("Malformed line %d in %s: %s", line_num, path, line)
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append({"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h})
            except ValueError:
                logger.warning("Invalid values at line %d in %s: %s", line_num, path, line)
    return annotations


def write_label_file(path: str | Path, annotations: list[dict]) -> None:
    """Write YOLO label file from list of {class_id, cx, cy, w, h}."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def count_class_distribution(label_dir: str | Path) -> dict[int, int]:
    """Count instance occurrences per class across all label files."""
    counts = Counter()
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return dict(counts)
    for label_file in label_dir.glob("*.txt"):
        for ann in parse_label_file(label_file):
            counts[ann["class_id"]] += 1
    return dict(sorted(counts.items()))


def count_class_presence(label_dir: str | Path) -> dict[int, int]:
    """Count how many images contain at least one instance of each class."""
    presence = Counter()
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return dict(presence)
    for label_file in label_dir.glob("*.txt"):
        classes_in_file = set()
        for ann in parse_label_file(label_file):
            classes_in_file.add(ann["class_id"])
        for cls in classes_in_file:
            presence[cls] += 1
    return dict(sorted(presence.items()))


def compute_cooccurrence(label_dir: str | Path) -> np.ndarray:
    """Compute class co-occurrence matrix (how often classes appear in same image)."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return matrix
    for label_file in label_dir.glob("*.txt"):
        classes_in_file = set()
        for ann in parse_label_file(label_file):
            classes_in_file.add(ann["class_id"])
        classes_list = sorted(classes_in_file)
        for i, cls_i in enumerate(classes_list):
            for cls_j in classes_list[i:]:
                matrix[cls_i][cls_j] += 1
                if cls_i != cls_j:
                    matrix[cls_j][cls_i] += 1
    return matrix


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    m = 0.5 * (p + q)
    # KL divergence with smoothing
    eps = 1e-10
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def compute_distribution_metrics(counts: dict[int, int]) -> dict:
    """Compute distribution quality metrics."""
    values = np.array([counts.get(i, 0) for i in range(NUM_CLASSES)], dtype=np.float64)
    total = values.sum()
    if total == 0:
        return {"std": 0, "cv": 0, "entropy": 0}
    probs = values / total
    std = float(np.std(values))
    mean = float(np.mean(values))
    cv = std / mean if mean > 0 else 0
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    return {"std": std, "cv": cv, "entropy": entropy, "mean": mean, "total": int(total)}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_class_distribution(
    counts: dict[int, int],
    title: str,
    save_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Plot and save a class distribution bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = class_names or CLASS_NAMES
    values = [counts.get(i, 0) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(NUM_CLASSES), values, color="steelblue")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cooccurrence_matrix(
    matrix: np.ndarray,
    save_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Plot and save a co-occurrence heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = class_names or CLASS_NAMES
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    plt.colorbar(im, ax=ax)
    ax.set_title("Class Co-occurrence Matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                        color="white" if val > matrix.max() * 0.6 else "black")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_augmentation_preview(
    images: list[np.ndarray],
    bboxes_list: list[list[dict]],
    titles: list[str],
    save_path: str | Path,
    max_cols: int = 5,
) -> None:
    """Render augmented samples with bounding boxes for sanity checking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx < n:
            img = images[idx].copy()
            h, w = img.shape[:2]
            for bbox in bboxes_list[idx]:
                cls_id = bbox["class_id"]
                cx, cy, bw, bh = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = (0, 255, 0) if cls_id < NUM_CLASSES else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else str(cls_id)
                cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(titles[idx], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

def validate_label_file(path: str | Path, num_classes: int = NUM_CLASSES) -> list[str]:
    """Validate a single YOLO label file. Returns list of issues."""
    issues = []
    annotations = parse_label_file(path)

    if not annotations:
        issues.append("Empty label file")
        return issues

    seen_boxes = []
    for i, ann in enumerate(annotations):
        cls = ann["class_id"]
        cx, cy, w, h = ann["cx"], ann["cy"], ann["w"], ann["h"]

        if cls < 0 or cls >= num_classes:
            issues.append(f"Line {i+1}: Invalid class ID {cls} (valid: 0-{num_classes-1})")

        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0 or val > 1:
                issues.append(f"Line {i+1}: {name}={val} outside [0,1]")

        if w <= 0 or h <= 0:
            issues.append(f"Line {i+1}: w={w}, h={h} must be > 0")

        # Check for duplicates
        for prev_cls, prev_cx, prev_cy in seen_boxes:
            if cls == prev_cls and abs(cx - prev_cx) < 0.01 and abs(cy - prev_cy) < 0.01:
                issues.append(f"Line {i+1}: Duplicate box for class {cls} near ({cx:.3f},{cy:.3f})")
        seen_boxes.append((cls, cx, cy))

    return issues


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd="/root/Safion"
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_gpu_info() -> dict:
    """Get GPU information."""
    info = {"available": False, "name": "N/A", "count": 0, "cuda_version": "N/A", "vram_mb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            info["vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except ImportError:
        pass
    return info


def compute_dataset_hash(data_dir: str | Path) -> str:
    """Compute a hash of the dataset for checkpoint validation."""
    data_dir = Path(data_dir)
    hasher = hashlib.md5()
    for split in ["train", "valid", "test"]:
        label_dir = data_dir / split / "labels"
        if label_dir.exists():
            for label_file in sorted(label_dir.glob("*.txt")):
                hasher.update(label_file.name.encode())
                with open(label_file, "rb") as f:
                    hasher.update(f.read())
    return hasher.hexdigest()


def get_experiment_metadata(config: dict, data_dir: str | Path) -> dict:
    """Collect full experiment metadata for reproducibility."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "dataset_hash": compute_dataset_hash(data_dir),
        "gpu_info": get_gpu_info(),
        "config": config,
    }


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def list_image_files(image_dir: str | Path) -> list[Path]:
    """List all image files in a directory."""
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])


def get_label_path_for_image(image_path: Path) -> Path:
    """Get the label file path for an image file."""
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def verify_image_label_pairs(image_dir: str | Path, label_dir: str | Path) -> tuple[list[str], list[str]]:
    """Verify image-label pairing. Returns (missing_labels, missing_images)."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_stems = {f.stem for f in list_image_files(image_dir)}
    label_stems = {f.stem for f in label_dir.glob("*.txt")} if label_dir.exists() else set()

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    return missing_labels, missing_images
