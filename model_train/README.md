# Safion Model Training Pipeline

End-to-end YOLO training pipeline for PPE (Personal Protective Equipment) detection. This pipeline handles data validation, preparation, training, evaluation, and model export.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Data Setup](#1-data-setup)
  - [2. Validate Dataset](#2-validate-dataset)
  - [3. Analyze Data](#3-analyze-data)
  - [4. Prepare Data](#4-prepare-data)
  - [5. Train Model](#5-train-model)
  - [6. Evaluate Model](#6-evaluate-model)
  - [7. Export Model](#7-export-model)
  - [8. Benchmark (Optional)](#8-benchmark-optional)
- [Configuration](#configuration)
- [Resuming Training](#resuming-training)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline trains a YOLOv11m model to detect 10 PPE-related classes in construction site images:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Hardhat | Person wearing hardhat |
| 1 | Mask | Person wearing mask |
| 2 | NO-Hardhat | Person without hardhat (violation) |
| 3 | NO-Mask | Person without mask (violation) |
| 4 | NO-Safety Vest | Person without safety vest (violation) |
| 5 | Person | General person detection |
| 6 | Safety Cone | Safety cone |
| 7 | Safety Vest | Person wearing safety vest |
| 8 | Machinery | Construction machinery |
| 9 | Vehicle | Vehicle |

---

## Requirements

### Python Dependencies

```bash
pip install ultralytics>=8.4.0 opencv-python-headless pyyaml scikit-learn matplotlib numpy
```

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended: 6GB+ VRAM)
- **RAM**: 8GB+ system RAM
- **Storage**: ~5GB for data + outputs

### Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Directory Structure

```
model_train/
├── config/
│   ├── data.yaml                 # Dataset paths and class names
│   ├── train_hyperparams.yaml    # Training hyperparameters
│   └── augmentation.yaml         # Data augmentation settings
├── scripts/
│   ├── utils.py                  # Shared utilities
│   ├── validate_labels.py        # Dataset validation (run first!)
│   ├── analyze_data.py           # Exploratory data analysis
│   ├── prepare_data.py           # Data splitting and augmentation
│   ├── train.py                  # Model training
│   ├── evaluate.py               # Model evaluation
│   ├── export_model.py           # ONNX export
│   └── benchmark.py              # Inference benchmarking
├── data/                         # [CREATED BY USER] Raw training data
│   ├── train/
│   │   ├── images/               # Training images (.jpg)
│   │   └── labels/               # Training labels (.txt, YOLO format)
│   ├── valid/
│   │   ├── images/               # Validation images
│   │   └── labels/               # Validation labels
│   └── test/
│       ├── images/               # Test images
│       └── labels/               # Test labels
├── data_prepared/                # [GENERATED] Stratified + augmented data
└── outputs/                      # [GENERATED] Training runs and results
    └── run_TIMESTAMP/
        ├── best.pt               # Best model weights
        ├── train/                # Training artifacts
        ├── evaluation/           # Evaluation results
        └── export/               # ONNX model
```

---

## Quick Start

```bash
cd /root/Safion/model_train/scripts

# 1. Validate data
python validate_labels.py --data-dir ../data --strict

# 2. Prepare data (stratified split + augmentation)
python prepare_data.py --data-dir ../data --output-dir ../data_prepared --config ../config/augmentation.yaml

# 3. Train model
python train.py --config ../config/train_hyperparams.yaml --data ../config/data.yaml --output-dir ../outputs/run_$(date +%Y%m%d_%H%M%S)

# 4. Evaluate model
python evaluate.py --model ../outputs/run_XXXX/best.pt --data ../config/data.yaml --output-dir ../outputs/run_XXXX/evaluation --sweep

# 5. Export to ONNX
python export_model.py --model ../outputs/run_XXXX/best.pt --output-dir ../outputs/run_XXXX/export --validate
```

---

## Step-by-Step Guide

### 1. Data Setup

#### Option A: Use Existing Dataset

If you have the Roboflow Construction Site Safety dataset, place it in `model_train/data/`:

```
data/
├── train/
│   ├── images/    # .jpg files
│   └── labels/    # .txt files (YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### Option B: Use Custom Data

Prepare your data in YOLO format. Each label file should contain one line per object:

```
<class_id> <center_x> <center_y> <width> <height>
```

Example (`image001.txt`):
```
0 0.4703125 0.4421875 0.86796875 0.7875
5 0.75625 0.4640625 0.33984375 0.3890625
```

All coordinates must be normalized to [0, 1].

#### Option C: Download from Roboflow

```bash
# Install roboflow
pip install roboflow

# Download dataset (requires API key)
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('roboflow-universe').project('construction-site-safety-image-dataset-roboflow')
version = project.version(2)
dataset = version.download('yolov11')
"
```

---

### 2. Validate Dataset

**Always run this first!** It checks for data quality issues that could waste training time.

```bash
cd /root/Safion/model_train/scripts

# Basic validation
python validate_labels.py --data-dir ../data

# Strict mode (exits with error on any issue)
python validate_labels.py --data-dir ../data --strict

# Save report to file
python validate_labels.py --data-dir ../data --output ../outputs/validation_report.json
```

**What it checks:**
- Image-label file pairing (missing labels or images)
- Corrupt images (unreadable files)
- Invalid class IDs (outside 0-9 range)
- Invalid bounding box coordinates (outside [0,1])
- Zero-size boxes (width or height <= 0)
- Duplicate annotations (same class at same location)
- Empty label files

**Example output:**
```
============================================================
  DATASET VALIDATION REPORT
============================================================

Total: 2801 images, 2801 labels, 24944 annotations
Critical issues: 0
Warnings: 620
Status: PASSED

--- TRAIN ---
  Images: 2605, Labels: 2605, Annotations: 24944
  Empty labels (42): ['img001', 'img002', ...]
  Invalid labels (578 files):
    img003.txt: Line 1: Duplicate box for class 5 near (0.470,0.442)

PASSED: Dataset validation successful.
```

**Fix any critical issues before proceeding!**

---

### 3. Analyze Data

Generate detailed statistics and visualizations about your dataset.

```bash
python analyze_data.py --data-dir ../data --output-dir ../outputs/eda
```

**Outputs:**
- `outputs/eda/distribution_train.png` - Class distribution chart
- `outputs/eda/cooccurrence_train.png` - Class co-occurrence heatmap
- `outputs/eda/analysis_report.json` - Detailed statistics

**Key metrics to check:**
- **Class imbalance**: Look for classes with very few instances
- **Co-occurrence**: Which classes appear together frequently
- **Jensen-Shannon Divergence (JSD)**: Should be < 0.10 between splits

---

### 4. Prepare Data

Create stratified train/valid/test split and augment minority classes.

```bash
python prepare_data.py \
  --data-dir ../data \
  --output-dir ../data_prepared \
  --config ../config/augmentation.yaml
```

**What it does:**

1. **Multilabel Stratified Split**
   - Combines all images from original splits
   - Redistributes using stratified sampling (preserves class distribution)
   - Target: 80% train, 15% valid, 5% test

2. **Data Augmentation** (for minority classes)
   - Horizontal flip (with correct bbox coordinate transform)
   - Brightness adjustment
   - Targets: Mask, Vehicle, Safety Cone classes

3. **Sanity Visualization**
   - Saves preview of augmented images with bounding boxes
   - Check `outputs/augmentation_preview.png` to verify labels are correct

**Example output:**
```
============================================================
  DATA PREPARATION
============================================================

Collecting all images...
  Total unique images: 2801

Performing multilabel stratified split...
  Train: 2240, Valid: 421, Test: 140

Augmenting minority classes...
  Augmenting class Mask: 321 source images, target 300
    Generated 300 augmented images
  Augmenting class Vehicle: 153 source images, target 300
    Generated 300 augmented images
  Augmenting class Safety Cone: 298 source images, target 200
    Generated 200 augmented images

Jensen-Shannon Divergence:
  Train vs Valid: 0.0014 (excellent)
  Train vs Test:  0.0006 (excellent)

============================================================
  DATA PREPARATION COMPLETE
============================================================
  Train: 3040 images
  Valid: 421 images
  Test:  140 images
```

---

### 5. Train Model

Train the YOLOv11m model with the prepared data.

```bash
# Basic training
python train.py \
  --config ../config/train_hyperparams.yaml \
  --data ../config/data.yaml \
  --output-dir ../outputs/run_$(date +%Y%m%d_%H%M%S)

# Training with nohup (persists if disconnected)
nohup python train.py \
  --config ../config/train_hyperparams.yaml \
  --data ../config/data.yaml \
  --output-dir ../outputs/run_$(date +%Y%m%d_%H%M%S) \
  > ../outputs/training.log 2>&1 &
```

**Training Parameters (from `train_hyperparams.yaml`):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | yolo11m.pt | Base model (medium) |
| `epochs` | 100 | Training epochs |
| `batch` | -1 | Auto batch size (adjusts to GPU) |
| `imgsz` | 832 | Input image size |
| `patience` | 35 | Early stopping patience |
| `cos_lr` | true | Cosine learning rate schedule |
| `close_mosaic` | 10 | Disable mosaic augmentation for last N epochs |
| `amp` | true | Mixed precision training |

**Monitor Training:**

```bash
# Watch live progress
tail -f ../outputs/run_XXXX/training.log

# Check current epoch
grep -oP '\d+/100' ../outputs/run_XXXX/training.log | tail -1

# View metrics
cat ../outputs/run_XXXX/train/results.csv
```

**Training Metrics (results.csv columns):**
- `train/box_loss` - Bounding box regression loss
- `train/cls_loss` - Classification loss
- `train/dfl_loss` - Distribution focal loss
- `metrics/precision(B)` - Validation precision
- `metrics/recall(B)` - Validation recall
- `metrics/mAP50(B)` - Mean AP at IoU=0.50
- `metrics/mAP50-95(B)` - Mean AP at IoU=0.50:0.95

**Expected Training Time:**
- GPU (NVIDIA RTX A2000): ~4 hours for 100 epochs
- GPU (NVIDIA RTX 3080): ~2 hours for 100 epochs
- CPU: Not recommended (20+ hours)

---

### 6. Evaluate Model

Run comprehensive evaluation on the test set.

```bash
python evaluate.py \
  --model ../outputs/run_XXXX/best.pt \
  --data ../config/data.yaml \
  --output-dir ../outputs/run_XXXX/evaluation \
  --sweep
```

**Options:**
- `--sweep` - Run confidence threshold sweep (finds optimal F1)
- `--tta` - Run Test-Time Augmentation evaluation

**Outputs:**
- `evaluation/evaluation_results.json` - Full metrics
- `evaluation/per_class_ap.png` - Per-class AP chart
- `evaluation/threshold_sweep.png` - Precision/Recall/F1 vs confidence

**Key Metrics:**
```
Overall:
  mAP50:    87.8%
  mAP50-95: 61.9%
  Precision: 93.3%
  Recall:   79.9%

Threshold Sweep:
  Best F1 threshold: 0.35 (F1=86.1%)
  Production threshold (0.4): P=94.2%, R=79.0%, F1=86.0%
```

**Per-Class Analysis:**
Check which classes have lower AP and may need more training data or augmentation.

---

### 7. Export Model

Export trained model to ONNX format for production deployment.

```bash
python export_model.py \
  --model ../outputs/run_XXXX/best.pt \
  --output-dir ../outputs/run_XXXX/export \
  --imgsz 832 \
  --validate
```

**Outputs:**
- `export/best.onnx` - ONNX model (76.8 MB)
- `export/model_metadata.json` - Class mapping and input shape

**Validate Export:**
```bash
# Test ONNX model
python -c "
from ultralytics import YOLO
model = YOLO('../outputs/run_XXXX/export/best.onnx')
results = model('test_image.jpg')
print(results[0].boxes)
"
```

---

### 8. Benchmark (Optional)

Compare inference speed between PyTorch and ONNX models.

```bash
python benchmark.py \
  --model ../outputs/run_XXXX/best.pt \
  --onnx ../outputs/run_XXXX/export/best.onnx \
  --imgsz 832 \
  --output ../outputs/run_XXXX/benchmark.json
```

**Example Output:**
```
============================================================
  INFERENCE BENCHMARK
============================================================

GPU: NVIDIA RTX A2000 (6138 MB)
CUDA: 12.8

--- PyTorch Benchmark ---
  Framework: pytorch
  Mean:      25.7 ms
  FPS:       38.9

--- ONNX Benchmark ---
  Framework: onnx
  Mean:      45.2 ms
  FPS:       22.1

--- Comparison ---
  PT vs ONNX speedup: 0.57x (PyTorch faster on GPU)
```

---

## Configuration

### train_hyperparams.yaml

```yaml
# Model
model: yolo11m.pt          # Options: yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt

# Training
epochs: 100
batch: -1                   # -1 = auto batch size
imgsz: 832                  # Input image size (832, 960, 1280)
patience: 35                # Early stopping patience
seed: 42                    # Random seed for reproducibility
deterministic: true         # Reproducible training

# Optimizer
optimizer: auto             # Auto-selects best optimizer
lr0: 0.01                   # Initial learning rate
lrf: 0.01                   # Final LR factor
momentum: 0.937
weight_decay: 0.0005

# Augmentation
mosaic: 1.0                 # Mosaic augmentation
mixup: 0.0                  # MixUp (set >0 to enable)
copy_paste: 0.0             # Copy-paste (set >0 to enable)
hsv_h: 0.015                # Hue augmentation
hsv_s: 0.7                  # Saturation
hsv_v: 0.4                  # Value/brightness
degrees: 10.0               # Rotation
fliplr: 0.5                 # Horizontal flip
```

### augmentation.yaml

```yaml
# Classes to augment (minority classes)
target_classes:
  - id: 1                    # Mask
    min_augmented_images: 300
  - id: 9                    # Vehicle
    min_augmented_images: 300
  - id: 6                    # Safety Cone
    min_augmented_images: 200

# Augmentation transforms
transforms:
  horizontal_flip: true
  brightness: true
  rotation: false            # Disabled until bbox transform validated

# Split ratios
split_ratios:
  train: 0.80
  valid: 0.15
  test: 0.05
```

### data.yaml

```yaml
path: /root/Safion/model_train/data_prepared
train: train/images
val: valid/images
test: test/images
nc: 10
names:
  - Hardhat
  - Mask
  - NO-Hardhat
  - NO-Mask
  - NO-Safety Vest
  - Person
  - Safety Cone
  - Safety Vest
  - Machinery
  - Vehicle
```

---

## Resuming Training

If training is interrupted, resume from the last checkpoint:

```bash
python train.py \
  --resume ../outputs/run_XXXX/train/weights/last.pt \
  --config ../config/train_hyperparams.yaml \
  --data ../config/data.yaml
```

**Checkpoint Files:**
- `best.pt` - Best model weights (lowest validation loss)
- `last.pt` - Latest checkpoint (for resuming)
- `epochXX.pt` - Periodic checkpoints (every 10 epochs)

**Resume Safety:**
The script automatically verifies:
- Model architecture matches
- Dataset hash matches
- Class mapping is consistent

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
# In train_hyperparams.yaml:
batch: 4                    # or batch: 2

# Or reduce image size
imgsz: 640
```

### Training Killed (OOM)

Reduce dataloader workers:
```yaml
# In train_hyperparams.yaml:
workers: 2                  # Default: 8
```

### Poor mAP on Minority Classes

1. Increase augmentation in `augmentation.yaml`:
   ```yaml
   target_classes:
     - id: 1
       min_augmented_images: 500  # Increase from 300
   ```

2. Add MixUp augmentation in `train_hyperparams.yaml`:
   ```yaml
   mixup: 0.15
   ```

### Low Recall

1. Lower confidence threshold during training:
   ```yaml
   conf: 0.001               # Very low for training
   ```

2. Use larger image size:
   ```yaml
   imgsz: 960                # or 1280
   ```

### Resume Not Working

Check if checkpoint exists:
```bash
ls -la ../outputs/run_XXXX/train/weights/
```

Verify args.yaml in checkpoint directory:
```bash
cat ../outputs/run_XXXX/train/args.yaml
```

---

## Deployment

After training, copy the best model to the project root:

```bash
cp ../outputs/run_XXXX/best.pt /root/Safion/best.pt
```

The backend (`backend/detection/yolo_service.py`) will automatically use the new model.

**Production Confidence Threshold:** 0.4 (optimized during evaluation)

---

## File Reference

| File | Purpose |
|------|---------|
| `config/data.yaml` | Dataset paths and class names |
| `config/train_hyperparams.yaml` | Training hyperparameters |
| `config/augmentation.yaml` | Augmentation settings |
| `scripts/utils.py` | Shared utilities |
| `scripts/validate_labels.py` | Dataset validation |
| `scripts/analyze_data.py` | Data analysis and EDA |
| `scripts/prepare_data.py` | Data splitting and augmentation |
| `scripts/train.py` | Model training |
| `scripts/evaluate.py` | Model evaluation |
| `scripts/export_model.py` | ONNX export |
| `scripts/benchmark.py` | Inference benchmarking |
