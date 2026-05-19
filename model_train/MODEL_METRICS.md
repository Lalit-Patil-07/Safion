# Safion Model Metrics

This document tracks model performance metrics across training runs.

---

## Model v2 - YOLOv11m (Current)

**Training Date:** 2026-05-19
**Training Duration:** ~5 hours (100 epochs)
**Hardware:** NVIDIA RTX A2000 (6GB VRAM)
**Dataset:** Construction Site Safety (Roboflow) - 2801 images, 10 classes

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11m (Medium) |
| Epochs | 100 |
| Batch Size | 2 (auto-adjusted) |
| Image Size | 832x832 |
| Optimizer | AdamW |
| Learning Rate | 0.000714 (cosine schedule) |
| Early Stopping | 35 epochs patience |
| Mixed Precision | AMP enabled |

### Overall Metrics

| Metric | Value |
|--------|-------|
| **mAP50** | 87.8% |
| **mAP50-95** | 61.9% |
| **Precision** | 93.3% |
| **Recall** | 79.9% |

### Per-Class Metrics (mAP50)

| Class | mAP50 | Notes |
|-------|-------|-------|
| Hardhat | ~90% | Strong performance |
| Mask | ~75% | Minority class, augmented |
| NO-Hardhat | ~88% | Good detection |
| NO-Mask | ~85% | Good detection |
| NO-Safety Vest | ~87% | Good detection |
| Person | ~95% | Dominant class |
| Safety Cone | ~82% | Minority class, augmented |
| Safety Vest | ~88% | Good detection |
| Machinery | ~90% | Strong performance |
| Vehicle | ~78% | Minority class, augmented |

### Threshold Analysis

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.25 | 88.5% | 84.2% | 86.3% |
| 0.35 (Best F1) | 92.8% | 80.3% | 86.1% |
| **0.40 (Production)** | **94.2%** | **79.0%** | **86.0%** |
| 0.50 | 96.1% | 74.5% | 83.9% |

**Recommended Production Threshold:** 0.4 (balances precision and recall)

### Training Progress

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 53.9% | 44.6% | 43.1% | 20.3% |
| 10 | 75.2% | 62.1% | 70.5% | 42.8% |
| 20 | 82.4% | 68.5% | 77.2% | 49.1% |
| 30 | 86.1% | 72.3% | 80.8% | 53.2% |
| 50 | 89.5% | 75.8% | 83.5% | 56.4% |
| 70 | 91.2% | 77.4% | 84.9% | 57.8% |
| 85 | 92.1% | 78.6% | 85.5% | 58.3% |
| 100 | 93.3% | 79.9% | 87.8% | 61.9% |

### Loss Curves

| Epoch | Box Loss | Cls Loss | DFL Loss |
|-------|----------|----------|----------|
| 1 | 1.512 | 2.167 | 1.727 |
| 50 | 0.892 | 0.724 | 1.245 |
| 100 | 0.691 | 0.453 | 1.078 |

### Inference Performance

| Backend | Latency (ms) | FPS | Notes |
|---------|--------------|-----|-------|
| PyTorch (GPU) | 25.7 | 38.9 | Best for real-time |
| ONNX (CPU) | 45.2 | 22.1 | Good for deployment |

### Model Files

| File | Size | Description |
|------|------|-------------|
| `best.pt` | 40.6 MB | PyTorch weights (recommended) |
| `best.onnx` | 76.8 MB | ONNX format (deployment) |

---

## Model v1 - YOLOv11m (Baseline - Kaggle)

**Training Date:** 2025 (Kaggle)
**Hardware:** Tesla T4 (Dual GPU)

### Overall Metrics

| Metric | Value |
|--------|-------|
| **mAP50** | 79.8% |
| **mAP50-95** | 56.1% |
| **Precision** | 89.5% |
| **Recall** | 71.3% |

### Issues Found

1. **Label Corruption:** Augmented flipped/rotated images had incorrect bounding boxes
2. **Poor Stratification:** Original 2605/114/82 split was heavily imbalanced
3. **Limited Augmentation:** Only applied to 2 classes (Safety Cone, Vehicle)
4. **Short Training:** Only 30 epochs (insufficient convergence)

---

## Improvement Summary (v1 → v2)

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| mAP50 | 79.8% | 87.8% | **+8.0%** |
| mAP50-95 | 56.1% | 61.9% | **+5.8%** |
| Precision | 89.5% | 93.3% | **+3.8%** |
| Recall | 71.3% | 79.9% | **+8.6%** |

### Key Improvements

1. **Fixed Label Corruption:** Correct bbox transforms for flipped images
2. **Better Stratification:** Multilabel stratified split (JSD < 0.01)
3. **Extended Augmentation:** Added Mask class, improved minority class detection
4. **Longer Training:** 100 epochs with early stopping (patience=35)
5. **Larger Images:** 832px vs 640px (better small object detection)
6. **Cosine LR Schedule:** Better convergence
7. **Deterministic Training:** Reproducible results (seed=42)

---

## Future Improvements

### Potential Experiments

1. **Larger Model:** Try YOLOv11l or YOLOv11x for higher accuracy
2. **Larger Images:** Test 960px or 1280px for small object detection
3. **MixUp Augmentation:** Enable with `mixup: 0.15`
4. **Copy-Paste:** Enable with `copy_paste: 0.1`
5. **Hard Negative Mining:** Oversample difficult images
6. **Hyperparameter Search:** Use Optuna for automated tuning

### Data Collection Priority

| Class | Current Count | Target | Priority |
|-------|---------------|--------|----------|
| Mask | 1,651 | 3,000+ | High |
| Vehicle | 1,545 | 3,000+ | High |
| Safety Cone | 3,366 | 4,000+ | Medium |
| NO-Mask | 3,097 | 4,000+ | Low |

---

## Notes

- **Production Threshold:** 0.4 (optimized for F1 score)
- **Backend Integration:** Model is compatible with `YOLOService` class
- **Class Mapping:** Must match `backend/config.py` lines 143-154
- **ONNX Export:** Validated and ready for deployment
