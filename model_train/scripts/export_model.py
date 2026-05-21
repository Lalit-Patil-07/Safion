"""
Export trained YOLO model to ONNX format for production deployment.

Usage:
    python scripts/export_model.py --model model_train/outputs/run_XXXX/weights/best.pt --output-dir model_train/outputs/run_XXXX/export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils import CLASS_NAMES, NUM_CLASSES, load_yaml


def export_to_onnx(model_path: str, output_dir: Path, imgsz: int = 832) -> Path:
    """Export model to ONNX format."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    print(f"\nExporting model to ONNX...")
    print(f"  Input size: {imgsz}x{imgsz}")

    export_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
    print(f"  Exported to: {export_path}")

    # Copy to output directory
    import shutil
    onnx_path = output_dir / "best.onnx"
    shutil.copy2(export_path, onnx_path)
    print(f"  Copied to: {onnx_path}")

    return onnx_path


def validate_onnx_export(model_path: str, onnx_path: str, imgsz: int = 832) -> bool:
    """Validate that ONNX model produces matching outputs."""
    import cv2

    print("\nValidating ONNX export...")

    # Load both models
    from ultralytics import YOLO
    pt_model = YOLO(model_path)
    onnx_model = YOLO(onnx_path)

    # Create test image
    test_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Run inference
    pt_results = pt_model(test_img, verbose=False)[0]
    onnx_results = onnx_model(test_img, verbose=False)[0]

    pt_boxes = pt_results.boxes.data.cpu().numpy()
    onnx_boxes = onnx_results.boxes.data.cpu().numpy()

    print(f"  PT detections:   {len(pt_boxes)}")
    print(f"  ONNX detections: {len(onnx_boxes)}")

    # Check if detections are similar (tolerance for floating point)
    if len(pt_boxes) == len(onnx_boxes) and len(pt_boxes) > 0:
        max_diff = np.max(np.abs(pt_boxes[:, :4] - onnx_boxes[:, :4]))
        print(f"  Max bbox difference: {max_diff:.6f}")
        if max_diff < 0.01:
            print("  Validation PASSED: Outputs match within tolerance")
            return True
        else:
            print("  WARNING: Large bbox difference detected")
            return False
    elif len(pt_boxes) == 0 and len(onnx_boxes) == 0:
        print("  Validation PASSED: Both models produce no detections (expected for random image)")
        return True
    else:
        print(f"  WARNING: Detection count mismatch")
        return False


def generate_metadata(model_path: str, onnx_path: str, output_dir: Path, imgsz: int) -> None:
    """Generate model metadata for deployment."""
    metadata = {
        "model_format": "onnx",
        "input_shape": [1, 3, imgsz, imgsz],
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_mapping": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "source_model": str(model_path),
        "onnx_model": str(onnx_path),
        "opset": 17,
        "simplified": True,
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model weights")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=832, help="Input image size")
    parser.add_argument("--validate", action="store_true", help="Validate ONNX export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MODEL EXPORT")
    print("=" * 60)

    # Export
    onnx_path = export_to_onnx(args.model, output_dir, args.imgsz)

    # Validate
    if args.validate:
        validate_onnx_export(args.model, str(onnx_path), args.imgsz)

    # Generate metadata
    generate_metadata(args.model, str(onnx_path), output_dir, args.imgsz)

    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE")
    print("=" * 60)
    print(f"  ONNX model: {onnx_path}")
    print(f"  Metadata:   {output_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
