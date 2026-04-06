"""
YOLOService
===========
Wraps the Ultralytics YOLO model as a thread-safe singleton.

Design decisions
----------------
- A single model instance is shared across all stream workers.
- A threading.Lock() serialises GPU calls — CUDA context sharing across
  threads without explicit management causes undefined behaviour.
- On CPU the lock still prevents GIL contention from multiplying memory
  pressure across many threads.
- inference() returns a clean list of detection dicts; callers never
  touch torch tensors directly.
"""

import logging
import threading
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class YOLOService:
    def __init__(self):
        self._model = None
        self._device: str = "cpu"
        self._confidence: float = 0.4
        self._ppe_classes: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_app(self, app) -> None:
        """Call from the app factory after Flask app is configured."""
        from ultralytics import YOLO

        model_path: str = app.config["MODEL_PATH"]
        self._confidence = app.config["CONFIDENCE_THRESHOLD"]
        self._ppe_classes = app.config["PPE_CLASSES"]

        yolo_device_cfg: str = app.config.get("YOLO_DEVICE", "auto")
        self._device = self._resolve_device(yolo_device_cfg)

        try:
            logger.info("Loading YOLO model from '%s' on device '%s'...", model_path, self._device)
            self._model = YOLO(model_path)
            self._model.to(self._device)
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            self._model = None

    @staticmethod
    def _resolve_device(cfg: str) -> str:
        if cfg == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA available — using GPU.")
                return "cuda"
            logger.warning("CUDA not available — falling back to CPU.")
            return "cpu"
        return cfg

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO detection on a single BGR frame.

        Parameters
        ----------
        frame : BGR numpy array (H, W, 3)

        Returns
        -------
        List of detection dicts:
        {
            "class_id"   : int,
            "class_name" : str,
            "confidence" : float,
            "bbox"       : [x1, y1, x2, y2],   # floats
            "color"      : "#rrggbb",
            "safe"       : bool,
        }
        """
        if self._model is None:
            return []

        with self._lock:
            results = self._model(
                frame,
                device=self._device,
                conf=self._confidence,
                verbose=False,
            )[0]

        detections: list[dict] = []
        for det in results.boxes.data:
            x1, y1, x2, y2, confidence, class_id = det.cpu().numpy()
            cls_info = self._ppe_classes.get(int(class_id))
            if cls_info is None:
                continue
            detections.append({
                "class_id": int(class_id),
                "class_name": cls_info["name"],
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "color": cls_info["color"],
                "safe": cls_info["safe"],
            })

        return detections
