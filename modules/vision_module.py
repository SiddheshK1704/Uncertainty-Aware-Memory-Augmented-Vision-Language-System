"""
Vision Module
=============
Handles object detection using YOLOv8.
Returns bounding boxes, class labels, and confidence scores.

Supported COCO indoor classes:
    chair, tv, sofa, dining table, bottle, cup, laptop, mouse, keyboard, book
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


# COCO class IDs for indoor/household objects we focus on
INDOOR_CLASSES = {
    56: "chair",
    57: "couch",        # sofa
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    39: "bottle",
    41: "cup",
    73: "book",
}


@dataclass
class Detection:
    """Represents a single detected object."""
    label: str                        # Human-readable class name
    class_id: int                     # COCO class index
    confidence: float                 # Detection confidence [0, 1]
    bbox: List[float]                 # [x1, y1, x2, y2] in pixels
    center: tuple = field(init=False) # (cx, cy) computed from bbox

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        """Return bounding box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "center": (round(self.center[0], 1), round(self.center[1], 1)),
        }


class VisionModule:
    """
    YOLOv8-based object detection module.

    Example usage::

        vision = VisionModule(model_size="n", conf_threshold=0.4)
        detections = vision.detect(image_bgr)
        for det in detections:
            print(det.label, det.confidence, det.bbox)
    """

    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        filter_indoor: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            model_size:      YOLOv8 variant — 'n' (nano), 's', 'm', 'l', 'x'
            conf_threshold:  Minimum confidence to keep a detection
            iou_threshold:   NMS IoU threshold
            filter_indoor:   If True, only return indoor COCO classes
            device:          'cpu' or 'cuda'
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_indoor = filter_indoor
        self.device = device
        self.model = None
        self._model_size = model_size
        self._load_model(model_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a BGR image (OpenCV format).

        Args:
            image: H×W×3 uint8 numpy array (BGR)

        Returns:
            List of Detection objects sorted by confidence (descending)
        """
        if image is None or image.size == 0:
            logger.warning("VisionModule.detect() received empty image.")
            return []

        if self.model is None:
            logger.error("Model not loaded. Returning empty detections.")
            return []

        try:
            # YOLOv8 expects RGB; convert from BGR
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(
                rgb,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
            )
            detections = self._parse_results(results)
            logger.debug(f"Detected {len(detections)} objects.")
            return detections

        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return []

    def detect_with_fallback(
        self, image: np.ndarray, fallback_image: Optional[np.ndarray] = None
    ) -> List[Detection]:
        """
        Attempt detection; if result is empty, try fallback_image.

        Useful when the main image is too degraded.
        """
        detections = self.detect(image)
        if not detections and fallback_image is not None:
            logger.info("Primary detection empty, trying fallback image.")
            detections = self.detect(fallback_image)
        return detections

    def get_labels(self, detections: List[Detection]) -> List[str]:
        """Return just the label strings from a list of detections."""
        return [d.label for d in detections]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_size: str):
        """Load YOLOv8 model from ultralytics."""
        try:
            from ultralytics import YOLO
            model_name = f"yolov8{model_size}.pt"
            logger.info(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)
            logger.success(f"YOLOv8-{model_size} loaded successfully.")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
        except Exception as exc:
            logger.error(f"Failed to load YOLOv8: {exc}")

    def _parse_results(self, results) -> List[Detection]:
        """Convert raw YOLOv8 results into Detection dataclass list."""
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Get label from model names dict
                label = result.names.get(class_id, f"class_{class_id}")

                # Optionally filter to indoor classes only
                if self.filter_indoor and class_id not in INDOOR_CLASSES:
                    continue

                det = Detection(
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=xyxy,
                )
                detections.append(det)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections