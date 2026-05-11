"""
Visualization Utilities
=======================
Drawing functions for detections, grounding results, confidence bars,
and memory recall overlays on images.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from loguru import logger

from modules.vision_module import Detection
from modules.grounding_module import GroundingResult
from modules.decision_module import ActionResult, ActionType
from modules.memory_module import MemoryEntry


# Color palette (BGR) indexed by label for consistency
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "chair":        (255, 128,   0),
    "couch":        (  0, 200, 255),
    "tv":           ( 50, 205,  50),
    "dining table": (255,  50, 255),
    "bottle":       ( 50, 150, 255),
    "cup":          (255, 200,   0),
    "laptop":       (  0, 255, 128),
    "mouse":        (255,  80, 180),
    "keyboard":     (100, 180, 255),
    "book":         (200, 100,  50),
}
DEFAULT_COLOR = (180, 180, 180)


def _get_color(label: str) -> Tuple[int, int, int]:
    return LABEL_COLORS.get(label.lower(), DEFAULT_COLOR)


class VisualizationUtils:
    """
    Static drawing methods for the UVLA pipeline.

    All methods accept and return BGR numpy arrays (OpenCV format).
    """

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        grounding_result: Optional[GroundingResult] = None,
        thickness: int = 2,
        font_scale: float = 0.55,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels for all detections.
        Highlights the grounded target with a thicker, brighter box.

        Args:
            image:            BGR image to draw on (will be copied)
            detections:       List of Detection objects
            grounding_result: If provided, highlights the grounded target
            thickness:        Box line thickness
            font_scale:       Label text size

        Returns:
            Annotated BGR image
        """
        canvas = image.copy()
        target_label = (
            grounding_result.target_detection.label
            if grounding_result and grounding_result.grounded and grounding_result.target_detection
            else None
        )

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = _get_color(det.label)
            is_target = det.label == target_label

            # Thicker box and brighter color for the grounded target
            line_w = thickness + 2 if is_target else thickness
            draw_color = color if not is_target else tuple(min(c + 80, 255) for c in color)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), draw_color, line_w)

            # Label background
            label_text = f"{det.label} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            pad = 4
            bg_y1 = max(y1 - th - 2 * pad, 0)
            cv2.rectangle(
                canvas,
                (x1, bg_y1),
                (x1 + tw + 2 * pad, y1),
                draw_color,
                -1,
            )
            cv2.putText(
                canvas,
                label_text,
                (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Star marker on grounded target
            if is_target:
                cx, cy = int(det.center[0]), int(det.center[1])
                cv2.drawMarker(
                    canvas, (cx, cy), (0, 255, 255),
                    cv2.MARKER_STAR, 20, 2
                )

        return canvas

    @staticmethod
    def draw_action_overlay(
        image: np.ndarray,
        action_result: ActionResult,
        position: Tuple[int, int] = (10, 30),
    ) -> np.ndarray:
        """
        Draw an action status banner at the top of the image.

        Args:
            image:         BGR image
            action_result: DecisionModule output
            position:      Top-left anchor for the banner

        Returns:
            Image with banner overlaid
        """
        canvas = image.copy()

        # Choose color based on action outcome
        if action_result.action_type == ActionType.SAFE_REJECT:
            color = (0, 0, 200)    # Red = danger
            prefix = "[REJECTED]"
        elif action_result.from_memory:
            color = (200, 150, 0)  # Orange = memory fallback
            prefix = "[MEMORY]"
        else:
            color = (0, 180, 0)    # Green = success
            prefix = "[ACTION]"

        text = (
            f"{prefix} {action_result.action_type.value.upper()} "
            f"→ {action_result.target_label or 'None'} "
            f"| Conf: {action_result.confidence:.2f}"
        )

        x, y = position
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x - 5, y - th - 8), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return canvas

    @staticmethod
    def draw_memory_overlay(
        image: np.ndarray,
        memory_entries: List[MemoryEntry],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Draw semi-transparent boxes at remembered object locations.

        Args:
            image:          BGR image
            memory_entries: List of MemoryEntry from MemoryModule.recall_all()
            alpha:          Transparency of memory boxes

        Returns:
            Image with memory overlays
        """
        canvas = image.copy()
        overlay = image.copy()

        for entry in memory_entries:
            x1, y1, x2, y2 = [int(v) for v in entry.bbox]
            color = _get_color(entry.label)
            # Dashed / faded box using filled rect at low alpha
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            age_text = f"[MEM] {entry.label} {entry.age_seconds:.0f}s ago"
            cv2.putText(
                canvas,
                age_text,
                (x1 + 5, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        return canvas

    @staticmethod
    def draw_confidence_hud(
        image: np.ndarray,
        quality_score: float,
        grounding_score: float,
        combined_score: float,
        position: Tuple[int, int] = (10, 60),
        bar_width: int = 150,
    ) -> np.ndarray:
        """
        Draw a HUD with colored confidence bars.
        """
        canvas = image.copy()

        metrics = [
            ("Quality  ", quality_score,   (255, 200,  50)),
            ("Grounding", grounding_score, ( 50, 200, 255)),
            ("Combined ", combined_score,  ( 50, 255, 100)),
        ]

        x, y = position
        row_h = 20

        for i, (label, score, color) in enumerate(metrics):
            row_y = y + i * row_h
            # Background bar
            cv2.rectangle(canvas, (x + 90, row_y), (x + 90 + bar_width, row_y + 14), (50, 50, 50), -1)
            # Filled bar proportional to score
            fill_w = int(bar_width * np.clip(score, 0.0, 1.0))
            cv2.rectangle(canvas, (x + 90, row_y), (x + 90 + fill_w, row_y + 14), color, -1)
            # Label
            cv2.putText(
                canvas,
                f"{label}: {score:.2f}",
                (x, row_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return canvas

    @staticmethod
    def make_comparison_grid(
        images: List[np.ndarray],
        titles: List[str],
        cols: int = 2,
        target_size: Tuple[int, int] = (400, 300),
    ) -> np.ndarray:
        """
        Arrange multiple images in a grid with titles.

        Args:
            images:      List of BGR images
            titles:      Title for each image
            cols:        Number of columns in the grid
            target_size: Resize each image to (width, height)

        Returns:
            Single BGR grid image
        """
        rows = (len(images) + cols - 1) // cols
        w, h = target_size
        cells = []

        for img, title in zip(images, titles):
            cell = cv2.resize(img, (w, h))
            # Dark title bar
            cv2.rectangle(cell, (0, 0), (w, 24), (30, 30, 30), -1)
            cv2.putText(cell, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (220, 220, 220), 1, cv2.LINE_AA)
            cells.append(cell)

        # Pad if needed
        while len(cells) < rows * cols:
            cells.append(np.zeros((h, w, 3), dtype=np.uint8))

        grid_rows = [
            np.hstack(cells[i * cols:(i + 1) * cols])
            for i in range(rows)
        ]
        return np.vstack(grid_rows)

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR (OpenCV) to RGB (Streamlit/matplotlib)."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)