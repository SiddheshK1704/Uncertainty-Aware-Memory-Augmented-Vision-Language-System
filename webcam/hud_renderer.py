"""
Webcam HUD Renderer
====================
All OpenCV drawing logic for the real-time webcam overlay.

Renders:
  - Detection bounding boxes + labels + confidence
  - Target object highlight (grounded from command)
  - Uncertainty / sharpness bar
  - Action status panel
  - Memory recall badge
  - FPS + system status strip
  - Corner scanline aesthetic (research terminal look)
"""

from __future__ import annotations
import cv2
import numpy as np
import time
from typing import List, Optional, Tuple

from modules.vision_module import Detection
from modules.memory_module import MemoryEntry


# ── Palette (BGR) ────────────────────────────────────────────────────
CLR_TARGET    = (0,   255, 120)   # bright green  — grounded target
CLR_DETECT    = (255, 180,  50)   # amber         — other detections
CLR_MEMORY    = (255, 200,   0)   # gold          — memory recall box
CLR_REJECT    = (50,   50, 220)   # red-ish       — safety reject
CLR_ACTION    = (0,   210, 255)   # cyan          — action approved
CLR_HUD_BG   = (10,   10,  18)   # near-black    — panel background
CLR_TEXT_PRI = (230, 230, 230)   # off-white     — primary text
CLR_TEXT_DIM = (110, 110, 120)   # dim gray      — secondary text
CLR_SCAN     = (20,   35,  20)   # dark green    — scanline tint

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN


def _alpha_rect(
    canvas: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw a semi-transparent filled rectangle."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def _draw_corner_marks(
    canvas: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    size: int = 14,
    thick: int = 2,
) -> None:
    """Draw 4-corner bracket marks instead of a full rectangle."""
    # top-left
    cv2.line(canvas, (x1, y1), (x1 + size, y1), color, thick)
    cv2.line(canvas, (x1, y1), (x1, y1 + size), color, thick)
    # top-right
    cv2.line(canvas, (x2, y1), (x2 - size, y1), color, thick)
    cv2.line(canvas, (x2, y1), (x2, y1 + size), color, thick)
    # bottom-left
    cv2.line(canvas, (x1, y2), (x1 + size, y2), color, thick)
    cv2.line(canvas, (x1, y2), (x1, y2 - size), color, thick)
    # bottom-right
    cv2.line(canvas, (x2, y2), (x2 - size, y2), color, thick)
    cv2.line(canvas, (x2, y2), (x2, y2 - size), color, thick)


def _text_with_shadow(
    canvas: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font=FONT_SMALL,
    scale: float = 0.55,
    color: Tuple[int, int, int] = CLR_TEXT_PRI,
    thickness: int = 1,
) -> None:
    """Draw text with a 1-px dark shadow for legibility on any background."""
    x, y = pos
    cv2.putText(canvas, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, pos,             font, scale, color,     thickness,     cv2.LINE_AA)


class HUDRenderer:
    """
    Stateful HUD renderer for the real-time webcam window.

    Call draw_frame() each tick; it modifies `canvas` in-place.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._fps_buf: List[float] = []
        self._last_tick = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        target_label: Optional[str],
        command: str,
        unc_confidence: float,
        unc_lap_var: float,
        action_status: str,           # "APPROVED" | "REJECTED" | "LOW_CONF" | "MEMORY"
        action_label: Optional[str],
        memory_entries: List[MemoryEntry],
        from_memory: bool,
        perception_tags: List[str],
    ) -> np.ndarray:
        """
        Compose the complete HUD onto `frame` (BGR uint8).

        Returns the annotated frame (same array, modified in-place + returned).
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 0. Scanline vignette (subtle research-terminal feel)
        canvas = self._scanline_vignette(canvas)

        # 1. Detection boxes
        self._draw_detections(canvas, detections, target_label)

        # 2. Memory ghost boxes (faded, dashed-style)
        self._draw_memory_ghosts(canvas, memory_entries, detections)

        # 3. Top status bar
        self._draw_top_bar(canvas, command, action_status, action_label, from_memory)

        # 4. Right-side info panel
        self._draw_right_panel(canvas, unc_confidence, unc_lap_var, detections, perception_tags)

        # 5. Bottom status strip (FPS + object count)
        fps = self._update_fps()
        self._draw_bottom_strip(canvas, fps, len(detections), len(memory_entries))

        # 6. Corner reticle (robot-eye aesthetic)
        self._draw_corner_reticle(canvas)

        return canvas

    # ──────────────────────────────────────────────────────────────────
    # Detection rendering
    # ──────────────────────────────────────────────────────────────────

    def _draw_detections(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        target_label: Optional[str],
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            is_target = (det.label == target_label)
            color = CLR_TARGET if is_target else CLR_DETECT
            thick = 2 if is_target else 1

            if is_target:
                # Filled semi-transparent highlight for the target
                canvas = _alpha_rect(canvas, x1, y1, x2, y2, CLR_TARGET, alpha=0.10)
                # Corner brackets (tech look)
                _draw_corner_marks(canvas, x1, y1, x2, y2, CLR_TARGET, size=18, thick=3)
                # Pulsing centre dot
                cx, cy = int(det.center[0]), int(det.center[1])
                cv2.circle(canvas, (cx, cy), 5, CLR_TARGET, -1)
                cv2.circle(canvas, (cx, cy), 10, CLR_TARGET, 1)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

            # Label pill
            label_txt = f"{det.label}  {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, FONT_SMALL, 0.50, 1)
            pad = 4
            lx1, ly1 = x1, max(y1 - th - 2 * pad, 0)
            lx2, ly2 = x1 + tw + 2 * pad, y1
            canvas = _alpha_rect(canvas, lx1, ly1, lx2, ly2, color, alpha=0.75)
            _text_with_shadow(
                canvas, label_txt, (lx1 + pad, ly2 - pad),
                scale=0.50, color=(5, 5, 5) if is_target else CLR_TEXT_PRI, thickness=1,
            )

            # "TARGET" beacon above the target box
            if is_target:
                beacon = f"◉ TARGET"
                _text_with_shadow(
                    canvas, beacon,
                    (x1, max(y1 - th - 2 * pad - 18, 10)),
                    font=FONT, scale=0.55, color=CLR_TARGET, thickness=1,
                )

    # ──────────────────────────────────────────────────────────────────
    # Memory ghost boxes
    # ──────────────────────────────────────────────────────────────────

    def _draw_memory_ghosts(
        self,
        canvas: np.ndarray,
        memory_entries: List[MemoryEntry],
        live_detections: List[Detection],
    ) -> None:
        """Draw dashed ghost boxes for objects in memory but not currently detected."""
        live_labels = {d.label for d in live_detections}
        for entry in memory_entries:
            if entry.label in live_labels:
                continue  # object is visible, no ghost needed
            x1, y1, x2, y2 = [int(v) for v in entry.bbox]
            # Dashed box via repeated line segments
            self._draw_dashed_rect(canvas, x1, y1, x2, y2, CLR_MEMORY, dash=10, gap=6)
            age_txt = f"[MEM] {entry.label}  {entry.age_seconds:.0f}s"
            _text_with_shadow(
                canvas, age_txt, (x1 + 4, y2 - 6),
                scale=0.42, color=CLR_MEMORY, thickness=1,
            )

    @staticmethod
    def _draw_dashed_rect(
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: Tuple[int, int, int],
        dash: int = 8,
        gap: int = 5,
    ) -> None:
        """Draw a dashed rectangle outline."""
        pts = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]
        for (sx, sy), (ex, ey) in pts:
            length = max(abs(ex - sx), abs(ey - sy))
            if length == 0:
                continue
            steps = length // (dash + gap)
            dx = (ex - sx) / max(length, 1)
            dy = (ey - sy) / max(length, 1)
            for i in range(int(steps)):
                t0 = i * (dash + gap)
                t1 = t0 + dash
                p0 = (int(sx + dx * t0), int(sy + dy * t0))
                p1 = (int(sx + dx * min(t1, length)), int(sy + dy * min(t1, length)))
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────
    # Top bar
    # ──────────────────────────────────────────────────────────────────

    def _draw_top_bar(
        self,
        canvas: np.ndarray,
        command: str,
        action_status: str,
        action_label: Optional[str],
        from_memory: bool,
    ) -> None:
        h, w = canvas.shape[:2]
        bar_h = 46

        # Background
        canvas = _alpha_rect(canvas, 0, 0, w, bar_h, CLR_HUD_BG, alpha=0.82)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (40, 60, 40), 1)

        # Left: system tag
        _text_with_shadow(canvas, "UVLA·SYS", (10, 30), font=FONT, scale=0.60,
                          color=(60, 160, 80), thickness=1)

        # Centre: command
        cmd_display = f"CMD ▸  {command}" if command else "CMD ▸  (none)"
        _text_with_shadow(canvas, cmd_display, (140, 30), font=FONT, scale=0.58,
                          color=CLR_TEXT_PRI, thickness=1)

        # Right: action status badge
        status_colors = {
            "APPROVED":  CLR_ACTION,
            "MEMORY":    CLR_MEMORY,
            "REJECTED":  CLR_REJECT,
            "LOW_CONF":  (100, 100, 220),
            "NO_TARGET": CLR_TEXT_DIM,
            "WAITING":   CLR_TEXT_DIM,
        }
        s_color = status_colors.get(action_status, CLR_TEXT_DIM)
        s_text = action_status
        if action_label:
            s_text += f"  →  {action_label}"
        if from_memory:
            s_text += "  🧠"

        (tw, _), _ = cv2.getTextSize(s_text, FONT, 0.55, 1)
        sx = w - tw - 20
        _text_with_shadow(canvas, s_text, (sx, 30), font=FONT, scale=0.55,
                          color=s_color, thickness=1)

    # ──────────────────────────────────────────────────────────────────
    # Right-side info panel
    # ──────────────────────────────────────────────────────────────────

    def _draw_right_panel(
        self,
        canvas: np.ndarray,
        unc_confidence: float,
        lap_var: float,
        detections: List[Detection],
        perception_tags: List[str],
    ) -> None:
        h, w = canvas.shape[:2]
        pw = 220
        px = w - pw - 8
        py = 58
        row = 22

        # Panel background
        canvas = _alpha_rect(canvas, px - 6, py - 4, w - 4, py + row * 9 + 8,
                             CLR_HUD_BG, alpha=0.78)
        cv2.line(canvas, (px - 6, py - 4), (px - 6, py + row * 9 + 8), (40, 60, 40), 1)

        # ── Uncertainty section ──
        _text_with_shadow(canvas, "UNCERTAINTY", (px, py + 14), scale=0.42,
                          color=(80, 180, 80), thickness=1)

        # Sharpness bar
        bar_y = py + row + 4
        bar_w = pw - 10
        cv2.rectangle(canvas, (px, bar_y), (px + bar_w, bar_y + 10), (30, 30, 30), -1)
        filled = int(bar_w * np.clip(unc_confidence, 0, 1))
        bar_color = (
            (50, 200, 50)  if unc_confidence > 0.65 else
            (50, 200, 200) if unc_confidence > 0.40 else
            (50, 50, 200)
        )
        cv2.rectangle(canvas, (px, bar_y), (px + filled, bar_y + 10), bar_color, -1)
        _text_with_shadow(
            canvas, f"conf  {unc_confidence:.3f}",
            (px, bar_y + 24), scale=0.44, color=CLR_TEXT_PRI,
        )
        _text_with_shadow(
            canvas, f"lap_var  {lap_var:.1f}",
            (px, bar_y + 40), scale=0.44, color=CLR_TEXT_DIM,
        )

        # Confidence label
        unc_label = (
            "SHARP" if unc_confidence > 0.65 else
            "MARGINAL" if unc_confidence > 0.40 else
            "LOW CONFIDENCE"
        )
        unc_color = (
            (50, 220, 50)  if unc_confidence > 0.65 else
            (50, 200, 200) if unc_confidence > 0.40 else
            (50, 50, 220)
        )
        _text_with_shadow(canvas, unc_label, (px, bar_y + 58), scale=0.48,
                          color=unc_color, thickness=1)

        # ── Detections section ──
        det_y = py + row * 5 + 6
        _text_with_shadow(canvas, "DETECTIONS", (px, det_y), scale=0.42,
                          color=(80, 180, 80), thickness=1)
        for i, det in enumerate(detections[:4]):
            dy = det_y + (i + 1) * 18
            _text_with_shadow(
                canvas,
                f"  {det.label:<14} {det.confidence:.2f}",
                (px, dy), scale=0.40, color=CLR_TEXT_PRI,
            )

        # ── Perception tags ──
        if perception_tags:
            tag_y = py + row * 9
            _text_with_shadow(
                canvas,
                "ENHANCE: " + " | ".join(perception_tags[:2]),
                (px, tag_y), scale=0.38, color=(120, 180, 120),
            )

    # ──────────────────────────────────────────────────────────────────
    # Bottom strip
    # ──────────────────────────────────────────────────────────────────

    def _draw_bottom_strip(
        self,
        canvas: np.ndarray,
        fps: float,
        n_detections: int,
        n_memory: int,
    ) -> None:
        h, w = canvas.shape[:2]
        strip_h = 28
        canvas = _alpha_rect(canvas, 0, h - strip_h, w, h, CLR_HUD_BG, alpha=0.80)
        cv2.line(canvas, (0, h - strip_h), (w, h - strip_h), (40, 60, 40), 1)

        _text_with_shadow(
            canvas,
            f"FPS {fps:4.1f}",
            (10, h - 8), scale=0.46, color=(80, 200, 80),
        )
        _text_with_shadow(
            canvas,
            f"OBJECTS {n_detections}   MEMORY {n_memory}",
            (110, h - 8), scale=0.46, color=CLR_TEXT_DIM,
        )
        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _text_with_shadow(
            canvas, ts, (w - 120, h - 8), scale=0.44, color=CLR_TEXT_DIM,
        )

    # ──────────────────────────────────────────────────────────────────
    # Aesthetic corner reticle
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_corner_reticle(canvas: np.ndarray, size: int = 30) -> None:
        """Draw small L-brackets in the four corners of the frame."""
        h, w = canvas.shape[:2]
        color = (40, 90, 40)
        t = 1
        # Top-left
        cv2.line(canvas, (0, 0), (size, 0), color, t)
        cv2.line(canvas, (0, 0), (0, size), color, t)
        # Top-right
        cv2.line(canvas, (w, 0), (w - size, 0), color, t)
        cv2.line(canvas, (w, 0), (w, size), color, t)
        # Bottom-left
        cv2.line(canvas, (0, h), (size, h), color, t)
        cv2.line(canvas, (0, h), (0, h - size), color, t)
        # Bottom-right
        cv2.line(canvas, (w, h), (w - size, h), color, t)
        cv2.line(canvas, (w, h), (w, h - size), color, t)

    # ──────────────────────────────────────────────────────────────────
    # Scanline vignette
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _scanline_vignette(frame: np.ndarray) -> np.ndarray:
        """Apply a very subtle dark vignette toward the edges."""
        h, w = frame.shape[:2]
        # Radial vignette
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        vignette = np.clip(1.0 - 0.30 * dist, 0.70, 1.0).astype(np.float32)
        frame = (frame.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        return frame

    # ──────────────────────────────────────────────────────────────────
    # FPS tracker
    # ──────────────────────────────────────────────────────────────────

    def _update_fps(self, max_samples: int = 30) -> float:
        now = time.perf_counter()
        self._fps_buf.append(now - self._last_tick)
        self._last_tick = now
        if len(self._fps_buf) > max_samples:
            self._fps_buf.pop(0)
        avg_dt = sum(self._fps_buf) / len(self._fps_buf)
        return 1.0 / max(avg_dt, 1e-6)