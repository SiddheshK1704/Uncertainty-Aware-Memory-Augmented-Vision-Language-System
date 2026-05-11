"""
Perception Module
=================
Adaptive image pre-processing pipeline that corrects for real-world
distribution shifts before sending frames to the Vision Module.

Supported enhancements:
  - Brightness correction  (CLAHE on LAB L-channel)
  - Contrast enhancement   (histogram equalization)
  - Denoising              (Non-local means / Gaussian fallback)
  - Sharpening             (Unsharp masking via Laplacian)
  - Gamma correction       (for low-light)

All transforms preserve the BGR uint8 format expected by YOLOv8.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Flag, auto
from typing import List, Optional
from loguru import logger


class Enhancement(Flag):
    """Bit-flag enum to select which enhancements to apply."""
    NONE       = 0
    BRIGHTNESS = auto()
    CONTRAST   = auto()
    DENOISE    = auto()
    SHARPEN    = auto()
    GAMMA      = auto()
    ALL = BRIGHTNESS | CONTRAST | DENOISE | SHARPEN | GAMMA


@dataclass
class PerceptionResult:
    """Holds the enhanced image and applied transforms."""
    enhanced_image: np.ndarray
    original_image: np.ndarray
    applied_transforms: List[str]
    quality_improved: bool  # Whether enhancement was triggered

    def to_dict(self) -> dict:
        return {
            "applied_transforms": self.applied_transforms,
            "quality_improved": self.quality_improved,
        }


class PerceptionModule:
    """
    Adaptive image quality enhancer.

    Intelligently detects image degradation and applies targeted corrections
    without over-processing already-good images.

    Example usage::

        perc = PerceptionModule()

        # Auto-detect and fix issues:
        result = perc.enhance(image_bgr)
        clean_image = result.enhanced_image

        # Or explicitly choose transforms:
        result = perc.enhance(image_bgr, mode=Enhancement.BRIGHTNESS | Enhancement.DENOISE)
    """

    def __init__(
        self,
        auto_enhance: bool = True,
        blur_threshold: float = 80.0,
        dark_threshold: float = 0.30,
        noise_threshold: float = 0.25,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        denoise_h: int = 10,
        sharpen_strength: float = 1.0,
    ):
        """
        Args:
            auto_enhance:      Auto-detect issues and select transforms
            blur_threshold:    Laplacian variance below this triggers sharpening
            dark_threshold:    Mean brightness below this triggers gamma correction
            noise_threshold:   Noise estimate above this triggers denoising
            clahe_clip:        CLAHE clip limit (contrast enhancement)
            clahe_grid:        CLAHE tile grid size
            denoise_h:         Non-local means filter strength
            sharpen_strength:  Unsharp mask alpha (0.5 = subtle, 2.0 = aggressive)
        """
        self.auto_enhance = auto_enhance
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.noise_threshold = noise_threshold
        self.sharpen_strength = sharpen_strength
        self.denoise_h = denoise_h

        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        image: np.ndarray,
        mode: Enhancement = Enhancement.NONE,
    ) -> PerceptionResult:
        """
        Enhance an image, either auto-selecting transforms or using `mode`.

        Args:
            image: H×W×3 uint8 BGR image
            mode:  Enhancement flags; if NONE and auto_enhance=True, auto-detect

        Returns:
            PerceptionResult with enhanced image and metadata
        """
        if image is None or image.size == 0:
            logger.warning("PerceptionModule received empty image.")
            return PerceptionResult(image, image, [], False)

        original = image.copy()

        # Auto-detect which enhancements are needed
        if mode == Enhancement.NONE and self.auto_enhance:
            mode = self._auto_select(image)

        if mode == Enhancement.NONE:
            logger.debug("No enhancement needed.")
            return PerceptionResult(image, original, [], False)

        # Apply selected enhancements in a sensible order
        enhanced = image.copy()
        applied: List[str] = []

        # 1. Gamma correction (low-light fix) — before CLAHE
        if Enhancement.GAMMA in mode:
            enhanced = self._apply_gamma(enhanced)
            applied.append("gamma_correction")

        # 2. Brightness / contrast via CLAHE
        if Enhancement.BRIGHTNESS in mode or Enhancement.CONTRAST in mode:
            enhanced = self._apply_clahe(enhanced)
            applied.append("clahe_brightness_contrast")

        # 3. Denoising — before sharpening to avoid amplifying noise
        if Enhancement.DENOISE in mode:
            enhanced = self._apply_denoise(enhanced)
            applied.append("denoising")

        # 4. Sharpening — last step to restore edge clarity
        if Enhancement.SHARPEN in mode:
            enhanced = self._apply_sharpen(enhanced)
            applied.append("sharpening")

        logger.info(f"Enhancements applied: {applied}")

        return PerceptionResult(
            enhanced_image=enhanced,
            original_image=original,
            applied_transforms=applied,
            quality_improved=bool(applied),
        )

    def enhance_for_condition(self, image: np.ndarray, condition: str) -> PerceptionResult:
        """
        Apply preset enhancement for a known degradation condition.

        Args:
            condition: One of 'blur', 'noise', 'low_light', 'occlusion'
        """
        presets = {
            "blur":       Enhancement.SHARPEN | Enhancement.CONTRAST,
            "noise":      Enhancement.DENOISE | Enhancement.CONTRAST,
            "low_light":  Enhancement.GAMMA | Enhancement.BRIGHTNESS | Enhancement.CONTRAST,
            "occlusion":  Enhancement.CONTRAST | Enhancement.SHARPEN,
            "clean":      Enhancement.NONE,
        }
        mode = presets.get(condition, Enhancement.NONE)
        return self.enhance(image, mode=mode)

    # ------------------------------------------------------------------
    # Enhancement implementations
    # ------------------------------------------------------------------

    def _apply_gamma(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Gamma correction to brighten low-light images.
        Auto-computes gamma from mean brightness if not specified.
        """
        if gamma is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean()) / 255.0
            # Adaptive gamma: darker image → more correction
            if mean_brightness < 1e-3:
                gamma = 0.5
            else:
                # gamma < 1 brightens; target brightness ~0.45
                gamma = np.log(0.45) / np.log(max(mean_brightness, 0.01))
                gamma = float(np.clip(gamma, 0.3, 2.5))

        # Build LUT for fast pixel-wise gamma
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, lut)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB L-channel.
        Boosts local contrast without over-saturating colors.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_ch)
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising (fastNlMeansDenoisingColored).
        Falls back to Gaussian blur if NLM fails.
        """
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                h=self.denoise_h,
                hColor=self.denoise_h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised
        except Exception:
            # Fast fallback
            return cv2.GaussianBlur(image, (5, 5), 0)

    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Unsharp masking: enhanced = original + alpha * (original - blurred)
        Highlights edges and fine detail.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(
            image, 1.0 + self.sharpen_strength,
            blurred, -self.sharpen_strength,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def _auto_select(self, image: np.ndarray) -> Enhancement:
        """
        Analyse image quality metrics to choose the right enhancements.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mode = Enhancement.NONE

        # --- Check brightness ---
        brightness = float(gray.mean()) / 255.0
        if brightness < self.dark_threshold:
            mode |= Enhancement.GAMMA | Enhancement.BRIGHTNESS
            logger.debug(f"Auto: low-light detected (brightness={brightness:.2f})")

        # --- Check blur ---
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.blur_threshold:
            mode |= Enhancement.SHARPEN
            logger.debug(f"Auto: blur detected (lap_var={lap_var:.1f})")

        # --- Check noise ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = float(np.abs(gray - blurred).std()) / 30.0
        noise_level = min(noise_level, 1.0)
        if noise_level > self.noise_threshold:
            mode |= Enhancement.DENOISE
            logger.debug(f"Auto: noise detected (level={noise_level:.2f})")

        # Always add contrast if we're doing any correction
        if mode != Enhancement.NONE:
            mode |= Enhancement.CONTRAST

        return mode