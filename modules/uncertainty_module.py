"""
Uncertainty Module
==================
Estimates image quality and uncertainty using the Laplacian variance method.

The Laplacian operator highlights edges; its variance measures image sharpness:
  - High variance → sharp, information-rich image → LOW uncertainty
  - Low variance  → blurry / noisy / dark image    → HIGH uncertainty

A second-pass detection confidence score is combined with sharpness to
produce a final uncertainty estimate used by the Decision Module.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


@dataclass
class UncertaintyResult:
    """Captures all uncertainty signals for a single image."""
    laplacian_variance: float       # Raw sharpness metric (higher = sharper)
    normalized_sharpness: float     # [0, 1] sharpness score
    brightness_score: float         # [0, 1] brightness quality
    noise_estimate: float           # [0, 1] estimated noise level (0=clean)
    overall_confidence: float       # [0, 1] combined quality score
    is_acceptable: bool             # True if image passes quality threshold
    rejection_reason: Optional[str] # Why the image was rejected (if any)

    def to_dict(self) -> dict:
        return {
            "laplacian_variance": round(self.laplacian_variance, 2),
            "normalized_sharpness": round(self.normalized_sharpness, 4),
            "brightness_score": round(self.brightness_score, 4),
            "noise_estimate": round(self.noise_estimate, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "is_acceptable": self.is_acceptable,
            "rejection_reason": self.rejection_reason,
        }


class UncertaintyModule:
    """
    Multi-signal image quality estimator for uncertainty-aware decision making.

    Uses three complementary signals:
        1. Laplacian variance (sharpness / focus quality)
        2. Mean luminance (brightness adequacy)
        3. Local noise estimation (SNR proxy)

    Example usage::

        unc = UncertaintyModule(blur_threshold=80.0)
        result = unc.estimate(image_bgr)

        if not result.is_acceptable:
            print("Image rejected:", result.rejection_reason)
        else:
            print("Confidence:", result.overall_confidence)
    """

    def __init__(
        self,
        blur_threshold: float = 80.0,
        brightness_min: float = 0.08,
        brightness_max: float = 0.92,
        noise_max: float = 0.6,
        min_overall_confidence: float = 0.30,
    ):
        """
        Args:
            blur_threshold:         Minimum Laplacian variance to pass (not blurry)
            brightness_min:         Minimum normalized brightness (avoid too dark)
            brightness_max:         Maximum normalized brightness (avoid overexposed)
            noise_max:              Maximum acceptable noise level
            min_overall_confidence: Minimum combined score to accept the image
        """
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.noise_max = noise_max
        self.min_overall_confidence = min_overall_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image: np.ndarray) -> UncertaintyResult:
        """
        Estimate image quality and uncertainty.

        Args:
            image: H×W×3 uint8 BGR image

        Returns:
            UncertaintyResult with all quality signals
        """
        if image is None or image.size == 0:
            return self._reject("Empty or null image received.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- Signal 1: Laplacian Variance (sharpness) ---
        lap_var = self._laplacian_variance(gray)
        norm_sharpness = self._normalize_sharpness(lap_var)

        # --- Signal 2: Brightness Quality ---
        brightness = float(gray.mean()) / 255.0
        brightness_score = self._score_brightness(brightness)

        # --- Signal 3: Noise Estimation ---
        noise_level = self._estimate_noise(gray)
        noise_score = max(0.0, 1.0 - noise_level)

        # --- Combined Score (weighted) ---
        overall = (
            0.50 * norm_sharpness +
            0.30 * brightness_score +
            0.20 * noise_score
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        # --- Rejection Check ---
        rejection_reason = self._check_rejection(
            lap_var, brightness, noise_level, overall
        )
        is_acceptable = rejection_reason is None

        if not is_acceptable:
            logger.warning(f"Image rejected: {rejection_reason}")
        else:
            logger.debug(
                f"Image accepted. Sharpness={lap_var:.1f}, "
                f"Brightness={brightness:.2f}, Overall={overall:.3f}"
            )

        return UncertaintyResult(
            laplacian_variance=lap_var,
            normalized_sharpness=norm_sharpness,
            brightness_score=brightness_score,
            noise_estimate=noise_level,
            overall_confidence=overall,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
        )

    def is_image_acceptable(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Quick check: returns (is_acceptable, confidence_score).

        Convenience method for the Decision Module.
        """
        result = self.estimate(image)
        return result.is_acceptable, result.overall_confidence

    # ------------------------------------------------------------------
    # Signal computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """
        Compute Laplacian variance — the classic blur detection metric.

        A sharp image has strong edges → high Laplacian response → high variance.
        A blurry image has soft edges → low response → low variance.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _normalize_sharpness(self, lap_var: float) -> float:
        """
        Map Laplacian variance to [0, 1] using a soft sigmoid-like curve.
        Reference: variance of ~200 is 'acceptably sharp'.
        """
        # Use log scale since variance spans many orders of magnitude
        if lap_var <= 0:
            return 0.0
        log_v = np.log1p(lap_var)
        log_thresh = np.log1p(self.blur_threshold)
        score = log_v / (log_thresh * 2.5)  # scale so threshold ≈ 0.4
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _score_brightness(brightness: float) -> float:
        """
        Score brightness on a trapezoidal curve peaking at [0.3, 0.7].
        Very dark or very bright images score low.
        """
        if 0.30 <= brightness <= 0.70:
            return 1.0
        elif brightness < 0.30:
            return float(brightness / 0.30)
        else:
            return float((1.0 - brightness) / 0.30)

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """
        Estimate noise level using the median absolute deviation of
        high-frequency residuals (Laplacian residuals).

        Returns a value in [0, 1] where 1 = very noisy.
        """
        # Apply Gaussian blur and subtract to get noise residual
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = np.abs(gray - blurred)
        noise_std = float(residual.std())
        # Normalize: std of ~30 in 8-bit image is quite noisy
        return float(np.clip(noise_std / 30.0, 0.0, 1.0))

    def _check_rejection(
        self,
        lap_var: float,
        brightness: float,
        noise_level: float,
        overall: float,
    ) -> Optional[str]:
        """
        Check each rejection condition and return a reason string if failed.
        Returns None if the image is acceptable.
        """
        if lap_var < self.blur_threshold:
            return (
                f"Image too blurry (Laplacian variance {lap_var:.1f} "
                f"< threshold {self.blur_threshold})"
            )
        if brightness < self.brightness_min:
            return f"Image too dark (brightness {brightness:.2f} < {self.brightness_min})"
        if brightness > self.brightness_max:
            return f"Image overexposed (brightness {brightness:.2f} > {self.brightness_max})"
        if noise_level > self.noise_max:
            return f"Image too noisy (noise {noise_level:.2f} > {self.noise_max})"
        if overall < self.min_overall_confidence:
            return (
                f"Overall image quality too low "
                f"(score {overall:.2f} < {self.min_overall_confidence})"
            )
        return None

    def _reject(self, reason: str) -> UncertaintyResult:
        """Create a rejection result with a specific reason."""
        return UncertaintyResult(
            laplacian_variance=0.0,
            normalized_sharpness=0.0,
            brightness_score=0.0,
            noise_estimate=1.0,
            overall_confidence=0.0,
            is_acceptable=False,
            rejection_reason=reason,
        )