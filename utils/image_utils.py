"""
Image Utilities
===============
Helper functions for loading, resizing, and applying synthetic
distribution-shift perturbations (noise, blur, low-light, occlusion).

Used primarily by the robustness evaluation pipeline.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


# ------------------------------------------------------------------
# Perturbation functions
# ------------------------------------------------------------------

def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 25.0,
    mean: float = 0.0,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to simulate sensor noise.

    Args:
        image: H×W×3 uint8 BGR image
        sigma: Standard deviation of noise (higher = noisier)
        mean:  Mean of noise (0 = unbiased)

    Returns:
        Noisy BGR image (clipped to [0, 255])
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Apply Gaussian blur to simulate motion blur or out-of-focus.

    Args:
        image:       H×W×3 uint8 BGR image
        kernel_size: Must be odd; larger = more blur
        sigma:       Gaussian sigma (0 = auto-computed from kernel_size)

    Returns:
        Blurred BGR image
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), sigma)


def apply_low_light(
    image: np.ndarray,
    gamma: float = 0.3,
    add_noise: bool = True,
    noise_sigma: float = 8.0,
) -> np.ndarray:
    """
    Simulate low-light conditions via gamma darkening + optional sensor noise.

    Args:
        image:       H×W×3 uint8 BGR image
        gamma:       Gamma value < 1 darkens, > 1 brightens (default 0.3 = dark)
        add_noise:   Add a small amount of noise typical of high-ISO sensors
        noise_sigma: Noise level to add

    Returns:
        Low-light BGR image
    """
    # Build gamma LUT
    inv_gamma = 1.0 / max(gamma, 0.01)
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    dark = cv2.LUT(image, lut)

    if add_noise:
        dark = apply_gaussian_noise(dark, sigma=noise_sigma)

    return dark


def apply_occlusion(
    image: np.ndarray,
    occlusion_fraction: float = 0.30,
    num_patches: int = 3,
    fill_value: int = 0,
) -> np.ndarray:
    """
    Randomly occlude rectangular regions of the image.

    Args:
        image:             H×W×3 uint8 BGR image
        occlusion_fraction: Fraction of image area to occlude total [0, 1]
        num_patches:        Number of random occlusion rectangles
        fill_value:         Pixel value to fill occluded areas (0=black)

    Returns:
        Occluded BGR image
    """
    occluded = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    pixels_per_patch = int(total_pixels * occlusion_fraction / num_patches)

    rng = np.random.default_rng(seed=42)  # Deterministic for reproducibility

    for _ in range(num_patches):
        # Random square patch sized to cover the target pixel count
        side = int(np.sqrt(pixels_per_patch))
        # Random top-left corner
        x1 = int(rng.integers(0, max(1, w - side)))
        y1 = int(rng.integers(0, max(1, h - side)))
        x2 = min(x1 + side, w)
        y2 = min(y1 + side, h)
        occluded[y1:y2, x1:x2] = fill_value

    return occluded


def apply_salt_pepper_noise(
    image: np.ndarray,
    density: float = 0.05,
) -> np.ndarray:
    """
    Apply salt-and-pepper noise (random white/black pixels).

    Args:
        image:   H×W×3 uint8 BGR image
        density: Fraction of pixels to corrupt

    Returns:
        Corrupted BGR image
    """
    noisy = image.copy()
    rng = np.random.default_rng()
    total = image.size // 3
    num_corrupt = int(total * density)

    # Salt (white)
    coords = rng.integers(0, image.shape[0], size=num_corrupt), \
             rng.integers(0, image.shape[1], size=num_corrupt)
    noisy[coords] = 255

    # Pepper (black)
    coords = rng.integers(0, image.shape[0], size=num_corrupt), \
             rng.integers(0, image.shape[1], size=num_corrupt)
    noisy[coords] = 0

    return noisy


# ------------------------------------------------------------------
# Image I/O helpers
# ------------------------------------------------------------------

def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Load an image from disk as BGR uint8 array.

    Args:
        path:        File path (jpg, png, bmp, etc.)
        target_size: Optional (width, height) to resize to

    Returns:
        BGR numpy array or None if loading fails
    """
    img_path = Path(path)
    if not img_path.exists():
        logger.error(f"Image not found: {path}")
        return None

    image = cv2.imread(str(img_path))
    if image is None:
        logger.error(f"OpenCV failed to load image: {path}")
        return None

    if target_size is not None:
        image = cv2.resize(image, target_size)

    logger.debug(f"Loaded image: {path} shape={image.shape}")
    return image


def resize_keep_aspect(
    image: np.ndarray,
    max_side: int = 640,
) -> np.ndarray:
    """
    Resize image so the longest side equals max_side, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def numpy_to_pil(image: np.ndarray):
    """Convert BGR numpy array to PIL Image (RGB)."""
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert PIL Image (RGB) to BGR numpy array."""
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_image_stats(image: np.ndarray) -> dict:
    """Return basic statistics about an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "shape": image.shape,
        "mean_brightness": float(gray.mean() / 255.0),
        "std": float(gray.std()),
        "laplacian_var": float(lap.var()),
        "min": int(image.min()),
        "max": int(image.max()),
    }