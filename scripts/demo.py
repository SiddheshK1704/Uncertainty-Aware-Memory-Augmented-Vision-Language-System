"""
Quick Demo Script
=================
Runs the UVLA pipeline on a single image with a natural language command
and prints the result. No GPU required.

Usage:
    python scripts/demo.py
    python scripts/demo.py --image path/to/your.jpg --command "navigate to the chair"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import numpy as np
from loguru import logger

from modules import (
    VisionModule,
    LanguageModule,
    GroundingModule,
    UncertaintyModule,
    MemoryModule,
    PerceptionModule,
    DecisionModule,
)
from utils.image_utils import load_image, resize_keep_aspect, get_image_stats
from utils.visualization import VisualizationUtils


def create_demo_image() -> np.ndarray:
    """Create a simple demo image for testing without a real image."""
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    # Fake objects as colored rectangles
    cv2.rectangle(img, (80, 200), (250, 380), (139, 90, 43), -1)   # chair
    cv2.rectangle(img, (350, 80), (580, 290), (50, 50, 180), -1)   # tv
    cv2.rectangle(img, (30, 290), (180, 460), (100, 160, 80), -1)  # bottle
    cv2.rectangle(img, (270, 330), (620, 470), (160, 100, 40), -1) # table
    return img


def run_pipeline(image: np.ndarray, command: str, verbose: bool = True) -> dict:
    """
    Run the complete UVLA pipeline and return results dict.

    Args:
        image:   BGR uint8 numpy image
        command: Natural language instruction
        verbose: Print detailed output

    Returns:
        Dictionary with all pipeline outputs
    """
    print("\n" + "="*60)
    print(f"🤖 UVLA PIPELINE")
    print(f"   Command: '{command}'")
    print("="*60)

    # ---- Initialize modules ----
    print("\n[1/7] Initializing modules...")
    vision      = VisionModule(model_size="n")
    language    = LanguageModule()
    grounding   = GroundingModule(language_module=language)
    uncertainty = UncertaintyModule()
    memory      = MemoryModule()
    perception  = PerceptionModule(auto_enhance=True)
    decision    = DecisionModule(memory_module=memory)

    # ---- Step 1: Adaptive Perception ----
    print("\n[2/7] Adaptive Perception Enhancement...")
    perc_result = perception.enhance(image)
    enhanced = perc_result.enhanced_image
    print(f"   Transforms applied: {perc_result.applied_transforms or ['none']}")

    # ---- Step 2: Uncertainty Estimation ----
    print("\n[3/7] Uncertainty Estimation...")
    unc_result = uncertainty.estimate(enhanced)
    print(f"   Laplacian variance:  {unc_result.laplacian_variance:.2f}")
    print(f"   Sharpness score:     {unc_result.normalized_sharpness:.3f}")
    print(f"   Brightness score:    {unc_result.brightness_score:.3f}")
    print(f"   Overall confidence:  {unc_result.overall_confidence:.3f}")
    print(f"   Acceptable:          {unc_result.is_acceptable}")
    if not unc_result.is_acceptable:
        print(f"   ⚠️  Rejection reason: {unc_result.rejection_reason}")

    # ---- Step 3: Object Detection ----
    print("\n[4/7] Object Detection (YOLOv8)...")
    detections = vision.detect(enhanced)
    if detections:
        for det in detections:
            print(f"   ✓ {det.label:<20} conf={det.confidence:.3f}  bbox={[round(v) for v in det.bbox]}")
    else:
        print("   No objects detected.")

    # ---- Step 4: Memory Update ----
    print("\n[5/7] Updating Memory...")
    memory.update(detections)
    mem_summary = memory.summary()
    print(f"   Memory contains {mem_summary['total_entries']} objects: {mem_summary['labels']}")

    # ---- Step 5: Language Grounding ----
    print("\n[6/7] Language Grounding...")
    ground_result = grounding.ground(command, detections)
    print(f"   Grounded: {ground_result.grounded}")
    if ground_result.grounded:
        print(f"   Target:   {ground_result.target_detection.label}")
        print(f"   Sim score:{ground_result.similarity_score:.3f}")
    if ground_result.all_scores:
        print("   All scores:")
        for label, score in ground_result.all_scores[:3]:
            print(f"      {label:<20} {score:.3f}")

    # ---- Step 6: Decision ----
    print("\n[7/7] Decision Module...")
    action = decision.execute(command, ground_result, unc_result)
    print(f"\n{action}")

    return {
        "perception": perc_result.to_dict(),
        "uncertainty": unc_result.to_dict(),
        "detections": [d.to_dict() for d in detections],
        "grounding": ground_result.to_dict(),
        "action": action.to_dict(),
        "memory": mem_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="UVLA Quick Demo")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to input image (uses synthetic if not provided)")
    parser.add_argument("--command", "-c", type=str,
                        default="navigate to the chair",
                        help="Natural language command")
    parser.add_argument("--save-output", "-s", type=str, default=None,
                        help="Save annotated output image to this path")
    args = parser.parse_args()

    # Load or create image
    if args.image:
        image = load_image(args.image)
        if image is None:
            logger.warning("Could not load image, using demo image.")
            image = create_demo_image()
    else:
        logger.info("No image provided, using synthetic demo image.")
        image = create_demo_image()

    image = resize_keep_aspect(image, max_side=640)

    # Print image stats
    stats = get_image_stats(image)
    print(f"\nImage stats: {image.shape}, brightness={stats['mean_brightness']:.2f}, "
          f"laplacian_var={stats['laplacian_var']:.1f}")

    # Run pipeline
    results = run_pipeline(image, args.command)

    print("\n✅ Demo complete!")
    print(f"   Action: {results['action']['action_type'].upper()}")
    print(f"   Target: {results['action']['target_label']}")
    print(f"   Confidence: {results['action']['confidence']:.3f}")


if __name__ == "__main__":
    main()