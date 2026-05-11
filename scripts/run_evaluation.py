"""
CLI Robustness Evaluation Runner
=================================
Run the full UVLA robustness evaluation from the command line.

Usage examples:
    # Evaluate on a single image with all conditions:
    python scripts/run_evaluation.py --image data/sample.jpg

    # Custom command:
    python scripts/run_evaluation.py --image data/sample.jpg --command "navigate to the chair"

    # Specific conditions only:
    python scripts/run_evaluation.py --image data/sample.jpg --conditions noise blur

    # Save results to JSON:
    python scripts/run_evaluation.py --image data/sample.jpg --output results.json
"""

import argparse
import json
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from loguru import logger

from evaluation.robustness_eval import RobustnessEvaluator, PERTURBATIONS, EVAL_COMMANDS
from utils.image_utils import load_image, resize_keep_aspect


def parse_args():
    parser = argparse.ArgumentParser(
        description="UVLA Robustness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--command", "-c", type=str, default=None,
        help="Single command to test (default: all EVAL_COMMANDS)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=list(PERTURBATIONS.keys()),
        default=list(PERTURBATIONS.keys()),
        help="Perturbation conditions to evaluate",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Disable adaptive perception enhancement",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Show progress bars",
    )
    return parser.parse_args()


def generate_synthetic_image() -> np.ndarray:
    """
    Generate a simple synthetic image for demo when no real image is available.
    Draws colored rectangles representing 'objects' in a scene.
    """
    image = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark gray background

    # Simulate objects
    objects = [
        ((100, 150, 300, 350), (139, 90, 43), "chair area"),
        ((350, 100, 550, 300), (60, 60, 200), "tv area"),
        ((50, 300, 250, 430), (100, 180, 100), "bottle area"),
        ((300, 320, 600, 460), (180, 100, 50), "table area"),
    ]

    for (x1, y1, x2, y2), color, label in objects:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.putText(
            image, label, (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    # Add lighting gradient
    gradient = np.linspace(0, 80, 640, dtype=np.uint8)
    gradient_img = np.tile(gradient, (480, 1))
    gradient_3ch = cv2.merge([gradient_img, gradient_img, gradient_img])
    image = np.clip(image.astype(int) + gradient_3ch.astype(int), 0, 255).astype(np.uint8)

    return image


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("UVLA Robustness Evaluation")
    logger.info("=" * 60)

    # Load image
    image = load_image(args.image)
    if image is None:
        logger.warning(f"Could not load '{args.image}'. Using synthetic demo image.")
        image = generate_synthetic_image()
        # Save it so pipeline can reference it
        demo_path = "data/demo_synthetic.jpg"
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(demo_path, image)
        args.image = demo_path

    image = resize_keep_aspect(image, max_side=640)
    logger.info(f"Image loaded: shape={image.shape}")

    # Choose commands
    commands = [args.command] if args.command else EVAL_COMMANDS
    logger.info(f"Commands: {commands}")
    logger.info(f"Conditions: {args.conditions}")

    # Save temp copy for evaluator
    img_path = args.image if os.path.exists(args.image) else "data/eval_input.jpg"
    cv2.imwrite(img_path, image)

    # Run evaluation
    evaluator = RobustnessEvaluator(
        conditions=args.conditions,
        enhance_before_detect=not args.no_enhance,
        verbose=args.verbose,
    )

    report = evaluator.run(
        images=[img_path],
        commands=commands,
    )

    # Print results
    report.print_table()

    # Save to JSON
    if args.output:
        result_dict = report.to_dict()
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.success(f"Results saved to: {args.output}")
    else:
        # Print JSON summary to stdout
        print("\n--- JSON Summary ---")
        print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()