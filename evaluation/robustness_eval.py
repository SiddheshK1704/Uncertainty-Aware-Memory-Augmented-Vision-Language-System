"""
Robustness Evaluation Pipeline
===============================
Systematically evaluates the UVLA system under four distribution shifts:
  1. Gaussian noise
  2. Blur
  3. Low light
  4. Occlusion

For each condition, runs N trials using the full pipeline and collects
metrics via MetricsCalculator.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm
from loguru import logger

from modules.vision_module import VisionModule
from modules.language_module import LanguageModule
from modules.grounding_module import GroundingModule
from modules.uncertainty_module import UncertaintyModule
from modules.memory_module import MemoryModule
from modules.perception_module import PerceptionModule
from modules.decision_module import DecisionModule

from utils.image_utils import (
    apply_gaussian_noise,
    apply_blur,
    apply_low_light,
    apply_occlusion,
    load_image,
    resize_keep_aspect,
)
from evaluation.metrics import MetricsCalculator, TrialResult, RobustnessReport


# Available perturbation conditions
PERTURBATIONS = {
    "clean":     lambda img: img,
    "noise":     lambda img: apply_gaussian_noise(img, sigma=25.0),
    "blur":      lambda img: apply_blur(img, kernel_size=15),
    "low_light": lambda img: apply_low_light(img, gamma=0.3),
    "occlusion": lambda img: apply_occlusion(img, occlusion_fraction=0.30),
}

# Sample evaluation commands targeting common indoor objects
EVAL_COMMANDS = [
    "navigate to the chair",
    "find the bottle on the table",
    "look at the tv",
    "pick up the book",
    "go to the sofa",
    "inspect the laptop",
    "move toward the cup",
    "navigate to the dining table",
]


class RobustnessEvaluator:
    """
    Full robustness evaluation harness.

    Runs the complete UVLA pipeline on each (image, command, perturbation)
    triple and aggregates metrics into a RobustnessReport.

    Example::

        evaluator = RobustnessEvaluator()
        report = evaluator.run(
            images=["path/to/img1.jpg", "path/to/img2.jpg"],
            commands=["navigate to the chair"],
        )
        report.print_table()
    """

    def __init__(
        self,
        conditions: Optional[List[str]] = None,
        enhance_before_detect: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            conditions:             Subset of PERTURBATIONS keys to test
            enhance_before_detect:  Apply PerceptionModule before detection
            verbose:                Show tqdm progress bars
        """
        self.conditions = conditions or list(PERTURBATIONS.keys())
        self.enhance_before_detect = enhance_before_detect
        self.verbose = verbose

        logger.info("Initializing UVLA pipeline for evaluation...")
        self._init_pipeline()
        self.calculator = MetricsCalculator()

    def run(
        self,
        images: List[str],
        commands: Optional[List[str]] = None,
    ) -> RobustnessReport:
        """
        Run full evaluation.

        Args:
            images:   List of image file paths
            commands: List of commands (defaults to EVAL_COMMANDS)

        Returns:
            RobustnessReport with all metrics
        """
        commands = commands or EVAL_COMMANDS
        self.calculator.reset()

        logger.info(
            f"Evaluation: {len(images)} images × "
            f"{len(commands)} commands × "
            f"{len(self.conditions)} conditions"
        )

        for condition in self.conditions:
            logger.info(f"Testing condition: {condition}")
            perturb_fn = PERTURBATIONS[condition]

            trials = []
            iterator = tqdm(
                [(img_path, cmd) for img_path in images for cmd in commands],
                desc=f"[{condition}]",
                disable=not self.verbose,
            )

            for img_path, command in iterator:
                trial = self._run_trial(img_path, command, condition, perturb_fn)
                if trial is not None:
                    trials.append(trial)

            self.calculator.add_many(trials)
            success_rate = sum(t.succeeded for t in trials) / max(len(trials), 1)
            logger.info(
                f"Condition '{condition}': "
                f"{len(trials)} trials, success={success_rate:.2%}"
            )

        return self.calculator.compute()

    def run_single(
        self,
        image: np.ndarray,
        command: str,
        condition: str = "clean",
    ) -> TrialResult:
        """
        Run a single trial on a pre-loaded numpy image.

        Args:
            image:     BGR uint8 image
            command:   Natural language instruction
            condition: Perturbation condition name

        Returns:
            TrialResult
        """
        perturb_fn = PERTURBATIONS.get(condition, PERTURBATIONS["clean"])
        return self._run_trial_on_image(image, command, condition, perturb_fn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_pipeline(self):
        """Initialize all UVLA pipeline modules."""
        self.vision = VisionModule(model_size="n", conf_threshold=0.35)
        self.language = LanguageModule()
        self.grounding = GroundingModule(language_module=self.language)
        self.uncertainty = UncertaintyModule(blur_threshold=50.0)
        self.memory = MemoryModule(ttl=60.0)
        self.perception = PerceptionModule(auto_enhance=True)
        self.decision = DecisionModule(
            memory_module=self.memory,
            confidence_threshold=0.25,
        )
        logger.success("Pipeline initialized.")

    def _run_trial(
        self,
        img_path: str,
        command: str,
        condition: str,
        perturb_fn,
    ) -> Optional[TrialResult]:
        """Load image and run trial."""
        image = load_image(img_path)
        if image is None:
            return None
        image = resize_keep_aspect(image, max_side=640)
        return self._run_trial_on_image(image, command, condition, perturb_fn)

    def _run_trial_on_image(
        self,
        image: np.ndarray,
        command: str,
        condition: str,
        perturb_fn,
    ) -> TrialResult:
        """Execute one full pipeline trial on a pre-loaded image."""
        # 1. Apply perturbation
        perturbed = perturb_fn(image)

        # 2. (Optional) Adaptive enhancement
        if self.enhance_before_detect and condition != "clean":
            perc_result = self.perception.enhance_for_condition(perturbed, condition)
            processed = perc_result.enhanced_image
        else:
            processed = perturbed

        # 3. Uncertainty estimation
        unc_result = self.uncertainty.estimate(processed)

        # 4. Object detection
        detections = self.vision.detect(processed)

        # 5. Memory update
        self.memory.update(detections)

        # 6. Language grounding
        ground_result = self.grounding.ground(command, detections)

        # 7. Decision
        action = self.decision.execute(command, ground_result, unc_result)

        return TrialResult(
            condition=condition,
            command=command,
            succeeded=action.succeeded,
            grounding_score=action.grounding_score,
            quality_score=action.quality_score,
            combined_confidence=action.confidence,
            action_type=action.action_type.value,
            from_memory=action.from_memory,
        )