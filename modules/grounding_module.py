"""
Grounding Module
================
Matches natural language command embeddings against detected object labels
using cosine similarity, identifying the most likely target object.

Example:
    command  = "navigate to the chair near the tv"
    objects  = [Detection(chair, 0.9), Detection(bottle, 0.7), Detection(tv, 0.85)]
    result   = Detection(chair, 0.9)  ← highest similarity
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

from .vision_module import Detection
from .language_module import LanguageModule


@dataclass
class GroundingResult:
    """Result of grounding a language command to a detected object."""
    target_detection: Optional[Detection]   # Best matching detection (or None)
    similarity_score: float                  # Cosine similarity [0, 1]
    all_scores: List[Tuple[str, float]]      # All (label, score) pairs ranked
    grounded: bool                           # Whether a target was found
    command: str                             # Original command text

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "command": self.command,
            "similarity_score": round(self.similarity_score, 4),
            "target": self.target_detection.to_dict() if self.target_detection else None,
            "all_scores": [
                {"label": lbl, "score": round(sc, 4)}
                for lbl, sc in self.all_scores
            ],
        }


class GroundingModule:
    """
    Vision-language grounding via cosine similarity.

    Pipeline:
        1. Encode the natural language command → command_embedding
        2. Encode each detected label → label_embeddings
        3. Compute pairwise cosine similarity
        4. Return the detection with the highest score above min_similarity

    Example usage::

        lang = LanguageModule()
        grounder = GroundingModule(language_module=lang)

        result = grounder.ground(
            command="pick up the bottle",
            detections=[...],
        )
        if result.grounded:
            print("Target:", result.target_detection.label)
    """

    def __init__(
        self,
        language_module: Optional[LanguageModule] = None,
        min_similarity: float = 0.25,
    ):
        """
        Args:
            language_module: Shared LanguageModule instance (creates one if None)
            min_similarity:  Minimum cosine similarity to accept a grounding match
        """
        self.lang = language_module or LanguageModule()
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(
        self,
        command: str,
        detections: List[Detection],
        command_embedding: Optional[np.ndarray] = None,
    ) -> GroundingResult:
        """
        Ground a natural language command to the best matching detection.

        Args:
            command:            Raw language instruction string
            detections:         List of Detection objects from VisionModule
            command_embedding:  Pre-computed embedding (skip re-encoding if given)

        Returns:
            GroundingResult with the best matching detection and all scores
        """
        # Handle empty detections
        if not detections:
            logger.warning("No detections to ground against.")
            return GroundingResult(
                target_detection=None,
                similarity_score=0.0,
                all_scores=[],
                grounded=False,
                command=command,
            )

        # Encode command
        if command_embedding is None:
            cmd_emb = self.lang.encode(command)
        else:
            cmd_emb = command_embedding

        # Encode all labels (with visual prompt prefix)
        labels = [d.label for d in detections]
        label_embs = self.lang.encode_labels(labels)  # shape (N, D)

        # Cosine similarity (embeddings are already unit-normalized)
        scores = self._cosine_similarity_batch(cmd_emb, label_embs)  # shape (N,)

        # Weight by detection confidence (reward high-confidence detections)
        confidences = np.array([d.confidence for d in detections])
        weighted_scores = scores * (0.7 + 0.3 * confidences)

        # Build ranked list
        ranked = sorted(
            zip(labels, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # Best match
        best_idx = int(np.argmax(weighted_scores))
        best_score = float(scores[best_idx])
        best_det = detections[best_idx]

        grounded = best_score >= self.min_similarity

        if grounded:
            logger.info(
                f"Grounded '{command}' → '{best_det.label}' "
                f"(sim={best_score:.3f}, det_conf={best_det.confidence:.3f})"
            )
        else:
            logger.warning(
                f"Grounding failed for '{command}'. "
                f"Best score {best_score:.3f} < threshold {self.min_similarity}"
            )

        return GroundingResult(
            target_detection=best_det if grounded else None,
            similarity_score=best_score,
            all_scores=ranked,
            grounded=grounded,
            command=command,
        )

    def ground_with_context(
        self,
        command: str,
        detections: List[Detection],
        memory_labels: Optional[List[str]] = None,
    ) -> GroundingResult:
        """
        Ground with optional memory context.
        Enriches command embedding by appending memory labels as context.

        Args:
            command:        Natural language command
            detections:     Current frame detections
            memory_labels:  Labels from memory module for context boost

        Returns:
            GroundingResult
        """
        enriched_command = command
        if memory_labels:
            ctx = ", ".join(memory_labels[:3])  # top-3 recent objects
            enriched_command = f"{command} (context: {ctx})"
            logger.debug(f"Enriched command: '{enriched_command}'")

        return self.ground(enriched_command, detections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity_batch(
        query: np.ndarray,
        keys: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized cosine similarity between a single query and N keys.

        Args:
            query: Shape (D,) — already normalized
            keys:  Shape (N, D) — already normalized

        Returns:
            Shape (N,) similarity scores in [-1, 1]
        """
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        keys_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9)
        similarities = keys_norm @ query_norm  # dot product = cosine sim when normalized
        # Clip to [0, 1] since labels are semantically positive
        return np.clip(similarities, 0.0, 1.0).astype(np.float32)