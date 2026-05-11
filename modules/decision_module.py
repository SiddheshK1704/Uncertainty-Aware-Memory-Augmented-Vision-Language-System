"""
Decision Module
===============
Gated execution pipeline that translates grounded vision-language results
into safe robot/navigation actions.

Core safety rule:
    Actions are ONLY executed when BOTH:
        1. Grounding confidence >= CONFIDENCE_THRESHOLD
        2. Image quality (uncertainty) confidence >= QUALITY_THRESHOLD

    Otherwise, the system falls back to memory or issues a safe rejection.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from loguru import logger

from .vision_module import Detection
from .grounding_module import GroundingResult
from .uncertainty_module import UncertaintyResult
from .memory_module import MemoryModule, MemoryEntry


class ActionType(str, Enum):
    """Enumeration of all possible action outputs."""
    NAVIGATE    = "navigate"       # Move toward target object
    PICK_UP     = "pick_up"        # Attempt to grasp target
    INSPECT     = "inspect"        # Look at / examine target
    WAIT        = "wait"           # Hold position, re-assess
    SAFE_REJECT = "safe_reject"    # Blocked by safety gate
    MEMORY_FALLBACK = "memory_fallback"  # Executed from memory (not live detection)


@dataclass
class ActionResult:
    """Describes the output decision from the pipeline."""
    action_type: ActionType
    target_label: Optional[str]           # What the action targets
    target_bbox: Optional[List[float]]    # Where the target is
    target_center: Optional[tuple]        # Center point
    confidence: float                      # Overall pipeline confidence
    grounding_score: float                 # Language-vision match score
    quality_score: float                   # Image quality score
    from_memory: bool                      # Whether target came from memory
    reason: str                            # Human-readable explanation
    timestamp: float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.action_type not in (ActionType.SAFE_REJECT, ActionType.WAIT)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "target_label": self.target_label,
            "target_bbox": (
                [round(v, 1) for v in self.target_bbox]
                if self.target_bbox else None
            ),
            "target_center": self.target_center,
            "confidence": round(self.confidence, 4),
            "grounding_score": round(self.grounding_score, 4),
            "quality_score": round(self.quality_score, 4),
            "from_memory": self.from_memory,
            "reason": self.reason,
            "succeeded": self.succeeded,
        }

    def __str__(self) -> str:
        icon = "✅" if self.succeeded else "🛑"
        return (
            f"{icon} Action: {self.action_type.value.upper()} | "
            f"Target: {self.target_label} | "
            f"Confidence: {self.confidence:.2f} | "
            f"Reason: {self.reason}"
        )


# Keyword maps for parsing natural language commands
_ACTION_KEYWORDS = {
    ActionType.NAVIGATE: ["go", "navigate", "move", "walk", "approach", "head", "travel"],
    ActionType.PICK_UP:  ["pick", "grab", "take", "lift", "hold", "fetch", "get"],
    ActionType.INSPECT:  ["look", "inspect", "examine", "check", "find", "locate", "show"],
}


class DecisionModule:
    """
    Gated action execution pipeline with safety guarantees.

    Example usage::

        memory = MemoryModule()
        decision = DecisionModule(memory_module=memory)

        action = decision.execute(
            command="navigate to the chair",
            grounding_result=grounding_result,
            uncertainty_result=uncertainty_result,
        )
        print(action)
    """

    def __init__(
        self,
        memory_module: Optional[MemoryModule] = None,
        confidence_threshold: float = 0.50,
        quality_threshold: float = 0.30,
        use_memory_fallback: bool = True,
    ):
        """
        Args:
            memory_module:         Shared MemoryModule instance
            confidence_threshold:  Min grounding score to allow action
            quality_threshold:     Min image quality score to allow action
            use_memory_fallback:   If True, use memory when live detection fails
        """
        self.memory = memory_module or MemoryModule()
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.use_memory_fallback = use_memory_fallback

        # Action history for logging/analysis
        self._history: List[ActionResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        grounding_result: GroundingResult,
        uncertainty_result: UncertaintyResult,
    ) -> ActionResult:
        """
        Main entry point: given grounding + uncertainty, decide on an action.

        Args:
            command:            Original natural language command
            grounding_result:   Output from GroundingModule.ground()
            uncertainty_result: Output from UncertaintyModule.estimate()

        Returns:
            ActionResult with the decided action
        """
        g_score = grounding_result.similarity_score
        q_score = uncertainty_result.overall_confidence
        combined = 0.6 * g_score + 0.4 * q_score

        action_type = self._parse_action_type(command)
        target = grounding_result.target_detection

        # --- Gate 1: Image quality check ---
        if not uncertainty_result.is_acceptable:
            reason = f"Unsafe: image rejected ({uncertainty_result.rejection_reason})"

            # Try memory fallback
            if self.use_memory_fallback and target is not None:
                mem_entry = self.memory.recall(target.label)
                if mem_entry:
                    return self._memory_action(
                        action_type, mem_entry, g_score, q_score, combined,
                        reason=f"Memory fallback (image quality low): last seen {mem_entry.age_seconds:.1f}s ago"
                    )

            return self._reject(reason, g_score, q_score, combined)

        # --- Gate 2: Grounding confidence check ---
        if not grounding_result.grounded or g_score < self.confidence_threshold:
            reason = (
                f"Unsafe: grounding confidence too low "
                f"({g_score:.2f} < {self.confidence_threshold})"
            )

            # Try memory fallback by extracting target noun from command
            if self.use_memory_fallback:
                fallback = self._memory_lookup_from_command(command)
                if fallback:
                    return self._memory_action(
                        action_type, fallback, g_score, q_score, combined,
                        reason=f"Memory fallback (low grounding): using last known '{fallback.label}'"
                    )

            return self._reject(reason, g_score, q_score, combined)

        # --- All gates passed: execute action ---
        if target is None:
            return self._reject(
                "No grounded target available.", g_score, q_score, combined
            )

        # Update memory with the freshly detected target
        self.memory.update([target])

        reason = (
            f"Action approved: grounding={g_score:.2f}, "
            f"quality={q_score:.2f}, combined={combined:.2f}"
        )

        result = ActionResult(
            action_type=action_type,
            target_label=target.label,
            target_bbox=target.bbox,
            target_center=target.center,
            confidence=combined,
            grounding_score=g_score,
            quality_score=q_score,
            from_memory=False,
            reason=reason,
        )

        logger.success(str(result))
        self._history.append(result)
        return result

    def get_history(self) -> List[ActionResult]:
        """Return list of all past action results."""
        return list(self._history)

    def get_success_rate(self) -> float:
        """Return fraction of actions that succeeded (not rejected)."""
        if not self._history:
            return 0.0
        successes = sum(1 for a in self._history if a.succeeded)
        return successes / len(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_action_type(self, command: str) -> ActionType:
        """Infer action type from command keywords."""
        cmd_lower = command.lower()
        for action_type, keywords in _ACTION_KEYWORDS.items():
            if any(kw in cmd_lower for kw in keywords):
                return action_type
        return ActionType.INSPECT  # Default: look at the object

    def _memory_lookup_from_command(self, command: str) -> Optional[MemoryEntry]:
        """Try to extract an object name from the command and recall from memory."""
        cmd_lower = command.lower()
        # Check each memory entry's label against the command text
        for entry in self.memory.recall_all():
            if entry.label.lower() in cmd_lower:
                return entry
        return None

    def _memory_action(
        self,
        action_type: ActionType,
        entry: MemoryEntry,
        g_score: float,
        q_score: float,
        combined: float,
        reason: str,
    ) -> ActionResult:
        """Create an ActionResult sourced from memory."""
        result = ActionResult(
            action_type=ActionType.MEMORY_FALLBACK,
            target_label=entry.label,
            target_bbox=entry.bbox,
            target_center=entry.center,
            confidence=combined * 0.8,  # Discount confidence for memory use
            grounding_score=g_score,
            quality_score=q_score,
            from_memory=True,
            reason=reason,
        )
        logger.info(f"Memory action: {result}")
        self._history.append(result)
        return result

    def _reject(
        self,
        reason: str,
        g_score: float,
        q_score: float,
        combined: float,
    ) -> ActionResult:
        """Create a safe rejection ActionResult."""
        result = ActionResult(
            action_type=ActionType.SAFE_REJECT,
            target_label=None,
            target_bbox=None,
            target_center=None,
            confidence=combined,
            grounding_score=g_score,
            quality_score=q_score,
            from_memory=False,
            reason=reason,
        )
        logger.warning(f"Action REJECTED: {reason}")
        self._history.append(result)
        return result