"""
Evaluation Metrics
==================
Computes task success rate, robustness scores, and confidence statistics
for the UVLA robustness evaluation pipeline.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class TrialResult:
    """Result of a single evaluation trial."""
    condition: str          # e.g. "clean", "noise", "blur"
    command: str
    succeeded: bool
    grounding_score: float
    quality_score: float
    combined_confidence: float
    action_type: str
    from_memory: bool


@dataclass
class ConditionMetrics:
    """Aggregated metrics for one perturbation condition."""
    condition: str
    num_trials: int
    task_success_rate: float         # [0, 1]
    mean_confidence: float           # [0, 1]
    std_confidence: float
    min_confidence: float
    max_confidence: float
    memory_fallback_rate: float      # Fraction that needed memory
    rejection_rate: float            # Fraction that were safety-rejected

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "num_trials": self.num_trials,
            "task_success_rate": round(self.task_success_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "std_confidence": round(self.std_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "memory_fallback_rate": round(self.memory_fallback_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
        }


@dataclass
class RobustnessReport:
    """Full evaluation report across all conditions."""
    per_condition: Dict[str, ConditionMetrics] = field(default_factory=dict)
    overall_robustness_score: float = 0.0
    baseline_success_rate: float = 0.0    # Clean condition
    degradation: Dict[str, float] = field(default_factory=dict)  # Drop vs clean

    def to_dict(self) -> dict:
        return {
            "overall_robustness_score": round(self.overall_robustness_score, 4),
            "baseline_success_rate": round(self.baseline_success_rate, 4),
            "per_condition": {
                k: v.to_dict() for k, v in self.per_condition.items()
            },
            "degradation_vs_clean": {
                k: round(v, 4) for k, v in self.degradation.items()
            },
        }

    def print_table(self):
        """Print a formatted results table to stdout."""
        header = f"{'Condition':<15} {'Success%':>9} {'Confidence':>12} {'Rejected%':>10} {'Memory%':>9}"
        print("\n" + "=" * 60)
        print("UVLA ROBUSTNESS EVALUATION RESULTS")
        print("=" * 60)
        print(header)
        print("-" * 60)
        for name, m in self.per_condition.items():
            print(
                f"{name:<15} "
                f"{m.task_success_rate * 100:>8.1f}% "
                f"{m.mean_confidence:>11.3f}  "
                f"{m.rejection_rate * 100:>9.1f}% "
                f"{m.memory_fallback_rate * 100:>8.1f}%"
            )
        print("-" * 60)
        print(f"Overall Robustness Score: {self.overall_robustness_score:.3f}")
        print(f"Baseline (Clean) Success: {self.baseline_success_rate * 100:.1f}%")
        print("=" * 60)


class MetricsCalculator:
    """
    Computes aggregated metrics from a list of TrialResult objects.

    Example::

        calc = MetricsCalculator()
        for trial in trials:
            calc.add(trial)
        report = calc.compute()
        report.print_table()
    """

    def __init__(self):
        self._trials: List[TrialResult] = []

    def add(self, trial: TrialResult):
        """Record a single trial result."""
        self._trials.append(trial)

    def add_many(self, trials: List[TrialResult]):
        """Record multiple trial results."""
        self._trials.extend(trials)

    def compute(self) -> RobustnessReport:
        """
        Aggregate all trials into a RobustnessReport.
        """
        if not self._trials:
            logger.warning("No trials recorded.")
            return RobustnessReport()

        # Group by condition
        conditions: Dict[str, List[TrialResult]] = {}
        for trial in self._trials:
            conditions.setdefault(trial.condition, []).append(trial)

        per_condition: Dict[str, ConditionMetrics] = {}
        for cond_name, trials in conditions.items():
            per_condition[cond_name] = self._aggregate(cond_name, trials)

        # Overall robustness = weighted average of success rates
        # (clean gets 0 weight, degraded conditions each get equal weight)
        degraded_metrics = [
            m for name, m in per_condition.items()
            if name != "clean"
        ]
        if degraded_metrics:
            robustness = float(
                np.mean([m.task_success_rate for m in degraded_metrics])
            )
        else:
            robustness = per_condition.get("clean", ConditionMetrics(
                "clean", 0, 0, 0, 0, 0, 0, 0, 0
            )).task_success_rate

        baseline = per_condition.get(
            "clean",
            ConditionMetrics("clean", 0, 0.0, 0, 0, 0, 0, 0, 0)
        ).task_success_rate

        # Degradation vs clean
        degradation = {}
        for name, m in per_condition.items():
            if name != "clean":
                degradation[name] = baseline - m.task_success_rate

        return RobustnessReport(
            per_condition=per_condition,
            overall_robustness_score=robustness,
            baseline_success_rate=baseline,
            degradation=degradation,
        )

    def reset(self):
        """Clear all recorded trials."""
        self._trials.clear()

    @staticmethod
    def _aggregate(condition: str, trials: List[TrialResult]) -> ConditionMetrics:
        """Compute summary statistics for one condition."""
        n = len(trials)
        if n == 0:
            return ConditionMetrics(condition, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        success_rate = sum(t.succeeded for t in trials) / n
        confidences = [t.combined_confidence for t in trials]
        memory_rate = sum(t.from_memory for t in trials) / n
        rejection_rate = sum(
            1 for t in trials if t.action_type == "safe_reject"
        ) / n

        return ConditionMetrics(
            condition=condition,
            num_trials=n,
            task_success_rate=success_rate,
            mean_confidence=float(np.mean(confidences)),
            std_confidence=float(np.std(confidences)),
            min_confidence=float(np.min(confidences)),
            max_confidence=float(np.max(confidences)),
            memory_fallback_rate=memory_rate,
            rejection_rate=rejection_rate,
        )