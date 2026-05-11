"""
Memory Module
=============
Maintains a persistent short-term memory of previously detected objects,
including their last known locations and confidence scores.

Key features:
  - Stores object detections across frames
  - Handles temporary occlusion by retaining stale entries
  - Provides fallback targets when current frame has no detections
  - TTL (time-to-live) based automatic expiration
"""

from __future__ import annotations
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .vision_module import Detection


@dataclass
class MemoryEntry:
    """A single memory record for a detected object."""
    label: str
    bbox: List[float]                  # Last known [x1, y1, x2, y2]
    center: Tuple[float, float]        # Last known center
    confidence: float                  # Detection confidence at time of storage
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0                  # Which frame this was seen in
    seen_count: int = 1                # How many times total this class was seen

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was last updated."""
        return time.time() - self.timestamp

    def refresh(self, detection: Detection, frame_id: int = 0):
        """Update entry with a fresh detection."""
        self.bbox = detection.bbox
        self.center = detection.center
        self.confidence = detection.confidence
        self.timestamp = time.time()
        self.frame_id = frame_id
        self.seen_count += 1

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox": [round(v, 1) for v in self.bbox],
            "center": (round(self.center[0], 1), round(self.center[1], 1)),
            "confidence": round(self.confidence, 4),
            "age_seconds": round(self.age_seconds, 2),
            "seen_count": self.seen_count,
        }


class MemoryModule:
    """
    Short-term object location memory for occlusion robustness.

    When an object disappears from the current frame (due to occlusion,
    motion blur, or detection failure), the system can fall back to the
    last known position stored here.

    Example usage::

        memory = MemoryModule(ttl=30.0, max_entries=20)

        # After each detection frame:
        memory.update(detections, frame_id=42)

        # Try to recall where 'chair' was:
        entry = memory.recall("chair")
        if entry:
            print(f"Chair was at {entry.center}")
    """

    def __init__(
        self,
        ttl: float = 30.0,
        max_entries: int = 50,
        max_history_per_class: int = 5,
    ):
        """
        Args:
            ttl:                   Time-to-live in seconds before entries expire
            max_entries:           Maximum total memory entries (oldest removed)
            max_history_per_class: How many historical positions to keep per class
        """
        self.ttl = ttl
        self.max_entries = max_entries
        self.max_history_per_class = max_history_per_class

        # Primary store: label → most recent MemoryEntry
        self._store: Dict[str, MemoryEntry] = {}

        # History store: label → deque of past positions (for trajectory)
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_per_class)
        )

        # Insertion order for eviction (oldest-first)
        self._insertion_order: deque = deque()

        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection], frame_id: Optional[int] = None):
        """
        Update memory with fresh detections from the current frame.

        Args:
            detections: List of Detection objects from VisionModule
            frame_id:   Optional frame index for tracking
        """
        if frame_id is None:
            frame_id = self._frame_counter
        self._frame_counter += 1

        for det in detections:
            label = det.label
            if label in self._store:
                # Save previous position to history
                old_entry = self._store[label]
                self._history[label].append(
                    {"center": old_entry.center, "time": old_entry.timestamp}
                )
                # Refresh existing entry
                self._store[label].refresh(det, frame_id)
                logger.debug(f"Memory updated: '{label}' @ {det.center}")
            else:
                # New object seen
                self._evict_if_full()
                entry = MemoryEntry(
                    label=label,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    frame_id=frame_id,
                )
                self._store[label] = entry
                self._insertion_order.append(label)
                logger.info(f"Memory: new object '{label}' stored at {det.center}")

        # Remove expired entries
        self._expire_old_entries()

    def recall(self, label: str) -> Optional[MemoryEntry]:
        """
        Recall the last known state of a specific object class.

        Args:
            label: Object class name (e.g. "chair")

        Returns:
            MemoryEntry if found and not expired, else None
        """
        entry = self._store.get(label)
        if entry is None:
            logger.debug(f"Memory miss: '{label}' not in memory.")
            return None

        if entry.age_seconds > self.ttl:
            logger.debug(f"Memory expired: '{label}' (age={entry.age_seconds:.1f}s)")
            self._remove(label)
            return None

        logger.info(
            f"Memory recall: '{label}' found, age={entry.age_seconds:.1f}s, "
            f"center={entry.center}"
        )
        return entry

    def recall_all(self) -> List[MemoryEntry]:
        """Return all non-expired memory entries."""
        self._expire_old_entries()
        return list(self._store.values())

    def get_recent_labels(self, n: int = 5) -> List[str]:
        """Return labels of the N most recently seen objects."""
        entries = sorted(
            self._store.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return [e.label for e in entries[:n]]

    def get_trajectory(self, label: str) -> List[Tuple[float, float]]:
        """
        Return historical center positions for a given label.

        Useful for estimating object motion direction.
        """
        if label not in self._history:
            return []
        return [h["center"] for h in self._history[label]]

    def clear(self):
        """Wipe all memory entries."""
        self._store.clear()
        self._history.clear()
        self._insertion_order.clear()
        logger.info("Memory cleared.")

    def summary(self) -> dict:
        """Return a summary of current memory contents."""
        self._expire_old_entries()
        return {
            "total_entries": len(self._store),
            "labels": list(self._store.keys()),
            "entries": [e.to_dict() for e in self._store.values()],
        }

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expire_old_entries(self):
        """Remove entries older than TTL."""
        expired = [
            label for label, entry in self._store.items()
            if entry.age_seconds > self.ttl
        ]
        for label in expired:
            self._remove(label)
            logger.debug(f"Expired memory entry: '{label}'")

    def _evict_if_full(self):
        """Remove oldest entry if at capacity."""
        while len(self._store) >= self.max_entries and self._insertion_order:
            oldest_label = self._insertion_order.popleft()
            self._store.pop(oldest_label, None)
            logger.debug(f"Evicted oldest memory entry: '{oldest_label}'")

    def _remove(self, label: str):
        """Remove a specific label from memory and insertion order."""
        self._store.pop(label, None)
        try:
            self._insertion_order.remove(label)
        except ValueError:
            pass