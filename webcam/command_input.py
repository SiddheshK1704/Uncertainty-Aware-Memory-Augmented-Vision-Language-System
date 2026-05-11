"""
Command Input Handler
======================
Provides non-blocking, thread-safe command input for the real-time loop.

The webcam loop runs at 20-30 FPS; we cannot block it waiting for keyboard
input.  This module runs a background thread that reads from stdin (or a
queue) and stores the latest command for the main loop to pick up.

Usage:
    handler = CommandInputHandler()
    handler.start()

    # In your main loop:
    cmd = handler.get_command()  # returns latest command, non-blocking

    handler.stop()
"""

from __future__ import annotations
import threading
import queue
import sys
from typing import Optional
from loguru import logger


# ── Preset shortcut commands (type the alias, get the full phrase) ───
COMMAND_ALIASES: dict[str, str] = {
    "chair":    "find chair",
    "bottle":   "locate bottle",
    "tv":       "move toward tv",
    "sofa":     "navigate to sofa",
    "table":    "find dining table",
    "cup":      "locate cup",
    "laptop":   "find laptop",
    "book":     "locate book",
    "person":   "follow person",
    "clear":    "",
    "reset":    "",
    "help":     "__help__",
}

HELP_TEXT = """
╔══════════════════════════════════════════════════╗
║          UVLA Live  —  Command Reference          ║
╠══════════════════════════════════════════════════╣
║  Type a full command or a quick alias:            ║
║    "find chair"        chair                      ║
║    "locate bottle"     bottle                     ║
║    "move toward tv"    tv                         ║
║    "navigate to sofa"  sofa                       ║
║    "find dining table" table                      ║
║    "find laptop"       laptop                     ║
║    "locate book"       book                       ║
║  Other:                                           ║
║    clear / reset  — clear command                 ║
║    help           — show this help                ║
║    q  (in window) — quit                          ║
╚══════════════════════════════════════════════════╝
"""


class CommandInputHandler:
    """
    Background-thread command reader.

    The thread blocks on input() while the main thread runs the video loop.
    Thread-safe via a queue.

    Example::

        handler = CommandInputHandler(default_command="find chair")
        handler.start()

        while running:
            cmd = handler.get_command()   # never blocks
            ...

        handler.stop()
    """

    def __init__(self, default_command: str = "find chair"):
        self._current_command: str = default_command
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ──────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background input thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="uvla-cmd-input",
        )
        self._thread.start()
        print(f"\n{'─'*52}")
        print(f"  UVLA Live System  —  Real-time Webcam Pipeline")
        print(f"{'─'*52}")
        print(f"  Active command : \"{self._current_command}\"")
        print(f"  Type a command and press Enter to update.")
        print(f"  Type 'help' for command list.")
        print(f"  Press 'q' in the video window to quit.")
        print(f"{'─'*52}\n")

    def stop(self) -> None:
        """Signal the input thread to stop (it will exit on next input)."""
        self._running = False

    def get_command(self) -> str:
        """
        Non-blocking: drain the queue and return the most recent command.
        Returns the previously active command if nothing new.
        """
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._current_command = latest
        return self._current_command

    # ──────────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Blocking stdin reader running in background thread."""
        while self._running:
            try:
                raw = input("cmd> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not raw:
                continue

            # Expand alias
            resolved = COMMAND_ALIASES.get(raw.lower(), raw)

            if resolved == "__help__":
                print(HELP_TEXT)
                continue

            if resolved == "":
                print("  ✓ Command cleared.")
                self._queue.put("")
                continue

            self._queue.put(resolved)
            print(f"  ✓ Command updated → \"{resolved}\"")