"""Bounded latest-frame workers for camera output backends."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OutputWorkerStats:
    enqueued_frames: int
    published_frames: int
    replaced_frames: int
    failures: int
    last_sequence: int
    last_queue_age_seconds: float
    last_processing_seconds: float
    last_publish_time_ns: int
    last_error: str


class LatestFrameOutputWorker:
    """Run a potentially blocking output on a capacity-one frame slot."""

    def __init__(
        self,
        *,
        name: str,
        publish: Callable[[np.ndarray], bool | None],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.name = str(name)
        self._publish = publish
        self._transform = transform or (lambda frame: frame)
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, int, float] | None = None
        self._stop = False
        self._enqueued = 0
        self._published = 0
        self._replaced = 0
        self._failures = 0
        self._last_sequence = 0
        self._last_queue_age = 0.0
        self._last_processing = 0.0
        self._last_publish_time_ns = 0
        self._last_error = ""
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.name}_latest_frame_output",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: np.ndarray, sequence: int) -> None:
        """Replace any frame waiting to be processed and return immediately."""
        with self._condition:
            if self._stop:
                return
            self._enqueued += 1
            if self._pending is not None:
                self._replaced += 1
            self._pending = (frame, int(sequence), time.monotonic())
            self._condition.notify()

    def stats(self) -> OutputWorkerStats:
        with self._condition:
            return OutputWorkerStats(
                enqueued_frames=self._enqueued,
                published_frames=self._published,
                replaced_frames=self._replaced,
                failures=self._failures,
                last_sequence=self._last_sequence,
                last_queue_age_seconds=self._last_queue_age,
                last_processing_seconds=self._last_processing,
                last_publish_time_ns=self._last_publish_time_ns,
                last_error=self._last_error,
            )

    def shutdown(self, timeout_seconds: float = 2.5) -> None:
        """Drop any pending frame and stop the worker."""
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=max(float(timeout_seconds), 0.0))

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                frame, sequence, enqueued_at = self._pending
                self._pending = None

            started = time.monotonic()
            queue_age = started - enqueued_at
            try:
                output = self._transform(frame)
                accepted = self._publish(output)
                processing = time.monotonic() - started
                with self._condition:
                    self._last_sequence = sequence
                    self._last_queue_age = queue_age
                    self._last_processing = processing
                    if accepted is not False:
                        self._published += 1
                        self._last_publish_time_ns = time.time_ns()
                        self._last_error = ""
                    else:
                        self._failures += 1
                        self._last_error = "output rejected frame"
            except Exception as exc:  # noqa: BLE001 - output boundary
                processing = time.monotonic() - started
                with self._condition:
                    self._failures += 1
                    self._last_sequence = sequence
                    self._last_queue_age = queue_age
                    self._last_processing = processing
                    self._last_error = str(exc)
