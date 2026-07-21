"""Bounded latest-observation worker that never blocks ROS subscription callbacks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .observation_adapter import ObservationSnapshot


class InferenceRuntime(Protocol):
    def predict(self, observation: ObservationSnapshot) -> object: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class WorkerStats:
    submitted: int
    completed: int
    dropped_or_replaced: int
    discarded_after_reset: int
    errors: int
    generation: int
    busy: bool


class LatestObservationWorker:
    """One daemon worker with a one-item pending slot and reset generations."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        on_result: Callable[[ObservationSnapshot, object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._runtime = runtime
        self._on_result = on_result
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending: tuple[int, ObservationSnapshot] | None = None
        self._generation = 0
        self._reset_requested = False
        self._stop = False
        self._busy = False
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._discarded = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="LeRobotInferenceWorker",
        )
        self._thread.start()

    def submit(self, observation: ObservationSnapshot) -> int:
        with self._condition:
            if self._stop:
                raise RuntimeError("inference worker is stopped")
            self._submitted += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = (self._generation, observation)
            self._condition.notify()
            return self._generation

    def reset(self) -> int:
        """Invalidate pending/in-flight work and reset policy queues on the worker."""
        with self._condition:
            self._generation += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = None
            self._reset_requested = True
            self._condition.notify()
            return self._generation

    def close(self, timeout_seconds: float = 3.0) -> bool:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=max(0.0, float(timeout_seconds)))
        return not self._thread.is_alive()

    def stats(self) -> WorkerStats:
        with self._condition:
            return WorkerStats(
                submitted=self._submitted,
                completed=self._completed,
                dropped_or_replaced=self._dropped,
                discarded_after_reset=self._discarded,
                errors=self._errors,
                generation=self._generation,
                busy=self._busy,
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._stop
                    and self._pending is None
                    and not self._reset_requested
                ):
                    self._condition.wait()
                if self._stop:
                    return
                if self._reset_requested:
                    self._reset_requested = False
                    do_reset = True
                    item = None
                else:
                    do_reset = False
                    item = self._pending
                    self._pending = None
                    self._busy = item is not None

            if do_reset:
                try:
                    self._runtime.reset()
                except Exception as exc:  # noqa: BLE001 - policy boundary
                    self._record_error(exc)
                continue
            if item is None:
                continue
            generation, observation = item
            try:
                result = self._runtime.predict(observation)
            except Exception as exc:  # noqa: BLE001 - CUDA/model boundary
                self._record_error(exc)
                continue
            with self._condition:
                self._busy = False
                if generation != self._generation or self._stop:
                    self._discarded += 1
                    continue
                self._completed += 1
            try:
                self._on_result(observation, result)
            except Exception as exc:  # noqa: BLE001 - callback boundary
                self._record_error(exc)

    def _record_error(self, error: Exception) -> None:
        with self._condition:
            self._busy = False
            self._errors += 1
        try:
            self._on_error(error)
        except Exception:
            time.sleep(0.01)
