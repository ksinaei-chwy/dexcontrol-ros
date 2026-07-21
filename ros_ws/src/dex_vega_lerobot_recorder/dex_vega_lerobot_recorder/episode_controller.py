"""Thread-safe candidate episode lifecycle independent of ROS and LeRobot."""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .dataset_writer import CommitResult


class EpisodeState(enum.Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    REVIEW_PENDING = "REVIEW_PENDING"
    SAVING = "SAVING"
    ERROR = "ERROR"


class InvalidTransition(RuntimeError):
    """Raised when a lifecycle command is invalid for the current state."""


class EpisodeValidationError(RuntimeError):
    """Raised when a pending candidate does not meet save requirements."""


class EpisodeWriter(Protocol):
    @property
    def committed_episodes(self) -> int: ...

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> CommitResult: ...

    def clear_episode_buffer(self) -> None: ...

    def finalize(self) -> None: ...


@dataclass(frozen=True)
class EpisodeSummary:
    state: EpisodeState
    frames: int
    duration_seconds: float
    dropped_samples: int
    stale_samples: int
    valid: bool
    validation_message: str


class EpisodeController:
    """Own the invariant that only an explicit save commits an episode."""

    def __init__(
        self,
        writer: EpisodeWriter,
        *,
        minimum_frames: int,
        minimum_duration_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.writer = writer
        self.minimum_frames = int(minimum_frames)
        self.minimum_duration_seconds = float(minimum_duration_seconds)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self.state = EpisodeState.IDLE
        self.frames = 0
        self.dropped_samples = 0
        self.stale_samples = 0
        self._start_time = 0.0
        self._stop_time = 0.0
        self.last_error = ""

    def start_episode(self) -> EpisodeSummary:
        with self._lock:
            self._require(EpisodeState.IDLE, "start_episode")
            self.frames = 0
            self.dropped_samples = 0
            self.stale_samples = 0
            self._start_time = self._monotonic()
            self._stop_time = 0.0
            self.last_error = ""
            self.state = EpisodeState.RECORDING
            return self.summary()

    def add_frame(self, frame: dict[str, Any]) -> bool:
        with self._lock:
            if self.state is not EpisodeState.RECORDING:
                return False
            try:
                self.writer.add_frame(frame)
                self.frames += 1
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.state = EpisodeState.ERROR
                raise

    def note_drop(self, *, stale: bool = False) -> None:
        with self._lock:
            if self.state is EpisodeState.RECORDING:
                self.dropped_samples += 1
                if stale:
                    self.stale_samples += 1

    def stop_episode(self) -> EpisodeSummary:
        with self._lock:
            self._require(EpisodeState.RECORDING, "stop_episode")
            self._stop_time = self._monotonic()
            self.state = EpisodeState.REVIEW_PENDING
            return self.summary()

    def save_episode(self) -> CommitResult:
        with self._lock:
            self._require(EpisodeState.REVIEW_PENDING, "save_episode")
            summary = self.summary()
            if not summary.valid:
                raise EpisodeValidationError(summary.validation_message)
            self.state = EpisodeState.SAVING
            try:
                result = self.writer.save_episode()
            except Exception as exc:
                self.last_error = str(exc)
                self.state = EpisodeState.ERROR
                raise
            self._reset_to_idle()
            return result

    def discard_episode(self) -> EpisodeSummary:
        with self._lock:
            if self.state not in {EpisodeState.RECORDING, EpisodeState.REVIEW_PENDING}:
                raise InvalidTransition(
                    f"discard_episode is invalid from {self.state.value}"
                )
            self.writer.clear_episode_buffer()
            self._reset_to_idle()
            return self.summary()

    def summary(self) -> EpisodeSummary:
        with self._lock:
            duration = self._duration_seconds()
            messages = []
            if self.frames < self.minimum_frames:
                messages.append(
                    f"{self.frames} frames is below minimum {self.minimum_frames}"
                )
            if duration < self.minimum_duration_seconds:
                messages.append(
                    f"{duration:.3f}s is below minimum "
                    f"{self.minimum_duration_seconds:.3f}s"
                )
            if self.frames <= 0:
                messages.append("episode is empty")
            return EpisodeSummary(
                state=self.state,
                frames=self.frames,
                duration_seconds=duration,
                dropped_samples=self.dropped_samples,
                stale_samples=self.stale_samples,
                valid=not messages,
                validation_message="; ".join(messages) if messages else "valid",
            )

    def shutdown(self, *, autosave: bool = False) -> str:
        """Resolve any candidate explicitly, then finalize committed data."""
        with self._lock:
            resolution = "no pending episode"
            if self.state is EpisodeState.RECORDING:
                if autosave:
                    self.stop_episode()
                    if self.summary().valid:
                        self.save_episode()
                        resolution = "autosaved pending recording"
                    else:
                        self.discard_episode()
                        resolution = "discarded invalid pending recording"
                else:
                    self.discard_episode()
                    resolution = "discarded unsaved active recording"
            elif self.state is EpisodeState.REVIEW_PENDING:
                if autosave and self.summary().valid:
                    self.save_episode()
                    resolution = "autosaved review-pending episode"
                else:
                    self.discard_episode()
                    resolution = "discarded unsaved review-pending episode"
            elif self.state is EpisodeState.ERROR:
                try:
                    self.writer.clear_episode_buffer()
                    resolution = "cleared pending data after recorder error"
                except Exception:
                    resolution = "recorder error; pending buffer cleanup failed"
            self.writer.finalize()
            return resolution

    def set_error(self, error: Exception | str) -> None:
        with self._lock:
            self.last_error = str(error)
            self.state = EpisodeState.ERROR

    def _duration_seconds(self) -> float:
        if self._start_time <= 0.0:
            return 0.0
        end = self._stop_time if self._stop_time > 0.0 else self._monotonic()
        return max(0.0, end - self._start_time)

    def _reset_to_idle(self) -> None:
        self.state = EpisodeState.IDLE
        self.frames = 0
        self.dropped_samples = 0
        self.stale_samples = 0
        self._start_time = 0.0
        self._stop_time = 0.0
        self.last_error = ""

    def _require(self, expected: EpisodeState, operation: str) -> None:
        if self.state is not expected:
            raise InvalidTransition(f"{operation} is invalid from {self.state.value}")
