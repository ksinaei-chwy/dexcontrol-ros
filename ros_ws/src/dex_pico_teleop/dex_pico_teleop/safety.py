"""Command safety helpers."""

from __future__ import annotations

import numpy as np


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    scale = 1.0 / max(1.0e-9, 1.0 - deadzone)
    return float(np.sign(value) * (abs(value) - deadzone) * scale)


def joystick_with_deadzone(joystick: np.ndarray, deadzone: float) -> np.ndarray:
    values = np.asarray(joystick, dtype=np.float64).reshape(2)
    return np.array([apply_deadzone(values[0], deadzone), apply_deadzone(values[1], deadzone)])


class VectorRateLimiter:
    """Limit per-element command changes per update."""

    def __init__(self, max_delta: float) -> None:
        self.max_delta = float(max_delta)
        self._last: np.ndarray | None = None

    def reset(self, value: np.ndarray | None = None) -> None:
        self._last = None if value is None else np.asarray(value, dtype=np.float64).copy()

    def limit(self, value: np.ndarray) -> np.ndarray:
        target = np.asarray(value, dtype=np.float64)
        if self._last is None or self._last.shape != target.shape:
            self._last = target.copy()
            return target
        delta = np.clip(target - self._last, -self.max_delta, self.max_delta)
        self._last = self._last + delta
        return self._last.copy()

