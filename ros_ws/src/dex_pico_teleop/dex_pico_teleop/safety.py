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


def base_twist_from_joysticks(
    left_joystick: np.ndarray,
    right_joystick: np.ndarray,
    deadzone: float,
    vx_scale: float,
    vy_scale: float,
    wz_scale: float,
) -> np.ndarray:
    """Map Pico controller joysticks to body-frame base velocity."""
    left = joystick_with_deadzone(left_joystick, deadzone)
    right = joystick_with_deadzone(right_joystick, deadzone)
    return np.array(
        [
            left[1] * float(vx_scale),
            -left[0] * float(vy_scale),
            -right[0] * float(wz_scale),
        ],
        dtype=np.float64,
    )


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
