"""Small state helpers for Pico teleoperation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def joint_values(
    names: tuple[str, ...],
    feedback_positions: Mapping[str, float],
    command_positions: Mapping[str, float],
    prefer_command: bool = True,
) -> np.ndarray:
    """Return joint values from command warm-starts and feedback fallbacks."""
    primary = command_positions if prefer_command else feedback_positions
    fallback = feedback_positions if prefer_command else command_positions
    return np.asarray(
        [primary.get(name, fallback.get(name, 0.0)) for name in names],
        dtype=np.float64,
    )
