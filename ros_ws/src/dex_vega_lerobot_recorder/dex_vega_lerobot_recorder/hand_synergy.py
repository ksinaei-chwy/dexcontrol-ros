"""F5D6 two-DoF hand synergy expansion and post-bridge reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .configuration import HandSynergyConfig


class HandSynergyError(ValueError):
    """Raised when six hand joints cannot represent the configured two-DoF hand."""


def expand_hand_synergy(
    config: HandSynergyConfig,
    open_close_ratio: float,
    thumb_opposition_ratio: float,
) -> np.ndarray:
    """Expand the logical hand ratios to the six F5D6 driver targets."""
    close = _bounded_ratio(open_close_ratio, "open_close_ratio")
    thumb = _bounded_ratio(thumb_opposition_ratio, "thumb_opposition_ratio")
    opened = np.asarray(config.open_positions, dtype=np.float64)
    closed = np.asarray(config.closed_positions, dtype=np.float64)
    result = opened.copy()
    result[:5] += close * (closed[:5] - opened[:5])
    result[5] += thumb * (closed[5] - opened[5])
    return result


def reconstruct_hand_synergy(
    config: HandSynergyConfig,
    positions: Mapping[str, float],
    *,
    require_exact_action: bool,
) -> tuple[float, float]:
    """
    Map measured/applied drivers to open-close and thumb-opposition ratios.

    Measured hand positions are projected to the two logical coordinates and
    clipped to their physical [0, 1] range. Applied commands are accepted only
    when all five flexion drivers agree on one ratio within the configured
    tolerance, so a lossy or off-synergy command is never labeled as two-DoF.
    """
    try:
        values = np.asarray(
            [positions[name] for name in config.joint_names], dtype=np.float64
        )
    except KeyError as exc:
        raise HandSynergyError(
            f"{config.side} hand is missing applied/measured joint {exc.args[0]}"
        ) from exc
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise HandSynergyError(f"{config.side} hand positions must be six finite values")

    opened = np.asarray(config.open_positions, dtype=np.float64)
    closed = np.asarray(config.closed_positions, dtype=np.float64)
    normalized = (values - opened) / (closed - opened)
    flexion = normalized[:5]
    open_close = float(np.mean(flexion))
    thumb = float(normalized[5])

    if require_exact_action:
        tolerance = config.action_ratio_tolerance
        disagreement = float(np.max(np.abs(flexion - open_close)))
        if disagreement > tolerance:
            raise HandSynergyError(
                f"{config.side} applied hand flexion is off the two-DoF synergy "
                f"(ratio disagreement {disagreement:.6f} > {tolerance:.6f})"
            )
        if open_close < -tolerance or open_close > 1.0 + tolerance:
            raise HandSynergyError(
                f"{config.side} applied open_close_ratio is outside [0, 1]: "
                f"{open_close:.6f}"
            )
        if thumb < -tolerance or thumb > 1.0 + tolerance:
            raise HandSynergyError(
                f"{config.side} applied thumb_opposition_ratio is outside [0, 1]: "
                f"{thumb:.6f}"
            )

    return float(np.clip(open_close, 0.0, 1.0)), float(np.clip(thumb, 0.0, 1.0))


def _bounded_ratio(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise HandSynergyError(f"{name} must be finite and within [0, 1]")
    return result
