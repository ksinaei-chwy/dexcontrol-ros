"""Pure proportional retargeting helpers for Dexmate F5D6 hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


F5D6_COMMAND_SUFFIXES = (
    "th_j1",
    "ff_j1",
    "mf_j1",
    "rf_j1",
    "lf_j1",
    "th_j0",
)
F5D6_OPEN_POSITIONS = (0.1834, 0.2891, 0.2801, 0.2840, 0.2811, -0.0158)
F5D6_CLOSED_POSITIONS = (-0.1, -1.0946, -1.0844, -1.0154, -1.0118, 0.84)

_SIDE_PREFIXES = {"left": "L", "right": "R"}
_FLEXION_INDICES = (0, 1, 2, 3, 4)
_OPPOSITION_INDEX = 5
_MIMIC_SPECS = (
    ("th_j2", 0, 1.35316, 0.00765),
    ("ff_j2", 1, 1.13028, -0.00053),
    ("mf_j2", 2, 1.13311, -0.00079),
    ("rf_j2", 3, 1.12935, 0.00065),
    ("lf_j2", 4, 1.15037, 0.00186),
)


def f5d6_joint_names(side: str) -> tuple[str, ...]:
    """Return the required six-command joint order for one F5D6 hand."""
    normalized_side = str(side).lower()
    try:
        prefix = _SIDE_PREFIXES[normalized_side]
    except KeyError as exc:
        raise ValueError(f"hand side must be 'left' or 'right', got {side!r}") from exc
    return tuple(f"{prefix}_{suffix}" for suffix in F5D6_COMMAND_SUFFIXES)


@dataclass(frozen=True)
class F5D6HandConfig:
    """Validated immutable endpoints for one six-command F5D6 hand."""

    side: str
    joint_names: tuple[str, ...]
    open_positions: tuple[float, ...]
    closed_positions: tuple[float, ...]

    @classmethod
    def from_values(
        cls,
        side: str,
        joint_names: Iterable[str],
        open_positions: Iterable[float],
        closed_positions: Iterable[float],
    ) -> "F5D6HandConfig":
        normalized_side = str(side).lower()
        expected_names = f5d6_joint_names(normalized_side)
        names = tuple(str(name) for name in joint_names)
        if names != expected_names:
            raise ValueError(
                f"{normalized_side} hand joint names must be {list(expected_names)}, "
                f"got {list(names)}"
            )

        open_values = _endpoint_values(
            open_positions,
            f"{normalized_side}_hand_open_positions",
        )
        closed_values = _endpoint_values(
            closed_positions,
            f"{normalized_side}_hand_closed_positions",
        )
        return cls(normalized_side, names, open_values, closed_values)


def retarget_f5d6_hand(
    config: F5D6HandConfig,
    trigger: float,
    grip: float,
) -> np.ndarray:
    """Interpolate flexion with trigger and thumb opposition with grip."""
    trigger_value = _finite_control_value(trigger, "trigger")
    grip_value = _finite_control_value(grip, "grip")
    open_positions = np.asarray(config.open_positions, dtype=np.float64)
    closed_positions = np.asarray(config.closed_positions, dtype=np.float64)
    target = open_positions.copy()
    target[list(_FLEXION_INDICES)] += trigger_value * (
        closed_positions[list(_FLEXION_INDICES)]
        - open_positions[list(_FLEXION_INDICES)]
    )
    target[_OPPOSITION_INDEX] += grip_value * (
        closed_positions[_OPPOSITION_INDEX] - open_positions[_OPPOSITION_INDEX]
    )
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{config.side} hand target contains non-finite values")
    return target


def f5d6_visual_joint_positions(
    side: str,
    command_positions: Iterable[float],
) -> dict[str, float]:
    """Expand six driver positions to the five URDF mimic joints for display."""
    names = f5d6_joint_names(side)
    values = np.asarray(tuple(command_positions), dtype=np.float64).reshape(-1)
    if values.size != len(names):
        raise ValueError(
            f"{side} hand expected {len(names)} values, got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{side} hand visualization values must be finite")

    result = {name: float(value) for name, value in zip(names, values)}
    prefix = _SIDE_PREFIXES[str(side).lower()]
    for suffix, driver_index, multiplier, offset in _MIMIC_SPECS:
        result[f"{prefix}_{suffix}"] = float(values[driver_index] * multiplier + offset)
    return result


def f5d6_visual_joint_names() -> tuple[str, ...]:
    """Return all driver and mimic joint names used by dry-run visualization."""
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f5d6_visual_joint_positions(side, np.zeros(6)).keys())
    return tuple(names)


def _endpoint_values(values: Iterable[float], parameter_name: str) -> tuple[float, ...]:
    try:
        endpoint = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must contain numeric values") from exc
    expected_length = len(F5D6_COMMAND_SUFFIXES)
    if endpoint.size != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values, got {endpoint.size}"
        )
    if not np.all(np.isfinite(endpoint)):
        raise ValueError(f"{parameter_name} must contain only finite values")
    return tuple(float(value) for value in endpoint)


def _finite_control_value(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"hand {name} must be finite")
    return float(np.clip(result, 0.0, 1.0))
