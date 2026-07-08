"""Posture retargeting helpers for Pico-to-Vega teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dex_pico_teleop.transforms import rot_y


@dataclass(frozen=True)
class ReachTarget:
    position: np.ndarray
    fraction: float


@dataclass(frozen=True)
class OperatorArmLength:
    value: float
    source: str


def side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def estimate_operator_arm_length(
    operator_height: float,
    ratio: float,
    minimum: float,
    maximum: float,
) -> float:
    scaled = float(operator_height) * float(ratio)
    return float(np.clip(scaled, float(minimum), float(maximum)))


def operator_arm_length_for_side(
    side: str,
    calibrated_lengths: dict[str, float],
    operator_height: float,
    ratio: float,
    minimum: float,
    maximum: float,
) -> OperatorArmLength:
    calibrated = calibrated_lengths.get(side)
    if calibrated is not None and np.isfinite(calibrated):
        return OperatorArmLength(
            value=float(np.clip(float(calibrated), float(minimum), float(maximum))),
            source="calibrated",
        )
    return OperatorArmLength(
        value=estimate_operator_arm_length(operator_height, ratio, minimum, maximum),
        source="height_estimate",
    )


def operator_shoulder_position(
    side: str,
    head_position: np.ndarray,
    shoulder_width: float,
    head_to_shoulder_z: float,
    shoulder_x: float,
) -> np.ndarray:
    head = np.asarray(head_position, dtype=np.float64).reshape(3)
    return np.array(
        [
            head[0] + float(shoulder_x),
            head[1] + side_sign(side) * 0.5 * float(shoulder_width),
            head[2] - float(head_to_shoulder_z),
        ],
        dtype=np.float64,
    )


def robot_shoulder_position(side: str, lateral_offset: float) -> np.ndarray:
    return np.array(
        [0.0, side_sign(side) * float(lateral_offset), 0.0],
        dtype=np.float64,
    )


def normalized_reach_target(
    controller_position: np.ndarray,
    operator_shoulder: np.ndarray,
    operator_arm_length: float,
    robot_shoulder: np.ndarray,
    robot_arm_reach: float,
) -> ReachTarget:
    vector = np.asarray(controller_position, dtype=np.float64).reshape(3) - np.asarray(
        operator_shoulder,
        dtype=np.float64,
    ).reshape(3)
    distance = float(np.linalg.norm(vector))
    if distance < 1.0e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = vector / distance
    fraction = float(np.clip(distance / max(float(operator_arm_length), 1.0e-6), 0.0, 1.0))
    target = np.asarray(robot_shoulder, dtype=np.float64).reshape(3) + (
        direction * fraction * float(robot_arm_reach)
    )
    return ReachTarget(position=target, fraction=fraction)


def fixed_head_rotation(pitch_down_deg: float) -> np.ndarray:
    return rot_y(np.deg2rad(float(pitch_down_deg)))


def fixed_head_joint_positions(pitch_down_deg: float) -> np.ndarray:
    return np.array([np.deg2rad(float(pitch_down_deg)), 0.0, 0.0], dtype=np.float64)
