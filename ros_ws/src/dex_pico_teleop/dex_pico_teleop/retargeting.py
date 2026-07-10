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


@dataclass(frozen=True)
class ShoulderReachCalibration:
    shoulder_offset: np.ndarray
    arm_length: float


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


def operator_shoulder_from_offset(
    head_position: np.ndarray,
    shoulder_offset: np.ndarray,
) -> np.ndarray:
    return np.asarray(head_position, dtype=np.float64).reshape(3) + np.asarray(
        shoulder_offset,
        dtype=np.float64,
    ).reshape(3)


def controller_hand_point(
    controller_position: np.ndarray,
    controller_rotation: np.ndarray,
    controller_to_hand_point: np.ndarray,
) -> np.ndarray:
    position = np.asarray(controller_position, dtype=np.float64).reshape(3)
    rotation = np.asarray(controller_rotation, dtype=np.float64).reshape(3, 3)
    offset = np.asarray(controller_to_hand_point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(offset)):
        raise ValueError("controller-to-hand-point offset must be finite")
    return position + rotation @ offset


def vector_in_arm_center(
    operator_vector: np.ndarray,
    arm_center_rotation: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(operator_vector, dtype=np.float64).reshape(3)
    rotation = np.asarray(arm_center_rotation, dtype=np.float64).reshape(3, 3)
    return rotation.T @ vector


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
    return normalized_reach_target_from_vector(
        vector,
        operator_arm_length,
        robot_shoulder,
        robot_arm_reach,
    )


def normalized_reach_target_from_vector(
    shoulder_to_hand_vector: np.ndarray,
    operator_arm_length: float,
    robot_shoulder: np.ndarray,
    robot_arm_reach: float,
) -> ReachTarget:
    vector = np.asarray(shoulder_to_hand_vector, dtype=np.float64).reshape(3)
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


def fit_two_pose_shoulder_reach(
    side: str,
    neutral_hand_relative_to_head: np.ndarray,
    reach_hand_relative_to_head: np.ndarray,
    shoulder_width: float,
    head_to_shoulder_z: float,
    arm_length_min: float,
    arm_length_max: float,
    minimum_forward_separation: float = 0.15,
    shoulder_x_min: float = -0.30,
    shoulder_x_max: float = 0.10,
) -> ShoulderReachCalibration:
    neutral = np.asarray(neutral_hand_relative_to_head, dtype=np.float64).reshape(3)
    reach = np.asarray(reach_hand_relative_to_head, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(neutral)) or not np.all(np.isfinite(reach)):
        raise ValueError(f"{side} calibration points must be finite")

    delta = reach - neutral
    if float(delta[0]) < float(minimum_forward_separation):
        raise ValueError(
            f"{side} hand must move at least {minimum_forward_separation:.2f} m forward"
        )

    shoulder_y = side_sign(side) * 0.5 * float(shoulder_width)
    shoulder_z = -float(head_to_shoulder_z)
    numerator = (
        float(np.dot(reach, reach) - np.dot(neutral, neutral))
        - 2.0 * float(delta[1]) * shoulder_y
        - 2.0 * float(delta[2]) * shoulder_z
    )
    shoulder_x = numerator / (2.0 * float(delta[0]))
    if not float(shoulder_x_min) <= shoulder_x <= float(shoulder_x_max):
        raise ValueError(
            f"{side} fitted shoulder x={shoulder_x:.3f} m is outside "
            f"[{shoulder_x_min:.2f}, {shoulder_x_max:.2f}] m"
        )

    shoulder = np.array([shoulder_x, shoulder_y, shoulder_z], dtype=np.float64)
    neutral_length = float(np.linalg.norm(neutral - shoulder))
    reach_length = float(np.linalg.norm(reach - shoulder))
    arm_length = 0.5 * (neutral_length + reach_length)
    if not float(arm_length_min) <= arm_length <= float(arm_length_max):
        raise ValueError(
            f"{side} fitted arm length={arm_length:.3f} m is outside "
            f"[{arm_length_min:.2f}, {arm_length_max:.2f}] m"
        )
    return ShoulderReachCalibration(shoulder_offset=shoulder, arm_length=arm_length)


def fixed_head_rotation(pitch_down_deg: float) -> np.ndarray:
    return rot_y(np.deg2rad(float(pitch_down_deg)))


def fixed_head_joint_positions(pitch_down_deg: float) -> np.ndarray:
    return np.array([np.deg2rad(float(pitch_down_deg)), 0.0, 0.0], dtype=np.float64)
