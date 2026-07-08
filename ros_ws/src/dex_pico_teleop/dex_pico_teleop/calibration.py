"""Calibration state for posture-based Pico teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dex_pico_teleop.transforms import Pose, rot_z, yaw_from_rotation
from dex_pico_teleop.xr_packet import PicoPacket


@dataclass
class CalibrationState:
    calibrated: bool = False
    operator_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    robot_from_operator_yaw: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    neutral_height_signal: float = 1.7
    neutral_arm_center_z: float = 0.873
    neutral_arm_center_pitch: float = 0.0
    neutral_arm_center_x: float = -0.305
    head_controller_to_chain_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    hand_open: dict[str, np.ndarray] = field(default_factory=dict)
    arm_controller_to_ee_rotation: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "left": np.eye(3, dtype=np.float64),
            "right": np.eye(3, dtype=np.float64),
        }
    )
    operator_arm_lengths: dict[str, float] = field(default_factory=dict)

    def calibrate(
        self,
        packet: PicoPacket,
        arm_center_position: np.ndarray,
        arm_center_rotation: np.ndarray,
        head_chain_rotation: np.ndarray,
        hand_positions: dict[str, np.ndarray],
        arm_end_effector_rotations: dict[str, np.ndarray],
    ) -> None:
        head = packet.head
        yaw = yaw_from_rotation(head.rotation)
        self.robot_from_operator_yaw = rot_z(yaw)
        self.operator_origin = head.position.copy()
        self.operator_origin[2] = 0.0
        self.neutral_height_signal = height_signal(packet)
        self.neutral_arm_center_z = float(arm_center_position[2])
        self.neutral_arm_center_x = float(arm_center_position[0])
        self.neutral_arm_center_pitch = float(
            np.arctan2(arm_center_rotation[0, 2], arm_center_rotation[0, 0])
        )
        neutral_head_pose = self.to_operator_pose(head)
        self.head_controller_to_chain_rotation = (
            neutral_head_pose.rotation.T
            @ np.asarray(head_chain_rotation, dtype=np.float64).reshape(3, 3)
        )
        self.hand_open = {side: values.copy() for side, values in hand_positions.items()}
        self.arm_controller_to_ee_rotation = {}
        for side, ee_rotation in arm_end_effector_rotations.items():
            controller_pose = self.to_operator_pose(packet.controllers[side].pose)
            self.arm_controller_to_ee_rotation[side] = (
                controller_pose.rotation.T @ np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3)
            )
        self.operator_arm_lengths = {}
        self.calibrated = True

    def set_operator_arm_lengths(self, lengths: dict[str, float]) -> None:
        self.operator_arm_lengths = {
            side: float(value)
            for side, value in lengths.items()
            if side in {"left", "right"} and np.isfinite(value)
        }

    def to_operator_pose(self, pose: Pose) -> Pose:
        yaw_inv = self.robot_from_operator_yaw.T
        position = yaw_inv @ (pose.position - self.operator_origin)
        rotation = yaw_inv @ pose.rotation
        return Pose(position=position, orientation=_matrix_to_pose_quat(rotation))

    def arm_target_rotation(self, side: str, controller_pose: Pose) -> np.ndarray:
        offset = self.arm_controller_to_ee_rotation.get(side)
        if offset is None:
            offset = np.eye(3, dtype=np.float64)
        return controller_pose.rotation @ offset

    def head_target_rotation(self, head_pose: Pose) -> np.ndarray:
        return head_pose.rotation @ self.head_controller_to_chain_rotation


def height_signal(packet: PicoPacket) -> float:
    left = packet.trackers.get("left_ankle")
    right = packet.trackers.get("right_ankle")
    if left is not None and right is not None and min(left.confidence, right.confidence) > 0.2:
        ankle_z = 0.5 * (left.pose.position[2] + right.pose.position[2])
        return float(packet.head.position[2] - ankle_z)
    return float(packet.head.position[2])


def _matrix_to_pose_quat(rotation: np.ndarray) -> np.ndarray:
    from dex_pico_teleop.transforms import matrix_to_quat

    return matrix_to_quat(rotation)
