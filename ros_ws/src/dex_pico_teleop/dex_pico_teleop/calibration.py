"""Calibration state for posture-based Pico teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dex_pico_teleop.retargeting import controller_hand_point
from dex_pico_teleop.transforms import Pose, matrix_to_quat, rot_z, yaw_from_rotation
from dex_pico_teleop.xr_packet import ControllerInput, PicoPacket


@dataclass(frozen=True)
class CalibrationSampleStats:
    sample_count: int
    controller_position_dispersion_m: dict[str, float]


def average_calibration_packets(
    packets: list[PicoPacket] | tuple[PicoPacket, ...],
) -> tuple[PicoPacket, CalibrationSampleStats]:
    if not packets:
        raise ValueError("no calibration packets provided")
    frames = {packet.frame for packet in packets}
    if len(frames) != 1:
        raise ValueError("calibration packets use inconsistent coordinate frames")

    latest = packets[-1]
    head_pose = _average_pose([packet.head for packet in packets])
    controllers: dict[str, ControllerInput] = {}
    dispersion: dict[str, float] = {}
    for side in ("left", "right"):
        poses = [packet.controllers[side].pose for packet in packets]
        averaged_pose = _average_pose(poses)
        positions = np.asarray([pose.position for pose in poses], dtype=np.float64)
        offsets = positions - averaged_pose.position
        dispersion[side] = float(np.sqrt(np.mean(np.sum(offsets * offsets, axis=1))))
        latest_controller = latest.controllers[side]
        controllers[side] = ControllerInput(
            pose=averaged_pose,
            trigger=latest_controller.trigger,
            grip=latest_controller.grip,
            joystick=latest_controller.joystick.copy(),
            buttons=latest_controller.buttons.copy(),
        )

    averaged = PicoPacket(
        timestamp_ns=latest.timestamp_ns,
        frame=latest.frame,
        head=head_pose,
        controllers=controllers,
        trackers=latest.trackers,
        sequence=latest.sequence,
    )
    return averaged, CalibrationSampleStats(
        sample_count=len(packets),
        controller_position_dispersion_m=dispersion,
    )


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
    arm_controller_to_ee_rotation: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "left": np.eye(3, dtype=np.float64),
            "right": np.eye(3, dtype=np.float64),
        }
    )
    operator_arm_lengths: dict[str, float] = field(default_factory=dict)
    operator_shoulder_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    neutral_hand_relative_to_head: dict[str, np.ndarray] = field(default_factory=dict)

    def calibrate(
        self,
        packet: PicoPacket,
        arm_center_position: np.ndarray,
        arm_center_rotation: np.ndarray,
        head_chain_rotation: np.ndarray,
        arm_end_effector_rotations: dict[str, np.ndarray],
        controller_to_hand_offsets: dict[str, np.ndarray] | None = None,
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
        self.arm_controller_to_ee_rotation = {}
        self.neutral_hand_relative_to_head = {}
        arm_center_from_operator = np.asarray(
            arm_center_rotation,
            dtype=np.float64,
        ).reshape(3, 3).T
        offsets = controller_to_hand_offsets or {}
        for side, ee_rotation in arm_end_effector_rotations.items():
            controller_pose = self.to_operator_pose(packet.controllers[side].pose)
            controller_rotation_arm_center = arm_center_from_operator @ controller_pose.rotation
            self.arm_controller_to_ee_rotation[side] = (
                controller_rotation_arm_center.T
                @ np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3)
            )
            hand_point = controller_hand_point(
                controller_pose.position,
                controller_pose.rotation,
                offsets.get(side, np.zeros(3, dtype=np.float64)),
            )
            self.neutral_hand_relative_to_head[side] = (
                hand_point - neutral_head_pose.position
            )
        self.calibrated = True

    def set_operator_arm_lengths(self, lengths: dict[str, float]) -> None:
        self.operator_arm_lengths = {
            side: float(value)
            for side, value in lengths.items()
            if side in {"left", "right"} and np.isfinite(value)
        }

    def set_operator_reach_calibration(
        self,
        shoulder_offsets: dict[str, np.ndarray],
        arm_lengths: dict[str, float],
    ) -> None:
        expected = {"left", "right"}
        if set(shoulder_offsets) != expected or set(arm_lengths) != expected:
            raise ValueError("reach calibration requires both left and right arms")
        candidate_offsets = {
            side: np.asarray(shoulder_offsets[side], dtype=np.float64).reshape(3).copy()
            for side in expected
        }
        candidate_lengths = {side: float(arm_lengths[side]) for side in expected}
        if not all(np.all(np.isfinite(value)) for value in candidate_offsets.values()):
            raise ValueError("shoulder calibration contains non-finite values")
        if not all(np.isfinite(value) for value in candidate_lengths.values()):
            raise ValueError("arm-length calibration contains non-finite values")
        self.operator_shoulder_offsets = candidate_offsets
        self.operator_arm_lengths = candidate_lengths

    def to_operator_pose(self, pose: Pose) -> Pose:
        yaw_inv = self.robot_from_operator_yaw.T
        position = yaw_inv @ (pose.position - self.operator_origin)
        rotation = yaw_inv @ pose.rotation
        return Pose(position=position, orientation=_matrix_to_pose_quat(rotation))

    def arm_target_rotation(
        self,
        side: str,
        controller_pose: Pose,
        arm_center_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        offset = self.arm_controller_to_ee_rotation.get(side)
        if offset is None:
            offset = np.eye(3, dtype=np.float64)
        if arm_center_rotation is None:
            arm_center_from_operator = np.eye(3, dtype=np.float64)
        else:
            arm_center_from_operator = np.asarray(
                arm_center_rotation,
                dtype=np.float64,
            ).reshape(3, 3).T
        return arm_center_from_operator @ controller_pose.rotation @ offset

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
    return matrix_to_quat(rotation)


def _average_pose(poses: list[Pose]) -> Pose:
    if not poses:
        raise ValueError("cannot average an empty pose list")
    positions = np.asarray([pose.position for pose in poses], dtype=np.float64)
    position = np.median(positions, axis=0)
    rotation_sum = np.sum([pose.rotation for pose in poses], axis=0)
    u, _singular_values, vh = np.linalg.svd(rotation_sum)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    return Pose(position=position, orientation=matrix_to_quat(rotation))
