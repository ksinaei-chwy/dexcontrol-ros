"""Small transform helpers for Pico-to-Vega teleoperation.

The code keeps quaternion order as xyzw at module boundaries because that is
what OpenXR, XRoboToolkit, and ROS geometry messages commonly expose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


OPENXR_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Pose:
    """A rigid pose represented as position plus xyzw quaternion."""

    position: np.ndarray
    orientation: np.ndarray

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, ...] | np.ndarray) -> "Pose":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != 7:
            raise ValueError(f"pose must contain 7 values, got {array.size}")
        if not np.all(np.isfinite(array)):
            raise ValueError("pose contains non-finite values")
        return cls(position=array[:3].copy(), orientation=normalize_quat(array[3:7]))

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0]))

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.orientation)

    @property
    def matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.position
        return mat

    def transformed(self, rotation: np.ndarray, translation: np.ndarray | None = None) -> "Pose":
        offset = np.zeros(3, dtype=np.float64) if translation is None else translation
        new_rot = rotation @ self.rotation @ rotation.T
        return Pose(rotation @ self.position + offset, matrix_to_quat(new_rot))


def normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid zero-length quaternion")
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return q


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([x, y, z, w], dtype=np.float64))


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_matrix(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    return rot_z(float(yaw)) @ rot_y(float(pitch)) @ rot_x(float(roll))


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vec / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1.0e-9:
        return 0.5 * np.array(
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            dtype=np.float64,
        )
    denom = 2.0 * math.sin(angle)
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    ) / denom
    return axis * angle


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return rotvec that rotates current orientation toward target."""
    return matrix_to_rotvec(np.asarray(target) @ np.asarray(current).T)


def yaw_from_rotation(rotation: np.ndarray) -> float:
    forward = np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 0]
    return math.atan2(float(forward[1]), float(forward[0]))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_openxr_to_robot(pose: Pose) -> Pose:
    return pose.transformed(OPENXR_TO_ROBOT)

