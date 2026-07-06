"""Lightweight Vega kinematics for dry-run teleop and Pink-compatible task logic.

This module deliberately keeps the task boundaries explicit: torso height,
head orientation, and left/right arm end-effector pose are solved separately.
That mirrors the planned Pinocchio/Pink integration while staying runnable in
the current development container where those packages are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dex_pico_teleop.transforms import (
    axis_angle_to_matrix,
    normalize_angle,
    rotation_error,
    rpy_to_matrix,
)


TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")
LEFT_ARM_JOINTS = tuple(f"L_arm_j{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"R_arm_j{i}" for i in range(1, 8))


@dataclass(frozen=True)
class JointSpec:
    name: str
    origin: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    limit: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self.limit is not None


@dataclass(frozen=True)
class IKSolution:
    q: np.ndarray
    success: bool
    error_norm: float
    iterations: int


class KinematicChain:
    def __init__(self, specs: list[JointSpec]) -> None:
        self.specs = specs
        self.joint_names = tuple(spec.name for spec in specs if spec.active)
        self.limits = np.asarray([spec.limit for spec in specs if spec.active], dtype=np.float64)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64).reshape(len(self.joint_names))
        return np.clip(values, self.limits[:, 0], self.limits[:, 1])

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.clamp(q)
        active_index = 0
        position = np.zeros(3, dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        for spec in self.specs:
            position = position + rotation @ spec.origin
            rotation = rotation @ rpy_to_matrix(spec.rpy)
            if spec.active:
                rotation = rotation @ axis_angle_to_matrix(spec.axis, float(values[active_index]))
                active_index += 1
        return position, rotation

    def solve_pose(
        self,
        q_seed: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        position_weight: float = 1.0,
        rotation_weight: float = 0.7,
        damping: float = 1.0e-3,
        max_step: float = 0.08,
        max_iterations: int = 40,
        tolerance: float = 2.0e-3,
    ) -> IKSolution:
        q = self.clamp(q_seed)
        target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)

        def error(values: np.ndarray) -> np.ndarray:
            pos, rot = self.forward(values)
            return np.concatenate(
                (
                    (target_position - pos) * position_weight,
                    rotation_error(rot, target_rotation) * rotation_weight,
                )
            )

        return _solve_error(q, self.clamp, error, damping, max_step, max_iterations, tolerance)


class VegaKinematics:
    """Numerical kinematics model for the Vega 1 Pro upper body."""

    def __init__(self) -> None:
        self.torso = KinematicChain(_torso_specs())
        self.head = KinematicChain(_head_specs())
        self.left_arm = KinematicChain(_left_arm_specs())
        self.right_arm = KinematicChain(_right_arm_specs())

    def arm_center_pose(self, torso_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.torso.forward(torso_q)

    def solve_torso_height(
        self,
        q_seed: np.ndarray,
        target_z: float,
        target_pitch: float = 0.0,
        target_x: float | None = None,
        max_iterations: int = 60,
    ) -> IKSolution:
        q = self.torso.clamp(q_seed)
        if target_x is None:
            target_x = float(self.torso.forward(q)[0][0])

        def error(values: np.ndarray) -> np.ndarray:
            pos, rot = self.torso.forward(values)
            pitch = np.arctan2(rot[0, 2], rot[0, 0])
            return np.array(
                [
                    float(target_z) - pos[2],
                    normalize_angle(float(target_pitch) - float(pitch)),
                    0.25 * (float(target_x) - pos[0]),
                ],
                dtype=np.float64,
            )

        return _solve_error(
            q,
            self.torso.clamp,
            error,
            damping=5.0e-3,
            max_step=0.04,
            max_iterations=max_iterations,
            tolerance=2.0e-3,
        )

    def solve_head_orientation(
        self,
        q_seed: np.ndarray,
        target_rotation: np.ndarray,
        max_iterations: int = 50,
    ) -> IKSolution:
        _, current_pos = np.zeros(3), None
        pos, _rot = self.head.forward(q_seed)
        current_pos = pos
        return self.head.solve_pose(
            q_seed,
            target_position=current_pos,
            target_rotation=target_rotation,
            position_weight=0.0,
            rotation_weight=1.0,
            damping=2.0e-3,
            max_step=0.06,
            max_iterations=max_iterations,
            tolerance=2.0e-3,
        )

    def solve_arm_pose(
        self,
        side: str,
        q_seed: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> IKSolution:
        chain = self.left_arm if side == "left" else self.right_arm
        return chain.solve_pose(
            q_seed,
            target_position=target_position,
            target_rotation=target_rotation,
            position_weight=1.0,
            rotation_weight=0.55,
            damping=5.0e-3,
            max_step=0.07,
            max_iterations=50,
            tolerance=4.0e-3,
        )


def _solve_error(
    q_seed: np.ndarray,
    clamp_fn,
    error_fn,
    damping: float,
    max_step: float,
    max_iterations: int,
    tolerance: float,
) -> IKSolution:
    q = clamp_fn(q_seed)
    last_norm = float("inf")
    for iteration in range(max_iterations):
        err = error_fn(q)
        err_norm = float(np.linalg.norm(err))
        last_norm = err_norm
        if err_norm <= tolerance:
            return IKSolution(q=q, success=True, error_norm=err_norm, iterations=iteration)

        jac = _finite_difference_jacobian(q, error_fn)
        lhs = jac @ jac.T + (damping * damping) * np.eye(jac.shape[0])
        try:
            step = -jac.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(jac) @ err
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step = step * (max_step / step_norm)
        q = clamp_fn(q + step)
    return IKSolution(q=q, success=last_norm <= tolerance * 2.0, error_norm=last_norm, iterations=max_iterations)


def _finite_difference_jacobian(q: np.ndarray, error_fn, eps: float = 1.0e-5) -> np.ndarray:
    base = error_fn(q)
    jac = np.zeros((base.size, q.size), dtype=np.float64)
    for index in range(q.size):
        perturbed = q.copy()
        perturbed[index] += eps
        jac[:, index] = (error_fn(perturbed) - base) / eps
    return jac


def _spec(name: str, xyz, axis, limit, rpy=(0.0, 0.0, 0.0)) -> JointSpec:
    return JointSpec(
        name=name,
        origin=np.asarray(xyz, dtype=np.float64),
        rpy=np.asarray(rpy, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
        limit=limit,
    )


def _fixed(name: str, xyz, rpy=(0.0, 0.0, 0.0)) -> JointSpec:
    return JointSpec(
        name=name,
        origin=np.asarray(xyz, dtype=np.float64),
        rpy=np.asarray(rpy, dtype=np.float64),
        axis=np.zeros(3, dtype=np.float64),
        limit=None,
    )


def _torso_specs() -> list[JointSpec]:
    return [
        _spec("torso_j1", [-0.235, 0.0, 0.248], [0.0, -1.0, 0.0], (0.0, 1.570)),
        _spec("torso_j2", [0.396, 0.0, 0.082], [0.0, 1.0, 0.0], (0.0, 3.141)),
        _spec("torso_j3", [-0.40718, 0.0, 0.09764], [0.0, -1.0, 0.0], (-1.570, 1.570)),
        _fixed("arm_center_j0", [-0.05908, 0.0, 0.44528]),
    ]


def _head_specs() -> list[JointSpec]:
    return [
        _spec("head_j1", [-0.0735, -0.0725, 0.014], [0.0, 1.0, 0.0], (-1.483, 1.483)),
        _spec("head_j2", [0.0, 0.0725, -0.0035], [0.0, 0.0, 1.0], (-2.792, 2.792)),
        _spec("head_j3", [0.0, 0.002, 0.0495], [0.0, -1.0, 0.0], (-1.378, 1.483)),
    ]


def _left_arm_specs() -> list[JointSpec]:
    return [
        _spec("L_arm_j1", [0.0, 0.16946, 0.0], [0.0, 1.0, 0.0], (-3.071, 3.071)),
        _spec("L_arm_j2", [0.04, 0.06, 0.0454], [0.0, 0.0, 1.0], (-0.453, 1.553)),
        _spec("L_arm_j3", [0.1644, 0.0, -0.043], [1.0, 0.0, 0.0], (-3.071, 3.071)),
        _spec("L_arm_j4", [0.113, 0.0433, 0.06], [0.0, 1.0, 0.0], (-3.071, 0.244)),
        _spec("L_arm_j5", [0.1938, -0.0434, -0.04], [1.0, 0.0, 0.0], (-3.071, 3.071)),
        _spec("L_arm_j6", [0.0762, 0.0319, 0.0], [0.0, 1.0, 0.0], (-1.396, 1.396)),
        _spec("L_arm_j7", [0.065, -0.032, 0.0319], [0.0, 0.0, 1.0], (-1.378, 1.117)),
        _fixed("L_arm_j8", [0.11597, 0.0, -0.032], (0.0, 1.57079, 0.0)),
        _fixed("L_ee_j0", [0.0, 0.0, 0.0], (0.0, 0.0, -1.57079)),
    ]


def _right_arm_specs() -> list[JointSpec]:
    return [
        _spec("R_arm_j1", [0.0, -0.16946, 0.0], [0.0, -1.0, 0.0], (-3.071, 3.071)),
        _spec("R_arm_j2", [0.04, -0.06, 0.0454], [0.0, 0.0, 1.0], (-1.553, 0.453)),
        _spec("R_arm_j3", [0.1644, 0.0, -0.043], [1.0, 0.0, 0.0], (-3.071, 3.071)),
        _spec("R_arm_j4", [0.113, 0.0433, 0.06], [0.0, 1.0, 0.0], (-3.071, 0.244)),
        _spec("R_arm_j5", [0.1938, -0.0434, -0.04], [1.0, 0.0, 0.0], (-3.071, 3.071)),
        _spec("R_arm_j6", [0.0762, -0.0319, 0.0], [0.0, -1.0, 0.0], (-1.396, 1.396)),
        _spec("R_arm_j7", [0.065, 0.032, 0.0319], [0.0, 0.0, 1.0], (-1.117, 1.378)),
        _fixed("R_arm_j8", [0.11597, 0.0, -0.032], (0.0, 1.57079, 0.0)),
        _fixed("R_ee_j0", [0.0, 0.0, 0.0], (0.0, 0.0, 1.57079)),
    ]

