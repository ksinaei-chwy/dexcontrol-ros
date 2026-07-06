"""Pinocchio/Pink kinematics backend for Vega teleoperation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dex_pico_teleop.kinematics import (
    HEAD_JOINTS,
    IKSolution,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
)
from dex_pico_teleop.transforms import normalize_angle, rotation_error


class PinkUnavailableError(RuntimeError):
    """Raised when the Pinocchio/Pink backend cannot be initialized."""


class PinkChain:
    """A reduced Pink model containing only the joints for one task group."""

    def __init__(
        self,
        urdf_path: str,
        active_joint_names: tuple[str, ...],
        frame_name: str,
        solver: str = "quadprog",
        dt: float = 0.02,
    ) -> None:
        try:
            import pinocchio as pin
            import pink
            import qpsolvers
            from pink import tasks
            from pink.limits import ConfigurationLimit, VelocityLimit
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            raise PinkUnavailableError(str(exc)) from exc

        if solver not in qpsolvers.available_solvers:
            raise PinkUnavailableError(
                f"QP solver '{solver}' is not available; installed solvers: "
                f"{qpsolvers.available_solvers}"
            )

        self.pin = pin
        self.pink = pink
        self.FrameTask = tasks.FrameTask
        self.PostureTask = tasks.PostureTask
        self.ConfigurationLimit = ConfigurationLimit
        self.VelocityLimit = VelocityLimit
        self.solver = solver
        self.dt = float(dt)
        self.joint_names = active_joint_names
        self.frame_name = frame_name

        full_model = pin.buildModelFromUrdf(str(urdf_path))
        full_neutral = pin.neutral(full_model)
        active = set(active_joint_names)
        joints_to_lock = [
            full_model.getJointId(name)
            for name in full_model.names
            if name != "universe" and name not in active
        ]
        self.model = pin.buildReducedModel(full_model, joints_to_lock, full_neutral)
        self.data = self.model.createData()
        self.neutral = pin.neutral(self.model)
        self.limits = [ConfigurationLimit(self.model), VelocityLimit(self.model)]
        self._indices = {
            name: int(self.model.joints[self.model.getJointId(name)].idx_q)
            for name in active_joint_names
        }

    def q_from_values(self, values: np.ndarray) -> np.ndarray:
        q = self.neutral.copy()
        for index, name in enumerate(self.joint_names):
            q[self._indices[name]] = float(values[index])
        return q

    def values_from_q(self, q: np.ndarray) -> np.ndarray:
        return np.asarray([q[self._indices[name]] for name in self.joint_names], dtype=np.float64)

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_from_values(values)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.model.getFrameId(self.frame_name)]
        return np.asarray(transform.translation).copy(), np.asarray(transform.rotation).copy()

    def solve_pose(
        self,
        q_seed: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        position_cost,
        orientation_cost,
        max_iterations: int = 20,
        tolerance: float = 3.0e-3,
    ) -> IKSolution:
        q_seed_full = self.q_from_values(q_seed)
        configuration = self.pink.Configuration(self.model, self.data, q_seed_full)

        frame_task = self.FrameTask(
            self.frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0e-6,
            gain=1.0,
        )
        target = self.pin.SE3(
            np.asarray(target_rotation, dtype=np.float64).reshape(3, 3),
            np.asarray(target_position, dtype=np.float64).reshape(3),
        )
        frame_task.set_target(target)

        posture_task = self.PostureTask(cost=1.0e-4)
        posture_task.set_target(q_seed_full)
        all_tasks = [frame_task, posture_task]

        position_weights = _weights(position_cost)
        orientation_weights = _weights(orientation_cost)
        last_error = float("inf")
        for iteration in range(max_iterations):
            current_position, current_rotation = self._forward_q(configuration.q)
            last_error = _weighted_pose_error_norm(
                current_position,
                current_rotation,
                np.asarray(target_position, dtype=np.float64).reshape(3),
                np.asarray(target_rotation, dtype=np.float64).reshape(3, 3),
                position_weights,
                orientation_weights,
            )
            if last_error <= tolerance:
                return IKSolution(
                    q=self.values_from_q(configuration.q),
                    success=True,
                    error_norm=last_error,
                    iterations=iteration,
                )
            velocity = self.pink.solve_ik(
                configuration,
                all_tasks,
                dt=self.dt,
                solver=self.solver,
                limits=self.limits,
                damping=1.0e-8,
            )
            configuration.integrate_inplace(velocity, self.dt)

        return IKSolution(
            q=self.values_from_q(configuration.q),
            success=last_error < 5.0 * tolerance,
            error_norm=last_error,
            iterations=max_iterations,
        )

    def _forward_q(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.model.getFrameId(self.frame_name)]
        return np.asarray(transform.translation).copy(), np.asarray(transform.rotation).copy()


class PinkVegaKinematics:
    """Vega kinematics implementation backed by Pinocchio and Pink."""

    def __init__(
        self,
        urdf_path: str | Path,
        solver: str = "quadprog",
        dt: float = 0.02,
    ) -> None:
        path = str(urdf_path)
        self.torso = PinkChain(path, TORSO_JOINTS, "arm_center", solver=solver, dt=dt)
        self.head = PinkChain(path, HEAD_JOINTS, "head_l3", solver=solver, dt=dt)
        self.left_arm = PinkChain(path, LEFT_ARM_JOINTS, "L_ee", solver=solver, dt=dt)
        self.right_arm = PinkChain(path, RIGHT_ARM_JOINTS, "R_ee", solver=solver, dt=dt)

    def arm_center_pose(self, torso_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.torso.forward(torso_q)

    def solve_torso_height(
        self,
        q_seed: np.ndarray,
        target_z: float,
        target_pitch: float = 0.0,
        target_x: float | None = None,
        max_iterations: int = 30,
    ) -> IKSolution:
        pos, rot = self.torso.forward(q_seed)
        if target_x is None:
            target_x = float(pos[0])
        target_pos = np.array([target_x, pos[1], target_z], dtype=np.float64)
        target_rot = _pitch_only_rotation(target_pitch)
        return self.torso.solve_pose(
            q_seed,
            target_pos,
            target_rot,
            position_cost=[0.25, 0.0, 1.0],
            orientation_cost=[0.0, 1.0, 0.0],
            max_iterations=max_iterations,
            tolerance=4.0e-3,
        )

    def solve_head_orientation(
        self,
        q_seed: np.ndarray,
        target_rotation: np.ndarray,
        max_iterations: int = 30,
    ) -> IKSolution:
        pos, _rot = self.head.forward(q_seed)
        return self.head.solve_pose(
            q_seed,
            pos,
            target_rotation,
            position_cost=0.0,
            orientation_cost=1.0,
            max_iterations=max_iterations,
            tolerance=4.0e-3,
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
            target_position,
            target_rotation,
            position_cost=1.0,
            orientation_cost=0.55,
            max_iterations=30,
            tolerance=6.0e-3,
        )


def _pitch_only_rotation(pitch: float) -> np.ndarray:
    c = np.cos(normalize_angle(float(pitch)))
    s = np.sin(normalize_angle(float(pitch)))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _weights(cost) -> np.ndarray:
    if isinstance(cost, (int, float)):
        return np.ones(3, dtype=np.float64) * float(cost)
    return np.asarray(cost, dtype=np.float64).reshape(3)


def _weighted_pose_error_norm(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    position_weights: np.ndarray,
    orientation_weights: np.ndarray,
) -> float:
    pos_error = (target_position - current_position) * position_weights
    rot_error = rotation_error(current_rotation, target_rotation) * orientation_weights
    return float(np.linalg.norm(np.concatenate((pos_error, rot_error))))
