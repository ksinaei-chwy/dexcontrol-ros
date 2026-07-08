"""Pinocchio/Pink kinematics backend for Vega teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dex_pico_teleop.kinematics import (
    IKSolution,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    ROBOT_SHOULDER_LATERAL_OFFSET_M,
    VegaKinematics,
)
from dex_pico_teleop.transforms import rotation_error


class PinkUnavailableError(RuntimeError):
    """Raised when the Pinocchio/Pink backend cannot be initialized."""


@dataclass(frozen=True)
class SelfCollisionOptions:
    """Pink self-collision barrier settings for a reduced chain."""

    srdf_path: Path
    package_dirs: tuple[Path, ...]
    collision_urdf_path: Path | None = None
    n_collision_pairs: int = 24
    gain: float = 1.0
    safe_displacement_gain: float = 0.0
    d_min: float = 0.04


class PinkChain:
    """A reduced Pink model containing only the joints for one task group."""

    def __init__(
        self,
        urdf_path: str,
        active_joint_names: tuple[str, ...],
        frame_name: str,
        solver: str = "quadprog",
        dt: float = 0.02,
        self_collision: SelfCollisionOptions | None = None,
        root_frame_name: str | None = None,
        velocity_limit_enabled: bool = False,
        task_gain: float = 1.0,
        lm_damping: float = 1.0e-6,
        solve_damping: float = 1.0e-8,
    ) -> None:
        try:
            import pinocchio as pin
            import pink
            import qpsolvers
            from pink.barriers import SelfCollisionBarrier
            from pink import tasks
            from pink.limits import ConfigurationLimit, VelocityLimit
            from pink.exceptions import NoSolutionFound
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            raise PinkUnavailableError(str(exc)) from exc

        if solver not in qpsolvers.available_solvers:
            raise PinkUnavailableError(
                f"QP solver '{solver}' is not available; installed solvers: "
                f"{qpsolvers.available_solvers}"
            )

        self.pin = pin
        self.pink = pink
        self.SelfCollisionBarrier = SelfCollisionBarrier
        self.FrameTask = tasks.FrameTask
        self.PostureTask = tasks.PostureTask
        self.ConfigurationLimit = ConfigurationLimit
        self.VelocityLimit = VelocityLimit
        self.NoSolutionFound = NoSolutionFound
        self.solver = solver
        self.dt = float(dt)
        self.joint_names = active_joint_names
        self.frame_name = frame_name
        self.root_frame_name = root_frame_name
        self.task_gain = float(task_gain)
        self.lm_damping = float(lm_damping)
        self.solve_damping = float(solve_damping)

        if self_collision is None:
            full_model = pin.buildModelFromUrdf(str(urdf_path))
            full_collision_model = None
        else:
            full_model = pin.buildModelFromUrdf(str(urdf_path))
            collision_urdf_path = self_collision.collision_urdf_path or Path(urdf_path)
            full_collision_model = pin.buildGeomFromUrdf(
                full_model,
                str(collision_urdf_path),
                pin.GeometryType.COLLISION,
                None,
                [str(path) for path in self_collision.package_dirs],
            )

        full_neutral = pin.neutral(full_model)
        active = set(active_joint_names)
        joints_to_lock = [
            full_model.getJointId(name)
            for name in full_model.names
            if name != "universe" and name not in active
        ]
        if full_collision_model is None:
            self.model = pin.buildReducedModel(full_model, joints_to_lock, full_neutral)
            self.collision_model = None
        else:
            self.model, self.collision_model = pin.buildReducedModel(
                full_model,
                full_collision_model,
                joints_to_lock,
                full_neutral,
            )
            self.collision_model.addAllCollisionPairs()
            pin.removeCollisionPairs(
                self.model,
                self.collision_model,
                str(self_collision.srdf_path),
                False,
            )
            self._remove_uncontrollable_collision_pairs(active_joint_names)

        self.data = self.model.createData()
        self.neutral = pin.neutral(self.model)
        self.limits = [ConfigurationLimit(self.model)]
        if velocity_limit_enabled:
            self.limits.append(VelocityLimit(self.model))
        self.barriers = self._make_barriers(self_collision)
        self.frame_id = self.model.getFrameId(self.frame_name)
        self.root_frame_id = (
            None if self.root_frame_name is None else self.model.getFrameId(self.root_frame_name)
        )
        self._indices = {
            name: int(self.model.joints[self.model.getJointId(name)].idx_q)
            for name in active_joint_names
        }
        self._lower_position_limits = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self._upper_position_limits = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        self.collision_pair_count = (
            0 if self.collision_model is None else len(self.collision_model.collisionPairs)
        )
        self.barrier_pair_count = sum(barrier.dim for barrier in self.barriers)

    def q_from_values(self, values: np.ndarray) -> np.ndarray:
        q = self.neutral.copy()
        joint_values = np.asarray(values, dtype=np.float64).reshape(len(self.joint_names))
        for index, name in enumerate(self.joint_names):
            q[self._indices[name]] = float(joint_values[index])
        return self._clip_full_q(q)

    def values_from_q(self, q: np.ndarray) -> np.ndarray:
        clipped_q = self._clip_full_q(q)
        return np.asarray(
            [clipped_q[self._indices[name]] for name in self.joint_names],
            dtype=np.float64,
        )

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_from_values(values)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self._frame_transform(self.data)
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
        collision_data = (
            self.pin.GeometryData(self.collision_model)
            if self.collision_model is not None
            else None
        )
        configuration = self.pink.Configuration(
            self.model,
            self.data,
            q_seed_full,
            collision_model=self.collision_model,
            collision_data=collision_data,
        )

        frame_task = self.FrameTask(
            self.frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=self.lm_damping,
            gain=self.task_gain,
        )
        target = self._target_transform(
            q_seed_full,
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
            try:
                velocity = self.pink.solve_ik(
                    configuration,
                    all_tasks,
                    dt=self.dt,
                    solver=self.solver,
                    limits=self.limits,
                    barriers=self.barriers,
                    damping=self.solve_damping,
                )
            except self.NoSolutionFound:
                return IKSolution(
                    q=self.values_from_q(configuration.q),
                    success=False,
                    error_norm=last_error,
                    iterations=iteration,
                )
            configuration.integrate_inplace(velocity, self.dt)
            configuration.update(self._clip_full_q(configuration.q))

        return IKSolution(
            q=self.values_from_q(configuration.q),
            success=last_error < 5.0 * tolerance,
            error_norm=last_error,
            iterations=max_iterations,
        )

    def _clip_full_q(self, q: np.ndarray) -> np.ndarray:
        clipped = np.asarray(q, dtype=np.float64).copy()
        limited = self._upper_position_limits > self._lower_position_limits
        clipped[limited] = np.clip(
            clipped[limited],
            self._lower_position_limits[limited],
            self._upper_position_limits[limited],
        )
        return clipped

    def _forward_q(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self._frame_transform(self.data)
        return np.asarray(transform.translation).copy(), np.asarray(transform.rotation).copy()

    def _frame_transform(self, data) -> object:
        transform = data.oMf[self.frame_id]
        if self.root_frame_id is None:
            return transform
        return data.oMf[self.root_frame_id].inverse() * transform

    def _target_transform(
        self,
        q_seed: np.ndarray,
        target_rotation: np.ndarray,
        target_position: np.ndarray,
    ) -> object:
        target = self.pin.SE3(target_rotation, target_position)
        if self.root_frame_id is None:
            return target
        self.pin.forwardKinematics(self.model, self.data, q_seed)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.root_frame_id] * target

    def _make_barriers(self, self_collision: SelfCollisionOptions | None) -> list[object]:
        if self_collision is None or self.collision_model is None:
            return []
        n_pairs = min(
            int(self_collision.n_collision_pairs),
            len(self.collision_model.collisionPairs),
        )
        if n_pairs <= 0:
            return []
        return [
            self.SelfCollisionBarrier(
                n_collision_pairs=n_pairs,
                gain=float(self_collision.gain),
                safe_displacement_gain=float(self_collision.safe_displacement_gain),
                d_min=float(self_collision.d_min),
            )
        ]

    def _remove_uncontrollable_collision_pairs(
        self,
        active_joint_names: tuple[str, ...],
    ) -> None:
        if self.collision_model is None:
            return
        active_joint_ids = {
            self.model.getJointId(name)
            for name in active_joint_names
            if self.model.existJointName(name)
        }
        for pair in list(self.collision_model.collisionPairs):
            geometry_1 = self.collision_model.geometryObjects[pair.first]
            geometry_2 = self.collision_model.geometryObjects[pair.second]
            active_related = (
                geometry_1.parentJoint in active_joint_ids
                or geometry_2.parentJoint in active_joint_ids
            )
            same_parent = geometry_1.parentJoint == geometry_2.parentJoint
            if not active_related or same_parent:
                self.collision_model.removeCollisionPair(pair)


class PinkVegaKinematics:
    """Vega kinematics implementation backed by Pinocchio and Pink."""

    def __init__(
        self,
        urdf_path: str | Path,
        solver: str = "quadprog",
        dt: float = 0.02,
        self_collision_components: tuple[str, ...] = (),
        self_collision_srdf_path: str | Path | None = None,
        self_collision_urdf_path: str | Path | None = None,
        collision_package_dirs: tuple[str | Path, ...] = (),
        self_collision_n_pairs: int = 24,
        self_collision_gain: float = 1.0,
        self_collision_safe_displacement_gain: float = 0.0,
        self_collision_d_min: float = 0.04,
        velocity_limit_enabled: bool = False,
        task_gain: float = 1.0,
        lm_damping: float = 1.0e-6,
        solve_damping: float = 1.0e-8,
        torso_max_iterations: int = 25,
        head_max_iterations: int = 8,
        arm_max_iterations: int = 20,
        arm_position_cost: float = 1.0,
        arm_orientation_cost: float = 0.1,
    ) -> None:
        path = str(urdf_path)
        self.torso_max_iterations = int(torso_max_iterations)
        self.head_max_iterations = int(head_max_iterations)
        self.arm_max_iterations = int(arm_max_iterations)
        self.arm_position_cost = float(arm_position_cost)
        self.arm_orientation_cost = float(arm_orientation_cost)
        self.velocity_limit_enabled = bool(velocity_limit_enabled)
        self._simple_kinematics = VegaKinematics()
        collision_components = {component.lower() for component in self_collision_components}
        valid_components = {"torso", "head", "left_arm", "right_arm"}
        unknown_components = collision_components - valid_components
        if unknown_components:
            raise ValueError(
                "unknown self-collision component(s): "
                f"{', '.join(sorted(unknown_components))}; valid components are "
                f"{', '.join(sorted(valid_components))}"
            )
        self.torso = self._simple_kinematics.torso
        self.head = self._simple_kinematics.head
        self.left_arm = PinkChain(
            path,
            LEFT_ARM_JOINTS,
            "L_ee",
            solver=solver,
            dt=dt,
            root_frame_name="arm_center",
            velocity_limit_enabled=self.velocity_limit_enabled,
            task_gain=task_gain,
            lm_damping=lm_damping,
            solve_damping=solve_damping,
            self_collision=_self_collision_options(
                "left_arm",
                collision_components,
                self_collision_srdf_path,
                self_collision_urdf_path,
                collision_package_dirs,
                self_collision_n_pairs,
                self_collision_gain,
                self_collision_safe_displacement_gain,
                self_collision_d_min,
            ),
        )
        self.right_arm = PinkChain(
            path,
            RIGHT_ARM_JOINTS,
            "R_ee",
            solver=solver,
            dt=dt,
            root_frame_name="arm_center",
            velocity_limit_enabled=self.velocity_limit_enabled,
            task_gain=task_gain,
            lm_damping=lm_damping,
            solve_damping=solve_damping,
            self_collision=_self_collision_options(
                "right_arm",
                collision_components,
                self_collision_srdf_path,
                self_collision_urdf_path,
                collision_package_dirs,
                self_collision_n_pairs,
                self_collision_gain,
                self_collision_safe_displacement_gain,
                self_collision_d_min,
            ),
        )

    def arm_center_pose(self, torso_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._simple_kinematics.arm_center_pose(torso_q)

    def arm_shoulder_position(self, side: str) -> np.ndarray:
        y = ROBOT_SHOULDER_LATERAL_OFFSET_M if side == "left" else -ROBOT_SHOULDER_LATERAL_OFFSET_M
        return np.array([0.0, y, 0.0], dtype=np.float64)

    def solve_torso_height(
        self,
        q_seed: np.ndarray,
        target_z: float,
        target_pitch: float = 0.0,
        target_x: float | None = None,
        max_iterations: int | None = None,
    ) -> IKSolution:
        return self._simple_kinematics.solve_torso_height(
            q_seed,
            target_z,
            target_pitch=target_pitch,
            target_x=target_x,
            max_iterations=self.torso_max_iterations if max_iterations is None else max_iterations,
        )

    def solve_head_orientation(
        self,
        q_seed: np.ndarray,
        target_rotation: np.ndarray,
        max_iterations: int | None = None,
    ) -> IKSolution:
        return self._simple_kinematics.solve_head_orientation(
            q_seed,
            target_rotation,
            max_iterations=self.head_max_iterations if max_iterations is None else max_iterations,
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
            position_cost=self.arm_position_cost,
            orientation_cost=self.arm_orientation_cost,
            max_iterations=self.arm_max_iterations,
            tolerance=6.0e-3,
        )


def _self_collision_options(
    component: str,
    enabled_components: set[str],
    srdf_path: str | Path | None,
    collision_urdf_path: str | Path | None,
    package_dirs: tuple[str | Path, ...],
    n_collision_pairs: int,
    gain: float,
    safe_displacement_gain: float,
    d_min: float,
) -> SelfCollisionOptions | None:
    if component not in enabled_components:
        return None
    if srdf_path is None:
        raise PinkUnavailableError("self-collision is enabled but no SRDF path was provided")
    srdf = Path(srdf_path)
    if not srdf.is_file():
        raise PinkUnavailableError(f"self-collision SRDF does not exist: {srdf}")
    collision_urdf = Path(collision_urdf_path) if collision_urdf_path else None
    if collision_urdf is not None and not collision_urdf.is_file():
        raise PinkUnavailableError(
            f"self-collision collision URDF does not exist: {collision_urdf}"
        )
    return SelfCollisionOptions(
        srdf_path=srdf,
        package_dirs=tuple(Path(path) for path in package_dirs),
        collision_urdf_path=collision_urdf,
        n_collision_pairs=n_collision_pairs,
        gain=gain,
        safe_displacement_gain=safe_displacement_gain,
        d_min=d_min,
    )


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
