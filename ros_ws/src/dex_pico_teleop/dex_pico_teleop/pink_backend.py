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
    TORSO_JOINTS,
    VegaKinematics,
)
from dex_pico_teleop.collision_profiles import filter_geometry_model
from dex_pico_teleop.transforms import rotation_error


class PinkUnavailableError(RuntimeError):
    """Raised when the Pinocchio/Pink backend cannot be initialized."""


@dataclass(frozen=True)
class SelfCollisionOptions:
    """Pink self-collision barrier settings for an IK chain."""

    srdf_path: Path
    package_dirs: tuple[Path, ...]
    collision_urdf_path: Path | None = None
    n_collision_pairs: int = 24
    # At 50 Hz, a gain of 6 permits responsive motion toward d_min without
    # changing the hard surface-distance boundary itself.
    gain: float = 6.0
    safe_displacement_gain: float = 0.0
    d_min: float = 0.04
    pipeline: str = "reduced_all_pairs"
    sphere_count: int = 30
    sphere_inflation: float = 1.0


class PinkChain:
    """A reduced Pink model containing only the joints for one task group."""

    def __init__(
        self,
        urdf_path: str,
        active_joint_names: tuple[str, ...],
        frame_name: str,
        solver: str = "proxqp",
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
        self.data = self.model.createData()
        self.collision_data = (
            self.pin.GeometryData(self.collision_model)
            if self.collision_model is not None
            else None
        )
        self.neutral = self._clip_full_q(pin.neutral(self.model))
        self.limits = [ConfigurationLimit(self.model)]
        if velocity_limit_enabled:
            self.limits.append(VelocityLimit(self.model))
        self.barriers = self._make_barriers(self_collision)
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
        target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        configuration = self.pink.Configuration(
            self.model,
            self.data,
            q_seed_full,
            collision_model=self.collision_model,
            collision_data=self.collision_data,
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
            target_rotation,
            target_position,
        )
        frame_task.set_target(target)

        posture_task = self.PostureTask(cost=1.0e-4)
        posture_task.set_target(q_seed_full)
        all_tasks = [frame_task, posture_task]

        position_weights = _weights(position_cost)
        orientation_weights = _weights(orientation_cost)
        initial_position, initial_rotation = self._forward_q(configuration.q)
        initial_position_error, initial_orientation_error = _pose_error_norms(
            initial_position,
            initial_rotation,
            target_position,
            target_rotation,
        )
        initial_error = _weighted_pose_error_norm(
            initial_position,
            initial_rotation,
            target_position,
            target_rotation,
            position_weights,
            orientation_weights,
        )
        best_q = configuration.q.copy()
        best_error = initial_error
        best_position_error = initial_position_error
        best_orientation_error = initial_orientation_error

        def result(success: bool, termination: str, iterations: int) -> IKSolution:
            return IKSolution(
                q=self.values_from_q(best_q),
                success=success,
                error_norm=best_error,
                iterations=iterations,
                termination=termination,
                initial_error_norm=initial_error,
                initial_position_error_norm=initial_position_error,
                initial_orientation_error_norm=initial_orientation_error,
                position_error_norm=best_position_error,
                orientation_error_norm=best_orientation_error,
            )

        for iteration in range(max_iterations):
            current_position, current_rotation = self._forward_q(configuration.q)
            position_error, orientation_error = _pose_error_norms(
                current_position,
                current_rotation,
                target_position,
                target_rotation,
            )
            current_error = _weighted_pose_error_norm(
                current_position,
                current_rotation,
                target_position,
                target_rotation,
                position_weights,
                orientation_weights,
            )
            if not all(np.isfinite(value) for value in (position_error, orientation_error, current_error)):
                return result(False, "nonfinite", iteration)
            if current_error < best_error:
                best_q = configuration.q.copy()
                best_error = current_error
                best_position_error = position_error
                best_orientation_error = orientation_error
            if current_error <= tolerance:
                return result(True, "converged", iteration)
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
                return result(False, "no_solution", iteration)
            except Exception:  # noqa: BLE001 - solver boundary must fail closed
                return result(False, "no_solution", iteration)
            velocity = np.asarray(velocity, dtype=np.float64)
            if not np.all(np.isfinite(velocity)):
                return result(False, "nonfinite", iteration)
            if float(np.linalg.norm(velocity) * self.dt) < 1.0e-10:
                return result(False, "stalled", iteration)
            configuration.integrate_inplace(velocity, self.dt)
            clipped_q = self._clip_full_q(configuration.q)
            if not np.all(np.isfinite(clipped_q)):
                return result(False, "nonfinite", iteration)
            configuration.update(clipped_q)

        final_position, final_rotation = self._forward_q(configuration.q)
        final_position_error, final_orientation_error = _pose_error_norms(
            final_position,
            final_rotation,
            target_position,
            target_rotation,
        )
        final_error = _weighted_pose_error_norm(
            final_position,
            final_rotation,
            target_position,
            target_rotation,
            position_weights,
            orientation_weights,
        )
        if not all(
            np.isfinite(value)
            for value in (final_position_error, final_orientation_error, final_error)
        ):
            return result(False, "nonfinite", max_iterations)
        converged = final_error <= tolerance
        return IKSolution(
            q=self.values_from_q(configuration.q),
            success=converged,
            error_norm=final_error,
            iterations=max_iterations,
            termination="converged" if converged else "max_iterations",
            initial_error_norm=initial_error,
            initial_position_error_norm=initial_position_error,
            initial_orientation_error_norm=initial_orientation_error,
            position_error_norm=final_position_error,
            orientation_error_norm=final_orientation_error,
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
        if any(name.startswith("L_arm_") for name in active_joint_names):
            own_side, other_side = "L", "R"
        elif any(name.startswith("R_arm_") for name in active_joint_names):
            own_side, other_side = "R", "L"
        else:
            return

        def link_name(geometry_name: str) -> str:
            return geometry_name.rsplit("_", maxsplit=1)[0]

        def arm_or_hand_base(geometry_name: str, side: str) -> bool:
            link = link_name(geometry_name)
            return link.startswith(f"{side}_arm_l") or link == f"{side}_hand_base"

        def fixed_obstacle(geometry_name: str) -> bool:
            link = link_name(geometry_name)
            return (
                link == "base"
                or link.startswith("torso_l")
                or link.startswith("head_l")
                or arm_or_hand_base(geometry_name, other_side)
            )

        retained: list[object] = []
        seen: set[tuple[int, int]] = set()
        for pair in self.collision_model.collisionPairs:
            geometry_1 = self.collision_model.geometryObjects[pair.first]
            geometry_2 = self.collision_model.geometryObjects[pair.second]
            key = (min(pair.first, pair.second), max(pair.first, pair.second))
            own_1 = arm_or_hand_base(geometry_1.name, own_side)
            own_2 = arm_or_hand_base(geometry_2.name, own_side)
            safety_relevant = (
                (own_1 and own_2)
                or (own_1 and fixed_obstacle(geometry_2.name))
                or (own_2 and fixed_obstacle(geometry_1.name))
            )
            if safety_relevant and key not in seen:
                retained.append(pair)
                seen.add(key)

        self.collision_model.removeAllCollisionPairs()
        for pair in retained:
            self.collision_model.addCollisionPair(pair)


class PinkArmView:
    """Compatibility view of one arm in the unified bimanual model."""

    def __init__(self, arms: "PinkBimanualChain", side: str) -> None:
        self._arms = arms
        self.side = side
        self.joint_names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        self._indices = {name: arms._indices[name] for name in self.joint_names}
        self._lower_position_limits = arms._lower_position_limits
        self._upper_position_limits = arms._upper_position_limits
        self.collision_pair_count = arms.collision_pair_count
        self.barrier_pair_count = arms.barrier_pair_count
        self.barriers = arms.barriers

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(7, dtype=np.float64)
        left = np.asarray(values, dtype=np.float64) if self.side == "left" else zeros
        right = np.asarray(values, dtype=np.float64) if self.side == "right" else zeros
        return self._arms.forward_arm(
            self.side,
            np.zeros(len(TORSO_JOINTS), dtype=np.float64),
            left,
            right,
        )


class PinkBimanualChain:
    """One Pink QP for both arms with a measured, velocity-fixed torso."""

    def __init__(
        self,
        urdf_path: str,
        solver: str,
        dt: float,
        self_collision: SelfCollisionOptions | None,
        velocity_limit_enabled: bool,
        task_gain: float,
        lm_damping: float,
        solve_damping: float,
        position_cost: float,
        orientation_cost: float,
    ) -> None:
        try:
            import pinocchio as pin
            import pink
            import qpsolvers
            from pink import tasks
            from pink.barriers import SelfCollisionBarrier
            from pink.exceptions import NoSolutionFound
            from pink.limits import ConfigurationLimit, VelocityLimit
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            raise PinkUnavailableError(str(exc)) from exc

        if solver not in qpsolvers.available_solvers:
            raise PinkUnavailableError(
                f"QP solver '{solver}' is not available; installed solvers: "
                f"{qpsolvers.available_solvers}"
            )

        class FixedJointVelocityLimit:
            """Constrain selected joint displacements to zero in the QP.

            Pink's ``quadprog`` backend can report an infeasible problem when
            an equality task is combined with velocity inequalities.  Two
            zero-width inequalities are mathematically equivalent here and
            remain compatible with Pink's configuration/velocity limits.
            """

            def __init__(self, indices: tuple[int, ...], nv: int) -> None:
                self.projection = np.eye(nv, dtype=np.float64)[list(indices)]

            def compute_qp_inequalities(self, _configuration, _dt):
                return (
                    np.vstack((self.projection, -self.projection)),
                    np.zeros(2 * self.projection.shape[0], dtype=np.float64),
                )

        class AllPairsSelfCollisionBarrier(SelfCollisionBarrier):
            """Pink collision barrier with a stable, precomputed pair order."""

            def compute_barrier(self, configuration) -> np.ndarray:
                return np.asarray(
                    [
                        result.min_distance - self.d_min
                        for result in configuration.collision_data.distanceResults
                    ],
                    dtype=np.float64,
                )

            def compute_jacobian(self, configuration) -> np.ndarray:
                model = configuration.model
                data = configuration.data
                collision_model = configuration.collision_model
                collision_data = configuration.collision_data
                jacobian = np.zeros((self.dim, model.nv), dtype=np.float64)
                for index, pair in enumerate(collision_model.collisionPairs):
                    result = collision_data.distanceResults[index]
                    geometry_1 = collision_model.geometryObjects[pair.first]
                    geometry_2 = collision_model.geometryObjects[pair.second]
                    joint_1 = geometry_1.parentJoint
                    joint_2 = geometry_2.parentJoint
                    point_1 = np.asarray(result.getNearestPoint1(), dtype=np.float64)
                    point_2 = np.asarray(result.getNearestPoint2(), dtype=np.float64)
                    if np.allclose(point_1, point_2):
                        continue
                    normal = (point_1 - point_2) / np.linalg.norm(point_1 - point_2)
                    offset_1 = point_1 - data.oMi[joint_1].translation
                    offset_2 = point_2 - data.oMi[joint_2].translation
                    joint_jacobian_1 = pin.getJointJacobian(
                        model,
                        data,
                        joint_1,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    joint_jacobian_2 = pin.getJointJacobian(
                        model,
                        data,
                        joint_2,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    row = normal.T @ joint_jacobian_1[:3, :]
                    row += (pin.skew(offset_1) @ normal).T @ joint_jacobian_1[3:, :]
                    row -= normal.T @ joint_jacobian_2[:3, :]
                    row -= (pin.skew(offset_2) @ normal).T @ joint_jacobian_2[3:, :]
                    jacobian[index] = row
                return np.nan_to_num(jacobian)

        self.pin = pin
        self.pink = pink
        self.NoSolutionFound = NoSolutionFound
        self.solver = solver
        self.dt = float(dt)
        self.solve_damping = float(solve_damping)
        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.joint_names = TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

        full_model = pin.buildModelFromUrdf(str(urdf_path))
        full_collision_model = None
        if self_collision is not None:
            collision_urdf_path = self_collision.collision_urdf_path or Path(urdf_path)
            full_collision_model = pin.buildGeomFromUrdf(
                full_model,
                str(collision_urdf_path),
                pin.GeometryType.COLLISION,
                None,
                [str(path) for path in self_collision.package_dirs],
            )
            if self_collision.pipeline == "reduced_all_pairs":
                self.selected_geometry_names = filter_geometry_model(
                    full_collision_model,
                    self_collision.sphere_count,
                    self_collision.sphere_inflation,
                )
            elif self_collision.pipeline == "closest_pairs":
                self.selected_geometry_names = tuple(
                    geometry.name for geometry in full_collision_model.geometryObjects
                )
            else:
                raise ValueError(
                    "pink collision pipeline must be 'reduced_all_pairs' or "
                    f"'closest_pairs', got {self_collision.pipeline!r}"
                )
        else:
            self.selected_geometry_names = ()

        active = set(self.joint_names)
        joints_to_lock = [
            full_model.getJointId(name)
            for name in full_model.names
            if name != "universe" and name not in active
        ]
        neutral = pin.neutral(full_model)
        if full_collision_model is None:
            self.model = pin.buildReducedModel(full_model, joints_to_lock, neutral)
            self.collision_model = None
        else:
            self.model, self.collision_model = pin.buildReducedModel(
                full_model,
                full_collision_model,
                joints_to_lock,
                neutral,
            )
            self.collision_model.addAllCollisionPairs()
            pin.removeCollisionPairs(
                self.model,
                self.collision_model,
                str(self_collision.srdf_path),
                False,
            )
            self._retain_relevant_collision_pairs()

        self._indices = {
            name: int(self.model.joints[self.model.getJointId(name)].idx_q)
            for name in self.joint_names
        }
        self._velocity_indices = {
            name: int(self.model.joints[self.model.getJointId(name)].idx_v)
            for name in self.joint_names
        }
        self._lower_position_limits = np.asarray(
            self.model.lowerPositionLimit,
            dtype=np.float64,
        )
        self._upper_position_limits = np.asarray(
            self.model.upperPositionLimit,
            dtype=np.float64,
        )
        self.neutral = self._clip_full_q(pin.neutral(self.model))
        self.data = self.model.createData()
        self.collision_data = (
            pin.GeometryData(self.collision_model)
            if self.collision_model is not None
            else None
        )
        self.configuration = pink.Configuration(
            self.model,
            self.data,
            self.neutral,
            copy_data=False,
            collision_model=self.collision_model,
            collision_data=self.collision_data,
        )
        self.frame_ids = {
            "left": self.model.getFrameId("L_ee"),
            "right": self.model.getFrameId("R_ee"),
        }
        self.root_frame_id = self.model.getFrameId("arm_center")
        self.frame_tasks = {
            "left": tasks.FrameTask(
                "L_ee",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
                gain=task_gain,
            ),
            "right": tasks.FrameTask(
                "R_ee",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
                gain=task_gain,
            ),
        }
        self.posture_task = tasks.PostureTask(cost=1.0e-4)
        torso_velocity_indices = tuple(self._velocity_indices[name] for name in TORSO_JOINTS)
        self.limits = [
            ConfigurationLimit(self.model),
            FixedJointVelocityLimit(torso_velocity_indices, self.model.nv),
        ]
        if velocity_limit_enabled:
            self.limits.append(VelocityLimit(self.model))

        self.barriers: list[object] = []
        if self_collision is not None and self.collision_model is not None:
            if self_collision.pipeline == "reduced_all_pairs":
                pair_count = len(self.collision_model.collisionPairs)
                barrier_type = AllPairsSelfCollisionBarrier
            else:
                pair_count = min(
                    int(self_collision.n_collision_pairs),
                    len(self.collision_model.collisionPairs),
                )
                barrier_type = SelfCollisionBarrier
            if pair_count > 0:
                self.barriers = [
                    barrier_type(
                        n_collision_pairs=pair_count,
                        gain=float(self_collision.gain),
                        safe_displacement_gain=float(
                            self_collision.safe_displacement_gain
                        ),
                        d_min=float(self_collision.d_min),
                    )
                ]
        self.collision_pipeline = (
            "disabled" if self_collision is None else self_collision.pipeline
        )
        self.collision_geometry_count = (
            0 if self.collision_model is None else self.collision_model.ngeoms
        )
        self.collision_pair_count = (
            0 if self.collision_model is None else len(self.collision_model.collisionPairs)
        )
        self.barrier_pair_count = sum(barrier.dim for barrier in self.barriers)
        self.last_collision_distance = float("inf")
        self.last_collision_pair = ("", "")

    def q_from_values(
        self,
        torso_q: np.ndarray,
        left_q: np.ndarray,
        right_q: np.ndarray,
    ) -> np.ndarray:
        q = self.neutral.copy()
        values = np.concatenate(
            (
                np.asarray(torso_q, dtype=np.float64).reshape(len(TORSO_JOINTS)),
                np.asarray(left_q, dtype=np.float64).reshape(len(LEFT_ARM_JOINTS)),
                np.asarray(right_q, dtype=np.float64).reshape(len(RIGHT_ARM_JOINTS)),
            )
        )
        for name, value in zip(self.joint_names, values):
            q[self._indices[name]] = float(value)
        return self._clip_full_q(q)

    def arm_values_from_q(self, q: np.ndarray, side: str) -> np.ndarray:
        names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        clipped = self._clip_full_q(q)
        return np.asarray([clipped[self._indices[name]] for name in names])

    def forward_arm(
        self,
        side: str,
        torso_q: np.ndarray,
        left_q: np.ndarray,
        right_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_from_values(torso_q, left_q, right_q)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        root = self.data.oMf[self.root_frame_id]
        transform = root.inverse() * self.data.oMf[self.frame_ids[side]]
        return (
            np.asarray(transform.translation, dtype=np.float64).copy(),
            np.asarray(transform.rotation, dtype=np.float64).copy(),
        )

    def solve(
        self,
        torso_q: np.ndarray,
        left_q: np.ndarray,
        right_q: np.ndarray,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        dt: float | None = None,
        tolerance: float = 6.0e-3,
    ) -> dict[str, IKSolution]:
        step_dt = self.dt if dt is None else float(dt)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("IK integration dt must be finite and positive")
        q_seed = self.q_from_values(torso_q, left_q, right_q)
        self.configuration.update(q_seed)
        root = self.configuration.data.oMf[self.root_frame_id]
        initial_errors: dict[str, tuple[float, float, float]] = {}
        for side in ("left", "right"):
            target_position = np.asarray(targets[side][0], dtype=np.float64).reshape(3)
            target_rotation = np.asarray(targets[side][1], dtype=np.float64).reshape(3, 3)
            self.frame_tasks[side].set_target(
                root * self.pin.SE3(target_rotation, target_position)
            )
            position, rotation = self._relative_frame_pose(side)
            position_error, orientation_error = _pose_error_norms(
                position,
                rotation,
                target_position,
                target_rotation,
            )
            weighted_error = _weighted_pose_error_norm(
                position,
                rotation,
                target_position,
                target_rotation,
                _weights(self.position_cost),
                _weights(self.orientation_cost),
            )
            initial_errors[side] = (weighted_error, position_error, orientation_error)

        self.posture_task.set_target(q_seed)
        try:
            velocity = self.pink.solve_ik(
                self.configuration,
                [*self.frame_tasks.values(), self.posture_task],
                dt=step_dt,
                solver=self.solver,
                limits=self.limits,
                barriers=self.barriers,
                damping=self.solve_damping,
            )
        except self.NoSolutionFound:
            return self._failed_solutions(q_seed, initial_errors, "no_solution")
        except Exception:  # noqa: BLE001 - solver boundary must fail closed
            return self._failed_solutions(q_seed, initial_errors, "no_solution")

        velocity = np.asarray(velocity, dtype=np.float64)
        if not np.all(np.isfinite(velocity)):
            return self._failed_solutions(q_seed, initial_errors, "nonfinite")
        self.configuration.integrate_inplace(velocity, step_dt)
        integrated_q = self._clip_full_q(self.configuration.q)
        for name in TORSO_JOINTS:
            integrated_q[self._indices[name]] = q_seed[self._indices[name]]
        if not np.all(np.isfinite(integrated_q)):
            return self._failed_solutions(q_seed, initial_errors, "nonfinite")
        self.configuration.update(integrated_q)
        self._update_collision_diagnostics()

        solutions: dict[str, IKSolution] = {}
        for side in ("left", "right"):
            target_position = np.asarray(targets[side][0], dtype=np.float64).reshape(3)
            target_rotation = np.asarray(targets[side][1], dtype=np.float64).reshape(3, 3)
            position, rotation = self._relative_frame_pose(side)
            position_error, orientation_error = _pose_error_norms(
                position,
                rotation,
                target_position,
                target_rotation,
            )
            error = _weighted_pose_error_norm(
                position,
                rotation,
                target_position,
                target_rotation,
                _weights(self.position_cost),
                _weights(self.orientation_cost),
            )
            initial_error, initial_position, initial_orientation = initial_errors[side]
            converged = error <= tolerance
            solutions[side] = IKSolution(
                q=self.arm_values_from_q(integrated_q, side),
                success=converged,
                error_norm=error,
                iterations=1,
                termination="converged" if converged else "integrated_step",
                initial_error_norm=initial_error,
                initial_position_error_norm=initial_position,
                initial_orientation_error_norm=initial_orientation,
                position_error_norm=position_error,
                orientation_error_norm=orientation_error,
            )
        return solutions

    def diagnostics(self) -> dict[str, object]:
        return {
            "collision_pipeline": self.collision_pipeline,
            "collision_geometry_count": self.collision_geometry_count,
            "collision_pair_count": self.collision_pair_count,
            "collision_barrier_pair_count": self.barrier_pair_count,
            "collision_min_distance": self.last_collision_distance,
            "collision_closest_pair": list(self.last_collision_pair),
        }

    def _failed_solutions(
        self,
        q: np.ndarray,
        initial_errors: dict[str, tuple[float, float, float]],
        termination: str,
    ) -> dict[str, IKSolution]:
        output: dict[str, IKSolution] = {}
        for side in ("left", "right"):
            error, position_error, orientation_error = initial_errors[side]
            output[side] = IKSolution(
                q=self.arm_values_from_q(q, side),
                success=False,
                error_norm=error,
                iterations=0,
                termination=termination,
                initial_error_norm=error,
                initial_position_error_norm=position_error,
                initial_orientation_error_norm=orientation_error,
                position_error_norm=position_error,
                orientation_error_norm=orientation_error,
            )
        return output

    def _relative_frame_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        root = self.configuration.data.oMf[self.root_frame_id]
        transform = root.inverse() * self.configuration.data.oMf[self.frame_ids[side]]
        return (
            np.asarray(transform.translation, dtype=np.float64).copy(),
            np.asarray(transform.rotation, dtype=np.float64).copy(),
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

    def _retain_relevant_collision_pairs(self) -> None:
        if self.collision_model is None:
            return

        def link_name(geometry_name: str) -> str:
            return geometry_name.rsplit("_", maxsplit=1)[0]

        def side_arm(link: str, side: str) -> bool:
            return link.startswith(f"{side}_arm_l") or link == f"{side}_hand_base"

        def body(link: str) -> bool:
            return link == "base" or link.startswith("torso_l") or link.startswith("head_l")

        retained: list[object] = []
        for pair in self.collision_model.collisionPairs:
            link_1 = link_name(self.collision_model.geometryObjects[pair.first].name)
            link_2 = link_name(self.collision_model.geometryObjects[pair.second].name)
            left_1, left_2 = side_arm(link_1, "L"), side_arm(link_2, "L")
            right_1, right_2 = side_arm(link_1, "R"), side_arm(link_2, "R")
            safety_relevant = (
                (left_1 and (left_2 or right_2 or body(link_2)))
                or (right_1 and (right_2 or left_2 or body(link_2)))
                or (left_2 and body(link_1))
                or (right_2 and body(link_1))
            )
            if safety_relevant:
                retained.append(pair)
        self.collision_model.removeAllCollisionPairs()
        for pair in retained:
            self.collision_model.addCollisionPair(pair)

    def _update_collision_diagnostics(self) -> None:
        if self.collision_model is None or self.collision_data is None:
            self.last_collision_distance = float("inf")
            self.last_collision_pair = ("", "")
            return
        distances = np.asarray(
            [result.min_distance for result in self.collision_data.distanceResults],
            dtype=np.float64,
        )
        if distances.size == 0:
            self.last_collision_distance = float("inf")
            self.last_collision_pair = ("", "")
            return
        index = int(np.argmin(distances))
        pair = self.collision_model.collisionPairs[index]
        self.last_collision_distance = float(distances[index])
        self.last_collision_pair = (
            self.collision_model.geometryObjects[pair.first].name,
            self.collision_model.geometryObjects[pair.second].name,
        )


class PinkVegaKinematics:
    """Vega kinematics implementation backed by Pinocchio and Pink."""

    def __init__(
        self,
        urdf_path: str | Path,
        solver: str = "proxqp",
        dt: float = 0.02,
        self_collision_components: tuple[str, ...] = (),
        self_collision_srdf_path: str | Path | None = None,
        self_collision_urdf_path: str | Path | None = None,
        collision_package_dirs: tuple[str | Path, ...] = (),
        self_collision_n_pairs: int = 24,
        self_collision_gain: float = 6.0,
        self_collision_safe_displacement_gain: float = 0.0,
        self_collision_d_min: float = 0.04,
        velocity_limit_enabled: bool = False,
        task_gain: float = 1.0,
        lm_damping: float = 1.0e-6,
        solve_damping: float = 1.0e-8,
        torso_max_iterations: int = 25,
        head_max_iterations: int = 8,
        arm_max_iterations: int = 20,
        collision_arm_max_iterations: int = 2,
        arm_position_cost: float = 1.0,
        arm_orientation_cost: float = 0.1,
        collision_pipeline: str = "reduced_all_pairs",
        collision_sphere_count: int = 18,
        collision_sphere_inflation: float = 1.0,
    ) -> None:
        path = str(urdf_path)
        self.torso_max_iterations = int(torso_max_iterations)
        self.head_max_iterations = int(head_max_iterations)
        self.arm_max_iterations = int(arm_max_iterations)
        self.collision_arm_max_iterations = int(collision_arm_max_iterations)
        if self.collision_arm_max_iterations <= 0:
            raise ValueError("pink_self_collision_arm_max_iterations must be positive")
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
        collision_options = _self_collision_options(
            collision_components,
            self_collision_srdf_path,
            self_collision_urdf_path,
            collision_package_dirs,
            self_collision_n_pairs,
            self_collision_gain,
            self_collision_safe_displacement_gain,
            self_collision_d_min,
            collision_pipeline,
            collision_sphere_count,
            collision_sphere_inflation,
        )
        self.arms = PinkBimanualChain(
            path,
            solver=solver,
            dt=dt,
            self_collision=collision_options,
            velocity_limit_enabled=self.velocity_limit_enabled,
            task_gain=task_gain,
            lm_damping=lm_damping,
            solve_damping=solve_damping,
            position_cost=self.arm_position_cost,
            orientation_cost=self.arm_orientation_cost,
        )
        self.left_arm = PinkArmView(self.arms, "left")
        self.right_arm = PinkArmView(self.arms, "right")

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
        """Compatibility wrapper; teleop uses :meth:`solve_bimanual_arm_poses`.

        This method intentionally performs one differential-IK integration,
        rather than reintroducing the old multi-iteration per-arm behavior.
        """
        zeros = np.zeros(7, dtype=np.float64)
        if side == "left":
            left_q, right_q = q_seed, zeros
        else:
            left_q, right_q = zeros, q_seed
        targets = {
            arm_side: self.arms.forward_arm(
                arm_side,
                np.zeros(len(TORSO_JOINTS), dtype=np.float64),
                left_q,
                right_q,
            )
            for arm_side in ("left", "right")
        }
        targets[side] = (target_position, target_rotation)
        return self.arms.solve(
            np.zeros(len(TORSO_JOINTS), dtype=np.float64),
            left_q,
            right_q,
            targets,
        )[side]

    def solve_bimanual_arm_poses(
        self,
        torso_q: np.ndarray,
        left_q: np.ndarray,
        right_q: np.ndarray,
        left_target_position: np.ndarray,
        left_target_rotation: np.ndarray,
        right_target_position: np.ndarray,
        right_target_rotation: np.ndarray,
        dt: float | None = None,
    ) -> dict[str, IKSolution]:
        """Solve both arms once from the measured whole-upper-body posture."""
        return self.arms.solve(
            torso_q,
            left_q,
            right_q,
            {
                "left": (left_target_position, left_target_rotation),
                "right": (right_target_position, right_target_rotation),
            },
            dt=dt,
        )

    def collision_diagnostics(self) -> dict[str, object]:
        """Return fixed-profile collision diagnostics for status telemetry."""
        return self.arms.diagnostics()


def _self_collision_options(
    enabled_components: set[str],
    srdf_path: str | Path | None,
    collision_urdf_path: str | Path | None,
    package_dirs: tuple[str | Path, ...],
    n_collision_pairs: int,
    gain: float,
    safe_displacement_gain: float,
    d_min: float,
    pipeline: str,
    sphere_count: int,
    sphere_inflation: float,
) -> SelfCollisionOptions | None:
    if not enabled_components:
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
        pipeline=str(pipeline).lower(),
        sphere_count=int(sphere_count),
        sphere_inflation=float(sphere_inflation),
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


def _pose_error_norms(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> tuple[float, float]:
    position_error = float(
        np.linalg.norm(
            np.asarray(target_position, dtype=np.float64).reshape(3)
            - np.asarray(current_position, dtype=np.float64).reshape(3)
        )
    )
    orientation_error = float(
        np.linalg.norm(rotation_error(current_rotation, target_rotation))
    )
    return position_error, orientation_error
