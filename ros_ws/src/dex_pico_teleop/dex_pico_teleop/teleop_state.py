"""Small state helpers for Pico teleoperation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from dex_pico_teleop.kinematics import IKSolution


@dataclass(frozen=True)
class IKAcceptance:
    accepted: bool
    mode: str


def assess_arm_ik_solution(
    solution: IKSolution,
) -> IKAcceptance:
    metrics = np.asarray(
        [
            solution.initial_error_norm,
            solution.error_norm,
            solution.initial_position_error_norm,
            solution.position_error_norm,
            solution.initial_orientation_error_norm,
            solution.orientation_error_norm,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(solution.q)) or not np.all(np.isfinite(metrics)):
        return IKAcceptance(False, "held_nonfinite")
    if solution.success and solution.termination == "converged":
        return IKAcceptance(True, "converged")
    if solution.termination in {"max_iterations", "integrated_step"}:
        # Differential IK is expected to retain Cartesian error after only one
        # or two integrations. A finite integrated step should still be sent so
        # successive control ticks can converge toward the moving reference.
        return IKAcceptance(True, "integrated_step")
    return IKAcceptance(False, f"held_{solution.termination}")


class PositionTargetPlant:
    """Small rate- and acceleration-limited position-servo model for dry runs.

    It is intentionally not a dynamics simulator.  Its job is to provide the
    controller with delayed posture feedback so MeshCat and the next IK tick do
    not pretend that a published position command was reached instantaneously.
    """

    def __init__(self, max_velocity: float, max_acceleration: float) -> None:
        if max_velocity <= 0.0 or max_acceleration <= 0.0:
            raise ValueError("dry-run plant velocity and acceleration must be positive")
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.positions: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.targets: dict[str, float] = {}

    def seed(self, names: tuple[str, ...], values: np.ndarray) -> None:
        for name, value in zip(names, np.asarray(values, dtype=np.float64).reshape(-1)):
            if not np.isfinite(value):
                continue
            self.positions[name] = float(value)
            self.targets[name] = float(value)
            self.velocities[name] = 0.0

    def set_target(self, names: tuple[str, ...], values: np.ndarray) -> None:
        for name, value in zip(names, np.asarray(values, dtype=np.float64).reshape(-1)):
            if not np.isfinite(value):
                continue
            value_float = float(value)
            if name not in self.positions:
                self.positions[name] = value_float
                self.velocities[name] = 0.0
            self.targets[name] = value_float

    def advance(self, dt: float) -> None:
        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            return
        for name, target in self.targets.items():
            position = self.positions[name]
            velocity = self.velocities.get(name, 0.0)
            error = target - position
            desired_velocity = float(
                np.clip(error / step, -self.max_velocity, self.max_velocity)
            )
            acceleration = float(
                np.clip(
                    desired_velocity - velocity,
                    -self.max_acceleration * step,
                    self.max_acceleration * step,
                )
            )
            next_velocity = velocity + acceleration
            next_position = position + next_velocity * step
            if (target - position) * (target - next_position) <= 0.0:
                next_position = target
                next_velocity = 0.0
            self.positions[name] = next_position
            self.velocities[name] = next_velocity

    def values(
        self,
        names: tuple[str, ...],
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        fallback_values = (
            np.zeros(len(names), dtype=np.float64)
            if fallback is None
            else np.asarray(fallback, dtype=np.float64).reshape(len(names))
        )
        return np.asarray(
            [self.positions.get(name, fallback_values[index]) for index, name in enumerate(names)],
            dtype=np.float64,
        )


def joint_values(
    names: tuple[str, ...],
    feedback_positions: Mapping[str, float],
    command_positions: Mapping[str, float],
    prefer_command: bool = True,
) -> np.ndarray:
    """Return joint values from command warm-starts and feedback fallbacks."""
    primary = command_positions if prefer_command else feedback_positions
    fallback = feedback_positions if prefer_command else command_positions
    return np.asarray(
        [primary.get(name, fallback.get(name, 0.0)) for name in names],
        dtype=np.float64,
    )
