"""Deterministic name-based conversion of ROS feedback and applied commands."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .configuration import HandSynergyConfig
from .hand_synergy import HandSynergyError, reconstruct_hand_synergy


class SnapshotValidationError(RuntimeError):
    """Raised when a synchronized state/action snapshot cannot be constructed."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))

    @property
    def stale(self) -> bool:
        return any("stale" in reason or "missing" in reason for reason in self.reasons)


@dataclass(frozen=True)
class FeatureSnapshot:
    state: np.ndarray
    action: np.ndarray
    state_age_seconds: float
    action_age_seconds: float


@dataclass(frozen=True)
class _JointSample:
    positions: dict[str, float]
    velocities: dict[str, float]
    stamp_ns: int


@dataclass(frozen=True)
class _VectorSample:
    values: tuple[float, float, float]
    stamp_ns: int


class RobotFeatureAdapter:
    """Hold the latest inputs and emit the fixed measured-state/applied-action layout."""

    def __init__(
        self,
        joint_names: Sequence[str],
        *,
        include_joint_velocities: bool = True,
        hand_synergies: Sequence[HandSynergyConfig] = (),
    ) -> None:
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        self.include_joint_velocities = include_joint_velocities
        self.hand_synergies = tuple(hand_synergies)
        hand_names = tuple(
            name for synergy in self.hand_synergies for name in synergy.joint_names
        )
        if len(hand_names) != len(set(hand_names)) or not set(hand_names).issubset(
            self.joint_names
        ):
            raise ValueError("hand synergy joint names must be unique configured joints")
        hand_name_set = set(hand_names)
        self.body_joint_names = tuple(
            name for name in self.joint_names if name not in hand_name_set
        )
        self._lock = threading.Lock()
        self._measured_joints: _JointSample | None = None
        self._applied_joints: _JointSample | None = None
        self._measured_base: _VectorSample | None = None
        self._applied_base: _VectorSample | None = None

    @property
    def state_dimension(self) -> int:
        compact_positions = len(self.body_joint_names) + 2 * len(self.hand_synergies)
        body_velocities = len(self.body_joint_names) if self.include_joint_velocities else 0
        return compact_positions + body_velocities + 3

    @property
    def action_dimension(self) -> int:
        return len(self.body_joint_names) + 2 * len(self.hand_synergies) + 3

    def update_measured_joints(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float],
        stamp_ns: int,
    ) -> None:
        try:
            sample = self._make_joint_sample(
                names,
                positions,
                velocities,
                stamp_ns,
                require_velocities=self.include_joint_velocities,
            )
        except ValueError:
            with self._lock:
                self._measured_joints = None
            raise
        with self._lock:
            self._measured_joints = sample

    def update_applied_joints(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        stamp_ns: int,
    ) -> None:
        try:
            sample = self._make_joint_sample(
                names, positions, (), stamp_ns, require_velocities=False
            )
        except ValueError:
            with self._lock:
                self._applied_joints = None
            raise
        with self._lock:
            self._applied_joints = sample

    def update_measured_base(self, values: Sequence[float], stamp_ns: int) -> None:
        try:
            sample = self._make_vector_sample(values, stamp_ns)
        except ValueError:
            with self._lock:
                self._measured_base = None
            raise
        with self._lock:
            self._measured_base = sample

    def update_applied_base(self, values: Sequence[float], stamp_ns: int) -> None:
        try:
            sample = self._make_vector_sample(values, stamp_ns)
        except ValueError:
            with self._lock:
                self._applied_base = None
            raise
        with self._lock:
            self._applied_base = sample

    def snapshot(
        self,
        now_ns: int,
        *,
        maximum_state_age_seconds: float,
        maximum_action_age_seconds: float,
    ) -> FeatureSnapshot:
        with self._lock:
            measured_joints = self._measured_joints
            applied_joints = self._applied_joints
            measured_base = self._measured_base
            applied_base = self._applied_base

        reasons: list[str] = []
        for label, sample in (
            ("measured joint state", measured_joints),
            ("measured base state", measured_base),
            ("applied joint action", applied_joints),
            ("applied base action", applied_base),
        ):
            if sample is None:
                reasons.append(f"missing {label}")
        if reasons:
            raise SnapshotValidationError(reasons)
        assert measured_joints and measured_base and applied_joints and applied_base

        state_age = max(
            _age_seconds(now_ns, measured_joints.stamp_ns),
            _age_seconds(now_ns, measured_base.stamp_ns),
        )
        action_age = max(
            _age_seconds(now_ns, applied_joints.stamp_ns),
            _age_seconds(now_ns, applied_base.stamp_ns),
        )
        if state_age > maximum_state_age_seconds:
            reasons.append(
                f"stale measured state ({state_age:.3f}s > "
                f"{maximum_state_age_seconds:.3f}s)"
            )
        if action_age > maximum_action_age_seconds:
            reasons.append(
                f"stale applied action ({action_age:.3f}s > "
                f"{maximum_action_age_seconds:.3f}s)"
            )
        if state_age < 0.0:
            reasons.append("measured state timestamp is in the future")
        if action_age < 0.0:
            reasons.append("applied action timestamp is in the future")
        if reasons:
            raise SnapshotValidationError(reasons)

        try:
            state_values = [
                measured_joints.positions[name] for name in self.body_joint_names
            ]
            for synergy in self.hand_synergies:
                state_values.extend(
                    reconstruct_hand_synergy(
                        synergy,
                        measured_joints.positions,
                        require_exact_action=False,
                    )
                )
            if self.include_joint_velocities:
                state_values.extend(
                    measured_joints.velocities[name] for name in self.body_joint_names
                )
            state_values.extend(measured_base.values)
            action_values = [
                applied_joints.positions[name] for name in self.body_joint_names
            ]
            for synergy in self.hand_synergies:
                action_values.extend(
                    reconstruct_hand_synergy(
                        synergy,
                        applied_joints.positions,
                        require_exact_action=True,
                    )
                )
            action_values.extend(applied_base.values)
        except KeyError as exc:
            raise SnapshotValidationError([f"required joint missing: {exc.args[0]}"]) from exc
        except HandSynergyError as exc:
            raise SnapshotValidationError([str(exc)]) from exc

        state = np.asarray(state_values, dtype=np.float32)
        action = np.asarray(action_values, dtype=np.float32)
        if state.shape != (self.state_dimension,) or action.shape != (
            self.action_dimension,
        ):
            raise SnapshotValidationError(["state/action dimension mismatch"])
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise SnapshotValidationError(["state/action contains non-finite values"])
        return FeatureSnapshot(state, action, state_age, action_age)

    @staticmethod
    def _make_joint_sample(
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float],
        stamp_ns: int,
        *,
        require_velocities: bool,
    ) -> _JointSample:
        if stamp_ns <= 0:
            raise ValueError("joint sample timestamp must be positive")
        if len(names) != len(set(names)):
            raise ValueError("joint sample contains duplicate names")
        if len(positions) != len(names):
            raise ValueError("joint names and positions differ in length")
        if require_velocities and len(velocities) != len(names):
            raise ValueError("measured joint velocities are missing or wrong-sized")
        if not all(math.isfinite(float(value)) for value in positions):
            raise ValueError("joint positions contain non-finite values")
        if require_velocities and not all(
            math.isfinite(float(value)) for value in velocities
        ):
            raise ValueError("joint velocities contain non-finite values")
        return _JointSample(
            positions={name: float(value) for name, value in zip(names, positions)},
            velocities={name: float(value) for name, value in zip(names, velocities)},
            stamp_ns=int(stamp_ns),
        )

    @staticmethod
    def _make_vector_sample(values: Sequence[float], stamp_ns: int) -> _VectorSample:
        if stamp_ns <= 0:
            raise ValueError("base sample timestamp must be positive")
        if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("base sample must contain three finite values")
        return _VectorSample(tuple(float(v) for v in values), int(stamp_ns))


def _age_seconds(now_ns: int, stamp_ns: int) -> float:
    return (int(now_ns) - int(stamp_ns)) / 1.0e9
