"""Postprocessor-output validation and bridge-native action adaptation."""

from __future__ import annotations

import math
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from dex_vega_lerobot_recorder.configuration import RecorderConfig
from dex_vega_lerobot_recorder.hand_synergy import HandSynergyError, expand_hand_synergy

from .contracts import (
    ACTION_DIMENSION,
    ACTION_NAMES,
    BODY_JOINT_NAMES,
    COMPONENT_JOINT_NAMES,
)
from .observation_adapter import validate_recorder_contract


class ActionValidationError(RuntimeError):
    """Raised before any ROS command is created or published."""


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float


@dataclass(frozen=True)
class ActionSafetyConfig:
    max_torso_target_delta_per_cycle: float = 0.02
    max_head_target_delta_per_cycle: float = 0.02
    max_arm_target_delta_per_cycle: float = 0.02
    max_hand_ratio_delta_per_cycle: float = 0.03
    max_base_linear_velocity: float = 0.10
    max_base_angular_velocity: float = 0.20
    max_base_linear_acceleration: float = 0.30
    max_base_angular_acceleration: float = 0.60

    def validate(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all action safety limits must be finite and positive")


@dataclass(frozen=True)
class AdaptedAction:
    policy_action: np.ndarray
    component_positions: dict[str, tuple[np.ndarray, np.ndarray]]
    base_twist: np.ndarray
    rate_limited: bool
    hand_clamped: bool
    joint_clamped: bool
    joint_clamps: dict[str, tuple[float, float]]
    base_clamped: bool


class ActionAdapter:
    """Validate physical 27-D actions, expand hands, and apply conservative slew limits."""

    def __init__(
        self,
        recorder_config: RecorderConfig,
        joint_limits: Mapping[str, JointLimit],
        safety: ActionSafetyConfig | None = None,
    ) -> None:
        validate_recorder_contract(recorder_config)
        self._synergies = recorder_config.robot_features.hand_synergies
        self._joint_limits = dict(joint_limits)
        self._safety = safety or ActionSafetyConfig()
        self._safety.validate()
        required = set(recorder_config.robot_features.joint_names)
        missing = sorted(required - self._joint_limits.keys())
        if missing:
            raise ValueError(f"URDF position limits missing for: {', '.join(missing)}")
        invalid = sorted(
            name
            for name in required
            if not math.isfinite(self._joint_limits[name].lower)
            or not math.isfinite(self._joint_limits[name].upper)
            or self._joint_limits[name].lower > self._joint_limits[name].upper
        )
        if invalid:
            raise ValueError(f"invalid URDF position limits for: {', '.join(invalid)}")
        self._lock = threading.Lock()
        self._previous_body: np.ndarray | None = None
        self._previous_ratios: np.ndarray | None = None
        self._previous_base = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        with self._lock:
            self._previous_body = None
            self._previous_ratios = None
            self._previous_base = np.zeros(3, dtype=np.float64)

    @staticmethod
    def validate_chunk(action_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIMENSION:
            raise ActionValidationError(
                f"postprocessed chunk must have shape [T, {ACTION_DIMENSION}], got {chunk.shape}"
            )
        if chunk.shape[0] < 1:
            raise ActionValidationError("postprocessed chunk is empty")
        if not np.all(np.isfinite(chunk)):
            raise ActionValidationError("postprocessed chunk contains NaN or Inf")
        return chunk

    def adapt(
        self,
        physical_action: np.ndarray,
        measured_state: np.ndarray,
        *,
        cycle_seconds: float,
    ) -> AdaptedAction:
        action = np.asarray(physical_action, dtype=np.float64).reshape(-1)
        state = np.asarray(measured_state, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIMENSION,):
            raise ActionValidationError(
                f"postprocessed action shape is {action.shape}, expected ({ACTION_DIMENSION},)"
            )
        if state.shape != (ACTION_DIMENSION,):
            raise ActionValidationError(
                f"measured state shape is {state.shape}, expected ({ACTION_DIMENSION},)"
            )
        if not np.all(np.isfinite(action)):
            raise ActionValidationError("postprocessed action contains NaN or Inf")
        if not np.all(np.isfinite(state)):
            raise ActionValidationError("measured state contains NaN or Inf")
        if not math.isfinite(cycle_seconds) or cycle_seconds <= 0.0:
            raise ActionValidationError("cycle_seconds must be finite and positive")

        raw_body = action[:20].copy()
        body, joint_clamps = self._clip_named_positions(BODY_JOINT_NAMES, raw_body)
        ratios = action[20:24].copy()
        raw_base = action[24:27].copy()
        hand_clamped = bool(np.any(ratios < 0.0) or np.any(ratios > 1.0))
        # Hand ratios are bounded physical coordinates. Preserve the raw model
        # output in policy_action for diagnostics, but saturate every finite
        # value before slew limiting and six-joint expansion. Non-finite values
        # have already failed above and remain a hard fault.
        ratios = np.clip(ratios, 0.0, 1.0)

        with self._lock:
            body_reference = (
                state[:20].copy() if self._previous_body is None else self._previous_body.copy()
            )
            ratio_reference = (
                state[20:24].copy()
                if self._previous_ratios is None
                else self._previous_ratios.copy()
            )
            base_reference = self._previous_base.copy()

            body_limited = self._limit_body_rate(body, body_reference)
            body_limited, limited_body_clamps = self._clip_named_positions(
                BODY_JOINT_NAMES, body_limited
            )
            joint_clamps.update(limited_body_clamps)
            ratios_limited = np.clip(
                ratios,
                ratio_reference - self._safety.max_hand_ratio_delta_per_cycle,
                ratio_reference + self._safety.max_hand_ratio_delta_per_cycle,
            )
            ratios_limited = np.clip(ratios_limited, 0.0, 1.0)

            base_velocity_limited = np.array(
                (
                    np.clip(
                        raw_base[0],
                        -self._safety.max_base_linear_velocity,
                        self._safety.max_base_linear_velocity,
                    ),
                    np.clip(
                        raw_base[1],
                        -self._safety.max_base_linear_velocity,
                        self._safety.max_base_linear_velocity,
                    ),
                    np.clip(
                        raw_base[2],
                        -self._safety.max_base_angular_velocity,
                        self._safety.max_base_angular_velocity,
                    ),
                ),
                dtype=np.float64,
            )
            acceleration_delta = np.array(
                (
                    self._safety.max_base_linear_acceleration * cycle_seconds,
                    self._safety.max_base_linear_acceleration * cycle_seconds,
                    self._safety.max_base_angular_acceleration * cycle_seconds,
                ),
                dtype=np.float64,
            )
            base_limited = np.clip(
                base_velocity_limited,
                base_reference - acceleration_delta,
                base_reference + acceleration_delta,
            )

            self._previous_body = body_limited.copy()
            self._previous_ratios = ratios_limited.copy()
            self._previous_base = base_limited.copy()

        component_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        offset = 0
        for component, names in COMPONENT_JOINT_NAMES.items():
            count = len(names)
            values = body_limited[slice(offset, offset + count)].copy()
            component_positions[component] = (
                np.asarray(names, dtype=object),
                values,
            )
            offset += count

        for synergy, ratio_offset in zip(self._synergies, (0, 2), strict=True):
            try:
                positions = expand_hand_synergy(
                    synergy,
                    ratios_limited[ratio_offset],
                    ratios_limited[ratio_offset + 1],
                )
            except HandSynergyError as exc:
                raise ActionValidationError(str(exc)) from exc
            positions, hand_joint_clamps = self._clip_named_positions(
                synergy.joint_names, positions
            )
            joint_clamps.update(hand_joint_clamps)
            component_positions[f"{synergy.side}_hand"] = (
                np.asarray(synergy.joint_names, dtype=object),
                positions.astype(np.float64, copy=False),
            )

        rate_limited = not (
            np.array_equal(body, body_limited) and np.array_equal(ratios, ratios_limited)
        )
        base_clamped = not np.array_equal(raw_base, base_limited)
        return AdaptedAction(
            policy_action=action.copy(),
            component_positions=component_positions,
            base_twist=base_limited,
            rate_limited=rate_limited,
            hand_clamped=hand_clamped,
            joint_clamped=bool(joint_clamps),
            joint_clamps=joint_clamps,
            base_clamped=base_clamped,
        )

    def _limit_body_rate(self, targets: np.ndarray, reference: np.ndarray) -> np.ndarray:
        limits = np.asarray(
            [
                self._safety.max_torso_target_delta_per_cycle
                if index < 3
                else self._safety.max_head_target_delta_per_cycle
                if index < 6
                else self._safety.max_arm_target_delta_per_cycle
                for index in range(20)
            ],
            dtype=np.float64,
        )
        return np.clip(targets, reference - limits, reference + limits)

    def _clip_named_positions(
        self, names: tuple[str, ...], positions: np.ndarray
    ) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.shape != (len(names),):
            raise ActionValidationError(
                f"joint position shape is {values.shape}, expected ({len(names)},)"
            )
        if not np.all(np.isfinite(values)):
            raise ActionValidationError("joint positions contain NaN or Inf")
        clipped = values.copy()
        clamps: dict[str, tuple[float, float]] = {}
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            limit = self._joint_limits[name]
            bounded = float(np.clip(value, limit.lower, limit.upper))
            clipped[index] = bounded
            if value != bounded:
                clamps[name] = (float(value), bounded)
        return clipped, clamps


def load_joint_limits_from_urdf(path: str | Path) -> dict[str, JointLimit]:
    """Load authoritative static position limits from the installed Vega URDF."""
    urdf_path = Path(path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Vega URDF not found: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    result: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is None or upper is None:
            continue
        result[name] = JointLimit(float(lower), float(upper))
    return result


def assert_action_order(names: tuple[str, ...]) -> None:
    if tuple(names) != ACTION_NAMES:
        raise ActionValidationError("unknown action ordering; refusing to adapt policy output")
