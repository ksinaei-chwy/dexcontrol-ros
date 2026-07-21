"""Exact recorder-compatible state and RGB observation construction."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dex_vega_lerobot_recorder.configuration import RecorderConfig
from dex_vega_lerobot_recorder.hand_synergy import (
    HandSynergyError,
    reconstruct_hand_synergy,
)

from .contracts import (
    ACTION_NAMES,
    BODY_JOINT_NAMES,
    HEAD_CAMERA_FEATURE,
    STATE_DIMENSION,
    STATE_NAMES,
    TASK,
)


class ObservationValidationError(RuntimeError):
    """Raised when a fresh, synchronized 27-D observation cannot be built."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))

    @property
    def stale(self) -> bool:
        return any(
            marker in reason
            for reason in self.reasons
            for marker in ("stale", "missing", "timestamp", "synchronization")
        )


@dataclass(frozen=True)
class CameraSample:
    rgb: np.ndarray
    source_stamp_ns: int
    receive_stamp_ns: int
    transport_delay_seconds: float = 0.0


@dataclass(frozen=True)
class ObservationSnapshot:
    state: np.ndarray
    rgb: np.ndarray
    task: str
    state_stamp_ns: int
    camera_stamp_ns: int
    receive_stamp_ns: int
    created_stamp_ns: int
    created_monotonic_ns: int
    state_age_seconds: float
    camera_capture_age_seconds: float
    camera_receive_age_seconds: float
    synchronization_skew_seconds: float

    def as_policy_observation(self) -> dict[str, Any]:
        return {
            "observation.state": self.state.copy(),
            HEAD_CAMERA_FEATURE: self.rgb.copy(),
        }


@dataclass(frozen=True)
class _JointSample:
    positions: dict[str, float]
    stamp_ns: int


@dataclass(frozen=True)
class _BaseSample:
    values: tuple[float, float, float]
    stamp_ns: int


class ObservationAdapter:
    """Latest-sample adapter using the recorder's feature config and hand projection."""

    def __init__(self, recorder_config: RecorderConfig) -> None:
        validate_recorder_contract(recorder_config)
        self._features = recorder_config.robot_features
        self._lock = threading.Lock()
        self._joints: _JointSample | None = None
        self._base: _BaseSample | None = None
        self._last_camera_stamp_ns: int | None = None

    def reset(self) -> None:
        with self._lock:
            self._joints = None
            self._base = None
            self._last_camera_stamp_ns = None

    def update_measured_joints(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        stamp_ns: int,
    ) -> None:
        if stamp_ns <= 0:
            raise ValueError("joint sample timestamp must be positive")
        if not names or len(names) != len(set(names)):
            raise ValueError("joint sample names must be non-empty and unique")
        if len(names) != len(positions):
            raise ValueError("joint names and positions differ in length")
        if not all(math.isfinite(float(value)) for value in positions):
            raise ValueError("joint positions contain non-finite values")
        sample = _JointSample(
            positions={name: float(value) for name, value in zip(names, positions)},
            stamp_ns=int(stamp_ns),
        )
        with self._lock:
            if self._joints is not None and sample.stamp_ns <= self._joints.stamp_ns:
                raise ValueError("joint sample timestamp is duplicate or out of order")
            self._joints = sample

    def update_measured_base(self, values: Sequence[float], stamp_ns: int) -> None:
        if stamp_ns <= 0:
            raise ValueError("base sample timestamp must be positive")
        if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("base sample must contain three finite values")
        sample = _BaseSample(tuple(float(value) for value in values), int(stamp_ns))
        with self._lock:
            if self._base is not None and sample.stamp_ns <= self._base.stamp_ns:
                raise ValueError("base sample timestamp is duplicate or out of order")
            self._base = sample

    def snapshot(
        self,
        camera: CameraSample,
        now_ns: int,
        *,
        maximum_state_age_seconds: float,
        maximum_receive_age_seconds: float,
        maximum_capture_age_seconds: float,
        maximum_transport_delay_seconds: float,
        maximum_synchronization_skew_seconds: float,
    ) -> ObservationSnapshot:
        _validate_camera(camera)
        if now_ns <= 0:
            raise ObservationValidationError(["current timestamp must be positive"])
        with self._lock:
            joints = self._joints
            base = self._base
            last_camera_stamp_ns = self._last_camera_stamp_ns

        reasons: list[str] = []
        if joints is None:
            reasons.append("missing measured joint state")
        if base is None:
            reasons.append("missing measured base velocity")
        if last_camera_stamp_ns is not None and camera.source_stamp_ns <= last_camera_stamp_ns:
            reasons.append("camera timestamp is duplicate or out of order")
        if reasons:
            raise ObservationValidationError(reasons)
        assert joints is not None and base is not None

        joint_age = _age_seconds(now_ns, joints.stamp_ns)
        base_age = _age_seconds(now_ns, base.stamp_ns)
        state_age = max(joint_age, base_age)
        capture_age = _age_seconds(now_ns, camera.source_stamp_ns)
        receive_age = _age_seconds(now_ns, camera.receive_stamp_ns)
        transport_delay = float(camera.transport_delay_seconds)
        stamps = (joints.stamp_ns, base.stamp_ns, camera.source_stamp_ns)
        sync_skew = (max(stamps) - min(stamps)) / 1e9

        for label, age in (
            ("joint state", joint_age),
            ("base velocity", base_age),
            ("camera capture", capture_age),
            ("camera receive", receive_age),
        ):
            if age < 0.0:
                reasons.append(f"{label} timestamp is in the future")
        if state_age > maximum_state_age_seconds:
            reasons.append(
                f"stale measured state ({state_age:.3f}s > {maximum_state_age_seconds:.3f}s)"
            )
        if capture_age > maximum_capture_age_seconds:
            reasons.append(
                f"stale camera capture ({capture_age:.3f}s > "
                f"{maximum_capture_age_seconds:.3f}s)"
            )
        if receive_age > maximum_receive_age_seconds:
            reasons.append(
                f"stale camera receive ({receive_age:.3f}s > "
                f"{maximum_receive_age_seconds:.3f}s)"
            )
        if transport_delay < 0.0:
            reasons.append("camera transport delay is negative")
        elif transport_delay > maximum_transport_delay_seconds:
            reasons.append(
                f"camera transport delay exceeds limit ({transport_delay:.3f}s > "
                f"{maximum_transport_delay_seconds:.3f}s)"
            )
        if sync_skew > maximum_synchronization_skew_seconds:
            reasons.append(
                f"state/image synchronization skew exceeds limit ({sync_skew:.3f}s > "
                f"{maximum_synchronization_skew_seconds:.3f}s)"
            )
        if reasons:
            raise ObservationValidationError(reasons)

        try:
            state_values = [joints.positions[name] for name in self._features.body_joint_names]
            for synergy in self._features.hand_synergies:
                state_values.extend(
                    reconstruct_hand_synergy(
                        synergy,
                        joints.positions,
                        require_exact_action=False,
                    )
                )
            state_values.extend(base.values)
        except KeyError as exc:
            raise ObservationValidationError(
                [f"required measured joint is missing: {exc.args[0]}"]
            ) from exc
        except HandSynergyError as exc:
            raise ObservationValidationError([str(exc)]) from exc

        state = np.asarray(state_values, dtype=np.float32)
        if state.shape != (STATE_DIMENSION,):
            raise ObservationValidationError(
                [f"state shape is {state.shape}, expected ({STATE_DIMENSION},)"]
            )
        if not np.all(np.isfinite(state)):
            raise ObservationValidationError(["state contains non-finite values"])

        with self._lock:
            self._last_camera_stamp_ns = camera.source_stamp_ns
        return ObservationSnapshot(
            state=state,
            rgb=np.ascontiguousarray(camera.rgb),
            task=TASK,
            state_stamp_ns=min(joints.stamp_ns, base.stamp_ns),
            camera_stamp_ns=camera.source_stamp_ns,
            receive_stamp_ns=camera.receive_stamp_ns,
            created_stamp_ns=int(now_ns),
            created_monotonic_ns=time.monotonic_ns(),
            state_age_seconds=state_age,
            camera_capture_age_seconds=capture_age,
            camera_receive_age_seconds=receive_age,
            synchronization_skew_seconds=sync_skew,
        )


def validate_recorder_contract(config: RecorderConfig) -> None:
    """Ensure runtime config remains byte-for-byte compatible in feature ordering."""
    features = config.robot_features
    if features.include_joint_velocities:
        raise ValueError("trained state excludes joint velocities")
    if tuple(features.body_joint_names) != BODY_JOINT_NAMES:
        raise ValueError("recorder body joint ordering differs from the trained contract")
    if tuple(features.state_names) != STATE_NAMES:
        raise ValueError("recorder state ordering differs from the trained contract")
    if tuple(features.action_names) != ACTION_NAMES:
        raise ValueError("recorder action ordering differs from the trained contract")
    if config.dataset.task_instruction != TASK:
        raise ValueError("recorder task text differs from the trained task")
    if config.head_camera.resolution.shape != (480, 640, 3):
        raise ValueError("trained head camera must be uint8 RGB 640x480")
    if len(features.hand_synergies) != 2:
        raise ValueError("trained contract requires left and right hand synergies")


def image_message_to_rgb(message: Any) -> np.ndarray:
    """Convert a ROS Image-like message to contiguous RGB without cv_bridge."""
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0:
        raise ObservationValidationError(["image dimensions must be positive"])
    if encoding not in {"rgb8", "bgr8"}:
        raise ObservationValidationError(
            [f"unsupported replay image encoding '{message.encoding}'; use rgb8 or bgr8"]
        )
    minimum_step = width * 3
    if step < minimum_step:
        raise ObservationValidationError(["image step is smaller than width*3"])
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size < height * step:
        raise ObservationValidationError(["image data is shorter than height*step"])
    rows = data[: height * step].reshape(height, step)
    image = rows[:, :minimum_step].reshape(height, width, 3)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _validate_camera(camera: CameraSample) -> None:
    if not isinstance(camera.rgb, np.ndarray):
        raise ObservationValidationError(["camera frame is not a numpy array"])
    if camera.rgb.dtype != np.uint8 or camera.rgb.shape != (480, 640, 3):
        raise ObservationValidationError(
            [
                "expected uint8 RGB camera shape (480, 640, 3), got "
                f"dtype={camera.rgb.dtype}, shape={camera.rgb.shape}"
            ]
        )
    if camera.source_stamp_ns <= 0 or camera.receive_stamp_ns <= 0:
        raise ObservationValidationError(["camera timestamps must be positive"])


def _age_seconds(now_ns: int, stamp_ns: int) -> float:
    return (int(now_ns) - int(stamp_ns)) / 1e9
