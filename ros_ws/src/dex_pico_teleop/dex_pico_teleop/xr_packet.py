"""Typed representation of the minimal Pico JSON stream."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dex_pico_teleop.transforms import Pose, pose_openxr_to_robot


@dataclass(frozen=True)
class ControllerInput:
    pose: Pose
    trigger: float = 0.0
    grip: float = 0.0
    joystick: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    buttons: dict[str, bool] = field(default_factory=dict)

    def button(self, name: str) -> bool:
        return bool(self.buttons.get(name.lower(), False))


@dataclass(frozen=True)
class TrackerInput:
    pose: Pose
    confidence: float = 1.0


@dataclass(frozen=True)
class PicoPacket:
    timestamp_ns: int
    frame: str
    head: Pose
    controllers: dict[str, ControllerInput]
    trackers: dict[str, TrackerInput]
    sequence: int | None = None

    @classmethod
    def from_json_bytes(cls, payload: bytes | str) -> "PicoPacket":
        if isinstance(payload, bytes):
            text = payload.decode("utf-8")
        else:
            text = payload
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_xrobotoolkit_tracking(cls, tracking: dict[str, Any]) -> "PicoPacket":
        """Build a packet from XRoboToolkit's "Tracking" function payload."""
        controllers_data = tracking.get("Controller", tracking.get("controller", {}))
        motion_data = tracking.get("Motion", tracking.get("motion", {}))
        data = {
            "timestamp_ns": int(tracking.get("timeStampNs", time.time_ns())),
            "frame": "openxr_y_up",
            "head": {"pose": _extract_xrobot_pose(tracking.get("Head", tracking.get("head", {})))},
            "controllers": {
                "left": _xrobot_controller(controllers_data.get("left", {}), "left"),
                "right": _xrobot_controller(controllers_data.get("right", {}), "right"),
            },
            "trackers": _xrobot_trackers(motion_data),
        }
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PicoPacket":
        frame = str(data.get("frame", "openxr_y_up")).lower()
        timestamp_ns = int(data.get("timestamp_ns", time.time_ns()))
        sequence = data.get("sequence")
        head = _extract_pose(data.get("head", {}))

        controllers_data = data.get("controllers", {})
        controllers = {
            "left": _extract_controller(controllers_data.get("left", {})),
            "right": _extract_controller(controllers_data.get("right", {})),
        }

        trackers_data = data.get("trackers", {})
        trackers = {
            name: _extract_tracker(trackers_data.get(name, {}))
            for name in ("left_ankle", "right_ankle")
            if name in trackers_data
        }

        if frame in {"openxr", "openxr_y_up", "pico", "xr"}:
            head = pose_openxr_to_robot(head)
            controllers = {
                side: ControllerInput(
                    pose=pose_openxr_to_robot(controller.pose),
                    trigger=controller.trigger,
                    grip=controller.grip,
                    joystick=controller.joystick,
                    buttons=controller.buttons,
                )
                for side, controller in controllers.items()
            }
            trackers = {
                name: TrackerInput(
                    pose=pose_openxr_to_robot(tracker.pose),
                    confidence=tracker.confidence,
                )
                for name, tracker in trackers.items()
            }
            frame = "robot_z_up"
        elif frame not in {"robot", "robot_z_up"}:
            raise ValueError(f"unsupported XR frame '{frame}'")
        else:
            frame = "robot_z_up"

        return cls(
            timestamp_ns=timestamp_ns,
            frame=frame,
            head=head,
            controllers=controllers,
            trackers=trackers,
            sequence=int(sequence) if sequence is not None else None,
        )


def _extract_pose(raw: Any) -> Pose:
    if isinstance(raw, dict):
        if "pose" in raw:
            return Pose.from_list(raw["pose"])
        if "position" in raw and "orientation" in raw:
            return Pose.from_list(list(raw["position"]) + list(raw["orientation"]))
    if isinstance(raw, str):
        return Pose.from_list(_float_csv(raw))
    if isinstance(raw, (list, tuple, np.ndarray)):
        return Pose.from_list(raw)
    return Pose.identity()


def _extract_controller(raw: dict[str, Any]) -> ControllerInput:
    pose = _extract_pose(raw)
    trigger = _clamp01(raw.get("trigger", raw.get("index_trigger", 0.0)))
    grip = _clamp01(raw.get("grip", raw.get("grip_trigger", 0.0)))
    joystick = np.asarray(raw.get("joystick", raw.get("thumbstick", [0.0, 0.0])), dtype=np.float64)
    if joystick.size != 2 or not np.all(np.isfinite(joystick)):
        joystick = np.zeros(2, dtype=np.float64)
    buttons = _normalize_buttons(raw.get("buttons", {}))
    for key in ("a", "b", "x", "y", "menu", "stick", "thumbstick_click"):
        if key in raw:
            buttons[key] = bool(raw[key])
    return ControllerInput(
        pose=pose,
        trigger=trigger,
        grip=grip,
        joystick=joystick.reshape(2),
        buttons=buttons,
    )


def _extract_tracker(raw: dict[str, Any]) -> TrackerInput:
    pose = _extract_pose(raw)
    confidence = _clamp01(raw.get("confidence", 1.0))
    return TrackerInput(pose=pose, confidence=confidence)


def _normalize_buttons(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    return {str(key).lower(): bool(value) for key, value in raw.items()}


def _clamp01(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return float(np.clip(result, 0.0, 1.0))


def _xrobot_controller(raw: dict[str, Any], side: str) -> dict[str, Any]:
    axis_click = bool(raw.get("axisClick", False))
    primary = bool(raw.get("primaryButton", False))
    secondary = bool(raw.get("secondaryButton", False))
    buttons = {
        "stick": axis_click,
        "thumbstick_click": axis_click,
        "primary": primary,
        "secondary": secondary,
        "menu": bool(raw.get("menuButton", False)),
    }
    if side == "left":
        buttons.update({"x": primary, "y": secondary})
    else:
        buttons.update({"a": primary, "b": secondary})
    return {
        "pose": _extract_xrobot_pose(raw),
        "trigger": raw.get("trigger", 0.0),
        "grip": raw.get("grip", 0.0),
        "joystick": [raw.get("axisX", 0.0), raw.get("axisY", 0.0)],
        "buttons": buttons,
    }


def _xrobot_trackers(raw: dict[str, Any]) -> dict[str, Any]:
    joints = raw.get("joints", []) if isinstance(raw, dict) else []
    trackers: dict[str, Any] = {}
    for name, joint in zip(("left_ankle", "right_ankle"), joints):
        trackers[name] = {
            "pose": _extract_xrobot_pose(joint),
            "confidence": 1.0,
        }
    return trackers


def _extract_xrobot_pose(raw: Any) -> list[float]:
    if isinstance(raw, dict):
        raw = raw.get("pose", raw.get("p", raw))
    if isinstance(raw, str):
        return _float_csv(raw)
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [float(value) for value in raw]
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _float_csv(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",")]
    if len(values) != 7:
        raise ValueError(f"pose CSV must contain 7 values, got {len(values)}")
    return values
