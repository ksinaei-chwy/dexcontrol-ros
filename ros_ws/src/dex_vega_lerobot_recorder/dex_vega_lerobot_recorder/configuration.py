"""Typed recorder configuration and startup validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


CAMERA_FEATURES = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
UPLOAD_POLICIES = {"manual", "each_episode", "on_session_end"}
INPUT_BACKENDS = {"disabled", "terminal", "linux_input_event"}
CAMERA_TRANSPORTS = {"zenoh", "rtc"}
CAMERA_CODECS = {"auto", "h264", "vp8"}


class ConfigurationError(ValueError):
    """Raised when recording configuration is unsafe or inconsistent."""


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    stream_name: str
    transport: str
    topic: str
    rtc_channel: str
    codec: str
    placeholder_enabled: bool
    resolution: Resolution


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    local_save_directory: Path
    recording_fps: int
    task_instruction: str
    robot_type: str
    use_videos: bool


@dataclass(frozen=True)
class HuggingFaceConfig:
    upload_enabled: bool
    namespace: str
    repo_id: str
    private: bool
    upload_policy: str


@dataclass(frozen=True)
class EpisodeControlConfig:
    start_key: str
    stop_key: str
    save_key: str
    discard_key: str
    debounce_seconds: float
    minimum_frames: int
    minimum_duration_seconds: float
    input_backend: str
    input_device: str
    autosave_on_shutdown: bool


@dataclass(frozen=True)
class ValidationConfig:
    maximum_state_age_seconds: float
    maximum_action_age_seconds: float
    maximum_receive_age_seconds: float
    maximum_capture_age_seconds: float
    maximum_transport_delay_seconds: float


@dataclass(frozen=True)
class TopicConfig:
    joint_states: str
    applied_joint_commands: str
    measured_base_twist: str
    applied_base_twist: str


@dataclass(frozen=True)
class HandSynergyConfig:
    side: str
    joint_names: tuple[str, ...]
    open_positions: tuple[float, ...]
    closed_positions: tuple[float, ...]
    action_ratio_tolerance: float

    @property
    def ratio_names(self) -> tuple[str, str]:
        return (
            f"{self.side}_hand.open_close_ratio",
            f"{self.side}_hand.thumb_opposition_ratio",
        )


@dataclass(frozen=True)
class RobotFeatureConfig:
    joint_names: tuple[str, ...]
    include_joint_velocities: bool
    hand_synergies: tuple[HandSynergyConfig, ...]

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name for synergy in self.hand_synergies for name in synergy.joint_names
        )

    @property
    def body_joint_names(self) -> tuple[str, ...]:
        hand_names = set(self.hand_joint_names)
        return tuple(name for name in self.joint_names if name not in hand_names)

    @property
    def hand_ratio_names(self) -> tuple[str, ...]:
        return tuple(name for synergy in self.hand_synergies for name in synergy.ratio_names)

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.body_joint_names + self.hand_ratio_names + (
            "base_vx",
            "base_vy",
            "base_wz",
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        names = tuple(f"{name}.position" for name in self.body_joint_names)
        names += self.hand_ratio_names
        if self.include_joint_velocities:
            names += tuple(f"{name}.velocity" for name in self.body_joint_names)
        return names + ("base_vx", "base_vy", "base_wz")


@dataclass(frozen=True)
class WriterConfig:
    frame_queue_size: int
    image_writer_threads: int
    streaming_encoding: bool


@dataclass(frozen=True)
class RecorderConfig:
    source_path: Path
    source_data: dict[str, Any]
    dataset: DatasetConfig
    hugging_face: HuggingFaceConfig
    head_camera: CameraConfig
    left_wrist_camera: CameraConfig
    right_wrist_camera: CameraConfig
    episode_control: EpisodeControlConfig
    validation: ValidationConfig
    topics: TopicConfig
    robot_features: RobotFeatureConfig
    writer: WriterConfig

    @property
    def local_dataset_path(self) -> Path:
        return self.dataset.local_save_directory / self.dataset.name

    @property
    def repo_id(self) -> str:
        explicit = self.hugging_face.repo_id.strip()
        if explicit:
            return explicit
        namespace = self.hugging_face.namespace.strip() or "local"
        return f"{namespace}/{self.dataset.name}"

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        head_shape = self.head_camera.resolution.shape
        shapes: dict[str, tuple[int, int, int]] = {}
        for key, camera in self.camera_configs.items():
            if not camera.enabled:
                continue
            shapes[key] = (
                head_shape
                if camera.placeholder_enabled
                else camera.resolution.shape
            )
        return shapes

    @property
    def camera_configs(self) -> dict[str, CameraConfig]:
        """Map stable dataset feature names to their configured camera sources."""
        return {
            CAMERA_FEATURES[0]: self.head_camera,
            CAMERA_FEATURES[1]: self.left_wrist_camera,
            CAMERA_FEATURES[2]: self.right_wrist_camera,
        }

    def lerobot_features(self) -> dict[str, dict[str, Any]]:
        visual_dtype = "video" if self.dataset.use_videos else "image"
        features: dict[str, dict[str, Any]] = {}
        for key, shape in self.camera_shapes.items():
            features[key] = {
                "dtype": visual_dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(self.robot_features.state_names),),
            "names": {"state": list(self.robot_features.state_names)},
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(self.robot_features.action_names),),
            "names": {"action": list(self.robot_features.action_names)},
        }
        return features

    def with_overrides(
        self,
        *,
        no_hf_upload: bool = False,
        input_backend: str | None = None,
        input_device: str | None = None,
    ) -> "RecorderConfig":
        from dataclasses import replace

        hf = self.hugging_face
        if no_hf_upload:
            hf = replace(hf, upload_enabled=False)
        controls = self.episode_control
        if input_backend is not None and input_backend != "":
            controls = replace(controls, input_backend=input_backend)
        if input_device is not None and input_device != "":
            controls = replace(controls, input_device=input_device)
        updated = replace(self, hugging_face=hf, episode_control=controls)
        validate_config(updated)
        return updated


def load_config(path: str | Path) -> RecorderConfig:
    source_path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")

    dataset_raw = _mapping(raw, "dataset")
    hf_raw = _mapping(raw, "hugging_face")
    cameras_raw = _mapping(raw, "cameras")
    controls_raw = _mapping(raw, "episode_control")
    validation_raw = _mapping(raw, "validation")
    topics_raw = _mapping(raw, "topics")
    features_raw = _mapping(raw, "robot_features")
    writer_raw = raw.get("writer", {})
    if not isinstance(writer_raw, dict):
        raise ConfigurationError("writer must be a mapping")

    save_directory = Path(
        os.path.expandvars(
            os.path.expanduser(str(_required(dataset_raw, "local_save_directory")))
        )
    )
    config = RecorderConfig(
        source_path=source_path,
        source_data=raw,
        dataset=DatasetConfig(
            name=str(_required(dataset_raw, "name")).strip(),
            local_save_directory=save_directory,
            recording_fps=int(_required(dataset_raw, "recording_fps")),
            task_instruction=str(_required(dataset_raw, "task_instruction")).strip(),
            robot_type=str(dataset_raw.get("robot_type", "dexmate_vega_1_pro")).strip(),
            use_videos=bool(dataset_raw.get("use_videos", True)),
        ),
        hugging_face=HuggingFaceConfig(
            upload_enabled=bool(hf_raw.get("upload_enabled", False)),
            namespace=str(hf_raw.get("namespace", "")).strip(),
            repo_id=str(hf_raw.get("repo_id", "")).strip(),
            private=bool(hf_raw.get("private", True)),
            upload_policy=str(hf_raw.get("upload_policy", "on_session_end")).strip(),
        ),
        head_camera=_camera_config(cameras_raw, "head", placeholder_default=False),
        left_wrist_camera=_camera_config(
            cameras_raw, "left_wrist", placeholder_default=True
        ),
        right_wrist_camera=_camera_config(
            cameras_raw, "right_wrist", placeholder_default=True
        ),
        episode_control=EpisodeControlConfig(
            start_key=str(controls_raw.get("start_key", "a")),
            stop_key=str(controls_raw.get("stop_key", "b")),
            save_key=str(controls_raw.get("save_key", "c")),
            discard_key=str(controls_raw.get("discard_key", "d")),
            debounce_seconds=float(controls_raw.get("debounce_seconds", 0.25)),
            minimum_frames=int(controls_raw.get("minimum_frames", 10)),
            minimum_duration_seconds=float(
                controls_raw.get("minimum_duration_seconds", 0.5)
            ),
            input_backend=str(controls_raw.get("input_backend", "terminal")),
            input_device=str(controls_raw.get("input_device", "")),
            autosave_on_shutdown=bool(
                controls_raw.get("autosave_on_shutdown", False)
            ),
        ),
        validation=ValidationConfig(
            maximum_state_age_seconds=float(
                _required(validation_raw, "maximum_state_age_seconds")
            ),
            maximum_action_age_seconds=float(
                _required(validation_raw, "maximum_action_age_seconds")
            ),
            maximum_receive_age_seconds=float(
                _required(validation_raw, "maximum_receive_age_seconds")
            ),
            maximum_capture_age_seconds=float(
                _required(validation_raw, "maximum_capture_age_seconds")
            ),
            maximum_transport_delay_seconds=float(
                _required(validation_raw, "maximum_transport_delay_seconds")
            ),
        ),
        topics=TopicConfig(
            joint_states=str(_required(topics_raw, "joint_states")),
            applied_joint_commands=str(
                _required(topics_raw, "applied_joint_commands")
            ),
            measured_base_twist=str(_required(topics_raw, "measured_base_twist")),
            applied_base_twist=str(_required(topics_raw, "applied_base_twist")),
        ),
        robot_features=RobotFeatureConfig(
            joint_names=tuple(str(name) for name in _required(features_raw, "joint_names")),
            include_joint_velocities=bool(
                features_raw.get("include_joint_velocities", True)
            ),
            hand_synergies=tuple(
                _hand_synergy_config(features_raw, side)
                for side in ("left", "right")
            ),
        ),
        writer=WriterConfig(
            frame_queue_size=int(writer_raw.get("frame_queue_size", 64)),
            image_writer_threads=int(writer_raw.get("image_writer_threads", 4)),
            streaming_encoding=bool(writer_raw.get("streaming_encoding", False)),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: RecorderConfig) -> None:
    dataset = config.dataset
    if not dataset.name or "/" in dataset.name or dataset.name in {".", ".."}:
        raise ConfigurationError("dataset.name must be a non-empty directory name")
    if not dataset.local_save_directory.is_absolute():
        raise ConfigurationError("dataset.local_save_directory must expand to an absolute path")
    if dataset.recording_fps <= 0:
        raise ConfigurationError("dataset.recording_fps must be positive")
    if not dataset.task_instruction:
        raise ConfigurationError("dataset.task_instruction must not be empty")
    if dataset.robot_type != "dexmate_vega_1_pro":
        raise ConfigurationError("dataset.robot_type must be dexmate_vega_1_pro")

    if config.hugging_face.upload_policy not in UPLOAD_POLICIES:
        raise ConfigurationError(
            f"hugging_face.upload_policy must be one of {sorted(UPLOAD_POLICIES)}"
        )
    if (
        config.hugging_face.upload_enabled
        and not config.hugging_face.repo_id
        and config.hugging_face.namespace in {"", "local"}
    ):
        raise ConfigurationError(
            "Hub upload requires a real hugging_face.namespace or explicit repo_id"
        )
    if config.hugging_face.upload_enabled and "/" not in config.repo_id:
        raise ConfigurationError("an upload repo_id must be namespace/dataset_name")

    for name, camera in (
        ("head", config.head_camera),
        ("left_wrist", config.left_wrist_camera),
        ("right_wrist", config.right_wrist_camera),
    ):
        if camera.resolution.width <= 0 or camera.resolution.height <= 0:
            raise ConfigurationError(f"cameras.{name}.resolution must be positive")
        if name == "head" and not camera.enabled:
            raise ConfigurationError("cameras.head.enabled must be true")
        if name == "head" and camera.placeholder_enabled:
            raise ConfigurationError("cameras.head cannot use placeholder mode")
        if not camera.enabled:
            continue
        if not camera.placeholder_enabled:
            if not camera.stream_name:
                raise ConfigurationError(f"cameras.{name}.stream_name must not be empty")
            if camera.transport not in CAMERA_TRANSPORTS:
                raise ConfigurationError(
                    f"cameras.{name}.transport must be one of "
                    f"{sorted(CAMERA_TRANSPORTS)}"
                )
            if camera.codec not in CAMERA_CODECS:
                raise ConfigurationError(
                    f"cameras.{name}.codec must be one of {sorted(CAMERA_CODECS)}"
                )
            if camera.transport == "zenoh" and not camera.topic:
                raise ConfigurationError(
                    f"cameras.{name}.topic is required for Zenoh transport"
                )
            if camera.transport == "rtc" and not camera.rtc_channel:
                raise ConfigurationError(
                    f"cameras.{name}.rtc_channel is required for RTC transport"
                )

    controls = config.episode_control
    keys = [controls.start_key, controls.stop_key, controls.save_key, controls.discard_key]
    if any(len(key) != 1 for key in keys) or len(set(keys)) != len(keys):
        raise ConfigurationError("episode control keys must be unique single characters")
    if controls.debounce_seconds < 0.0:
        raise ConfigurationError("episode_control.debounce_seconds cannot be negative")
    if controls.minimum_frames <= 0 or controls.minimum_duration_seconds < 0.0:
        raise ConfigurationError("episode minimums must be positive/non-negative")
    if controls.input_backend not in INPUT_BACKENDS:
        raise ConfigurationError(
            f"episode_control.input_backend must be one of {sorted(INPUT_BACKENDS)}"
        )
    if controls.input_backend == "linux_input_event" and not controls.input_device:
        raise ConfigurationError("linux_input_event requires episode_control.input_device")

    ages = (
        config.validation.maximum_state_age_seconds,
        config.validation.maximum_action_age_seconds,
        config.validation.maximum_receive_age_seconds,
        config.validation.maximum_capture_age_seconds,
        config.validation.maximum_transport_delay_seconds,
    )
    if any(age <= 0.0 for age in ages):
        raise ConfigurationError("all validation maximum ages must be positive")
    if any(not topic.startswith("/") for topic in vars(config.topics).values()):
        raise ConfigurationError("all configured ROS topics must be absolute")

    names = config.robot_features.joint_names
    if not names or len(names) != len(set(names)):
        raise ConfigurationError("robot_features.joint_names must be non-empty and unique")
    if len(names) != 32:
        raise ConfigurationError("robot_features.joint_names must contain 32 Vega joints")
    synergy_sides = tuple(synergy.side for synergy in config.robot_features.hand_synergies)
    if synergy_sides != ("left", "right"):
        raise ConfigurationError("hand synergies must be ordered left then right")
    hand_names = config.robot_features.hand_joint_names
    if len(hand_names) != 12 or len(set(hand_names)) != len(hand_names):
        raise ConfigurationError("hand synergies must contain 12 unique driver joints")
    if not set(hand_names).issubset(names):
        raise ConfigurationError("hand synergy joints must be present in joint_names")
    if len(config.robot_features.body_joint_names) != 20:
        raise ConfigurationError("compact Vega schema must contain 20 non-hand joints")
    for synergy in config.robot_features.hand_synergies:
        if len(synergy.joint_names) != 6:
            raise ConfigurationError(f"{synergy.side} hand synergy must contain six joints")
        if len(synergy.open_positions) != 6 or len(synergy.closed_positions) != 6:
            raise ConfigurationError(f"{synergy.side} hand endpoints must contain six values")
        endpoints = synergy.open_positions + synergy.closed_positions
        if not all(math.isfinite(value) for value in endpoints):
            raise ConfigurationError(f"{synergy.side} hand endpoints must be finite")
        if any(
            math.isclose(opened, closed, abs_tol=1.0e-12)
            for opened, closed in zip(
                synergy.open_positions, synergy.closed_positions
            )
        ):
            raise ConfigurationError(
                f"{synergy.side} hand open/closed endpoints must differ"
            )
        if not 0.0 < synergy.action_ratio_tolerance <= 0.25:
            raise ConfigurationError(
                f"{synergy.side} action_ratio_tolerance must be in (0, 0.25]"
            )
    feature_keys = set(config.lerobot_features())
    expected_keys = set(config.camera_shapes) | {"observation.state", "action"}
    if feature_keys != expected_keys:
        raise ConfigurationError("dataset feature names are missing or duplicated")
    if config.writer.frame_queue_size <= 0 or config.writer.image_writer_threads < 0:
        raise ConfigurationError("writer queue/thread values are invalid")


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(parent, key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required(parent: dict[str, Any], key: str) -> Any:
    if key not in parent:
        raise ConfigurationError(f"missing required configuration key: {key}")
    return parent[key]


def _camera_config(
    cameras: dict[str, Any], name: str, *, placeholder_default: bool
) -> CameraConfig:
    raw = _mapping(cameras, name)
    resolution_raw = _mapping(raw, "resolution")
    return CameraConfig(
        enabled=bool(raw.get("enabled", True)),
        stream_name=str(raw.get("stream_name", name)).strip(),
        transport=str(raw.get("transport", "zenoh")).strip().lower(),
        topic=str(raw.get("topic", "")).strip(),
        rtc_channel=str(raw.get("rtc_channel", "")).strip(),
        codec=str(raw.get("codec", "auto")).strip().lower(),
        placeholder_enabled=bool(
            raw.get("placeholder_enabled", placeholder_default)
        ),
        resolution=Resolution(
            width=int(_required(resolution_raw, "width")),
            height=int(_required(resolution_raw, "height")),
        ),
    )


def _hand_synergy_config(
    robot_features: dict[str, Any], side: str
) -> HandSynergyConfig:
    synergies = _mapping(robot_features, "hand_synergies")
    raw = _mapping(synergies, side)
    return HandSynergyConfig(
        side=side,
        joint_names=tuple(str(name) for name in _required(raw, "joint_names")),
        open_positions=tuple(float(value) for value in _required(raw, "open_positions")),
        closed_positions=tuple(
            float(value) for value in _required(raw, "closed_positions")
        ),
        action_ratio_tolerance=float(raw.get("action_ratio_tolerance", 0.02)),
    )
