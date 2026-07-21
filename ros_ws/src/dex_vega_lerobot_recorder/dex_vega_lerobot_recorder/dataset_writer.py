"""LeRobotDataset v3 writer adapter with local-first upload semantics."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .configuration import RecorderConfig


class DatasetDependencyError(RuntimeError):
    """Raised when the installed LeRobot package cannot provide the writer API."""


class ExistingDatasetError(RuntimeError):
    """Raised when a dataset path exists without an explicit safe policy."""


@dataclass(frozen=True)
class CommitResult:
    episode_index: int
    local_path: Path


class LeRobotDatasetWriter:
    """Small testable facade around the installed LeRobotDataset implementation."""

    def __init__(
        self,
        config: RecorderConfig,
        *,
        overwrite: bool = False,
        resume: bool = False,
        dataset_class: type | None = None,
        log_info: Callable[[str], None] | None = None,
        log_warn: Callable[[str], None] | None = None,
    ) -> None:
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        self.config = config
        self.local_path = config.local_dataset_path
        self.repo_id = config.repo_id
        self._log_info = log_info or (lambda _message: None)
        self._log_warn = log_warn or (lambda _message: None)
        self._dataset_class = dataset_class or _load_lerobot_dataset_class()
        self._finalized = False

        self._prepare_local_path(overwrite=overwrite, resume=resume)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            self._dataset = self._resume_dataset()
            self._validate_resumed_dataset()
        else:
            self._dataset = self._create_dataset()
        self._write_recording_metadata()

        if not config.hugging_face.upload_enabled:
            self._log_info(
                f"Hugging Face upload disabled; saving locally only: {self.local_path}"
            )

    @property
    def committed_episodes(self) -> int:
        return _dataset_episode_count(self._dataset)

    @property
    def pending_frames(self) -> int:
        writer = getattr(self._dataset, "writer", None)
        buffer = getattr(writer, "episode_buffer", None)
        if isinstance(buffer, dict):
            return int(buffer.get("size", 0))
        legacy_buffer = getattr(self._dataset, "episode_buffer", None)
        if isinstance(legacy_buffer, dict):
            return int(legacy_buffer.get("size", 0))
        return 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self._validate_frame(frame)
        self._dataset.add_frame(frame)

    def save_episode(self) -> CommitResult:
        before = self.committed_episodes
        self._dataset.save_episode()
        after = self.committed_episodes
        if after != before + 1:
            raise RuntimeError(
                f"LeRobot save_episode changed episode count {before} -> {after}; expected +1"
            )
        result = CommitResult(episode_index=before, local_path=self.local_path)
        if (
            self.config.hugging_face.upload_enabled
            and self.config.hugging_face.upload_policy == "each_episode"
        ):
            self._finalize_upload_and_resume()
        return result

    def clear_episode_buffer(self) -> None:
        clear = getattr(self._dataset, "clear_episode_buffer", None)
        if clear is None:
            raise DatasetDependencyError(
                "installed LeRobotDataset lacks clear_episode_buffer(); "
                "discard-safe recording requires LeRobot >= 0.4"
            )
        clear(delete_images=True)

    def finalize(self) -> None:
        if self._finalized:
            return
        finalize = getattr(self._dataset, "finalize", None)
        if finalize is None:
            finalize = getattr(self._dataset, "consolidate", None)
        if finalize is None:
            raise DatasetDependencyError(
                "installed LeRobotDataset has neither finalize() nor consolidate()"
            )
        finalize()
        self._finalized = True
        if (
            self.config.hugging_face.upload_enabled
            and self.config.hugging_face.upload_policy == "on_session_end"
        ):
            self.upload()

    def upload(self) -> None:
        if not self.config.hugging_face.upload_enabled:
            return
        if not self._finalized:
            raise RuntimeError("dataset must be finalized before Hub upload")
        try:
            self._log_info(
                f"starting Hub upload of finalized committed data to {self.repo_id}; "
                "keep this process running until the completion message"
            )
            self._dataset.push_to_hub(private=self.config.hugging_face.private)
            self._log_info(f"uploaded committed dataset to {self.repo_id}")
        except Exception as exc:  # noqa: BLE001 - authentication/network boundary
            self._log_warn(
                f"Hub upload failed; complete local dataset remains at "
                f"{self.local_path}: {exc}"
            )

    def _prepare_local_path(self, *, overwrite: bool, resume: bool) -> None:
        if not self.local_path.exists():
            if resume:
                raise ExistingDatasetError(
                    f"cannot resume missing dataset: {self.local_path}"
                )
            return
        nonempty = any(self.local_path.iterdir()) if self.local_path.is_dir() else True
        if not nonempty:
            # LeRobotDataset.create() requires the dataset root itself not to
            # exist. Removing an empty directory is safe and deterministic.
            self.local_path.rmdir()
            return
        if overwrite:
            if self.local_path.is_symlink():
                self.local_path.unlink()
            elif self.local_path.is_dir():
                shutil.rmtree(self.local_path)
            else:
                self.local_path.unlink()
            return
        if not resume:
            raise ExistingDatasetError(
                f"dataset already exists at {self.local_path}; use --resume or --overwrite"
            )

    def _create_dataset(self) -> Any:
        create = getattr(self._dataset_class, "create", None)
        if create is None:
            raise DatasetDependencyError("installed LeRobotDataset has no create()")
        kwargs = {
            "repo_id": self.repo_id,
            "root": self.local_path,
            "fps": self.config.dataset.recording_fps,
            "robot_type": self.config.dataset.robot_type,
            "features": self.config.lerobot_features(),
            "use_videos": self.config.dataset.use_videos,
            "image_writer_processes": 0,
            "image_writer_threads": self.config.writer.image_writer_threads,
            "streaming_encoding": self.config.writer.streaming_encoding,
            "encoder_queue_maxsize": self.config.writer.frame_queue_size,
        }
        return create(**_supported_kwargs(create, kwargs))

    def _resume_dataset(self) -> Any:
        resume = getattr(self._dataset_class, "resume", None)
        kwargs = {
            "repo_id": self.repo_id,
            "root": self.local_path,
            "image_writer_processes": 0,
            "image_writer_threads": self.config.writer.image_writer_threads,
            "streaming_encoding": self.config.writer.streaming_encoding,
            "encoder_queue_maxsize": self.config.writer.frame_queue_size,
        }
        if resume is not None:
            return resume(**_supported_kwargs(resume, kwargs))

        # LeRobot 0.4.4 resumes recording through its public constructor rather
        # than a classmethod. Preflight every mandatory local metadata input so
        # LeRobot cannot interpret a partial local dataset as a cache miss and
        # fall through to a Hub lookup. A zero-episode writer has only
        # info.json and must be explicitly replaced rather than resumed.
        self._validate_local_resume_metadata()
        constructor_kwargs = {
            "repo_id": self.repo_id,
            "root": self.local_path,
            "streaming_encoding": self.config.writer.streaming_encoding,
            "encoder_queue_maxsize": self.config.writer.frame_queue_size,
        }
        dataset = self._dataset_class(
            **_supported_kwargs(self._dataset_class, constructor_kwargs)
        )
        if getattr(dataset, "episode_buffer", None) is None:
            create_buffer = getattr(dataset, "create_episode_buffer", None)
            if create_buffer is None:
                raise DatasetDependencyError(
                    "installed LeRobotDataset cannot initialize a resumed episode"
                )
            dataset.episode_buffer = create_buffer()
        start_image_writer = getattr(dataset, "start_image_writer", None)
        if start_image_writer is not None and self.config.writer.image_writer_threads:
            start_image_writer(
                num_processes=0,
                num_threads=self.config.writer.image_writer_threads,
            )
        return dataset

    def _validate_local_resume_metadata(self) -> None:
        meta = self.local_path / "meta"
        info_path = meta / "info.json"
        tasks_path = meta / "tasks.parquet"
        if not info_path.is_file():
            raise ExistingDatasetError(
                f"cannot resume an unfinalized or invalid local dataset: "
                f"missing {info_path}"
            )
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            episode_count = int(info.get("total_episodes", 0))
        except (OSError, ValueError, TypeError) as exc:
            raise ExistingDatasetError(
                f"cannot resume local dataset with invalid metadata "
                f"{info_path}: {exc}"
            ) from exc
        if episode_count <= 0:
            raise ExistingDatasetError(
                f"cannot resume local dataset with zero committed episodes at "
                f"{self.local_path}; use --overwrite to replace this empty "
                "dataset or configure a new dataset name"
            )
        if not tasks_path.is_file():
            raise ExistingDatasetError(
                f"cannot resume incomplete local dataset: missing {tasks_path}"
            )
        episode_files = tuple((meta / "episodes").glob("**/*.parquet"))
        if not episode_files:
            raise ExistingDatasetError(
                f"cannot resume incomplete local dataset: no episode metadata "
                f"under {meta / 'episodes'}"
            )

    def _finalize_upload_and_resume(self) -> None:
        finalize = getattr(self._dataset, "finalize", None)
        if finalize is None:
            raise DatasetDependencyError(
                "each_episode upload requires LeRobotDataset.finalize()"
            )
        finalize()
        self._finalized = True
        self.upload()
        self._dataset = self._resume_dataset()
        self._validate_resumed_dataset()
        self._finalized = False

    def _validate_resumed_dataset(self) -> None:
        features = getattr(self._dataset, "features", None)
        if not isinstance(features, dict):
            raise DatasetDependencyError(
                "resumed LeRobot dataset does not expose feature metadata"
            )
        for key, expected in self.config.lerobot_features().items():
            actual = features.get(key)
            if not isinstance(actual, dict):
                raise ExistingDatasetError(
                    f"resumed dataset is missing required feature {key}"
                )
            if actual.get("dtype") != expected["dtype"] or tuple(
                actual.get("shape", ())
            ) != tuple(expected["shape"]):
                raise ExistingDatasetError(
                    f"resumed dataset feature {key} has dtype/shape "
                    f"{actual.get('dtype')}/{actual.get('shape')}; expected "
                    f"{expected['dtype']}/{expected['shape']}"
                )
            if actual.get("names") != expected.get("names"):
                raise ExistingDatasetError(
                    f"resumed dataset feature {key} has a different ordering"
                )
        fps = getattr(self._dataset, "fps", None)
        if fps is not None and int(fps) != self.config.dataset.recording_fps:
            raise ExistingDatasetError(
                f"resumed dataset FPS {fps} differs from configured "
                f"{self.config.dataset.recording_fps}"
            )
        meta = getattr(self._dataset, "meta", None)
        robot_type = getattr(meta, "robot_type", None)
        if robot_type is not None and robot_type != self.config.dataset.robot_type:
            raise ExistingDatasetError(
                f"resumed dataset robot_type {robot_type} differs from configured "
                f"{self.config.dataset.robot_type}"
            )

    def _validate_frame(self, frame: dict[str, Any]) -> None:
        expected = set(self.config.camera_shapes) | {
            "observation.state",
            "action",
            "task",
        }
        if set(frame) != expected:
            raise ValueError(
                f"frame keys differ from fixed schema: got {sorted(frame)}, "
                f"expected {sorted(expected)}"
            )
        if frame["task"] != self.config.dataset.task_instruction:
            raise ValueError("frame task differs from configured task instruction")
        for key, shape in self.config.camera_shapes.items():
            image = frame[key]
            if not isinstance(image, np.ndarray):
                raise ValueError(f"{key} must be a numpy array")
            if image.dtype != np.uint8 or image.shape != shape:
                raise ValueError(
                    f"{key} expected uint8 {shape}, got {image.dtype} {image.shape}"
                )
        state = np.asarray(frame["observation.state"])
        action = np.asarray(frame["action"])
        if state.dtype != np.float32 or state.shape != (
            len(self.config.robot_features.state_names),
        ):
            raise ValueError("observation.state dtype or shape is invalid")
        if action.dtype != np.float32 or action.shape != (
            len(self.config.robot_features.action_names),
        ):
            raise ValueError("action dtype or shape is invalid")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError("state/action contains non-finite values")

    def _write_recording_metadata(self) -> None:
        meta = self.local_path / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        effective_config = deepcopy(self.config.source_data)
        effective_config["dataset"]["local_save_directory"] = str(
            self.config.dataset.local_save_directory
        )
        effective_config["hugging_face"]["upload_enabled"] = (
            self.config.hugging_face.upload_enabled
        )
        effective_config["hugging_face"]["repo_id"] = self.config.repo_id
        effective_config["episode_control"]["input_backend"] = (
            self.config.episode_control.input_backend
        )
        effective_config["episode_control"]["input_device"] = (
            self.config.episode_control.input_device
        )
        with (meta / "vega_recording_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(effective_config, stream, sort_keys=False)
        features = self.config.robot_features
        body_joint_names = list(features.body_joint_names)
        body_count = len(body_joint_names)
        state_segments = [
            {
                "indices": [0, body_count],
                "values": body_joint_names,
                "units": "rad",
                "reference_frame": "joint_local",
                "absolute_or_relative": "absolute",
                "measured_or_commanded": "measured",
                "quantity": "joint_position",
            }
        ]
        state_cursor = body_count
        for synergy in features.hand_synergies:
            state_segments.append(
                {
                    "indices": [state_cursor, state_cursor + 2],
                    "values": list(synergy.ratio_names),
                    "units": "unit_interval",
                    "reference_frame": f"{synergy.side}_hand_synergy",
                    "absolute_or_relative": "absolute_ratio",
                    "measured_or_commanded": (
                        "measured_derived_from_six_driver_positions"
                    ),
                    "quantity": "hand_open_close_and_thumb_opposition",
                    "range": [0.0, 1.0],
                    "source_joint_names": list(synergy.joint_names),
                }
            )
            state_cursor += 2
        if features.include_joint_velocities:
            state_segments.append(
                {
                    "indices": [state_cursor, state_cursor + body_count],
                    "values": body_joint_names,
                    "units": "rad/s",
                    "reference_frame": "joint_local",
                    "absolute_or_relative": "rate",
                    "measured_or_commanded": "measured",
                    "quantity": "joint_velocity",
                }
            )
            state_cursor += body_count
        state_base_start = state_cursor
        state_segments.extend(
            [
                {
                    "indices": [state_base_start, state_base_start + 2],
                    "values": ["base_vx", "base_vy"],
                    "units": "m/s",
                    "reference_frame": "base (+x forward, +y left)",
                    "absolute_or_relative": "rate",
                    "measured_or_commanded": "measured",
                    "quantity": "base_linear_velocity",
                },
                {
                    "indices": [state_base_start + 2, state_base_start + 3],
                    "values": ["base_wz"],
                    "units": "rad/s",
                    "reference_frame": "base (+z up)",
                    "absolute_or_relative": "rate",
                    "measured_or_commanded": "measured",
                    "quantity": "base_angular_velocity",
                },
            ]
        )
        action_segments = [
            {
                "indices": [0, body_count],
                "values": body_joint_names,
                "units": "rad",
                "reference_frame": "joint_local",
                "absolute_or_relative": "absolute",
                "measured_or_commanded": "commanded_after_bridge_clipping",
                "quantity": "joint_position_target",
            }
        ]
        action_cursor = body_count
        for synergy in features.hand_synergies:
            action_segments.append(
                {
                    "indices": [action_cursor, action_cursor + 2],
                    "values": list(synergy.ratio_names),
                    "units": "unit_interval",
                    "reference_frame": f"{synergy.side}_hand_synergy",
                    "absolute_or_relative": "absolute_ratio",
                    "measured_or_commanded": (
                        "reconstructed_from_post_bridge_applied_joint_targets"
                    ),
                    "quantity": "hand_open_close_and_thumb_opposition",
                    "range": [0.0, 1.0],
                    "source_joint_names": list(synergy.joint_names),
                    "maximum_ratio_disagreement": synergy.action_ratio_tolerance,
                }
            )
            action_cursor += 2
        action_segments.extend(
            [
                {
                    "indices": [action_cursor, action_cursor + 2],
                    "values": ["base_vx", "base_vy"],
                    "units": "m/s",
                    "reference_frame": "base (+x forward, +y left)",
                    "absolute_or_relative": "rate",
                    "measured_or_commanded": "commanded_to_vendor_api",
                    "quantity": "base_linear_velocity",
                },
                {
                    "indices": [action_cursor + 2, action_cursor + 3],
                    "values": ["base_wz"],
                    "units": "rad/s",
                    "reference_frame": "base (+z up)",
                    "absolute_or_relative": "rate",
                    "measured_or_commanded": "commanded_to_vendor_api",
                    "quantity": "base_angular_velocity",
                },
            ]
        )
        specification = {
            "robot_type": self.config.dataset.robot_type,
            "observation.state": {
                "dimension": len(features.state_names),
                "dtype": "float32",
                "ordered_names": list(features.state_names),
                "segments": state_segments,
            },
            "action": {
                "dimension": len(features.action_names),
                "dtype": "float32",
                "ordered_names": list(features.action_names),
                "segments": action_segments,
            },
            "hand_synergy_definitions": {
                synergy.side: {
                    "ordered_joint_names": list(synergy.joint_names),
                    "open_positions_rad": list(synergy.open_positions),
                    "closed_positions_rad": list(synergy.closed_positions),
                    "ordered_ratios": list(synergy.ratio_names),
                    "ratio_semantics": (
                        "open_close: 0=open, 1=closed; thumb_opposition: "
                        "0=unopposed, 1=opposed"
                    ),
                    "action_ratio_tolerance": synergy.action_ratio_tolerance,
                }
                for synergy in features.hand_synergies
            },
            "camera_features": {
                key: {
                    "shape": list(shape),
                    "dtype": "uint8",
                    "channel_order": "RGB",
                    "storage": "video" if self.config.dataset.use_videos else "image",
                    "placeholder": self.config.camera_configs[key].placeholder_enabled,
                }
                for key, shape in self.config.camera_shapes.items()
            },
            "coordinate_convention": "ROS base frame: +x forward, +y left, +z up",
            "source_document": "dex_vega_lerobot_recorder/docs/ROS_INTERFACE_DISCOVERY.md",
        }
        with (meta / "vega_feature_specification.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(specification, stream, indent=2, sort_keys=True)
            stream.write("\n")


def upload_existing_dataset(
    local_directory: str | Path,
    repo_id: str,
    *,
    private: bool,
    dataset_class: type | None = None,
) -> None:
    """Load a finalized local dataset and push it using standard HF auth."""
    root = Path(local_directory).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise ValueError(f"not a finalized LeRobot dataset directory: {root}")
    cls = dataset_class or _load_lerobot_dataset_class()
    dataset = cls(repo_id=repo_id, root=root)
    dataset.push_to_hub(private=private)


def _load_lerobot_dataset_class() -> type:
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DatasetDependencyError(
            "LeRobot is not installed. Install lerobot>=0.4 with its dataset/video "
            "dependencies before starting the recorder."
        ) from exc
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError as exc:
            raise DatasetDependencyError(
                f"LeRobot {version} does not expose LeRobotDataset"
            ) from exc
    return LeRobotDataset


def _supported_kwargs(callable_object: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_object)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _dataset_episode_count(dataset: Any) -> int:
    for attribute in ("num_episodes",):
        value = getattr(dataset, attribute, None)
        if value is not None:
            return int(value)
    meta = getattr(dataset, "meta", None)
    for attribute in ("total_episodes", "num_episodes"):
        value = getattr(meta, attribute, None)
        if value is not None:
            return int(value)
    return 0
