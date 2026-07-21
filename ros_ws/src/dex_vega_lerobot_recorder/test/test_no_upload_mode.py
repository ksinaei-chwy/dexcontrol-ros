from dataclasses import replace
import importlib.util
import json

import numpy as np
import pytest
import yaml

from dex_vega_lerobot_recorder.dataset_writer import (
    ExistingDatasetError,
    LeRobotDatasetWriter,
)
from dex_vega_lerobot_recorder.recorder_node import _parse_arguments

from helpers import config_for_directory


class FakeDataset:
    push_calls = 0

    def __init__(self):
        self.num_episodes = 0
        self.frames = []
        self.writer = type("Writer", (), {"episode_buffer": {"size": 0}})()

    def add_frame(self, frame):
        self.frames.append(frame)
        self.writer.episode_buffer["size"] += 1

    def save_episode(self):
        self.num_episodes += 1
        self.frames.clear()
        self.writer.episode_buffer["size"] = 0

    def clear_episode_buffer(self, delete_images=True):
        assert delete_images
        self.frames.clear()
        self.writer.episode_buffer["size"] = 0

    def finalize(self):
        pass

    def push_to_hub(self, **_kwargs):
        type(self).push_calls += 1


class FakeDatasetClass:
    @classmethod
    def create(cls, **_kwargs):
        return FakeDataset()


class UploadTrackingDataset(FakeDataset):
    finalize_calls = 0

    def finalize(self):
        type(self).finalize_calls += 1


class UploadTrackingDatasetClass:
    create_kwargs = {}

    @classmethod
    def create(cls, **kwargs):
        cls.create_kwargs = kwargs
        return cls._dataset()

    @classmethod
    def resume(cls, **_kwargs):
        return cls._dataset()

    @classmethod
    def _dataset(cls):
        dataset = UploadTrackingDataset()
        dataset.features = cls.create_kwargs["features"]
        dataset.fps = cls.create_kwargs["fps"]
        dataset.meta = type(
            "Meta", (), {"robot_type": cls.create_kwargs["robot_type"]}
        )()
        return dataset


def test_local_only_mode_never_calls_hub(tmp_path):
    FakeDataset.push_calls = 0
    config = config_for_directory(tmp_path)
    config = replace(
        config,
        hugging_face=replace(config.hugging_face, upload_enabled=False),
    )
    writer = LeRobotDatasetWriter(config, dataset_class=FakeDatasetClass)
    shape = config.head_camera.resolution.shape
    frame = {
        "observation.images.head": np.zeros(shape, dtype=np.uint8),
        "observation.images.left_wrist": np.zeros(shape, dtype=np.uint8),
        "observation.images.right_wrist": np.zeros(shape, dtype=np.uint8),
        "observation.state": np.zeros(27, dtype=np.float32),
        "action": np.zeros(27, dtype=np.float32),
        "task": config.dataset.task_instruction,
    }
    writer.add_frame(frame)
    writer.save_episode()
    writer.finalize()
    writer.upload()
    assert FakeDataset.push_calls == 0


def test_cli_no_upload_flag_is_consumed_and_true():
    options, remaining = _parse_arguments(
        ["--no-hf-upload", "--ros-args", "-r", "__node:=test"]
    )
    assert options.no_hf_upload
    assert remaining == ["--ros-args", "-r", "__node:=test"]


@pytest.mark.parametrize(
    ("policy", "pushes_after_save", "pushes_after_finalize"),
    [
        ("each_episode", 1, 1),
        ("on_session_end", 0, 1),
        ("manual", 0, 0),
    ],
)
def test_upload_policy_only_pushes_finalized_committed_data(
    tmp_path, policy, pushes_after_save, pushes_after_finalize
):
    UploadTrackingDataset.push_calls = 0
    UploadTrackingDataset.finalize_calls = 0
    config = config_for_directory(tmp_path, name=f"upload_{policy}")
    config = replace(
        config,
        hugging_face=replace(
            config.hugging_face,
            upload_enabled=True,
            namespace="example",
            repo_id=f"example/upload_{policy}",
            upload_policy=policy,
        ),
    )
    writer = LeRobotDatasetWriter(
        config, dataset_class=UploadTrackingDatasetClass
    )
    shape = config.head_camera.resolution.shape
    writer.add_frame(
        {
            "observation.images.head": np.zeros(shape, dtype=np.uint8),
            "observation.images.left_wrist": np.zeros(shape, dtype=np.uint8),
            "observation.images.right_wrist": np.zeros(shape, dtype=np.uint8),
            "observation.state": np.zeros(27, dtype=np.float32),
            "action": np.zeros(27, dtype=np.float32),
            "task": config.dataset.task_instruction,
        }
    )
    writer.save_episode()
    assert UploadTrackingDataset.push_calls == pushes_after_save
    writer.finalize()
    assert UploadTrackingDataset.push_calls == pushes_after_finalize


def test_effective_no_upload_override_is_saved_in_dataset_metadata(tmp_path):
    config = config_for_directory(tmp_path, name="effective_config")
    config = replace(
        config,
        hugging_face=replace(
            config.hugging_face,
            upload_enabled=False,
            namespace="example",
            repo_id="example/effective_config",
        ),
    )
    LeRobotDatasetWriter(config, dataset_class=FakeDatasetClass)
    saved = yaml.safe_load(
        (config.local_dataset_path / "meta" / "vega_recording_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert saved["hugging_face"]["upload_enabled"] is False
    assert saved["hugging_face"]["repo_id"] == "example/effective_config"
    specification = json.loads(
        (config.local_dataset_path / "meta" / "vega_feature_specification.json").read_text(
            encoding="utf-8"
        )
    )
    assert specification["observation.state"]["dimension"] == 27
    assert specification["action"]["dimension"] == 27
    assert specification["action"]["ordered_names"][20:24] == [
        "left_hand.open_close_ratio",
        "left_hand.thumb_opposition_ratio",
        "right_hand.open_close_ratio",
        "right_hand.thumb_opposition_ratio",
    ]
    assert set(specification["hand_synergy_definitions"]) == {"left", "right"}


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="LeRobot is not installed in this environment",
)
def test_real_local_writer_does_not_construct_hugging_face_api(
    tmp_path, monkeypatch
):
    import huggingface_hub
    from lerobot.datasets import lerobot_dataset, utils as dataset_utils

    class ForbiddenHfApi:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("HfApi must not be constructed in local-only mode")

    monkeypatch.setattr(huggingface_hub, "HfApi", ForbiddenHfApi)
    monkeypatch.setattr(lerobot_dataset, "HfApi", ForbiddenHfApi)
    monkeypatch.setattr(dataset_utils, "HfApi", ForbiddenHfApi)
    config = config_for_directory(tmp_path, name="real_no_hub")
    writer = LeRobotDatasetWriter(config)
    writer.finalize()
    with pytest.raises(ExistingDatasetError, match="zero committed episodes"):
        LeRobotDatasetWriter(config, resume=True)
