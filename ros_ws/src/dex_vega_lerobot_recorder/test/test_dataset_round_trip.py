from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from dex_vega_lerobot_recorder.configuration import Resolution
from dex_vega_lerobot_recorder.dataset_writer import LeRobotDatasetWriter

from helpers import config_for_directory


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="LeRobot is not installed in this environment",
)


def test_finalized_video_dataset_reloads_with_all_features_and_task(
    tmp_path, monkeypatch
):
    datasets_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    from datasets import config as datasets_config

    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(datasets_cache))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = config_for_directory(tmp_path, name="round_trip")
    resolution = Resolution(width=32, height=24)
    config = replace(
        config,
        dataset=replace(config.dataset, recording_fps=10, use_videos=True),
        head_camera=replace(config.head_camera, resolution=resolution),
        left_wrist_camera=replace(
            config.left_wrist_camera, resolution=resolution
        ),
        right_wrist_camera=replace(
            config.right_wrist_camera, resolution=resolution
        ),
    )
    writer = LeRobotDatasetWriter(config)
    for index in range(3):
        head = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        black = np.zeros_like(head)
        writer.add_frame(
            {
                "observation.images.head": head,
                "observation.images.left_wrist": black,
                "observation.images.right_wrist": black,
                "observation.state": np.full(27, index, dtype=np.float32),
                "action": np.full(27, index, dtype=np.float32),
                "task": config.dataset.task_instruction,
            }
        )
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.local_dataset_path)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert set(config.camera_shapes).issubset(dataset.features)
    assert dataset.features["observation.state"]["shape"] == (27,)
    assert dataset.features["action"]["shape"] == (27,)
    state_names = dataset.features["observation.state"]["names"]["state"]
    action_names = dataset.features["action"]["names"]["action"]
    expected_hand_names = [
        "left_hand.open_close_ratio",
        "left_hand.thumb_opposition_ratio",
        "right_hand.open_close_ratio",
        "right_hand.thumb_opposition_ratio",
    ]
    assert state_names[20:24] == expected_hand_names
    assert action_names[20:24] == expected_hand_names
    assert not any("_ff_j1" in name or "_th_j" in name for name in action_names)
    frame = dataset[0]
    for key in config.camera_shapes:
        assert key in frame
    assert config.dataset.task_instruction in dataset.meta.tasks.index


def test_finalized_head_only_video_dataset_reloads(tmp_path, monkeypatch):
    datasets_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    from datasets import config as datasets_config

    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(datasets_cache))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = config_for_directory(tmp_path, name="head_only_round_trip")
    resolution = Resolution(width=32, height=20)
    config = replace(
        config,
        dataset=replace(config.dataset, recording_fps=30, use_videos=True),
        head_camera=replace(config.head_camera, resolution=resolution),
        left_wrist_camera=replace(config.left_wrist_camera, enabled=False),
        right_wrist_camera=replace(config.right_wrist_camera, enabled=False),
    )
    writer = LeRobotDatasetWriter(config)
    for index in range(3):
        writer.add_frame(
            {
                "observation.images.head": np.full(
                    (20, 32, 3), index * 20, dtype=np.uint8
                ),
                "observation.state": np.full(27, index, dtype=np.float32),
                "action": np.full(27, index, dtype=np.float32),
                "task": config.dataset.task_instruction,
            }
        )
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.local_dataset_path)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert set(dataset.features) >= set(config.lerobot_features())
    assert "observation.images.left_wrist" not in dataset.features
    assert "observation.images.right_wrist" not in dataset.features
    assert "observation.images.head" in dataset[0]


def test_finalized_video_dataset_safely_resumes_and_appends(tmp_path, monkeypatch):
    datasets_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    from datasets import config as datasets_config

    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(datasets_cache))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = config_for_directory(tmp_path, name="resume_round_trip")
    resolution = Resolution(width=32, height=24)
    config = replace(
        config,
        dataset=replace(config.dataset, recording_fps=10, use_videos=True),
        head_camera=replace(config.head_camera, resolution=resolution),
        left_wrist_camera=replace(
            config.left_wrist_camera, resolution=resolution
        ),
        right_wrist_camera=replace(
            config.right_wrist_camera, resolution=resolution
        ),
    )

    def add_episode(writer, value):
        for _index in range(2):
            image = np.full((24, 32, 3), value, dtype=np.uint8)
            writer.add_frame(
                {
                    "observation.images.head": image,
                    "observation.images.left_wrist": np.zeros_like(image),
                    "observation.images.right_wrist": np.zeros_like(image),
                    "observation.state": np.full(27, value, dtype=np.float32),
                    "action": np.full(27, value, dtype=np.float32),
                    "task": config.dataset.task_instruction,
                }
            )
        writer.save_episode()
        writer.finalize()

    first = LeRobotDatasetWriter(config)
    add_episode(first, 1)
    second = LeRobotDatasetWriter(config, resume=True)
    add_episode(second, 2)

    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.local_dataset_path)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 4
