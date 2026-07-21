from dataclasses import replace

import pytest

from dex_vega_lerobot_recorder.dataset_writer import (
    ExistingDatasetError,
    LeRobotDatasetWriter,
)

from helpers import config_for_directory
from test_no_upload_mode import FakeDataset, FakeDatasetClass


class ResumableFakeDatasetClass(FakeDatasetClass):
    @classmethod
    def resume(cls, **_kwargs):
        dataset = FakeDataset()
        config = config_for_directory(_kwargs["root"].parent, name=_kwargs["root"].name)
        dataset.features = config.lerobot_features()
        dataset.fps = config.dataset.recording_fps
        dataset.meta = type(
            "Meta", (), {"robot_type": config.dataset.robot_type}
        )()
        return dataset


def test_existing_dataset_requires_explicit_policy(tmp_path):
    config = config_for_directory(tmp_path)
    config.local_dataset_path.mkdir(parents=True)
    (config.local_dataset_path / "existing").write_text("data", encoding="utf-8")
    with pytest.raises(ExistingDatasetError):
        LeRobotDatasetWriter(config, dataset_class=FakeDatasetClass)


def test_empty_existing_directory_is_accepted_for_new_dataset(tmp_path):
    config = config_for_directory(tmp_path, name="empty")
    config.local_dataset_path.mkdir(parents=True)
    LeRobotDatasetWriter(config, dataset_class=FakeDatasetClass)
    assert (config.local_dataset_path / "meta").is_dir()


def test_resume_uses_supported_writer_and_overwrite_replaces(tmp_path):
    config = config_for_directory(tmp_path, name="resume")
    config.local_dataset_path.mkdir(parents=True)
    (config.local_dataset_path / "existing").write_text("data", encoding="utf-8")
    writer = LeRobotDatasetWriter(
        config, resume=True, dataset_class=ResumableFakeDatasetClass
    )
    assert writer.committed_episodes == 0

    overwrite_config = replace(
        config, dataset=replace(config.dataset, name="overwrite")
    )
    overwrite_config.local_dataset_path.mkdir(parents=True)
    marker = overwrite_config.local_dataset_path / "marker"
    marker.write_text("old", encoding="utf-8")
    LeRobotDatasetWriter(
        overwrite_config, overwrite=True, dataset_class=FakeDatasetClass
    )
    assert not marker.exists()


def test_resume_rejects_schema_mismatch(tmp_path):
    class WrongSchemaDatasetClass(ResumableFakeDatasetClass):
        @classmethod
        def resume(cls, **kwargs):
            dataset = super().resume(**kwargs)
            dataset.features["action"] = {
                "dtype": "float32",
                "shape": (34,),
                "names": {"action": ["wrong"] * 34},
            }
            return dataset

    config = config_for_directory(tmp_path, name="wrong_schema")
    config.local_dataset_path.mkdir(parents=True)
    (config.local_dataset_path / "existing").write_text("data", encoding="utf-8")
    with pytest.raises(ExistingDatasetError):
        LeRobotDatasetWriter(
            config, resume=True, dataset_class=WrongSchemaDatasetClass
        )
