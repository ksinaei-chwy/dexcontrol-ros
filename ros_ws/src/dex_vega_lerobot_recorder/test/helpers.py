from dataclasses import replace
from pathlib import Path

from dex_vega_lerobot_recorder.configuration import load_config
from dex_vega_lerobot_recorder.dataset_writer import CommitResult


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "vega_lerobot_recording.yaml"
)


class FakeEpisodeWriter:
    def __init__(self, *_args, **_kwargs):
        self.frames = []
        self.committed = []
        self.clear_count = 0
        self.finalize_count = 0

    @property
    def committed_episodes(self):
        return len(self.committed)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        index = len(self.committed)
        self.committed.append(list(self.frames))
        self.frames.clear()
        return CommitResult(index, Path("/tmp/fake_dataset"))

    def clear_episode_buffer(self):
        self.frames.clear()
        self.clear_count += 1

    def finalize(self):
        self.finalize_count += 1


def config_for_directory(directory: Path, *, name="test_dataset"):
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        dataset=replace(
            config.dataset,
            name=name,
            local_save_directory=directory,
        ),
    )
