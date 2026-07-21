import numpy as np
import pytest

from dex_vega_lerobot_recorder.episode_controller import (
    EpisodeController,
    EpisodeState,
    EpisodeValidationError,
    InvalidTransition,
)

from helpers import FakeEpisodeWriter


def make_controller(minimum_frames=1, minimum_duration=0.0):
    current = [10.0]

    def clock():
        return current[0]

    writer = FakeEpisodeWriter()
    controller = EpisodeController(
        writer,
        minimum_frames=minimum_frames,
        minimum_duration_seconds=minimum_duration,
        monotonic=clock,
    )
    return controller, writer, current


def test_stop_does_not_commit_and_save_commits_exactly_one():
    controller, writer, clock = make_controller()
    controller.start_episode()
    controller.add_frame({"value": np.array([1])})
    clock[0] += 1.0
    summary = controller.stop_episode()
    assert summary.state is EpisodeState.REVIEW_PENDING
    assert writer.committed_episodes == 0

    result = controller.save_episode()
    assert result.episode_index == 0
    assert writer.committed_episodes == 1
    assert controller.state is EpisodeState.IDLE


@pytest.mark.parametrize("stop_first", [False, True])
def test_discard_commits_zero_and_clears_pending_data(stop_first):
    controller, writer, _clock = make_controller()
    controller.start_episode()
    controller.add_frame({"value": 1})
    if stop_first:
        controller.stop_episode()
    controller.discard_episode()
    assert writer.committed_episodes == 0
    assert writer.frames == []
    assert writer.clear_count == 1
    assert controller.state is EpisodeState.IDLE


def test_invalid_transitions_are_rejected():
    controller, _writer, _clock = make_controller()
    with pytest.raises(InvalidTransition):
        controller.stop_episode()
    with pytest.raises(InvalidTransition):
        controller.save_episode()
    with pytest.raises(InvalidTransition):
        controller.discard_episode()
    controller.start_episode()
    with pytest.raises(InvalidTransition):
        controller.start_episode()


def test_empty_or_short_episode_cannot_be_saved():
    controller, writer, clock = make_controller(
        minimum_frames=2, minimum_duration=0.5
    )
    controller.start_episode()
    controller.add_frame({"value": 1})
    clock[0] += 0.1
    controller.stop_episode()
    with pytest.raises(EpisodeValidationError):
        controller.save_episode()
    assert writer.committed_episodes == 0
    assert controller.state is EpisodeState.REVIEW_PENDING


def test_shutdown_discards_pending_by_default_and_finalizes():
    controller, writer, _clock = make_controller()
    controller.start_episode()
    controller.add_frame({"value": 1})
    resolution = controller.shutdown(autosave=False)
    assert "discarded unsaved" in resolution
    assert writer.committed_episodes == 0
    assert writer.clear_count == 1
    assert writer.finalize_count == 1
