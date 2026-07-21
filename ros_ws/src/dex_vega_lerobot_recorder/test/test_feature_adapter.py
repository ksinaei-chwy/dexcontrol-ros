import numpy as np
import pytest

from dex_vega_lerobot_recorder.robot_feature_adapter import (
    RobotFeatureAdapter,
    SnapshotValidationError,
)


def test_state_and_action_are_reordered_by_explicit_joint_names():
    adapter = RobotFeatureAdapter(("j1", "j2"), include_joint_velocities=True)
    stamp = 1_000_000_000
    adapter.update_measured_joints(
        ("j2", "j1"), (20.0, 10.0), (2.0, 1.0), stamp
    )
    adapter.update_applied_joints(("j2", "j1"), (200.0, 100.0), stamp)
    adapter.update_measured_base((0.1, 0.2, 0.3), stamp)
    adapter.update_applied_base((0.4, 0.5, 0.6), stamp)
    result = adapter.snapshot(
        stamp + 10_000_000,
        maximum_state_age_seconds=0.1,
        maximum_action_age_seconds=0.1,
    )
    np.testing.assert_array_equal(
        result.state,
        np.array([10, 20, 1, 2, 0.1, 0.2, 0.3], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.action,
        np.array([100, 200, 0.4, 0.5, 0.6], dtype=np.float32),
    )


@pytest.mark.parametrize("stale_input", ["image-independent-state", "action"])
def test_stale_state_and_action_are_rejected(stale_input):
    adapter = RobotFeatureAdapter(("j1",))
    stamp = 1_000_000_000
    fresh = stamp + 150_000_000 if stale_input == "image-independent-state" else stamp
    adapter.update_measured_joints(("j1",), (1.0,), (0.0,), stamp)
    adapter.update_measured_base((0.0, 0.0, 0.0), stamp)
    adapter.update_applied_joints(("j1",), (1.0,), fresh)
    adapter.update_applied_base((0.0, 0.0, 0.0), fresh)
    with pytest.raises(SnapshotValidationError) as error:
        adapter.snapshot(
            stamp + 200_000_000,
            maximum_state_age_seconds=0.1,
            maximum_action_age_seconds=0.1,
        )
    assert error.value.stale


def test_missing_and_invalid_samples_are_rejected():
    adapter = RobotFeatureAdapter(("j1",))
    with pytest.raises(SnapshotValidationError):
        adapter.snapshot(
            1_000_000_000,
            maximum_state_age_seconds=0.1,
            maximum_action_age_seconds=0.1,
        )
    with pytest.raises(ValueError):
        adapter.update_measured_joints(("j1",), (1.0,), (), 1)


def test_invalid_or_partial_applied_message_clears_recent_action():
    adapter = RobotFeatureAdapter(("j1", "j2"))
    stamp = 1_000_000_000
    adapter.update_measured_joints(
        ("j1", "j2"), (1.0, 2.0), (0.1, 0.2), stamp
    )
    adapter.update_measured_base((0.0, 0.0, 0.0), stamp)
    adapter.update_applied_joints(("j1", "j2"), (3.0, 4.0), stamp)
    adapter.update_applied_base((0.0, 0.0, 0.0), stamp)

    adapter.update_applied_joints(("j1",), (5.0,), stamp + 1)
    with pytest.raises(SnapshotValidationError, match="required joint missing"):
        adapter.snapshot(
            stamp + 2,
            maximum_state_age_seconds=0.1,
            maximum_action_age_seconds=0.1,
        )

    adapter.update_applied_joints(("j1", "j2"), (3.0, 4.0), stamp + 3)
    with pytest.raises(ValueError):
        adapter.update_applied_base((np.nan, 0.0, 0.0), stamp + 3)
    with pytest.raises(SnapshotValidationError, match="missing applied base action"):
        adapter.snapshot(
            stamp + 4,
            maximum_state_age_seconds=0.1,
            maximum_action_age_seconds=0.1,
        )


def test_position_only_state_does_not_require_velocity_feedback():
    adapter = RobotFeatureAdapter(("j1",), include_joint_velocities=False)
    stamp = 1_000_000_000
    adapter.update_measured_joints(("j1",), (1.0,), (), stamp)
    adapter.update_measured_base((0.1, 0.2, 0.3), stamp)
    adapter.update_applied_joints(("j1",), (2.0,), stamp)
    adapter.update_applied_base((0.4, 0.5, 0.6), stamp)
    snapshot = adapter.snapshot(
        stamp + 1,
        maximum_state_age_seconds=0.1,
        maximum_action_age_seconds=0.1,
    )
    np.testing.assert_array_equal(
        snapshot.state, np.array([1.0, 0.1, 0.2, 0.3], dtype=np.float32)
    )
