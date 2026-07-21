from types import SimpleNamespace

import numpy as np
import pytest

from dex_vega_lerobot_inference.observation_adapter import (
    CameraSample,
    ObservationAdapter,
    ObservationValidationError,
    image_message_to_rgb,
)
from dex_vega_lerobot_recorder.hand_synergy import expand_hand_synergy
from dex_vega_lerobot_recorder.robot_feature_adapter import RobotFeatureAdapter


def _joint_values(config, left=(0.25, 0.75), right=(0.6, 0.1)):
    values = {name: 0.0 for name in config.robot_features.body_joint_names}
    for synergy, ratios in zip(config.robot_features.hand_synergies, (left, right), strict=True):
        values.update(
            zip(
                synergy.joint_names,
                expand_hand_synergy(synergy, ratios[0], ratios[1]),
                strict=True,
            )
        )
    names = config.robot_features.joint_names
    return names, [values[name] for name in names]


def test_observation_matches_recorder_adapter(recorder_config):
    names, positions = _joint_values(recorder_config)
    stamp_ns = 9_950_000_000
    now_ns = 10_000_000_000
    base = (0.03, -0.02, 0.07)

    reference = RobotFeatureAdapter(
        recorder_config.robot_features.joint_names,
        include_joint_velocities=False,
        hand_synergies=recorder_config.robot_features.hand_synergies,
    )
    reference.update_measured_joints(names, positions, (), stamp_ns)
    reference.update_applied_joints(names, positions, stamp_ns)
    reference.update_measured_base(base, stamp_ns)
    reference.update_applied_base(base, stamp_ns)
    expected = reference.snapshot(
        now_ns,
        maximum_state_age_seconds=0.1,
        maximum_action_age_seconds=0.1,
    )

    adapter = ObservationAdapter(recorder_config)
    adapter.update_measured_joints(names, positions, stamp_ns)
    adapter.update_measured_base(base, stamp_ns)
    snapshot = adapter.snapshot(
        CameraSample(
            np.zeros((480, 640, 3), dtype=np.uint8),
            stamp_ns,
            stamp_ns,
        ),
        now_ns,
        maximum_state_age_seconds=0.1,
        maximum_receive_age_seconds=0.1,
        maximum_capture_age_seconds=0.3,
        maximum_transport_delay_seconds=0.25,
        maximum_synchronization_skew_seconds=0.1,
    )
    np.testing.assert_allclose(snapshot.state, expected.state, atol=1e-7)
    assert snapshot.state.shape == (27,)
    assert snapshot.rgb.flags.c_contiguous


def test_missing_stale_and_duplicate_timestamps_fail(recorder_config):
    adapter = ObservationAdapter(recorder_config)
    camera = CameraSample(
        np.zeros((480, 640, 3), dtype=np.uint8),
        9_950_000_000,
        9_950_000_000,
    )
    kwargs = dict(
        maximum_state_age_seconds=0.1,
        maximum_receive_age_seconds=0.1,
        maximum_capture_age_seconds=0.3,
        maximum_transport_delay_seconds=0.25,
        maximum_synchronization_skew_seconds=0.1,
    )
    with pytest.raises(ObservationValidationError, match="missing"):
        adapter.snapshot(camera, 10_000_000_000, **kwargs)

    names, positions = _joint_values(recorder_config)
    adapter.update_measured_joints(names, positions, 9_000_000_000)
    adapter.update_measured_base((0.0, 0.0, 0.0), 9_000_000_000)
    with pytest.raises(ObservationValidationError, match="stale measured state"):
        adapter.snapshot(camera, 10_000_000_000, **kwargs)

    fresh = ObservationAdapter(recorder_config)
    fresh.update_measured_joints(names, positions, 9_950_000_000)
    fresh.update_measured_base((0.0, 0.0, 0.0), 9_950_000_000)
    fresh.snapshot(camera, 10_000_000_000, **kwargs)
    with pytest.raises(ObservationValidationError, match="duplicate"):
        fresh.snapshot(camera, 10_000_000_000, **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        fresh.update_measured_base((0.0, 0.0, 0.0), 9_950_000_000)


def test_rgb_and_bgr_replay_conversion():
    rgb_pixel = bytes((1, 2, 3))
    rgb = image_message_to_rgb(
        SimpleNamespace(height=1, width=1, step=3, encoding="rgb8", data=rgb_pixel)
    )
    bgr = image_message_to_rgb(
        SimpleNamespace(height=1, width=1, step=3, encoding="bgr8", data=rgb_pixel)
    )
    assert rgb.tolist() == [[[1, 2, 3]]]
    assert bgr.tolist() == [[[3, 2, 1]]]
    assert rgb.flags.c_contiguous and bgr.flags.c_contiguous


def test_invalid_image_encoding_is_rejected():
    with pytest.raises(ObservationValidationError, match="unsupported"):
        image_message_to_rgb(
            SimpleNamespace(height=1, width=1, step=3, encoding="mono8", data=b"123")
        )
