from dataclasses import replace

import pytest

from dex_vega_lerobot_recorder.configuration import (
    ConfigurationError,
    load_config,
    validate_config,
)

from helpers import CONFIG_PATH


def test_default_configuration_has_fixed_dimensions_and_camera_names():
    config = load_config(CONFIG_PATH)
    assert len(config.robot_features.action_names) == 27
    assert len(config.robot_features.state_names) == 27
    assert config.topics.joint_states == "/joint_states"
    assert config.robot_features.include_joint_velocities is False
    assert config.head_camera.transport == "zenoh"
    assert config.head_camera.topic == "sensors/head_camera/left_rgb"
    assert config.validation.maximum_receive_age_seconds == 0.1
    assert config.validation.maximum_capture_age_seconds == 0.3
    assert config.validation.maximum_transport_delay_seconds == 0.25
    assert set(config.camera_shapes) == {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }


def test_head_only_camera_configuration_is_valid():
    config = load_config(CONFIG_PATH)
    head_only = replace(
        config,
        left_wrist_camera=replace(config.left_wrist_camera, enabled=False),
        right_wrist_camera=replace(config.right_wrist_camera, enabled=False),
    )
    validate_config(head_only)
    assert set(head_only.camera_shapes) == {"observation.images.head"}
    assert set(head_only.lerobot_features()) == {
        "observation.images.head",
        "observation.state",
        "action",
    }


def test_vega_state_and_action_feature_order_is_exact():
    config = load_config(CONFIG_PATH)
    action = config.robot_features.action_names
    assert action[:6] == (
        "torso_j1",
        "torso_j2",
        "torso_j3",
        "head_j1",
        "head_j2",
        "head_j3",
    )
    assert action[6:13] == tuple(f"L_arm_j{index}" for index in range(1, 8))
    assert action[13:20] == tuple(f"R_arm_j{index}" for index in range(1, 8))
    assert action[20:24] == (
        "left_hand.open_close_ratio",
        "left_hand.thumb_opposition_ratio",
        "right_hand.open_close_ratio",
        "right_hand.thumb_opposition_ratio",
    )
    assert action[24:] == ("base_vx", "base_vy", "base_wz")
    state = config.robot_features.state_names
    assert state[:20] == tuple(f"{name}.position" for name in action[:20])
    assert state[20:] == action[20:]


def test_duplicate_joint_or_control_names_are_rejected():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ConfigurationError):
        validate_config(
            replace(
                config,
                robot_features=replace(
                    config.robot_features, joint_names=("same", "same")
                ),
            )
        )
    with pytest.raises(ConfigurationError):
        validate_config(
            replace(
                config,
                episode_control=replace(
                    config.episode_control, start_key="b"
                ),
            )
        )


def test_no_hf_upload_override_wins_over_yaml():
    config = load_config(CONFIG_PATH)
    enabled = replace(
        config,
        hugging_face=replace(config.hugging_face, upload_enabled=True),
    )
    assert enabled.with_overrides(no_hf_upload=True).hugging_face.upload_enabled is False


def test_upload_requires_nonlocal_namespace_or_explicit_repo_id():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ConfigurationError):
        validate_config(
            replace(
                config,
                hugging_face=replace(
                    config.hugging_face,
                    upload_enabled=True,
                    namespace="local",
                    repo_id="",
                ),
            )
        )
