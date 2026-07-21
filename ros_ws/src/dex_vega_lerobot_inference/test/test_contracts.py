import numpy as np

from dex_vega_lerobot_inference.contracts import (
    ACTION_DIMENSION,
    ACTION_NAMES,
    BODY_JOINT_NAMES,
    COMMAND_TOPICS,
    STATE_DIMENSION,
    STATE_NAMES,
    TASK,
)
from dex_vega_lerobot_inference.observation_adapter import validate_recorder_contract


def test_fixed_contract_lengths_and_order(recorder_config):
    validate_recorder_contract(recorder_config)
    assert STATE_DIMENSION == ACTION_DIMENSION == 27
    assert len(BODY_JOINT_NAMES) == 20
    assert tuple(recorder_config.robot_features.state_names) == STATE_NAMES
    assert tuple(recorder_config.robot_features.action_names) == ACTION_NAMES
    assert recorder_config.dataset.task_instruction == TASK


def test_contract_has_mixed_base_tail():
    assert STATE_NAMES[-3:] == ("base_vx", "base_vy", "base_wz")
    assert ACTION_NAMES[-3:] == ("base_vx", "base_vy", "base_wz")
    assert np.array_equal(np.arange(27)[24:27], [24, 25, 26])


def test_telemetry_topics_are_never_command_destinations():
    destinations = set(COMMAND_TOPICS.values())
    assert "/dexcontrol/applied_joint_commands" not in destinations
    assert "/dexcontrol/applied_base_twist" not in destinations
