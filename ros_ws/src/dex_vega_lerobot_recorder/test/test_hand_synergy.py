import numpy as np
import pytest

from dex_vega_lerobot_recorder.configuration import load_config
from dex_vega_lerobot_recorder.hand_synergy import (
    HandSynergyError,
    expand_hand_synergy,
    reconstruct_hand_synergy,
)
from dex_vega_lerobot_recorder.robot_feature_adapter import (
    RobotFeatureAdapter,
    SnapshotValidationError,
)

from helpers import CONFIG_PATH


def test_two_dof_hand_expansion_and_post_bridge_reconstruction_round_trip():
    synergy = load_config(CONFIG_PATH).robot_features.hand_synergies[0]
    expanded = expand_hand_synergy(synergy, 0.35, 0.75)
    positions = dict(zip(synergy.joint_names, expanded))
    reconstructed = reconstruct_hand_synergy(
        synergy, positions, require_exact_action=True
    )
    np.testing.assert_allclose(reconstructed, [0.35, 0.75], atol=1.0e-7)


def test_post_bridge_action_rejects_off_synergy_finger_targets():
    synergy = load_config(CONFIG_PATH).robot_features.hand_synergies[0]
    expanded = expand_hand_synergy(synergy, 0.5, 0.5)
    expanded[1] += 0.1 * (
        synergy.closed_positions[1] - synergy.open_positions[1]
    )
    with pytest.raises(HandSynergyError, match="off the two-DoF synergy"):
        reconstruct_hand_synergy(
            synergy,
            dict(zip(synergy.joint_names, expanded)),
            require_exact_action=True,
        )


def test_compact_adapter_orders_two_ratios_per_hand_in_state_and_action():
    features = load_config(CONFIG_PATH).robot_features
    adapter = RobotFeatureAdapter(
        features.joint_names,
        include_joint_velocities=False,
        hand_synergies=features.hand_synergies,
    )
    measured = {name: float(index) for index, name in enumerate(features.joint_names)}
    applied = {name: float(index + 100) for index, name in enumerate(features.joint_names)}
    for synergy, state_ratios, action_ratios in zip(
        features.hand_synergies,
        ((0.1, 0.2), (0.3, 0.4)),
        ((0.5, 0.6), (0.7, 0.8)),
    ):
        measured.update(
            zip(synergy.joint_names, expand_hand_synergy(synergy, *state_ratios))
        )
        applied.update(
            zip(synergy.joint_names, expand_hand_synergy(synergy, *action_ratios))
        )

    stamp = 1_000_000_000
    adapter.update_measured_joints(
        features.joint_names,
        [measured[name] for name in features.joint_names],
        (),
        stamp,
    )
    adapter.update_applied_joints(
        features.joint_names,
        [applied[name] for name in features.joint_names],
        stamp,
    )
    adapter.update_measured_base((0.1, 0.2, 0.3), stamp)
    adapter.update_applied_base((0.4, 0.5, 0.6), stamp)
    snapshot = adapter.snapshot(
        stamp + 1,
        maximum_state_age_seconds=0.1,
        maximum_action_age_seconds=0.1,
    )

    assert snapshot.state.shape == (27,)
    assert snapshot.action.shape == (27,)
    np.testing.assert_allclose(snapshot.state[20:24], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(snapshot.action[20:24], [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(snapshot.state[-3:], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(snapshot.action[-3:], [0.4, 0.5, 0.6])


def test_compact_adapter_drops_off_synergy_applied_action():
    features = load_config(CONFIG_PATH).robot_features
    adapter = RobotFeatureAdapter(
        features.joint_names,
        include_joint_velocities=False,
        hand_synergies=features.hand_synergies,
    )
    positions = {name: 0.0 for name in features.joint_names}
    applied = {name: 0.0 for name in features.joint_names}
    for synergy in features.hand_synergies:
        measured_values = expand_hand_synergy(synergy, 0.2, 0.3)
        applied_values = expand_hand_synergy(synergy, 0.4, 0.5)
        positions.update(zip(synergy.joint_names, measured_values))
        applied.update(zip(synergy.joint_names, applied_values))
    first = features.hand_synergies[0]
    applied[first.joint_names[2]] += 0.1 * (
        first.closed_positions[2] - first.open_positions[2]
    )

    stamp = 1_000_000_000
    adapter.update_measured_joints(
        features.joint_names,
        [positions[name] for name in features.joint_names],
        (),
        stamp,
    )
    adapter.update_applied_joints(
        features.joint_names,
        [applied[name] for name in features.joint_names],
        stamp,
    )
    adapter.update_measured_base((0.0, 0.0, 0.0), stamp)
    adapter.update_applied_base((0.0, 0.0, 0.0), stamp)
    with pytest.raises(SnapshotValidationError, match="off the two-DoF synergy"):
        adapter.snapshot(
            stamp + 1,
            maximum_state_age_seconds=0.1,
            maximum_action_age_seconds=0.1,
        )
