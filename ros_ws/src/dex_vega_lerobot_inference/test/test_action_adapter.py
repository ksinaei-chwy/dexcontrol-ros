import numpy as np
import pytest

from dex_vega_lerobot_inference.action_adapter import (
    ActionAdapter,
    ActionValidationError,
    JointLimit,
    load_joint_limits_from_urdf,
)
from dex_vega_lerobot_recorder.hand_synergy import (
    HandSynergyError,
    expand_hand_synergy,
    reconstruct_hand_synergy,
)


def _adapter(recorder_config, vega_urdf_path):
    return ActionAdapter(
        recorder_config,
        load_joint_limits_from_urdf(vega_urdf_path),
    )


def test_hand_expansion_uses_recorder_definition(recorder_config, vega_urdf_path):
    adapter = _adapter(recorder_config, vega_urdf_path)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = (0.2, 0.3, 0.4, 0.5)
    action = state.copy()
    action[20:24] = state[20:24]
    adapted = adapter.adapt(action, state, cycle_seconds=1.0 / 30.0)
    for synergy, offset in zip(
        recorder_config.robot_features.hand_synergies, (0, 2), strict=True
    ):
        expected = expand_hand_synergy(
            synergy,
            action[20 + offset],
            action[21 + offset],
        )
        names, actual = adapted.component_positions[f"{synergy.side}_hand"]
        assert tuple(names) == synergy.joint_names
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_recorder_hand_tolerance_and_round_trip_are_unchanged(recorder_config):
    for synergy in recorder_config.robot_features.hand_synergies:
        assert synergy.action_ratio_tolerance == pytest.approx(0.02)
        positions = expand_hand_synergy(synergy, 0.4, 0.7)
        mapped = dict(zip(synergy.joint_names, positions, strict=True))
        assert reconstruct_hand_synergy(
            synergy,
            mapped,
            require_exact_action=True,
        ) == pytest.approx((0.4, 0.7))

        delta = 0.03 * (synergy.closed_positions[0] - synergy.open_positions[0])
        mapped[synergy.joint_names[0]] += delta
        with pytest.raises(HandSynergyError, match="ratio disagreement"):
            reconstruct_hand_synergy(synergy, mapped, require_exact_action=True)


def test_rate_and_base_limits_are_enforced(recorder_config, vega_urdf_path):
    adapter = _adapter(recorder_config, vega_urdf_path)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = 0.5
    action = state.copy()
    action[0] = 0.5
    action[6] = 0.5
    action[20:24] = 0.9
    action[24:27] = (1.0, -1.0, 2.0)
    adapted = adapter.adapt(action, state, cycle_seconds=0.1)
    assert adapted.rate_limited
    assert adapted.base_clamped
    assert adapted.component_positions["torso"][1][0] == pytest.approx(0.02)
    assert adapted.component_positions["left_arm"][1][0] == pytest.approx(0.02)
    assert adapted.base_twist.tolist() == pytest.approx([0.03, -0.03, 0.06])


def test_all_finite_postprocessed_hand_overshoot_is_clamped(
    recorder_config, vega_urdf_path
):
    adapter = _adapter(recorder_config, vega_urdf_path)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = (0.0, 1.0, 0.0, 1.0)
    action = state.copy()
    action[20:24] = (-10.0, 10.0, -0.021181, 1.5)

    adapted = adapter.adapt(action, state, cycle_seconds=1.0 / 30.0)

    assert adapted.hand_clamped
    left_positions = adapted.component_positions["left_hand"][1]
    right_positions = adapted.component_positions["right_hand"][1]
    expected_left = expand_hand_synergy(
        recorder_config.robot_features.hand_synergies[0], 0.0, 1.0
    )
    expected_right = expand_hand_synergy(
        recorder_config.robot_features.hand_synergies[1], 0.0, 1.0
    )
    np.testing.assert_allclose(left_positions, expected_left, atol=1e-12)
    np.testing.assert_allclose(right_positions, expected_right, atol=1e-12)
    np.testing.assert_allclose(adapted.policy_action[20:24], action[20:24])


def test_finite_body_joint_overshoot_is_clipped_to_urdf(
    recorder_config, vega_urdf_path
):
    limits = load_joint_limits_from_urdf(vega_urdf_path)
    adapter = ActionAdapter(recorder_config, limits)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = 0.5
    action = state.copy()
    action[0] = limits["torso_j1"].lower - 0.1

    adapted = adapter.adapt(action, state, cycle_seconds=1.0 / 30.0)

    assert adapted.joint_clamped
    assert adapted.joint_clamps["torso_j1"] == pytest.approx(
        (action[0], limits["torso_j1"].lower)
    )
    torso_names, torso_positions = adapted.component_positions["torso"]
    torso = dict(zip(torso_names, torso_positions, strict=True))
    assert torso["torso_j1"] == pytest.approx(limits["torso_j1"].lower)


def test_expanded_hand_boundary_roundoff_is_clipped_to_urdf(
    recorder_config, vega_urdf_path
):
    limits = load_joint_limits_from_urdf(vega_urdf_path)
    adapter = ActionAdapter(recorder_config, limits)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = (0.0, 0.0, 1.0, 0.0)

    adapted = adapter.adapt(state.copy(), state, cycle_seconds=1.0 / 30.0)

    raw, clipped = adapted.joint_clamps["R_ff_j1"]
    assert raw == -1.0946000000000002
    assert clipped == limits["R_ff_j1"].lower
    right_names, right_positions = adapted.component_positions["right_hand"]
    right_hand = dict(zip(right_names, right_positions, strict=True))
    assert right_hand["R_ff_j1"] == limits["R_ff_j1"].lower


def test_invalid_urdf_joint_limits_remain_a_hard_error(
    recorder_config, vega_urdf_path
):
    limits = load_joint_limits_from_urdf(vega_urdf_path)
    limits["torso_j1"] = JointLimit(float("nan"), 1.0)
    with pytest.raises(ValueError, match="invalid URDF position limits"):
        ActionAdapter(recorder_config, limits)


@pytest.mark.parametrize(
    "bad_action,match",
    [
        (np.full(27, np.nan), "NaN or Inf"),
        (np.zeros(26), "shape"),
    ],
)
def test_invalid_actions_are_rejected(
    recorder_config, vega_urdf_path, bad_action, match
):
    adapter = _adapter(recorder_config, vega_urdf_path)
    state = np.zeros(27)
    state[20:24] = 0.5
    with pytest.raises(ActionValidationError, match=match):
        adapter.adapt(bad_action, state, cycle_seconds=1.0 / 30.0)


def test_postprocessed_chunk_shape_and_finiteness():
    valid = ActionAdapter.validate_chunk(np.zeros((50, 27), dtype=np.float32))
    assert valid.shape == (50, 27)
    with pytest.raises(ActionValidationError, match="shape"):
        ActionAdapter.validate_chunk(np.zeros((50, 32)))
    invalid = np.zeros((50, 27))
    invalid[3, 4] = np.inf
    with pytest.raises(ActionValidationError, match="NaN or Inf"):
        ActionAdapter.validate_chunk(invalid)
