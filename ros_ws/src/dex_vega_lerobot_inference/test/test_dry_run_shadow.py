from collections import deque
from types import SimpleNamespace

import numpy as np

from dex_vega_lerobot_inference.inference_node import (
    InferenceNode,
    _PredictionEnvelope,
)


def _shadow_fixture(adapted_action):
    parameters = {
        "action_wait_timeout_seconds": 0.75,
        "control_frequency_hz": 30.0,
    }
    fake = SimpleNamespace(
        _latest_snapshot=SimpleNamespace(state=np.zeros(27, dtype=np.float32)),
        _last_action_available_ns=1,
        _shadow_action_errors=0,
        _shadow_actions_evaluated=0,
        _shadow_base_clamped_actions=0,
        _shadow_hand_clamped_actions=0,
        _shadow_joint_clamped_actions=0,
        _shadow_queue_starvations=0,
        _shadow_rate_limited_actions=0,
        _shadow_stale_observation_actions=0,
        _shadow_stale_queue_actions=0,
        _shadow_wait_timeout_recorded=False,
        _last_shadow_error="",
        _last_shadow_hand_clamp=None,
        _last_shadow_joint_clamp=None,
        _action_adapter=SimpleNamespace(adapt=lambda *_args, **_kwargs: adapted_action),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    return fake


def test_dry_run_shadow_adapts_without_any_publication_interface():
    raw = np.zeros(27, dtype=np.float64)
    raw[20] = 1.2
    adapted = SimpleNamespace(
        policy_action=raw,
        rate_limited=True,
        hand_clamped=True,
        joint_clamped=False,
        joint_clamps={},
        base_clamped=True,
    )
    fake = _shadow_fixture(adapted)
    fake._pop_next_action = lambda: (object(), raw)
    fake._prediction_age_gate_failure = lambda *_args: ""

    InferenceNode._evaluate_next_shadow_action(fake, 10)

    assert fake._shadow_actions_evaluated == 1
    assert fake._shadow_rate_limited_actions == 1
    assert fake._shadow_hand_clamped_actions == 1
    assert fake._shadow_base_clamped_actions == 1
    assert fake._last_shadow_hand_clamp["left_hand.open_close_ratio"] == {
        "raw": 1.2,
        "clamped": 1.0,
    }
    assert not hasattr(fake, "_command_publishers")
    assert not hasattr(fake, "_publish_action")


def test_dry_run_shadow_counts_each_queue_starvation_once():
    fake = _shadow_fixture(None)
    fake._pop_next_action = lambda: (None, None)

    InferenceNode._evaluate_next_shadow_action(fake, 1_000_000_001)
    InferenceNode._evaluate_next_shadow_action(fake, 2_000_000_001)

    assert fake._shadow_queue_starvations == 1
    assert fake._last_shadow_error == "no fresh action chunk available"


def test_prediction_queue_pop_removes_only_the_configured_shadow_prefix():
    fake = SimpleNamespace()
    import threading

    fake._prediction_lock = threading.Lock()
    fake._prediction = _PredictionEnvelope(
        observation=SimpleNamespace(),
        prediction=SimpleNamespace(),
        actions=deque(
            np.full(27, value, dtype=np.float64) for value in (1.0, 2.0)
        ),
        received_monotonic_ns=1,
    )

    first_envelope, first = InferenceNode._pop_next_action(fake)
    second_envelope, second = InferenceNode._pop_next_action(fake)

    assert first_envelope is second_envelope
    np.testing.assert_array_equal(first, np.ones(27))
    np.testing.assert_array_equal(second, np.full(27, 2.0))
    assert fake._prediction is None
